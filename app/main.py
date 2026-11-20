import hashlib
import json
import logging
import re
import sqlite3
import time
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from queue import Empty, Queue
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.agents.graph import audit_graph
from app.agents.intent import classify_intent
from app.agents.nodes.intent_router import classify_semantic_intent
from app.gateway.client import get_extraction_llm, sanitize_error_message, use_user_api_keys
from app.ingestion.processor import (
    PDF_INDEX_VERSION,
    _compute_doc_id,
    ingest_pdf_document,
)
from app.agents.nodes.scope_router import resolve_audit_scope
from app.config import get_settings
from app.guardrails.rails import YOUTUBE_URL_RE, sanitize_transcript_text
from app.services.db.database import init_db
from app.services.db.repository import (
    count_completed_llm_turns,
    get_document,
    get_session_messages,
    get_system_metrics,
    log_audit_event,
    list_approvals,
    list_documents,
    list_recent_sessions,
    resolve_approval,
    save_chat_message,
    save_feedback,
)
from app.services.job_tracker import get_job_state, set_job_stage

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as exc:
        logger.warning("DB init error: %s", exc)
    yield


app = FastAPI(title="Syllabus & Book Gap Finder API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MAX_GUEST_CALLS = 3  # Strict 3-call budget (covers gap audits + Q&A) per guest unless BYOK keys are supplied
MAX_FREE_LLM_TURNS = MAX_GUEST_CALLS
WHOLE_DOC_PHRASES = {
    "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead", "proceed",
    "whole book", "entire book", "complete book", "full book",
    "whole document", "entire document", "complete document",
    "whole pdf", "entire pdf", "complete pdf", "full pdf",
    "all", "everything", "all chapters",
}
ACTIVE_JOBS: Dict[str, str] = {}
ACTIVE_JOB_IDS: Dict[str, str] = {}
ACTIVE_JOB_RESULTS: Dict[str, Dict[str, Any]] = {}
ACTIVE_JOB_USERS: Dict[str, str] = {}
ACTIVE_SYNC_RESERVATIONS: Dict[str, str] = {}
ACTIVE_STREAM_SESSIONS: set[str] = set()
ACTIVE_JOBS_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


# --- Request Schemas ---
class ChatTurnRequest(BaseModel):
    session_id: str
    user_id: str = "default_user"
    doc_id: Optional[str] = None
    message: str
    video_url: Optional[str] = None
    prior_clarification_asked: bool = False
    user_gemini_key: Optional[str] = None
    user_groq_key: Optional[str] = None
    raw_transcript: Optional[str] = None


class TranscribeRequest(BaseModel):
    video_url: str
    force_refresh: bool = False


class ApprovalDecisionRequest(BaseModel):
    decision: str  # "confirmed_gap" | "marked_covered"
    reviewer: str = "teacher"


class FeedbackRequest(BaseModel):
    message_id: str
    user_id: str = "default_user"
    rating: int  # 1 or -1
    comment: str = ""


def _extract_url_from_text(text: str) -> Optional[str]:
    for token in text.split():
        cleaned = token.strip("()[]<>.,")
        if YOUTUBE_URL_RE.match(cleaned):
            return cleaned
    return None


def _strip_mode_tag(text: str) -> str:
    return re.sub(r"\[Mode:[^\]]+\]\s*", "", text).strip()


def _is_greeting(text: str) -> bool:
    return bool(
        re.match(
            r"^(hi+|hello+|hey+|yo+|good\s+(morning|afternoon|evening)|help|start)[\s!?.]*$",
            text,
            re.I,
        )
    )


def _is_video_audit_turn(req: ChatTurnRequest) -> bool:
    clean_msg = _strip_mode_tag(req.message)
    if _is_greeting(clean_msg) and not _extract_url_from_text(clean_msg):
        return False
    if _extract_url_from_text(clean_msg):
        return bool(req.doc_id)
    return bool(req.doc_id and req.prior_clarification_asked and req.video_url)


def _classify_intent(
    message: str, has_active_doc: bool, prior_clarification: bool
) -> str:
    """Apply the shared audit > project guide > PDF Q&A routing priority."""
    return classify_semantic_intent(
        query=message,
        has_active_doc=has_active_doc,
        prior_clarification_asked=prior_clarification,
    )


def _has_byok(req: ChatTurnRequest) -> bool:
    return bool((req.user_gemini_key or "").strip() or (req.user_groq_key or "").strip())


def _request_may_use_llm(req: ChatTurnRequest) -> bool:
    clean_msg = _strip_mode_tag(sanitize_transcript_text(req.message))
    intent = _classify_intent(
        message=clean_msg,
        has_active_doc=bool(req.doc_id),
        prior_clarification=req.prior_clarification_asked,
    )
    if intent == "audit":
        if not req.doc_id:
            return False
        has_scope_hint = bool(
            re.search(
                r"\b(chapter|unit|module|section|part|page|whole|entire|full|all)\b",
                clean_msg,
                re.I,
            )
        )
        return req.prior_clarification_asked or has_scope_hint or not _extract_url_from_text(clean_msg)
    return not _is_greeting(clean_msg)


def _redact_sensitive_text(text: str, req: ChatTurnRequest) -> str:
    for secret in (req.user_gemini_key, req.user_groq_key):
        normalized = (secret or "").strip()
        if normalized:
            text = text.replace(normalized, "[REDACTED]")
    return sanitize_error_message(text)


def _update_session_turn_context(session_id: str, result: Dict[str, Any]) -> None:
    with ACTIVE_JOBS_LOCK:
        job_result = ACTIVE_JOB_RESULTS.setdefault(session_id, {})
        awaiting_scope = bool(result.get("awaiting_scope", False))
        job_result["awaiting_scope"] = awaiting_scope
        if result.get("video_url"):
            job_result["video_url"] = result["video_url"]
        elif not awaiting_scope:
            job_result.pop("video_url", None)
        if result.get("video_id"):
            job_result["video_id"] = result["video_id"]
        elif not awaiting_scope:
            job_result.pop("video_id", None)


def _format_audit_markdown(state: Dict[str, Any]) -> str:
    if state.get("guard_blocked"):
        return (
            f"⚠️ **Audit Blocked by Safety/Quality Guard:** {state.get('guard_reason')}"
        )

    doc_name = state.get("doc_name") or "Uploaded Document"
    raw_scoped = state.get("scoped_section_titles") or []
    scope_label = (
        ", ".join(raw_scoped)
        if raw_scoped
        else f"Full Document ({doc_name})"
    )

    missed_gaps = (
        state.get("final_missed_topics")
        or state.get("drafted_gaps")
        or state.get("missed_topics")
        or []
    )
    escalated = state.get("escalated_items", [])
    eval_passed = (
        state.get("evaluator_passed")
        if "evaluator_passed" in state
        else (state.get("final_status") == "passed" or state.get("evaluator_verdict") == "PASS")
    )
    retry_count = state.get("retry_count") if "retry_count" in state else state.get("attempt", 0)

    lines = [
        f"### 📋 Gap Audit Results — `{scope_label}`",
        f"**Evaluator Status:** `{'PASSED' if eval_passed else 'FAILED'}` (Retries: `{retry_count}`) · **Missed Topics Found:** `{len(missed_gaps)}`",
        "",
    ]

    retrieved_sections = state.get("retrieved_sections") or []
    if not missed_gaps and not escalated:
        if not retrieved_sections:
            lines.append(
                "⚠️ **No sections matched that scope in Pinecone; auditing full document instead.**"
            )
        else:
            lines.append(
                "✅ **All scoped concepts were covered!** "
                "No major syllabus gaps were detected in this video for the selected scope."
            )
    else:
        for idx, gap in enumerate(missed_gaps, start=1):
            title = gap.get("topic_title") or gap.get("section_title") or f"Topic {idx}"
            sec = gap.get("section_title", "")
            pg = gap.get("page_number", "?")
            desc = gap.get("description") or gap.get("explanation") or gap.get("summary") or ""
            vid_note = gap.get("closest_timestamp") or gap.get("video_check_note") or "Not covered in video transcript"

            lines.append(f"**{idx}. {title}**  ·  `Page {pg}` *({sec})*")
            lines.append(f"- 📄 **Missing from Video:** {desc}")
            lines.append(f"- 🎥 **Closest Video Check:** {vid_note}")
            lines.append("")

        if escalated:
            lines.append(
                f"#### ⚠️ Sent to Approvals — Low Confidence ({len(escalated)})"
            )
            for item in escalated:
                ts_hint = (
                    f" (closest match at `{item['closest_video_timestamp']}`)"
                    if item.get("closest_video_timestamp")
                    else ""
                )
                lines.append(
                    f"- **{item['topic_title']}** *(Page {item['page_number']})*{ts_hint} — routed to the **Approvals** queue for human review."
                )
            lines.append("")

    lines.append(
        "💡 *Would you like a quick breakdown of any missed topic above, or a ready-to-record script covering all of them?*"
    )
    return "\n".join(lines)


def _generate_suggested_prompts(
    filename: str, chunks: List[Dict[str, Any]]
) -> List[str]:
    """Create four concise starter prompts grounded in the uploaded PDF."""
    doc_label = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    section_title = str(
        chunks[0].get("section_title") if chunks else "Chapter 1"
    ).strip() or "Chapter 1"
    fallback_prompts = [
        f"Audit a YouTube lecture against {doc_label}",
        f"Which key concepts in {doc_label} are often skipped?",
        f"Check video coverage of {section_title}",
        f"Summarize evaluation rules from {doc_label}",
    ]
    fallback_prompts = [prompt[:84].rstrip() for prompt in fallback_prompts]
    sample_sections = "\n".join(
        (
            f"- [Page {c.get('page_number', 1)} | {c.get('section_title', '')}]: "
            f"{(c.get('child_text') or c.get('text') or c.get('parent_text') or '')[:220]}"
        )
        for c in chunks[:10]
    )

    try:
        response = get_extraction_llm().invoke(
            [
                SystemMessage(
                    content=(
                        "Generate exactly 4 concise, high-value questions or syllabus "
                        "audit prompts derived ONLY from the PDF excerpts below. "
                        "Return ONLY a JSON array of 4 strings, each under 85 characters."
                    )
                ),
                HumanMessage(content=f"Document: {filename}\nExcerpts:\n{sample_sections}"),
            ]
        )
        content = response.content
        raw = content if isinstance(content, str) else str(content)
        start, end = raw.find("["), raw.rfind("]")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
            if (
                isinstance(parsed, list)
                and len(parsed) == 4
                and all(isinstance(question, str) for question in parsed)
            ):
                questions = [
                    re.sub(r"\s+", " ", question).strip() for question in parsed
                ]
                if all(question and len(question) < 85 for question in questions):
                    return questions
        logger.warning(
            "Suggested prompt generation returned invalid output for %s; using grounded fallback prompts.",
            filename,
        )
    except Exception:
        logger.exception(
            "Could not generate suggested prompts for %s; using grounded fallback prompts.",
            filename,
        )
    return fallback_prompts


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _get_total_llm_turns(user_id: str) -> int:
    try:
        return count_completed_llm_turns(user_id)
    except Exception as exc:
        logger.warning("Quota DB lookup unavailable for %s (%s); defaulting to 0.", user_id, exc)
        try:
            from app.services.db.database import init_db
            init_db()
        except Exception:
            pass
        return 0


def _active_user_reservations(user_id: str) -> int:
    running_jobs = sum(
        1
        for session_id, status in ACTIVE_JOBS.items()
        if status == "running" and ACTIVE_JOB_USERS.get(session_id) == user_id
    )
    synchronous_jobs = sum(
        1
        for reserved_user in ACTIVE_SYNC_RESERVATIONS.values()
        if reserved_user == user_id
    )
    return running_jobs + synchronous_jobs


def _reserve_llm_quota(user_id: str) -> str:
    with ACTIVE_JOBS_LOCK:
        used = _get_total_llm_turns(user_id)
        if used + _active_user_reservations(user_id) >= MAX_GUEST_CALLS:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Free demo limit reached ({MAX_GUEST_CALLS}/{MAX_GUEST_CALLS} LLM calls used). "
                    "Enter your own Gemini or Groq API key in the sidebar under Inference Engine to continue."
                ),
            )
        reservation_id = uuid.uuid4().hex
        ACTIVE_SYNC_RESERVATIONS[reservation_id] = user_id
        return reservation_id


def _process_chat_turn_with_quota(
    req: ChatTurnRequest, *, already_reserved: bool = False
) -> Dict[str, Any]:
    has_byok = _has_byok(req)
    reservation_id = None
    if _request_may_use_llm(req) and not has_byok and not already_reserved:
        reservation_id = _reserve_llm_quota(req.user_id)

    try:
        with use_user_api_keys(req.user_gemini_key, req.user_groq_key):
            result = _process_chat_turn(req)

        llm_used = bool(result.pop("_llm_used", False))
        route = result.pop("_route_taken", result.get("route_taken", "project_guide"))
        if llm_used and not has_byok:
            try:
                log_audit_event(
                    action="llm_turn_completed",
                    run_id=result.get("run_id"),
                    user_id=req.user_id,
                    detail={"route": route},
                )
            except Exception as exc:
                logger.warning(
                    "Could not record completed LLM turn for user %s (%s).", req.user_id, exc
                )
        return result
    finally:
        if reservation_id:
            with ACTIVE_JOBS_LOCK:
                ACTIVE_SYNC_RESERVATIONS.pop(reservation_id, None)


def _invoke_graph_turn(
    turn_input: Dict[str, Any], session_id: str
) -> Dict[str, Any]:
    config = {"configurable": {"thread_id": session_id}}
    snapshot = audit_graph.get_state(config)
    is_paused_at_guide = bool(
        snapshot and snapshot.next and "await_user_input" in snapshot.next
    )

    if is_paused_at_guide:
        return audit_graph.invoke(Command(resume=turn_input), config=config)
    return audit_graph.invoke(turn_input, config=config)


@app.get("/quota/{user_id}")
def api_get_quota(user_id: str) -> Dict[str, Any]:
    used = _get_total_llm_turns(user_id)
    return {
        "user_id": user_id,
        "used": used,
        "limit": MAX_FREE_LLM_TURNS,
        "remaining": max(0, MAX_FREE_LLM_TURNS - used),
    }


@app.get("/engine/status")
def api_engine_status() -> Dict[str, Any]:
    settings = get_settings()
    return {
        "api_status": "online",
        "auditor_model": settings.AUDITOR_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "pinecone_configured": bool(settings.PINECONE_API_KEY),
    }


@app.get("/documents")
def api_list_documents() -> List[Dict[str, Any]]:
    return list_documents()


@app.get("/documents/{doc_id}/chunks")
def api_get_document_chunks(doc_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    from app.services.vector.pinecone_store import fetch_scoped_chunks_with_vectors

    chunks = fetch_scoped_chunks_with_vectors(doc_id=doc_id, top_k=max(limit, 50))
    out: List[Dict[str, Any]] = []
    for ch in chunks[:limit]:
        txt = str(ch.get("text") or "").strip()
        out.append(
            {
                "id": ch.get("chunk_id", ""),
                "page_number": ch.get("page_number", 1),
                "section_title": ch.get("section_title", ""),
                "chars": len(txt),
                "text": txt,
                "has_tables": bool(ch.get("has_tables", False)),
                "ocr_applied": bool(ch.get("ocr_applied", False)),
            }
        )
    return out


@app.post("/documents/clear-user-uploads")
def api_clear_user_uploads(preserve_sample_prefix: str = "harness-engineering") -> Dict[str, Any]:
    """Deletes user-uploaded documents from Pinecone & SQLite while keeping built-in demo sample books intact."""
    from app.services.db.crud import list_documents, delete_document
    from app.services.vector.pinecone_store import get_pinecone_index

    idx = get_pinecone_index()
    removed = 0
    for doc in list_documents():
        did = doc["doc_id"]
        if preserve_sample_prefix and did.startswith(preserve_sample_prefix):
            continue
        try:
            idx.delete(filter={"doc_id": {"$eq": did}})
        except Exception:
            pass
        delete_document(did)
        removed += 1
    return {"status": "cleared", "removed_count": removed}


@app.post("/documents/upload")
async def api_upload_document(
    file: UploadFile = File(...),
    uploaded_by: str = Form("default_user"),
) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filename = Path(file.filename.replace("\\", "/")).name
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    dest = get_settings().DATA_DIR / filename
    started_at = time.perf_counter()
    try:
        dest.write_bytes(raw_bytes)
        doc_id = _compute_doc_id(dest)
        result = get_document(doc_id)
        if (
            not result
            or result.get("status") != "ready"
            or result.get("index_version") != PDF_INDEX_VERSION
        ):
            result = await run_in_threadpool(
                ingest_pdf_document,
                dest,
                uploaded_by,
                _generate_suggested_prompts,
            )
    except Exception as exc:
        logger.exception("Synchronous PDF ingestion failed for %s.", filename)
        raise HTTPException(
            status_code=500,
            detail=f"PDF indexing failed ({type(exc).__name__}): {exc}",
        ) from exc

    return {
        "status": "ready",
        "doc_id": result["doc_id"],
        "doc_name": result["doc_name"],
        "page_count": result["page_count"],
        "chunk_count": result["chunk_count"],
        "sections": result["sections"],
        "suggested_questions": result["suggested_questions"],
        "elapsed_sec": round(time.perf_counter() - started_at, 2),
    }


@app.post("/documents/upload/stream")
async def api_upload_document_stream(
    file: UploadFile = File(...),
    user_id: str = Form("default_user"),
) -> StreamingResponse:
    raw_bytes = await file.read()
    filename = file.filename or "uploaded.pdf"

    def event_stream() -> Generator[str, None, None]:
        t0 = time.perf_counter()
        try:
            if not filename.lower().endswith(".pdf") or not raw_bytes:
                yield f"data: {json.dumps({'type': 'error', 'text': 'Invalid or empty PDF file.'})}\n\n"
                return

            content_hash = hashlib.sha256(raw_bytes).hexdigest()[:10]
            stem = Path(filename).stem.lower().replace(" ", "-")
            doc_id = f"{stem}-v3-{content_hash}"

            from app.services.db.repository import get_document
            existing_doc = get_document(doc_id)
            if existing_doc and existing_doc.get("status") == "ready" and existing_doc.get("chunk_count", 0) > 0:
                payload = {
                    "status": "ready",
                    "doc_id": doc_id,
                    "filename": filename,
                    "doc_name": filename,
                    "page_count": existing_doc.get("page_count", 0),
                    "chunk_count": existing_doc.get("chunk_count", 0),
                    "sections": existing_doc.get("sections", [])[:12],
                    "suggested_questions": existing_doc.get("suggested_questions", []),
                    "elapsed_sec": round(time.perf_counter() - t0, 2),
                    "from_cache": True,
                }
                yield f"data: {json.dumps({'type': 'done', 'payload': payload})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'step', 'step': 1, 'text': 'Extracting PDF sections & pages...'})}\n\n"

            dest = get_settings().DATA_DIR / Path(filename.replace("\\", "/")).name
            dest.write_bytes(raw_bytes)

            from app.ingestion.loaders.pdf_loader import load_pdf_sections
            from app.ingestion.chunking.splitter import build_hierarchical_chunks

            parsed_pdf = load_pdf_sections(dest)
            sections = parsed_pdf["sections"]
            page_count = parsed_pdf["page_count"]
            ocr_pages = parsed_pdf.get("ocr_pages_processed", 0)
            step1_msg = (
                f"Extracted {page_count} pages ({ocr_pages} image/diagram pages via OCR)..."
                if ocr_pages
                else f"Extracted {page_count} native pages..."
            )
            yield f"data: {json.dumps({'type': 'step', 'step': 1, 'ocr_pages': ocr_pages, 'text': step1_msg})}\n\n"

            yield f"data: {json.dumps({'type': 'step', 'step': 2, 'text': 'Filtering boilerplate, headers & stub pages...'})}\n\n"

            child_chunks = build_hierarchical_chunks(doc_id=doc_id, sections=sections)
            # Drop image-only divider pages & header-only stubs (e.g., "DailyDoseofDS.com 210")
            clean_chunks = [
                c for c in child_chunks
                if len(str(c.get("text") or c.get("parent_text") or "").strip()) >= 180
            ]
            if not clean_chunks:
                clean_chunks = child_chunks

            if not clean_chunks:
                yield f"data: {json.dumps({'type': 'error', 'text': 'No substantive selectable text found in this PDF.'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'step', 'step': 3, 'chunk_count': len(clean_chunks), 'text': f'Coalesced {len(clean_chunks)} adaptive chunks...'})}\n\n"

            yield f"data: {json.dumps({'type': 'step', 'step': 4, 'text': 'Generating 768-d BGE embeddings (parallel)...'})}\n\n"
            from app.services.vector.pinecone_store import upsert_document_chunks
            chunk_count = upsert_document_chunks(
                doc_id=doc_id,
                filename=filename,
                chunks=clean_chunks,
            )

            yield f"data: {json.dumps({'type': 'step', 'step': 5, 'text': f'Indexed {chunk_count} vectors into Pinecone...'})}\n\n"

            BOILERPLATE = {"figures and tables", "table of contents", "contents", "overview", "dailydoseofds.com"}
            section_titles = list(
                dict.fromkeys(
                    s["section_title"]
                    for s in sections
                    if s.get("section_title")
                    and len(s["section_title"].strip()) >= 5
                    and s["section_title"].strip().lower() not in BOILERPLATE
                    and "dailydoseofds" not in s["section_title"].lower()
                )
            )
            top_sec = section_titles[0] if section_titles else "the core chapters"
            second_sec = section_titles[1] if len(section_titles) > 1 else "key architectures"

            starter_prompts = [
                f"Summarize the key topics covered in {filename}",
                f"Explain '{top_sec}' from this PDF",
                f"What are the key concepts in '{second_sec}'?",
                "How does this Syllabus Gap Audit AI project work?",
            ]

            from app.services.db.crud import register_document
            register_document(
                doc_id=doc_id,
                filename=filename,
                page_count=page_count,
                chunk_count=chunk_count,
                section_titles=section_titles,
                status="ready",
                uploaded_by=user_id,
                suggested_questions=starter_prompts,
            )

            payload = {
                "status": "ready",
                "doc_id": doc_id,
                "filename": filename,
                "doc_name": filename,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "ocr_pages_processed": ocr_pages,
                "sections": section_titles[:12],
                "suggested_questions": starter_prompts,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
            yield f"data: {json.dumps({'type': 'done', 'payload': payload})}\n\n"
        except Exception as exc:
            traceback.print_exc()
            cleaned_err = sanitize_error_message(f"{type(exc).__name__}: {exc}")
            yield f"data: {json.dumps({'type': 'error', 'text': cleaned_err})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/sessions")
def api_list_sessions(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    sessions = list_recent_sessions(limit=20, user_id=user_id)
    known_sessions = {session["session_id"] for session in sessions}
    with ACTIVE_JOBS_LOCK:
        active_jobs = list(ACTIVE_JOBS.items())[::-1]
        active_job_ids = dict(ACTIVE_JOB_IDS)
        active_job_results = {
            session_id: dict(result)
            for session_id, result in ACTIVE_JOB_RESULTS.items()
        }
        active_job_users = dict(ACTIVE_JOB_USERS)
    for session_id, status in active_jobs:
        if user_id is not None and active_job_users.get(session_id) != user_id:
            continue
        if session_id in known_sessions:
            continue
        if status not in {"running", "done", "error"}:
            continue
        job = active_job_results.get(session_id, {})
        video_title = job.get("video_title", "Video audit")
        if status == "running":
            video_title = f"In progress: {video_title}"
        sessions.insert(
            0,
            {
                "session_id": session_id,
                "run_id": active_job_ids.get(session_id, session_id),
                "video_title": video_title,
                "target_scope": "",
                "evaluator_status": status,
                "created_at": "",
            },
        )
        known_sessions.add(session_id)
    return sessions[:20]


@app.get("/sessions/{session_id}/messages")
def api_get_messages(session_id: str) -> List[Dict[str, Any]]:
    return get_session_messages(session_id)


@app.get("/videos")
def api_list_videos() -> List[Dict[str, Any]]:
    settings = get_settings()
    found: Dict[str, Dict[str, Any]] = {}

    for f in sorted(
        settings.PROCESSED_VIDEOS_DIR.glob("video_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise TypeError("Video payload must be a JSON object.")
            vid = data.get("video_id", f.stem.replace("video_", ""))
            if not isinstance(vid, str) or not vid:
                raise TypeError("Video payload has an invalid video_id.")
            found[vid] = {
                "video_id": vid,
                "title": data.get("title", "YouTube Lecture"),
                "duration_sec": data.get("duration_sec", 0),
                "window_count": len(data.get("windows", [])),
                "slide_count": len(data.get("ocr_slides", [])),
            }
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Skipping invalid processed video file %s: %s", f, exc)

    if settings.CACHE_DB_PATH.exists():
        try:
            with sqlite3.connect(settings.CACHE_DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT video_id, payload_json FROM video_cache"
                ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table: video_cache" not in str(exc):
                logger.exception("Could not read the video cache.")
            rows = []
        except sqlite3.Error:
            logger.exception("Could not read the video cache.")
            rows = []

        for vid, raw in rows:
            if vid in found:
                continue
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise TypeError("Video payload must be a JSON object.")
                found[vid] = {
                    "video_id": vid,
                    "title": data.get("title", "YouTube Lecture"),
                    "duration_sec": data.get("duration_sec", 0),
                    "window_count": len(data.get("windows", [])),
                    "slide_count": len(data.get("ocr_slides", [])),
                }
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Skipping invalid cached video %s: %s", vid, exc)

    return list(found.values())


@app.get("/videos/{video_id}")
def api_get_video_transcript(video_id: str) -> Dict[str, Any]:
    def public_payload(data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=500, detail="Invalid cached video transcript."
            )
        return {
            key: value
            for key, value in data.items()
            if key not in {"window_embeddings", "window_embeddings_key"}
        }

    settings = get_settings()
    target = settings.PROCESSED_VIDEOS_DIR / f"video_{video_id}.json"
    if target.exists():
        return public_payload(json.loads(target.read_text(encoding="utf-8")))

    if settings.CACHE_DB_PATH.exists():
        try:
            with sqlite3.connect(settings.CACHE_DB_PATH) as conn:
                row = conn.execute(
                    "SELECT payload_json FROM video_cache WHERE video_id = ?",
                    (video_id,),
                ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table: video_cache" in str(exc):
                row = None
            else:
                logger.exception("Could not read transcript for video %s.", video_id)
                raise HTTPException(
                    status_code=500, detail="Could not read the video cache."
                ) from exc
        except sqlite3.Error as exc:
            logger.exception("Could not read transcript for video %s.", video_id)
            raise HTTPException(
                status_code=500, detail="Could not read the video cache."
            ) from exc
        if row:
            return public_payload(json.loads(row[0]))

    raise HTTPException(status_code=404, detail="Transcribed video windows not found.")


@app.post("/videos/transcribe")
def api_transcribe_video(req: TranscribeRequest) -> Dict[str, Any]:
    from app.ingestion.processor import get_or_parse_video

    parsed = get_or_parse_video(
        req.video_url.strip(), force_refresh=req.force_refresh
    )
    if not parsed.get("passed_guard"):
        raise HTTPException(
            status_code=400,
            detail=parsed.get("guard_reason", "Video failed guard checks."),
        )
    return parsed


@app.post("/chat/turn")
def api_chat_turn(req: ChatTurnRequest) -> Dict[str, Any]:
    try:
        result = _process_chat_turn_with_quota(req)
        _update_session_turn_context(req.session_id, result)
        return result
    except HTTPException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=_redact_sensitive_text(str(exc.detail), req),
        ) from None
    except Exception as exc:
        logger.error(
            "Unexpected error while processing chat turn for session %s (%s).",
            req.session_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=_redact_sensitive_text(f"{type(exc).__name__}: {exc}", req),
        ) from None


@app.post("/chat/turn/stream")
def api_chat_turn_stream(req: ChatTurnRequest) -> StreamingResponse:
    with ACTIVE_JOBS_LOCK:
        if (
            req.session_id in ACTIVE_STREAM_SESSIONS
            or ACTIVE_JOBS.get(req.session_id) == "running"
        ):
            raise HTTPException(
                status_code=409,
                detail="A chat turn is already running for this session.",
            )

        used = _get_total_llm_turns(req.user_id)
        if (
            not _has_byok(req)
            and _request_may_use_llm(req)
            and used + _active_user_reservations(req.user_id) >= MAX_GUEST_CALLS
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Free demo limit reached ({MAX_GUEST_CALLS}/{MAX_GUEST_CALLS} LLM calls used). "
                    "Enter your own Gemini or Groq API key in the sidebar under Inference Engine to continue."
                ),
            )
        ACTIVE_STREAM_SESSIONS.add(req.session_id)

    result_queue: Queue[Tuple[str, Any]] = Queue(maxsize=1)

    def run_turn() -> None:
        try:
            result = api_chat_turn(req)
            _update_session_turn_context(req.session_id, result)
            result_queue.put(("done", result))
        except HTTPException as exc:
            result_queue.put(
                (
                    "http_error",
                    {
                        "status_code": exc.status_code,
                        "detail": _redact_sensitive_text(str(exc.detail), req),
                    },
                )
            )
        except Exception as exc:
            logger.error(
                "Unexpected error while streaming chat turn for session %s (%s).",
                req.session_id,
                type(exc).__name__,
            )
            result_queue.put(
                (
                    "error",
                    _redact_sensitive_text(
                        f"{type(exc).__name__}: {exc}", req
                    ),
                )
            )
        finally:
            with ACTIVE_JOBS_LOCK:
                ACTIVE_STREAM_SESSIONS.discard(req.session_id)

    def encode_event(event_type: str, **payload: Any) -> str:
        return (
            f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}"
            "\n\n"
        )

    set_job_stage(req.session_id, "🚀 Starting chat turn...")
    worker = threading.Thread(target=run_turn, daemon=True)
    worker.start()

    def event_stream() -> Generator[str, None, None]:
        yield encode_event(
            "stage", text="🚀 Starting chat turn..."
        )
        last_stage = "🚀 Starting chat turn..."

        while True:
            stage = get_job_state(req.session_id).get("stage", "")
            if stage and stage != last_stage:
                yield encode_event("stage", text=stage)
                last_stage = stage

            try:
                outcome, payload = result_queue.get(timeout=0.25)
            except Empty:
                continue

            if outcome == "done":
                set_job_stage(req.session_id, "✅ Chat turn complete", status="done")
                yield encode_event("stage", text="✅ Chat turn complete")
                yield encode_event("done", payload=payload)
            elif outcome == "http_error":
                detail = payload.get("detail", "Chat turn failed.")
                set_job_stage(req.session_id, str(detail), status="error")
                yield encode_event(
                    "error",
                    text=str(detail),
                    status_code=payload.get("status_code", 500),
                )
            else:
                set_job_stage(req.session_id, str(payload), status="error")
                yield encode_event("error", text=str(payload))
            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _process_chat_turn(req: ChatTurnRequest) -> Dict[str, Any]:
    doc = get_document(req.doc_id) if req.doc_id else None
    sanitized_msg = sanitize_transcript_text(req.message)
    clean_msg = _strip_mode_tag(sanitized_msg)
    message_url = _extract_url_from_text(clean_msg)
    is_greeting = _is_greeting(clean_msg) and not message_url
    detected_url = (
        None
        if is_greeting
        else message_url
        or (req.video_url if req.prior_clarification_asked else None)
    )

    save_chat_message(
        session_id=req.session_id,
        role="user",
        content=clean_msg,
        user_id=req.user_id,
        video_url=detected_url or "",
    )

    if is_greeting:
        if doc:
            reply = (
                f"Hi! 👋 Your active reference document is **`{doc['doc_name']}`** "
                f"({doc['page_count']} pages, indexed and ready).\n\n"
                "Please paste the **YouTube lecture URL** you want me to audit "
                "against this document (or upload a different PDF above first)."
            )
        else:
            reply = (
                "Hi! 👋 To get started with your Syllabus Gap Audit:\n"
                "1. **Upload a reference PDF** (or pick one of the indexed sample books above).\n"
                "2. **Paste the YouTube lecture URL** you want to check against that document."
            )
        msg_id = save_chat_message(
            session_id=req.session_id,
            role="assistant",
            content=reply,
            user_id=req.user_id,
        )
        return {
            "type": "chat",
            "message_id": msg_id,
            "reply": reply,
            "awaiting_scope": False,
            "_llm_used": False,
            "_route_taken": "project_guide",
        }

    # Case A: Video URL present -> Scope Clarification or Full LangGraph Audit
    if detected_url:
        if not doc:
            reply = (
                "Please **upload or select a reference PDF** first. Then paste the "
                "YouTube lecture URL again so I can audit it against that document."
            )
            msg_id = save_chat_message(
                session_id=req.session_id,
                role="assistant",
                content=reply,
                user_id=req.user_id,
            )
            return {
                "type": "chat",
                "message_id": msg_id,
                "reply": reply,
                "awaiting_scope": False,
                "_llm_used": False,
                "_route_taken": "audit",
            }

        # Fast deterministic check on Turn 1 (< 10ms, zero LLM wait!)
        has_scope_hint = bool(
            re.search(
                r"\b(chapter|unit|module|section|part|page|whole|entire|full|all)\b",
                clean_msg,
                re.I,
            )
        )
        if (
            not req.prior_clarification_asked
            and not has_scope_hint
        ):
            toc_examples = (
                ", ".join(s["section_title"] for s in doc["sections"][:3])
                or "Chapter 1"
            )
            reply = (
                f"Would you like to analyze this video against the **entire book** "
                f"(`{doc['doc_name']}`) or a specific section "
                f"(such as *{toc_examples}*)?"
            )
            msg_id = save_chat_message(
                session_id=req.session_id,
                role="assistant",
                content=reply,
                user_id=req.user_id,
            )
            return {
                "type": "clarification",
                "message_id": msg_id,
                "reply": reply,
                "video_url": detected_url,
                "awaiting_scope": True,
                "_llm_used": False,
                "_route_taken": "audit",
            }

        from app.agents.nodes.scope_router import WHOLE_DOC_PHRASES

        clean_low = clean_msg.lower().strip()
        if any(p in clean_low for p in WHOLE_DOC_PHRASES):
            scope_res = {
                "needs_clarification": False,
                "clarifying_question": "",
                "target_scope": "Entire Document",
                "scoped_section_titles": [],
            }
        else:
            scope_res = resolve_audit_scope(
                user_message=clean_msg,
                doc_name=doc["doc_name"],
                doc_sections=doc["sections"],
                prior_clarification_asked=req.prior_clarification_asked or has_scope_hint,
            )

        if scope_res["needs_clarification"]:
            reply = scope_res["clarifying_question"]
            msg_id = save_chat_message(
                session_id=req.session_id,
                role="assistant",
                content=reply,
                user_id=req.user_id,
            )
            return {
                "type": "clarification",
                "message_id": msg_id,
                "reply": reply,
                "video_url": detected_url,
                "awaiting_scope": True,
                "_llm_used": True,
                "_route_taken": "audit",
            }

        # Execute full LangGraph Audit Pipeline
        t0 = time.perf_counter()
        final_state = _invoke_graph_turn(
            {
                "session_id": req.session_id,
                "user_id": req.user_id,
                "doc_id": doc["doc_id"],
                "doc_name": doc["doc_name"],
                "video_url": detected_url,
                "raw_transcript": req.raw_transcript,
                "user_query": sanitized_msg,
                "prior_clarification_asked": bool(
                    req.prior_clarification_asked or has_scope_hint
                ),
                "target_scope": scope_res["target_scope"],
                "scoped_section_titles": scope_res["scoped_section_titles"],
                "attempt": 0,
                "max_attempts": 1,
            },
            req.session_id,
        )
        if final_state.get("route_taken") in ("project_guide", "pdf_qa") or "__interrupt__" in final_state:
            reply_text = final_state.get("chat_response", "")
            if not reply_text:
                raise RuntimeError("The chat graph returned an empty response.")
            citations = (
                final_state.get("retrieved_evidence", [])
                if final_state.get("route_taken") == "pdf_qa"
                else []
            )
            msg_id = save_chat_message(
                session_id=req.session_id,
                role="assistant",
                content=reply_text,
                user_id=req.user_id,
                citations=citations,
            )
            return {
                "type": "pdf_qa" if final_state.get("route_taken") == "pdf_qa" else "chat",
                "message_id": msg_id,
                "reply": reply_text,
                "citations": citations,
                "awaiting_scope": False,
                "_llm_used": bool(final_state.get("llm_used", True)),
                "_route_taken": final_state.get("route_taken", "project_guide"),
            }
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        final_state["latency_ms"] = elapsed_ms

        reply_md = _format_audit_markdown(final_state)
        all_chunks = (
            final_state.get("retrieved_evidence")
            or final_state.get("cited_chunks")
            or []
        )
        missed_gaps = (
            final_state.get("final_missed_topics")
            or final_state.get("drafted_gaps")
            or final_state.get("missed_topics")
            or []
        )
        if missed_gaps and all_chunks:
            gap_pages = [int(g["page_number"]) for g in missed_gaps if g.get("page_number") is not None]
            page_order = {pg: idx for idx, pg in enumerate(dict.fromkeys(gap_pages))}
            matched = [c for c in all_chunks if int(c.get("page_number") or 0) in page_order]
            matched.sort(key=lambda c: page_order[int(c.get("page_number") or 0)])
            remaining = [c for c in all_chunks if c not in matched]
            citations = (matched + remaining)[:6]
        else:
            citations = all_chunks[:6]

        enriched_citations = []
        for c in citations:
            c_dict = dict(c)
            txt = (
                c_dict.get("text")
                or c_dict.get("evidence_quote")
                or c_dict.get("snippet")
                or c_dict.get("parent_text")
                or c_dict.get("summary")
                or ""
            )
            c_dict["text"] = txt
            c_dict["snippet"] = txt[:600]
            c_dict["evidence_quote"] = c_dict.get("evidence_quote") or txt[:140]
            enriched_citations.append(c_dict)

        if len(enriched_citations) < 3 and missed_gaps:
            doc_name = final_state.get("doc_name") or "Uploaded Document"
            for idx, g in enumerate(missed_gaps):
                if len(enriched_citations) >= 6:
                    break
                txt = (
                    g.get("evidence_quote")
                    or g.get("summary")
                    or g.get("description")
                    or g.get("explanation")
                    or ""
                )
                if txt:
                    enriched_citations.append(
                        {
                            "chunk_id": g.get("chunk_id", f"gap_{idx}"),
                            "doc_name": doc_name,
                            "section_title": g.get("section_title", ""),
                            "page_number": g.get("page_number", 1),
                            "text": txt,
                            "snippet": txt[:600],
                            "evidence_quote": txt[:140],
                            "score": 0.45,
                        }
                    )
        citations = enriched_citations

        msg_id = save_chat_message(
            session_id=req.session_id,
            role="assistant",
            content=reply_md,
            user_id=req.user_id,
            citations=citations,
            run_id=final_state.get("run_id"),
        )

        return {
            "type": "audit_report",
            "message_id": msg_id,
            "run_id": final_state.get("run_id"),
            "video_id": final_state.get("video_id", ""),
            "reply": reply_md,
            "citations": citations,
            "trace": {
                "video_title": final_state.get("video_title", ""),
                "windows_count": len(final_state.get("video_windows", [])),
                "slides_count": len(final_state.get("ocr_slides", [])),
                "sections_checked": len(final_state.get("retrieved_sections", [])),
                "evaluator_status": final_state.get("final_status", "passed"),
                "retry_count": final_state.get("attempt", 0),
                "evaluation_notes": final_state.get("evaluation_notes", ""),
                "repair_notes": final_state.get("repair_notes", ""),
                "step_timings": final_state.get("step_timings", {}),
                "latency_ms": elapsed_ms,
                
            },
            "awaiting_scope": False,
            "_llm_used": final_state.get("final_status") != "blocked",
            "_route_taken": "audit",
        }

    # Project guidance and document Q&A pause at the same graph input node.
    guide_state = _invoke_graph_turn(
        {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "doc_id": doc["doc_id"] if doc else None,
            "doc_name": doc["doc_name"] if doc else "",
            "page_count": doc.get("page_count", 0) if doc else 0,
            "chunk_count": doc.get("chunk_count", 0) if doc else 0,
            "user_query": sanitized_msg,
            "video_url": "",
            "prior_clarification_asked": False,
        },
        req.session_id,
    )
    if guide_state.get("route_taken") not in ("project_guide", "pdf_qa"):
        raise RuntimeError("The chat graph did not produce a guide or PDF answer.")

    reply_text = guide_state.get("chat_response") or ""
    if not reply_text:
        raise RuntimeError("The chat graph returned an empty response.")
    citations = (
        guide_state.get("retrieved_evidence", [])
        if guide_state.get("route_taken") == "pdf_qa"
        else []
    )
    msg_id = save_chat_message(
        session_id=req.session_id,
        role="assistant",
        content=reply_text,
        user_id=req.user_id,
        citations=citations,
    )
    return {
        "type": "pdf_qa" if guide_state.get("route_taken") == "pdf_qa" else "chat",
        "message_id": msg_id,
        "reply": reply_text,
        "citations": citations,
        "awaiting_scope": False,
        "_llm_used": bool(guide_state.get("llm_used", True)),
        "_route_taken": guide_state.get("route_taken", "project_guide"),
    }


def _run_audit_background(req: ChatTurnRequest, job_id: str) -> None:
    session_id = req.session_id
    with ACTIVE_JOBS_LOCK:
        if ACTIVE_JOB_IDS.get(session_id) != job_id:
            return
        ACTIVE_JOBS[session_id] = "running"
    set_job_stage(session_id, "🚀 Starting audit pipeline...")
    try:
        result = _process_chat_turn_with_quota(req, already_reserved=True)
        with ACTIVE_JOBS_LOCK:
            if ACTIVE_JOB_IDS.get(session_id) != job_id:
                return
            job_result = ACTIVE_JOB_RESULTS.get(session_id, {})
            job_result["awaiting_scope"] = bool(
                result.get("awaiting_scope", False)
            )
            job_result["message_id"] = result.get("message_id")
            job_result["trace"] = result.get("trace")
            for key in ("video_id", "video_url"):
                if result.get(key):
                    job_result[key] = result[key]
            ACTIVE_JOB_RESULTS[session_id] = job_result
            ACTIVE_JOBS[session_id] = "done"
        if result.get("awaiting_scope"):
            set_job_stage(
                session_id,
                "Waiting for the requested audit scope...",
                status="done",
            )
        elif result.get("final_status") == "blocked":
            set_job_stage(
                session_id,
                "⚠️ Video blocked by the media quality guard.",
                status="done",
            )
        else:
            set_job_stage(session_id, "✅ Audit complete", status="done")
    except Exception as exc:
        import traceback

        tb = _redact_sensitive_text(traceback.format_exc(), req)
        print(f"\n[BACKGROUND AUDIT ERROR]\n{tb}")
        err_msg = _redact_sensitive_text(
            f"⚠️ **Audit Error (`{type(exc).__name__}`):** {exc}", req
        )
        try:
            save_chat_message(
                session_id=session_id,
                role="assistant",
                content=err_msg,
                user_id=req.user_id,
            )
        except Exception:
            logger.exception(
                "Could not save background audit error for session %s.", session_id
            )
        update_stage = False
        with ACTIVE_JOBS_LOCK:
            if ACTIVE_JOB_IDS.get(session_id) == job_id:
                job_result = ACTIVE_JOB_RESULTS.get(session_id, {})
                job_result.update({"error": err_msg, "awaiting_scope": False})
                ACTIVE_JOB_RESULTS[session_id] = job_result
                ACTIVE_JOBS[session_id] = "error"
                update_stage = True
        if update_stage:
            set_job_stage(session_id, err_msg, status="error")


@app.post("/chat/turn/async")
def api_chat_turn_async(
    req: ChatTurnRequest, background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    try:
        if not _is_video_audit_turn(req) or not _request_may_use_llm(req):
            with ACTIVE_JOBS_LOCK:
                if (
                    ACTIVE_JOBS.get(req.session_id) == "running"
                    or req.session_id in ACTIVE_STREAM_SESSIONS
                ):
                    raise HTTPException(
                        status_code=409,
                        detail="An audit is already running for this session.",
                    )
            result = _process_chat_turn_with_quota(req)
            _update_session_turn_context(req.session_id, result)
            return result

        with ACTIVE_JOBS_LOCK:
            if (
                ACTIVE_JOBS.get(req.session_id) == "running"
                or req.session_id in ACTIVE_STREAM_SESSIONS
            ):
                raise HTTPException(
                    status_code=409,
                    detail="An audit is already running for this session.",
                )

            used = _get_total_llm_turns(req.user_id)
            if (
                not _has_byok(req)
                and used + _active_user_reservations(req.user_id)
                >= MAX_GUEST_CALLS
            ):
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Free demo limit reached ({MAX_GUEST_CALLS}/{MAX_GUEST_CALLS} LLM calls used). "
                        "Enter your own Gemini or Groq API key in the sidebar under Inference Engine to continue."
                    ),
                )

            job_id = uuid.uuid4().hex
            ACTIVE_JOB_IDS[req.session_id] = job_id
            if _has_byok(req):
                ACTIVE_JOB_USERS.pop(req.session_id, None)
            else:
                ACTIVE_JOB_USERS[req.session_id] = req.user_id
            previous_result = ACTIVE_JOB_RESULTS.get(req.session_id, {})
            job_result = {
                key: previous_result[key]
                for key in ("video_title", "video_url", "video_id")
                if previous_result.get(key)
            }
            job_result["video_title"] = (
                _extract_url_from_text(req.message)
                or req.video_url
                or job_result.get("video_title")
                or "Video audit"
            )
            ACTIVE_JOB_RESULTS[req.session_id] = job_result
            ACTIVE_JOBS[req.session_id] = "running"

        set_job_stage(req.session_id, "🚀 Starting audit pipeline...")
        background_tasks.add_task(_run_audit_background, req, job_id)
        return {"status": "running", "session_id": req.session_id, "job_id": job_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error while accepting chat turn for session %s (%s).",
            req.session_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=_redact_sensitive_text(f"{type(exc).__name__}: {exc}", req),
        ) from exc


@app.get("/sessions/{session_id}/status")
def api_session_status(session_id: str) -> Dict[str, Any]:
    with ACTIVE_JOBS_LOCK:
        result: Dict[str, Any] = {
            "session_id": session_id,
            "status": ACTIVE_JOBS.get(session_id, "idle"),
        }
        job_id = ACTIVE_JOB_IDS.get(session_id)
        if job_id:
            result["job_id"] = job_id
        result.update(ACTIVE_JOB_RESULTS.get(session_id, {}))
        result["stage"] = get_job_state(session_id).get("stage", "")
        return result


@app.get("/approvals")
def api_list_approvals(status: Optional[str] = None) -> List[Dict[str, Any]]:
    return list_approvals(status_filter=status)


@app.post("/approvals/{approval_id}")
def api_resolve_approval(
    approval_id: str, req: ApprovalDecisionRequest
) -> Dict[str, Any]:
    if req.decision not in {"confirmed_gap", "marked_covered"}:
        raise HTTPException(
            status_code=400,
            detail="Decision must be 'confirmed_gap' or 'marked_covered'.",
        )
    ok = resolve_approval(approval_id, decision=req.decision, reviewer=req.reviewer)
    if not ok:
        raise HTTPException(status_code=404, detail="Approval item not found.")
    return {"status": "updated", "approval_id": approval_id, "decision": req.decision}


@app.post("/feedback")
def api_submit_feedback(req: FeedbackRequest) -> Dict[str, str]:
    fb_id = save_feedback(
        message_id=req.message_id,
        rating=req.rating,
        user_id=req.user_id,
        comment=req.comment,
    )
    return {"status": "saved", "feedback_id": fb_id}


@app.get("/metrics")
def api_metrics() -> Dict[str, Any]:
    return get_system_metrics()
