import time
import json
import logging
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from app.agents.graph import audit_graph
from app.ingestion.processor import _compute_doc_id
from app.services.db.repository import save_document_record
from app.agents.nodes.scope_router import load_agents_contract, resolve_audit_scope
from app.config import get_settings
from app.gateway.client import get_auditor_llm
from app.guardrails.rails import YOUTUBE_URL_RE, sanitize_transcript_text
from app.ingestion.processor import ingest_pdf_document
from app.services.db.database import init_db
from app.services.db.repository import (
    count_completed_audits,
    get_document,
    get_session_messages,
    get_system_metrics,
    list_approvals,
    list_documents,
    list_recent_sessions,
    log_audit_event,
    resolve_approval,
    save_chat_message,
    save_feedback,
)
from app.services.retrieval.pinecone_service import PineconeService

app = FastAPI(title="Syllabus & Book Gap Finder API", version="1.0.0")
MAX_FREE_TURNS = 5
ACTIVE_JOBS: Dict[str, str] = {}
ACTIVE_JOB_IDS: Dict[str, str] = {}
ACTIVE_JOB_RESULTS: Dict[str, Dict[str, Any]] = {}
ACTIVE_JOB_USERS: Dict[str, str] = {}
ACTIVE_SYNC_RESERVATIONS: Dict[str, str] = {}
ACTIVE_JOBS_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# --- Request Schemas ---
class ChatTurnRequest(BaseModel):
    session_id: str
    user_id: str = "default_user"
    doc_id: str
    message: str
    video_url: Optional[str] = None
    prior_clarification_asked: bool = False


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


def _format_audit_markdown(state: Dict[str, Any]) -> str:
    if state.get("guard_blocked"):
        return (
            f"⚠️ **Audit Blocked by Safety/Quality Guard:** {state.get('guard_reason')}"
        )

    missed = state.get("final_missed_topics", [])
    escalated = state.get("escalated_items", [])
    scope = state.get("target_scope", "Entire Document")
    status = state.get("final_status", "passed")
    retries = state.get("attempt", 0)

    lines = [
        f"### Gap Audit Results — *{scope}*",
        f"**Evaluator Status:** `{status.upper()}` (Retries: `{retries}`)",
        "",
    ]

    if not missed and not escalated:
        lines.append(
            "✅ **All scoped concepts were covered!** "
            "No major syllabus gaps were detected in this video for the selected scope."
        )
    else:
        if missed:
            lines.append(f"#### Confirmed Missed Topics ({len(missed)})")
            for idx, item in enumerate(missed, 1):
                lines.append(
                    f"{idx}. **{item['topic_title']}** *(Section: {item['section_title']} — Page {item['page_number']})*\n"
                    f"   {item['summary']}"
                )
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

    if missed:
        lines.append(
            "Would you like me to briefly explain each of these missed topics, or generate a quick summary script covering them?"
        )
    return "\n".join(lines)


def _extract_llm_text(content: Any) -> str:
    """Unwrap Gemini text content blocks into Markdown text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _get_user_turn_count(user_id: str) -> int:
    try:
        return count_completed_audits(user_id)
    except SQLAlchemyError as exc:
        logger.exception("Could not read completed audit count for user %s.", user_id)
        raise HTTPException(
            status_code=503, detail="Audit quota service is temporarily unavailable."
        ) from exc


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


def _reserve_sync_quota(user_id: str) -> str:
    with ACTIVE_JOBS_LOCK:
        used = _get_user_turn_count(user_id)
        if used + _active_user_reservations(user_id) >= MAX_FREE_TURNS:
            raise HTTPException(
                status_code=429,
                detail=f"Free trial limit reached ({MAX_FREE_TURNS}/{MAX_FREE_TURNS} audits used).",
            )
        reservation_id = uuid.uuid4().hex
        ACTIVE_SYNC_RESERVATIONS[reservation_id] = user_id
        return reservation_id


@app.get("/quota/{user_id}")
def api_get_quota(user_id: str) -> Dict[str, Any]:
    used = _get_user_turn_count(user_id)
    return {
        "user_id": user_id,
        "used": used,
        "limit": MAX_FREE_TURNS,
        "remaining": max(0, MAX_FREE_TURNS - used),
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


@app.post("/documents/upload")
async def api_upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    uploaded_by: str = Form("default_user"),
) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    settings = get_settings()
    dest = settings.DATA_DIR / file.filename
    dest.write_bytes(await file.read())
    doc_id = _compute_doc_id(dest)

    # Immediately create a 'processing' row in Neon PostgreSQL so UI shows it right away
    save_document_record(
        doc_id=doc_id,
        doc_name=file.filename,
        file_path=str(dest),
        uploaded_by=uploaded_by,
        status="processing",
    )
    background_tasks.add_task(ingest_pdf_document, dest, uploaded_by)
    return {
        "status": "processing",
        "doc_id": doc_id,
        "doc_name": file.filename,
    }


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
    reservation_id = _reserve_sync_quota(req.user_id)
    try:
        return _process_chat_turn(req)
    finally:
        with ACTIVE_JOBS_LOCK:
            ACTIVE_SYNC_RESERVATIONS.pop(reservation_id, None)


def _process_chat_turn(req: ChatTurnRequest) -> Dict[str, Any]:
    doc = get_document(req.doc_id)
    if not doc:
        raise HTTPException(
            status_code=404, detail="Selected reference document not found."
        )

    clean_msg = sanitize_transcript_text(req.message)
    detected_url = req.video_url or _extract_url_from_text(clean_msg)
    url_from_history = False
    if not detected_url:
        for past in reversed(get_session_messages(req.session_id)):
            found_url = _extract_url_from_text(past.get("content", ""))
            if found_url:
                detected_url = found_url
                url_from_history = True
                break

    # Persist user message
    save_chat_message(
        session_id=req.session_id,
        role="user",
        content=clean_msg,
        user_id=req.user_id,
        video_url=detected_url or "",
    )

    # Case A: Video URL present -> Scope Clarification or Full LangGraph Audit
    if detected_url:
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
            and not url_from_history
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
            }

        scope_res = resolve_audit_scope(
            user_message=clean_msg,
            doc_name=doc["doc_name"],
            doc_sections=doc["sections"],
            prior_clarification_asked=(
                req.prior_clarification_asked or has_scope_hint or url_from_history
            ),
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
            }

        # Execute full LangGraph Audit Pipeline
        t0 = time.perf_counter()
        final_state = audit_graph.invoke(
            {
                "session_id": req.session_id,
                "user_id": req.user_id,
                "doc_id": doc["doc_id"],
                "doc_name": doc["doc_name"],
                "video_url": detected_url,
                "user_query": clean_msg,
                "target_scope": scope_res["target_scope"],
                "scoped_section_titles": scope_res["scoped_section_titles"],
                "attempt": 0,
                "max_attempts": get_settings().MAX_CRITIC_RETRIES,
            }
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        final_state["latency_ms"] = elapsed_ms

        reply_md = _format_audit_markdown(final_state)
        citations = final_state.get("cited_chunks", [])
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
        }

    # Case B: Conversational Follow-Up Q&A grounded in the uploaded book
    pc = PineconeService()
    hits = pc.search_and_rerank(
        doc_id=doc["doc_id"], query=clean_msg, top_k_dense=8, top_n_final=4
    )
    context_str = "\n\n".join(
        f"[{h['doc_name']} — Page {h['page_number']} | {h['section_title']}]\n{h['parent_text']}"
        for h in hits
    )

    history = get_session_messages(req.session_id)[-6:]
    hist_text = "\n".join(f"{m['role'].upper()}: {m['content'][:600]}" for m in history)

    llm = get_auditor_llm()
    sys_prompt = (
        f"{load_agents_contract()}\n\n"
        "You are answering a follow-up question about a reference book and recent video gap audit. "
        "Ground your explanation strictly in the provided Reference Book Context and cite page numbers."
    )
    user_prompt = (
        f"Recent Conversation:\n{hist_text}\n\n"
        f"Reference Book Context:\n{context_str}\n\n"
        f"User Question: {clean_msg}"
    )
    ai_resp = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    )
    reply_text = _extract_llm_text(ai_resp.content)

    citations = [
        {
            "chunk_id": h["chunk_id"],
            "doc_name": h.get("doc_name", doc["doc_name"]),
            "section_title": h["section_title"],
            "page_number": h["page_number"],
            "score": h.get("rerank_score", h.get("dense_score", 0.0)),
            "snippet": h["parent_text"][:600],
        }
        for h in hits
    ]

    msg_id = save_chat_message(
        session_id=req.session_id,
        role="assistant",
        content=reply_text,
        user_id=req.user_id,
        citations=citations,
    )
    log_audit_event("followup_chat", {"doc_id": doc["doc_id"]}, user_id=req.user_id)

    return {
        "type": "followup",
        "message_id": msg_id,
        "reply": reply_text,
        "citations": citations,
        "awaiting_scope": False,
    }


def _run_audit_background(req: ChatTurnRequest, job_id: str) -> None:
    session_id = req.session_id
    with ACTIVE_JOBS_LOCK:
        if ACTIVE_JOB_IDS.get(session_id) != job_id:
            return
        ACTIVE_JOBS[session_id] = "running"
    try:
        result = _process_chat_turn(req)
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
    except Exception as exc:
        logger.exception("Background chat turn failed for session %s.", session_id)
        error_message = f"⚠️ **Audit Error:** {exc}"
        try:
            save_chat_message(
                session_id=session_id,
                role="assistant",
                content=error_message,
                user_id=req.user_id,
            )
        except Exception:
            logger.exception(
                "Could not save background audit error for session %s.", session_id
            )
        with ACTIVE_JOBS_LOCK:
            if ACTIVE_JOB_IDS.get(session_id) == job_id:
                job_result = ACTIVE_JOB_RESULTS.get(session_id, {})
                job_result.update({"error": error_message, "awaiting_scope": False})
                ACTIVE_JOB_RESULTS[session_id] = job_result
                ACTIVE_JOBS[session_id] = "error"


@app.post("/chat/turn/async")
def api_chat_turn_async(
    req: ChatTurnRequest, background_tasks: BackgroundTasks
) -> Dict[str, str]:
    with ACTIVE_JOBS_LOCK:
        if ACTIVE_JOBS.get(req.session_id) == "running":
            raise HTTPException(
                status_code=409,
                detail="An audit is already running for this session.",
            )

        used = _get_user_turn_count(req.user_id)
        if used + _active_user_reservations(req.user_id) >= MAX_FREE_TURNS:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Free trial limit reached ({MAX_FREE_TURNS}/"
                    f"{MAX_FREE_TURNS} audits used)."
                ),
            )

        job_id = uuid.uuid4().hex
        ACTIVE_JOB_IDS[req.session_id] = job_id
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

    background_tasks.add_task(_run_audit_background, req, job_id)
    return {"status": "running", "session_id": req.session_id, "job_id": job_id}


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
