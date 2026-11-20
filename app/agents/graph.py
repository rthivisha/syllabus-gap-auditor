import time
from typing import Any, Dict, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from app.agents.nodes.evaluator import evaluator_critic_node
from app.agents.nodes.gap_auditor import gap_auditor_worker_node
from app.agents.nodes.intent_router import classify_semantic_intent
from app.agents.nodes.pdf_qa import pdf_qa_node
from app.agents.nodes.project_guide import project_guide_node
from app.agents.nodes.repair import targeted_repair_node
from app.agents.nodes.syllabus_retriever import retrieve_syllabus_evidence_node
from app.agents.state import AuditGraphState
from app.config import get_settings
from app.ingestion.loaders.video_loader import parse_raw_transcript_text
from app.ingestion.processor import get_or_parse_video
from app.services.db.repository import enqueue_approval, log_audit_event, save_audit_run
from app.services.job_tracker import set_job_stage


def media_ingestion_node(state: AuditGraphState) -> Dict[str, Any]:
    set_job_stage(
        state.get("session_id", ""),
        "1/4 Fetching captions and extracting 25-second video windows...",
    )
    t0 = time.perf_counter()
    if state.get("raw_transcript"):
        parsed = parse_raw_transcript_text(state["raw_transcript"])
    else:
        parsed = get_or_parse_video(state.get("video_url", ""))
    timings = dict(parsed.get("step_timings") or {})
    if parsed.get("from_cache"):
        timings["total_ingestion_sec"] = round(time.perf_counter() - t0, 2)
        timings["transcription_mode"] = "sqlite_cache_hit"

    if not parsed.get("passed_guard"):
        return {
            "guard_blocked": True,
            "guard_reason": parsed.get("guard_reason", "Media guard rejected video."),
            "final_status": "blocked",
            "step_timings": timings,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "route_taken": "audit_pipeline",
            "awaiting_video": False,
        }

    return {
        "guard_blocked": False,
        "video_id": parsed["video_id"],
        "video_title": parsed["title"],
        "duration_sec": parsed["duration_sec"],
        "video_windows": parsed["windows"],
        "ocr_slides": parsed["ocr_slides"],
        "step_timings": timings,
        "route_taken": "audit_pipeline",
        "awaiting_video": False,
    }


def route_entry_intent(state: AuditGraphState) -> str:
    """Returns a string route key for add_conditional_edges."""
    return classify_semantic_intent(
        query=state.get("user_query", ""),
        has_active_doc=bool(state.get("doc_id")),
        prior_clarification_asked=bool(state.get("prior_clarification_asked")),
    )


def await_user_input_node(state: AuditGraphState) -> Dict[str, Any]:
    next_message = interrupt(
        {
            "prompt": (
                "Apply a YouTube video URL to try, or ask another question "
                "about the project."
            ),
            "chat_response": state.get("chat_response", ""),
        }
    )
    if isinstance(next_message, dict):
        return next_message
    return {"user_query": str(next_message)}


def finalize_and_escalate_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Step 5 & 6 Finalizer:
      - Separates high-confidence missed topics from unresolved max-retry borderline items (0.42 - 0.55).
      - Routes borderline items to PostgreSQL `approvals` (status='needs_human_review').
      - Persists complete `audit_runs` & `audit_logs` records in Neon PostgreSQL.
    """
    settings = get_settings()
    draft = (
        state.get("drafted_gaps")
        or state.get("draft_missed_topics")
        or state.get("missed_topics")
        or []
    )
    verdict = state.get("evaluator_verdict", "FAIL")
    attempt = state.get("attempt", 0)
    flagged_set = {
        (t.get("topic_title", "") if isinstance(t, dict) else str(t)).lower()
        for t in state.get("flagged_topics", [])
    }
    eval_passed = verdict == "PASS"
    final_verdict = "passed" if eval_passed else "failed"

    confirmed_missed: List[Dict[str, Any]] = []
    escalated_items: List[Dict[str, Any]] = []

    for item in draft:
        conf = float(item.get("confidence", 0.8))
        is_borderline = verdict == "FAIL" and (
            item["topic_title"].lower() in flagged_set
            or (settings.AMBIGUITY_LOW <= conf <= settings.AMBIGUITY_HIGH)
        )
        if is_borderline:
            escalated_items.append(item)
        else:
            confirmed_missed.append(item)

    missed_gaps = state.get("drafted_gaps") or state.get("missed_topics") or confirmed_missed or []
    all_chunks = state.get("retrieved_evidence") or state.get("cited_chunks") or []

    if missed_gaps and all_chunks:
        gap_pages = [int(g["page_number"]) for g in missed_gaps if g.get("page_number") is not None]
        page_order = {pg: idx for idx, pg in enumerate(dict.fromkeys(gap_pages))}
        matched = [c for c in all_chunks if int(c.get("page_number") or 0) in page_order]
        matched.sort(key=lambda c: page_order[int(c.get("page_number") or 0)])
        remaining = [c for c in all_chunks if c not in matched]
        prioritized_sources = (matched + remaining)[:6]
    else:
        prioritized_sources = all_chunks[:6]

    run_id = save_audit_run(
        {
            "session_id": state["session_id"],
            "user_id": state.get("user_id", "default_user"),
            "doc_id": state["doc_id"],
            "video_id": state.get("video_id", ""),
            "video_url": state["video_url"],
            "video_title": state.get("video_title", ""),
            "target_scope": state.get("target_scope", "Entire Document"),
            "evaluator_status": final_verdict,
            "retry_count": attempt,
            "latency_ms": state.get("latency_ms", 0),
            "missed_topics_json": confirmed_missed,
            "cited_chunks_json": prioritized_sources,
            "evaluation_notes": state.get("evaluation_notes", ""),
            "repair_notes": state.get("repair_notes", ""),
        }
    )

    for esc in escalated_items:
        enqueue_approval(
            {
                "run_id": run_id,
                "doc_id": state["doc_id"],
                "topic_title": esc["topic_title"],
                "book_snippet": esc["summary"],
                "page_number": int(esc.get("page_number", 1)),
                "matched_timestamp": esc.get("closest_video_timestamp", ""),
                "video_snippet": esc.get("closest_video_snippet", ""),
                "similarity_score": float(esc.get("confidence", 0.5)),
                "evaluator_notes": state.get("evaluation_notes", ""),
                "status": "needs_human_review",
            }
        )

    log_audit_event(
        action="audit_completed",
        run_id=run_id,
        user_id=state.get("user_id", "default_user"),
        detail={
            "status": final_verdict,
            "retries": attempt,
            "confirmed_gaps": len(confirmed_missed),
            "escalated_hitl": len(escalated_items),
        },
    )

    return {
        "run_id": run_id,
        "final_missed_topics": confirmed_missed,
        "escalated_items": escalated_items,
        "final_status": final_verdict,
        "route_taken": "audit_pipeline",
        "awaiting_video": False,
        "retrieved_sections": state.get("retrieved_sections", []),
        "retrieved_evidence": prioritized_sources,
        "cited_chunks": prioritized_sources,
    }


def _route_after_media(state: AuditGraphState) -> str:
    return END if state.get("guard_blocked") else "syllabus_retriever"


def _route_after_evaluator(state: AuditGraphState) -> str:
    settings = get_settings()
    if state.get("evaluator_verdict") == "PASS":
        return "finalizer"
    max_attempts = state.get("max_attempts", settings.MAX_CRITIC_RETRIES)
    if state.get("attempt", 0) < max_attempts:
        return "targeted_repair"
    return "finalizer"


def build_audit_graph():
    workflow = StateGraph(AuditGraphState)
    workflow.add_node("project_guide", project_guide_node)
    workflow.add_node("pdf_qa", pdf_qa_node)
    workflow.add_node("await_user_input", await_user_input_node)
    workflow.add_node("media_ingestion", media_ingestion_node)
    workflow.add_node("syllabus_retriever", retrieve_syllabus_evidence_node)
    workflow.add_node("gap_auditor", gap_auditor_worker_node)
    workflow.add_node("evaluator_critic", evaluator_critic_node)
    workflow.add_node("targeted_repair", targeted_repair_node)
    workflow.add_node("finalizer", finalize_and_escalate_node)

    workflow.add_conditional_edges(
        START,
        route_entry_intent,
        {
            "project_guide": "project_guide",
            "pdf_qa": "pdf_qa",
            "rag_qa": "pdf_qa",
            "audit": "media_ingestion",
            "gap_audit": "media_ingestion",
            "media_ingestion": "media_ingestion",
        },
    )
    workflow.add_edge("project_guide", END)
    workflow.add_edge("pdf_qa", END)
    workflow.add_conditional_edges(
        "await_user_input",
        route_entry_intent,
        {
            "project_guide": "project_guide",
            "pdf_qa": "pdf_qa",
            "rag_qa": "pdf_qa",
            "audit": "media_ingestion",
            "gap_audit": "media_ingestion",
            "media_ingestion": "media_ingestion",
        },
    )
    workflow.add_conditional_edges("media_ingestion", _route_after_media)
    workflow.add_edge("syllabus_retriever", "gap_auditor")
    workflow.add_edge("gap_auditor", "evaluator_critic")
    workflow.add_conditional_edges("evaluator_critic", _route_after_evaluator)
    workflow.add_edge("targeted_repair", "evaluator_critic")
    workflow.add_edge("finalizer", END)

    return workflow.compile(checkpointer=MemorySaver())


audit_graph = build_audit_graph()
