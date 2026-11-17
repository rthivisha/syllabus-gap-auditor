import re
import time
from typing import Any, Dict, List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from app.agents.nodes.evaluator import evaluator_critic_node
from app.agents.nodes.gap_auditor import gap_auditor_worker_node
from app.agents.nodes.project_guide import project_guide_node
from app.agents.nodes.repair import targeted_repair_node
from app.agents.nodes.syllabus_retriever import retrieve_syllabus_evidence_node
from app.agents.state import AuditGraphState
from app.config import get_settings
from app.ingestion.processor import get_or_parse_video
from app.services.db.repository import enqueue_approval, log_audit_event, save_audit_run
from app.services.job_tracker import set_job_stage


def media_ingestion_node(state: AuditGraphState) -> Dict[str, Any]:
    set_job_stage(
        state.get("session_id", ""),
        "1/4 Fetching captions and extracting 25-second video windows...",
    )
    t0 = time.perf_counter()
    parsed = get_or_parse_video(state["video_url"])
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


def _has_youtube_url(text: str) -> bool:
    return bool(
        re.search(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/\S+", text or "")
    )


def route_entry_intent(state: AuditGraphState) -> str:
    query = state.get("user_query", "")
    if _has_youtube_url(query) or (
        state.get("prior_clarification_asked") and state.get("video_url")
    ):
        return "media_ingestion"
    return "project_guide"


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
    flagged_set = {t.lower() for t in state.get("flagged_topics", [])}
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
            "cited_chunks_json": state.get("cited_chunks", []),
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
            "media_ingestion": "media_ingestion",
        },
    )
    workflow.add_edge("project_guide", "await_user_input")
    workflow.add_conditional_edges(
        "await_user_input",
        route_entry_intent,
        {
            "project_guide": "project_guide",
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
