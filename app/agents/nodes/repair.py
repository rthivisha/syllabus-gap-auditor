import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.agents.nodes.grounding import (
    match_citation,
    state_citation_records,
)
from app.agents.nodes.evaluator import load_skills_rubric
from app.agents.nodes.scope_router import load_agents_contract
from app.agents.state import AuditGraphState, MissedTopicItem
from app.gateway.client import get_structured_auditor_llm
from app.services.job_tracker import set_job_stage


class BatchAuditResult(BaseModel):
    repaired_missed_topics: List[MissedTopicItem] = Field(default_factory=list)
    repair_notes: str = Field(
        description="Clear summary of what was removed, added, or adjusted and why."
    )


def _ground_repaired_topic(
    item: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    *,
    allow_correction: bool,
) -> Optional[Dict[str, Any]]:
    matched = match_citation(item, evidence, allow_correction=allow_correction)
    if matched is None:
        return None
    summary = str(item.get("summary") or item.get("description") or "")
    closest_timestamp = str(
        item.get("closest_video_timestamp") or item.get("video_check_note") or ""
    )
    return {
        **item,
        "chunk_id": matched["chunk_id"],
        "section_title": matched["section_title"],
        "page_number": matched["page_number"],
        "description": summary,
        "explanation": summary,
        "closest_timestamp": closest_timestamp,
    }


def targeted_repair_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Make one targeted repair using only the original retrieved evidence.
    """
    t0 = time.perf_counter()
    step_timings = dict(state.get("step_timings") or {})
    set_job_stage(
        state.get("session_id", ""),
        "4/4 Repairing topics flagged by the evaluator...",
    )
    next_attempt = state.get("attempt", 0) + 1
    eval_notes = state.get("evaluation_notes", "")
    flagged_topics = {
        str(topic).casefold() for topic in state.get("flagged_topics", [])
    }
    draft = (
        state.get("drafted_gaps")
        or state.get("draft_missed_topics")
        or state.get("missed_topics")
        or []
    )
    evidence = state_citation_records(state)
    evidence_payload = [
        {
            "chunk_id": item["chunk_id"],
            "section_title": item["section_title"],
            "page_number": item["page_number"],
            "text": str(
                item.get("parent_text")
                or item.get("summary")
                or item.get("snippet")
                or item.get("text")
                or ""
            )[:1400],
        }
        for item in evidence[:30]
    ]
    video_context = [
        {
            "timestamp": str(window.get("timestamp") or ""),
            "text": str(window.get("combined_text") or "")[:320],
        }
        for window in state.get("video_windows", [])[:20]
    ]
    slide_context = [
        {
            "timestamp": str(slide.get("timestamp") or ""),
            "text": str(slide.get("text") or "")[:320],
        }
        for slide in state.get("ocr_slides", [])[:20]
    ]

    llm = get_structured_auditor_llm(BatchAuditResult)
    sys_prompt = (
        f"{load_agents_contract()}\n\n"
        f"{load_skills_rubric()}\n\n"
        "You are the Targeted Repair Node. Fix ONLY the issues identified in the Evaluator's feedback.\n"
        "Use only the supplied original retrieved evidence. Preserve unflagged valid topics; "
        "correct a citation only when the exact source chunk is identifiable, and add a topic "
        "only when its omission is supported by that evidence. Never invent or use another "
        "retrieval result. For every retained topic, include an exact chunk_id, section_title, "
        "page_number, a concise summary, a verbatim evidence_quote, and the closest checked "
        "video_check_note timestamp. Document every change in repair_notes."
    )

    user_prompt = (
        f"Attempt: {next_attempt}\n"
        f"Flagged Topics:\n{json.dumps(sorted(flagged_topics), indent=2)}\n\n"
        f"Evaluator Feedback:\n{eval_notes}\n\n"
        f"Current Draft:\n{json.dumps(draft, indent=2)}\n\n"
        f"Original Retrieved Evidence:\n{json.dumps(evidence_payload, indent=2)}\n\n"
        f"Video Transcript Windows:\n{json.dumps(video_context, indent=2)}\n\n"
        f"Slide OCR Windows:\n{json.dumps(slide_context, indent=2)}"
    )

    repaired: BatchAuditResult = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    )
    repaired_list: List[Dict[str, Any]] = []
    rejected_topics: List[str] = []
    seen_topics = set()
    preserved_count = 0

    def append_grounded(item: Dict[str, Any], *, allow_correction: bool) -> bool:
        grounded = _ground_repaired_topic(
            item, evidence, allow_correction=allow_correction
        )
        if grounded is None:
            return False
        topic_key = (
            str(grounded.get("topic_title") or "").casefold(),
            grounded["chunk_id"],
        )
        if topic_key not in seen_topics:
            seen_topics.add(topic_key)
            repaired_list.append(grounded)
        return True

    for item in draft:
        if str(item.get("topic_title") or "").casefold() not in flagged_topics:
            previous_count = len(repaired_list)
            if append_grounded(item, allow_correction=False):
                preserved_count += len(repaired_list) - previous_count

    for topic in repaired.repaired_missed_topics:
        item = topic.model_dump()
        if not append_grounded(item, allow_correction=True):
            rejected_topics.append(item.get("topic_title", "Untitled topic"))
    repair_notes = repaired.repair_notes
    if preserved_count:
        repair_notes = (
            f"{repair_notes} Preserved {preserved_count} unflagged grounded topic(s)."
        ).strip()
    if rejected_topics:
        repair_notes = (
            f"{repair_notes} "
            f"Discarded ungrounded items: {', '.join(rejected_topics)}."
        ).strip()

    step_timings["targeted_repair_sec"] = round(time.perf_counter() - t0, 2)

    return {
        "attempt": next_attempt,
        "draft_missed_topics": repaired_list,
        "drafted_gaps": repaired_list,
        "missed_topics": repaired_list,
        "repair_notes": repair_notes,
        "step_timings": step_timings,
    }
