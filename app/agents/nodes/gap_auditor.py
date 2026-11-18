import time
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.nodes.grounding import match_citation, normalize_citation_records
from app.agents.nodes.scope_router import load_agents_contract
from app.agents.state import AuditGraphState, WorkerDraftOutput
from app.gateway.client import get_structured_auditor_llm
from app.services.job_tracker import set_job_stage


def _format_video_window_str(val: Any) -> str:
    if isinstance(val, dict):
        ts = val.get("timestamp", "")
        txt = val.get("combined_text") or val.get("text") or ""
        return f"[{ts}] {txt}"[:140] if (ts or txt) else "None"
    if isinstance(val, str) and val:
        return val[:140]
    return "None"


def _format_missed_topics(
    items: List[Dict[str, Any]], candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    evidence = normalize_citation_records(candidates)
    formatted: List[Dict[str, Any]] = []
    for item in items:
        matched = match_citation(item, evidence, allow_correction=True)
        summary = str(
            item.get("summary")
            or item.get("description")
            or item.get("explanation")
            or ""
        )
        closest_timestamp = str(
            item.get("closest_video_timestamp") or item.get("video_check_note") or ""
        )
        formatted.append(
            {
                **item,
                "chunk_id": (matched or {}).get("chunk_id", item.get("chunk_id", "")),
                "section_title": (matched or {}).get(
                    "section_title", item.get("section_title", "")
                ),
                "topic_title": item.get("topic_title", ""),
                "page_number": (matched or {}).get(
                    "page_number", item.get("page_number", 1)
                ),
                "description": summary,
                "explanation": summary,
                "closest_timestamp": closest_timestamp,
            }
        )
    return formatted


def gap_auditor_worker_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Step 4.1 Worker Node (Single Compact Pass):
      - Compares top 6 lowest-similarity textbook sections against closest video matches in 1 call.
      - ~1,400 tokens total (instead of 10,000+ across multi-batches) for ~2.5-3.5s runtime.
    """
    t0 = time.perf_counter()
    step_timings = dict(state.get("step_timings") or {})
    session_id = state.get("session_id", "")
    candidates = state.get("candidate_gaps", [])[:12]

    if not candidates:
        step_timings["gap_auditor_sec"] = round(time.perf_counter() - t0, 2)
        return {
            "draft_missed_topics": [],
            "drafted_gaps": [],
            "missed_topics": [],
            "draft_covered_topics": [],
            "attempt": state.get("attempt", 0),
            "step_timings": step_timings,
        }

    set_job_stage(
        session_id,
        f"3/4 Auditing {len(candidates)} candidate sections in a single fast pass...",
    )

    compact_sections = "\n".join(
        f"- [Chunk: {c.get('chunk_id', '')} | Page {c.get('page_number', 1)} | {c.get('section_title', '')} | "
        f"Closest Video Match: {_format_video_window_str(c.get('closest_video_window') or c.get('closest_video_text') or c.get('closest_timestamp'))}]: "
        f"{str(c.get('text') or c.get('parent_text') or '')[:350]}"
        for c in candidates
    )

    llm = get_structured_auditor_llm(WorkerDraftOutput)
    system_prompt = (
        "You are a strict Syllabus & Document Coverage Auditor.\n"
        "Compare the PDF sections below against the YouTube video transcript windows.\n"
        "RULES:\n"
        "1. Identify EVERY substantive topic, rule, eligibility criterion, or mechanism in the PDF that the video completely skipped or failed to explain.\n"
        "2. IGNORE non-content boilerplate (Table of Contents, Figures and Tables, QR code/Scan pages, cover pages).\n"
        "3. Keep 'topic_title' crisp (4-8 words) and 'description' under 22 words so teachers/viewers can scan every missed topic instantly.\n"
        "4. Cite the exact 'page_number' and 'section_title' from the provided chunks."
    )
    user_prompt = (
        f"Document: {state.get('doc_name')} | Scope: {state.get('target_scope')}\n"
        f"Video Title: {state.get('video_title')}\n\n"
        f"=== CANDIDATE BOOK SECTIONS & CLOSEST VIDEO MATCHES ===\n"
        f"{compact_sections}"
    )

    draft_result: WorkerDraftOutput = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    missed_topics = []
    covered_topics = []
    seen_missed_topics = set()
    seen_covered_topics = set()

    for item in (getattr(draft_result, "missed_topics", None) or []):
        topic_key = item.topic_title.strip().casefold()
        if topic_key not in seen_missed_topics:
            seen_missed_topics.add(topic_key)
            missed_topics.append(item.model_dump() if hasattr(item, "model_dump") else dict(item))

    for item in (getattr(draft_result, "covered_topics", None) or []):
        topic_key = item.topic_title.strip().casefold()
        if topic_key not in seen_covered_topics:
            seen_covered_topics.add(topic_key)
            covered_topics.append(item.model_dump() if hasattr(item, "model_dump") else dict(item))

    covered_topics = [
        item
        for item in covered_topics
        if item["topic_title"].strip().casefold() not in seen_missed_topics
    ]
    formatted_missed_topics = _format_missed_topics(missed_topics, candidates)

    step_timings["gap_auditor_sec"] = round(time.perf_counter() - t0, 2)

    return {
        "draft_missed_topics": formatted_missed_topics,
        "drafted_gaps": formatted_missed_topics,
        "missed_topics": formatted_missed_topics,
        "draft_covered_topics": covered_topics,
        "attempt": state.get("attempt", 0),
        "step_timings": step_timings,
    }
