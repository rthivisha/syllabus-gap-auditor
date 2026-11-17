import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.nodes.grounding import match_citation, normalize_citation_records
from app.agents.nodes.scope_router import load_agents_contract
from app.agents.state import AuditGraphState, WorkerDraftOutput
from app.gateway.client import get_structured_auditor_llm
from app.services.job_tracker import set_job_stage


def _chunk_list(
    items: List[Dict[str, Any]], batch_size: int = 3
) -> List[List[Dict[str, Any]]]:
    return [
        items[index : index + batch_size]
        for index in range(0, len(items), batch_size)
    ]


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


def _audit_single_batch(
    batch_idx: int,
    evidence_batch: List[Dict[str, Any]],
    video_digest: str,
    state: AuditGraphState,
) -> WorkerDraftOutput:
    books_payload = json.dumps(
        [
            {
                "section_title": candidate["section_title"],
                "page_number": candidate["page_number"],
                "chunk_id": candidate["chunk_id"],
                "best_video_similarity": candidate["best_video_similarity"],
                "closest_timestamp": candidate["closest_timestamp"],
                "closest_video_text": candidate["closest_video_text"],
                "book_parent_text": candidate["parent_text"][:1400],
            }
            for candidate in evidence_batch
        ],
        indent=2,
    )
    llm = get_structured_auditor_llm(WorkerDraftOutput)
    system_prompt = (
        f"{load_agents_contract()}\n\n"
        "You are the Syllabus Gap Worker Node. Compare these reference book chunks "
        "against the timestamped video transcript and slide OCR windows.\n"
        "Classify each important concept as missed or covered; do not emit unrelated topics.\n"
        "For every missed or covered topic, cite the exact section_title, page_number, and "
        "chunk_id supplied; provide an evidence_quote of at most 20 words copied "
        "verbatim from that chunk; and provide a video_check_note containing the "
        "closest checked MM:SS timestamp. Keep each summary to 1-2 sentences and "
        "no more than 30 words. A synonym or clearly explained slide concept counts "
        "as covered; a bare term mention does not. Do not excuse omitted mechanisms "
        "because the lecture is introductory. Put each candidate in exactly one of "
        "missed_topics or covered_topics."
    )
    user_prompt = (
        f"Document: {state.get('doc_name')} | Scope: {state.get('target_scope')}\n"
        f"Video Title: {state.get('video_title')}\n\n"
        f"=== VIDEO WINDOWS ===\n{video_digest}\n\n"
        f"=== CANDIDATE BOOK SECTIONS (BATCH {batch_idx + 1}) ===\n"
        f"{books_payload}"
    )
    return llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )


def gap_auditor_worker_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Step 4.1 Worker Node:
      - Loads invariant `agents.md` contract (does NOT load `skills.md` yet).
      - Diffs candidate book parent chunks against the video's 25s speech + Slide OCR windows.
      - Drafts concise, self-verifying missed and covered topic records.
    """
    t0 = time.perf_counter()
    step_timings = dict(state.get("step_timings") or {})
    session_id = state.get("session_id", "")
    candidates = state.get("candidate_gaps", [])[:18]
    batches = _chunk_list(candidates)
    set_job_stage(
        session_id,
        f"3/4 Auditing {len(candidates)} candidate sections across "
        f"{len(batches)} batches...",
    )
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

    video_digest = "\n".join(
        f"[{w['timestamp']}] {w['combined_text'][:320]}"
        for w in state.get("video_windows", [])[:40]
    )
    results: Dict[int, WorkerDraftOutput] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _audit_single_batch, batch_idx, batch, video_digest, state
            ): batch_idx
            for batch_idx, batch in enumerate(batches)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            batch_idx = futures[future]
            results[batch_idx] = future.result()
            set_job_stage(
                session_id,
                f"3/4 Auditing candidate sections... "
                f"{completed}/{len(batches)} batches complete.",
            )

    missed_topics = []
    covered_topics = []
    seen_missed_topics = set()
    seen_covered_topics = set()
    for batch_idx in sorted(results):
        for item in results[batch_idx].missed_topics:
            topic_key = item.topic_title.strip().casefold()
            if topic_key not in seen_missed_topics:
                seen_missed_topics.add(topic_key)
                missed_topics.append(item.model_dump())
        for item in results[batch_idx].covered_topics:
            topic_key = item.topic_title.strip().casefold()
            if topic_key not in seen_covered_topics:
                seen_covered_topics.add(topic_key)
                covered_topics.append(item.model_dump())
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
