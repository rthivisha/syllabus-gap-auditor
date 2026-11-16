import json
import time
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.nodes.scope_router import load_agents_contract
from app.agents.state import AuditGraphState, WorkerDraftOutput
from app.gateway.client import get_auditor_llm


def gap_auditor_worker_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Step 4.1 Worker Node:
      - Loads invariant `agents.md` contract (does NOT load `skills.md` yet).
      - Diffs candidate book parent chunks against the video's 25s speech + Slide OCR windows.
      - Drafts a clean, concise list of missed topics with page numbers & chunk IDs.
    """
    t0 = time.perf_counter()
    step_timings = dict(state.get("step_timings") or {})
    candidates = state.get("candidate_gaps", [])
    if not candidates:
        step_timings["gap_auditor_sec"] = round(time.perf_counter() - t0, 2)
        return {
            "draft_missed_topics": [],
            "attempt": state.get("attempt", 0),
            "step_timings": step_timings,
        }

    video_digest = "\n".join(
        f"[{w['timestamp']}] {w['combined_text'][:320]}"
        for w in state.get("video_windows", [])[:40]
    )

    books_payload = json.dumps(
        [
            {
                "section_title": c["section_title"],
                "page_number": c["page_number"],
                "chunk_id": c["chunk_id"],
                "best_video_similarity": c["best_video_similarity"],
                "closest_timestamp": c["closest_timestamp"],
                "closest_video_text": c["closest_video_text"],
                "book_parent_text": c["parent_text"][:1400],
            }
            for c in candidates[:18]
        ],
        indent=2,
    )

    llm = get_auditor_llm().with_structured_output(WorkerDraftOutput)
    sys_prompt = (
        f"{load_agents_contract()}\n\n"
        "You are the Syllabus Gap Worker Node.\n"
        "Compare the Reference Book Parent Chunks against the Video Transcript & Slide OCR Windows.\n"
        "Identify ONLY the important book topics/concepts that were genuinely MISSED in the video.\n"
        "Rules:\n"
        "1. A concept is COVERED only when the speaker explains it (including with a true synonym) or slide OCR shows it with supporting context; a bare term mention is insufficient.\n"
        "2. Every missed topic MUST cite the exact `section_title`, `page_number`, and `chunk_id` provided.\n"
        "3. Write a crisp 1-2 sentence summary of what the book teaches for each missed topic.\n"
        "4. Do not mark a section covered because the video mentions only its broad domain; "
        "compare the specific mechanisms and engineering rules the book teaches.\n"
        "5. When supplied book chunks teach specific mechanisms such as Bash/Git safety "
        "constraints, concurrency-safe tool scheduling, or recovery/autocompact boundaries, "
        "flag them if absent from the transcript and slide OCR.\n"
        "6. An introductory-video label never excuses a missing mechanism. Only omit a "
        "candidate when the video genuinely explains its core mechanism."
    )

    user_prompt = (
        f"Document: {state.get('doc_name')} | Scope: {state.get('target_scope')}\n"
        f"Video Title: {state.get('video_title')}\n\n"
        f"=== VIDEO TRANSCRIPT & SLIDE OCR WINDOWS ===\n{video_digest}\n\n"
        f"=== CANDIDATE BOOK SECTIONS TO AUDIT ===\n{books_payload}"
    )

    result: WorkerDraftOutput = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    )
    step_timings["gap_auditor_sec"] = round(time.perf_counter() - t0, 2)

    return {
        "draft_missed_topics": [item.model_dump() for item in result.missed_topics],
        "attempt": state.get("attempt", 0),
        "step_timings": step_timings,
    }
