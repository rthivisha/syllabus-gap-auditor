from typing import Any, Dict, List

from app.agents.nodes.grounding import match_citation, normalize_citation_records
from app.agents.state import AuditGraphState
from app.config import get_settings
from app.services.job_tracker import set_job_stage


def load_skills_rubric() -> str:
    """Load the verification and repair policy for the Evaluator and Repair nodes."""
    path = get_settings().SKILLS_MD_PATH
    return path.read_text(encoding="utf-8")


def evaluator_critic_node(state: AuditGraphState) -> Dict[str, Any]:
    """Pass grounded drafts immediately; otherwise request one targeted repair."""
    session_id = state.get("session_id", "")
    set_job_stage(session_id, "4/4 Verifying missed topics against retrieved PDF chunks...")

    evidence = normalize_citation_records(
        state.get("retrieved_evidence")
        or state.get("retrieved_sections")
        or state.get("candidate_gaps")
        or state.get("cited_chunks")
        or []
    )
    retrieved = (
        state.get("retrieved_evidence")
        or state.get("retrieved_sections")
        or state.get("candidate_gaps")
        or []
    )
    drafted = (
        state.get("drafted_gaps")
        or state.get("draft_missed_topics")
        or state.get("missed_topics")
        or []
    )
    issues: List[str] = []
    invalid_topics: List[str] = []

    if not drafted:
        if len(retrieved) >= 3:
            issues.append(
                "Zero missed topics were drafted despite retrieving at least three "
                "textbook sections. Re-examine retrieved_evidence for specific "
                "mechanisms omitted from the video."
            )
        else:
            issues.append(
                "No missed topics were drafted. Re-examine retrieved_evidence and "
                "add only concepts that the video skipped or did not explain."
            )

    for item in drafted:
        if match_citation(item, evidence) is None:
            topic_title = str(item.get("topic_title") or "Untitled topic")
            invalid_topics.append(topic_title)
            issues.append(
                f"Topic '{topic_title}' must cite an exact retrieved chunk_id, "
                "section_title, and page_number; supplied citation was "
                f"({item.get('chunk_id')!r}, {item.get('section_title')!r}, "
                f"{item.get('page_number')!r})."
            )

    step_timings = dict(state.get("step_timings") or {})
    step_timings["evaluator_sec"] = 0.0
    if not issues:
        set_job_stage(session_id, "4/4 Verified missed-topic citations.")
        return {
            "draft_missed_topics": drafted,
            "drafted_gaps": drafted,
            "missed_topics": drafted,
            "evaluator_passed": True,
            "evaluator_verdict": "PASS",
            "evaluator_confidence": 1.0,
            "evaluation_notes": (
                f"All {len(drafted)} missed topics exactly match retrieved evidence."
            ),
            "flagged_topics": [],
            "evaluator_mode": "grounding_fast_pass",
            "step_timings": step_timings,
        }

    flagged_topics = list(dict.fromkeys(invalid_topics))
    if not flagged_topics:
        flagged_topics = list(
            dict.fromkeys(
                str(record.get("section_title") or "")
                for record in retrieved
                if record.get("section_title")
            )
        )[:5]
    return {
        "draft_missed_topics": drafted,
        "drafted_gaps": drafted,
        "missed_topics": drafted,
        "evaluator_passed": False,
        "evaluator_verdict": "FAIL",
        "evaluator_confidence": 0.0,
        "evaluation_notes": " ".join(issues),
        "flagged_topics": flagged_topics,
        "evaluator_mode": "grounding_repair_required",
        "step_timings": step_timings,
    }
