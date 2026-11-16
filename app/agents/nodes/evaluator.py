import json
import time
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.nodes.scope_router import load_agents_contract
from app.agents.state import AuditGraphState, EvaluatorVerdict
from app.config import get_settings
from app.gateway.client import get_auditor_llm


def load_skills_rubric() -> str:
    """Progressive Disclosure: loads `skills/skills.md` ONLY inside Evaluator & Repair nodes."""
    path = get_settings().SKILLS_MD_PATH
    return path.read_text(encoding="utf-8") if path.exists() else ""


def evaluator_critic_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Step 4.2 Evaluator Node:
      - Dynamically loads `skills/skills.md` rubric.
      - Runs deterministic chunk-ID grounding check + LLM rubric audit.
      - Outputs PASS or FAIL with actionable `evaluation_notes`.
    """
    t0 = time.perf_counter()
    step_timings = dict(state.get("step_timings") or {})
    draft = state.get("draft_missed_topics", [])
    valid_chunk_ids = {c["chunk_id"] for c in state.get("candidate_gaps", [])} | {
        c["chunk_id"] for c in state.get("cited_chunks", [])
    }

    # 1. Deterministic Closed-World Chunk ID Check
    ungrounded = [
        d["topic_title"]
        for d in draft
        if valid_chunk_ids and d.get("chunk_id") not in valid_chunk_ids
    ]
    if ungrounded:
        step_timings["evaluator_sec"] = round(time.perf_counter() - t0, 2)
        return {
            "evaluator_verdict": "FAIL",
            "evaluator_confidence": 0.40,
            "evaluation_notes": f"FAIL [RULE_02_GROUNDEDNESS]: Ungrounded chunk_ids for topics: {ungrounded}. Remove or remap to valid retrieved chunk IDs.",
            "flagged_topics": ungrounded,
            "step_timings": step_timings,
        }

    # 2. LLM Rubric Audit governed by `skills.md`
    all_slide_ocr = "\n".join(
        f"[{s['timestamp']}] {s['text']}" for s in state.get("ocr_slides", [])[:50]
    )
    all_speech = "\n".join(
        f"[{w['timestamp']}] {w['speech_text']}"
        for w in state.get("video_windows", [])[:40]
    )

    llm = get_auditor_llm().with_structured_output(EvaluatorVerdict)
    sys_prompt = (
        f"{load_agents_contract()}\n\n"
        "=== LOADED EVALUATION RUBRIC (skills.md) ===\n"
        f"{load_skills_rubric()}\n\n"
        "Audit the Worker's drafted Missed Topics list strictly against every rule in skills.md.\n"
        "Return verdict='PASS' only if all rules pass. Otherwise return verdict='FAIL' with specific, actionable remediation instructions in evaluation_notes."
    )

    user_prompt = (
        f"Drafted Missed Topics:\n{json.dumps(draft, indent=2)}\n\n"
        f"Raw Video Slide OCR:\n{all_slide_ocr[:4000]}\n\n"
        f"Spoken Video Transcript:\n{all_speech[:4000]}"
    )

    verdict_obj: EvaluatorVerdict = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    )
    step_timings["evaluator_sec"] = round(time.perf_counter() - t0, 2)

    return {
        "evaluator_verdict": verdict_obj.verdict.upper(),
        "evaluator_confidence": verdict_obj.confidence_score,
        "evaluation_notes": verdict_obj.evaluation_notes,
        "flagged_topics": verdict_obj.flagged_topics,
        "step_timings": step_timings,
    }
