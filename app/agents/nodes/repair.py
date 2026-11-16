import json
import time
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.agents.nodes.evaluator import load_skills_rubric
from app.agents.nodes.scope_router import load_agents_contract
from app.agents.state import AuditGraphState, MissedTopicItem
from app.gateway.client import get_auditor_llm
from app.services.retrieval.pinecone_service import PineconeService


class RepairOutput(BaseModel):
    repaired_missed_topics: List[MissedTopicItem] = Field(default_factory=list)
    repair_notes: str = Field(
        description="Clear summary of what was removed, added, or adjusted and why."
    )


def targeted_repair_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Step 4.3 Targeted Re-Retrieval & Repair Node (bounded by max_attempts=2):
      - Executes expanded keyword re-retrieval against Pinecone & raw Slide OCR for flagged topics.
      - Repairs the draft based on Evaluator's `evaluation_notes` and `skills.md`.
    """
    t0 = time.perf_counter()
    step_timings = dict(state.get("step_timings") or {})
    next_attempt = state.get("attempt", 0) + 1
    flagged = state.get("flagged_topics", [])
    eval_notes = state.get("evaluation_notes", "")
    draft = state.get("draft_missed_topics", [])

    # Targeted Re-Retrieval on flagged topics
    pc = PineconeService()
    re_retrieved_evidence: List[Dict[str, Any]] = []
    for topic in flagged[:5]:
        hits = pc.search_and_rerank(
            doc_id=state["doc_id"],
            query=topic,
            section_titles=state.get("scoped_section_titles") or None,
            top_k_dense=10,
            top_n_final=2,
        )
        re_retrieved_evidence.extend(hits)

    # Direct multi-word phrase check in raw Slide OCR & Transcript
    ocr_hits = []
    for topic in flagged:
        tokens = [tok for tok in topic.lower().split() if len(tok) > 3]
        for slide in state.get("ocr_slides", []):
            s_low = slide["text"].lower()
            matches = sum(1 for tok in tokens if tok in s_low)
            if tokens and (matches / len(tokens)) >= 0.6:
                ocr_hits.append(f"[{slide['timestamp']}] {slide['text']}")

    llm = get_auditor_llm().with_structured_output(RepairOutput)
    sys_prompt = (
        f"{load_agents_contract()}\n\n"
        f"{load_skills_rubric()}\n\n"
        "You are the Targeted Repair Node. Fix ONLY the issues identified in the Evaluator's feedback.\n"
        "If a flagged topic is confirmed present in Slide OCR or spoken audio, remove it from missed_topics. "
        "Document every change in `repair_notes`."
    )

    user_prompt = (
        f"Attempt: {next_attempt}\n"
        f"Evaluator Feedback:\n{eval_notes}\n\n"
        f"Current Draft:\n{json.dumps(draft, indent=2)}\n\n"
        f"Re-Retrieved Book Chunks:\n{json.dumps(re_retrieved_evidence, indent=2)[:3000]}\n\n"
        f"Matching Slide OCR Lines:\n" + "\n".join(ocr_hits[:15])
    )

    repaired: RepairOutput = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
    )
    step_timings["targeted_repair_sec"] = round(time.perf_counter() - t0, 2)

    return {
        "attempt": next_attempt,
        "draft_missed_topics": [
            t.model_dump() for t in repaired.repaired_missed_topics
        ],
        "repair_notes": repaired.repair_notes,
        "step_timings": step_timings,
    }
