from typing import Dict, List
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from app.gateway.client import get_fast_groq_llm

_INTENT_ANCHORS: Dict[str, List[str]] = {
    "project_guide": [
        "Tell me about this project and what this application does",
        "How does Syllabus Gap Audit AI work, what is its architecture and tech stack?",
        "How do I use this tool and how does the LangGraph workflow run?",
    ],
    "pdf_qa": [
        "What is inside this uploaded PDF document and what are its main chapters?",
        "Summarize the concepts, sections, and pages from the reference book",
        "Explain the topics covered in the textbook I uploaded",
    ],
}

_ANCHOR_VECS: Dict[str, np.ndarray] = {}


def _get_anchor_vectors() -> Dict[str, np.ndarray]:
    if not _ANCHOR_VECS:
        from app.services.vector.embeddings import batch_encode_texts
        for intent, phrases in _INTENT_ANCHORS.items():
            mat = batch_encode_texts(phrases)
            _ANCHOR_VECS[intent] = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return _ANCHOR_VECS


def classify_semantic_intent(
    query: str,
    has_active_doc: bool,
    prior_clarification_asked: bool,
) -> str:
    """
    1ms semantic intent classifier using pre-embedded BGE anchors + Groq tie-breaker.
    Routes 'project_guide', 'pdf_qa', or 'audit' accurately regardless of phrasing.
    """
    q = (query or "").strip()
    q_low = q.lower()

    # 1. Deterministic Video Audit check
    if prior_clarification_asked or "youtube.com/" in q_low or "youtu.be/" in q_low:
        return "audit"

    # 2. Fast Semantic Similarity against Project vs. PDF anchors (~1ms)
    from app.services.vector.embeddings import batch_encode_texts
    q_vec = batch_encode_texts([q])[0]
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

    anchors = _get_anchor_vectors()
    proj_score = float(np.max(anchors["project_guide"] @ q_vec))
    pdf_score = float(np.max(anchors["pdf_qa"] @ q_vec)) if has_active_doc else 0.0

    # Clear semantic winner (margin >= 0.06)
    if proj_score >= pdf_score + 0.06:
        return "project_guide"
    if has_active_doc and pdf_score >= proj_score + 0.06:
        return "pdf_qa"

    # 3. Borderline semantic tie -> 0.25s Groq classification
    if not has_active_doc:
        return "project_guide"
    try:
        llm = get_fast_groq_llm()
        resp = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Classify the user query into ONE word:\n"
                        "- PROJECT: asking about this software app, its features, workflow, or how to use it.\n"
                        "- PDF: asking about the contents, chapters, or concepts inside the uploaded PDF document.\n"
                        "Reply ONLY with PROJECT or PDF."
                    )
                ),
                HumanMessage(content=q),
            ]
        )
        label = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip().upper()
        return "pdf_qa" if "PDF" in label else "project_guide"
    except Exception:
        return "project_guide"
