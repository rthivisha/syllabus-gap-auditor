from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AuditGraphState
from app.gateway.client import get_fast_groq_llm
from app.services.vector.embeddings import batch_encode_texts
from app.services.vector.pinecone_store import get_pinecone_index


def pdf_qa_node(state: AuditGraphState) -> Dict[str, Any]:
    doc_id = state.get("doc_id")
    if not doc_id:
        raise ValueError("A reference document is required for PDF Q&A.")

    query = state.get("user_query", "").strip()
    if not query:
        raise ValueError("A question is required for PDF Q&A.")

    doc_name = state.get("doc_name") or "Unknown PDF"
    page_count = state.get("page_count") or "unknown"

    q_vec = batch_encode_texts([query])[0].tolist()
    res = get_pinecone_index().query(
        vector=q_vec,
        filter={"doc_id": {"$eq": doc_id}},
        top_k=6,
        include_metadata=True,
        include_values=False,
    )

    raw_matches = getattr(res, "matches", None)
    if raw_matches is None and isinstance(res, dict):
        raw_matches = res.get("matches", [])
    matches = raw_matches or []

    evidence: List[Dict[str, Any]] = []
    context: List[str] = []
    for index, match in enumerate(matches, start=1):
        md = getattr(match, "metadata", None)
        if md is None and isinstance(match, dict):
            md = match.get("metadata", {})
        md = md or {}

        section_title = str(md.get("section_title") or "Unlabeled section")
        page_number = int(md.get("page_number") or 1)
        text = str(
            md.get("parent_text") or md.get("child_text") or md.get("text") or ""
        ).strip()
        match_id = str(
            getattr(match, "id", None)
            or (match.get("id") if isinstance(match, dict) else "")
            or ""
        )
        score = getattr(match, "score", None)
        if score is None and isinstance(match, dict):
            score = match.get("score", 0.0)
        score = round(float(score or 0.0), 4)

        evidence.append(
            {
                "chunk_id": match_id,
                "doc_name": str(md.get("doc_name") or md.get("filename") or doc_name),
                "section_title": section_title,
                "page_number": page_number,
                "score": score,
                "snippet": text[:700],
            }
        )
        context.append(
            f"[Source {index}: {section_title}, p. {page_number}]\n{text[:1800]}"
        )

    context_block = "\n\n".join(context) if context else "No direct excerpts retrieved from Pinecone."
    llm = get_fast_groq_llm()
    resp = llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are the Syllabus Gap Audit AI assistant.\n"
                    f"CURRENTLY ACTIVE REFERENCE PDF: '{doc_name}' (Total Pages: {page_count}, Doc ID: {doc_id}).\n\n"
                    "Answer the user's question using the active PDF metadata above and the retrieved document excerpts below. "
                    "Cite exact page numbers (e.g., Page 4) when referencing content.\n\n"
                    "MANDATORY FOLLOW-UP:\n"
                    "End your response with a concise follow-up inviting the user to **paste a YouTube lecture URL** "
                    f"to audit which topics from '{doc_name}' the video missed.\n\n"
                    f"=== RETRIEVED EXCERPTS FROM {doc_name} ===\n{context_block}"
                )
            ),
            HumanMessage(content=query),
        ]
    )

    answer = resp.content
    if isinstance(answer, list):
        answer = "\n".join(
            str(block.get("text") or "")
            if isinstance(block, dict) and block.get("type") == "text"
            else block
            for block in answer
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") == "text")
        )
    answer = str(answer or "").strip()
    if not answer:
        raise RuntimeError("The PDF Q&A model returned an empty response.")

    return {
        "chat_response": answer,
        "retrieved_evidence": evidence,
        "route_taken": "pdf_qa",
        "llm_used": True,
    }
