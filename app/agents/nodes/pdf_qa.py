from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import AuditGraphState
from app.gateway.client import get_fast_groq_llm
from app.services.job_tracker import set_job_stage
from app.services.vector.embeddings import batch_encode_texts
from app.services.vector.pinecone_store import get_pinecone_index


def pdf_qa_node(state: AuditGraphState) -> Dict[str, Any]:
    """Sub-2s grounded PDF Q&A: single Pinecone HNSW query (no FlashRank) + Groq synthesis."""
    session_id = state.get("session_id", "")
    doc_id = state.get("doc_id", "")
    doc_name = state.get("doc_name") or "Active PDF"
    page_count = state.get("page_count") or "unknown"
    query = state.get("user_query", "")

    if not doc_id:
        return {
            "chat_response": "No reference PDF is active. Please select or upload a PDF in Step 1 first.",
            "route_taken": "pdf_qa",
        }

    set_job_stage(session_id, f"⚡ Querying {doc_name}...")
    q_vec = batch_encode_texts([query])[0].tolist()
    res = get_pinecone_index().query(
        vector=q_vec,
        filter={"doc_id": {"$eq": doc_id}},
        top_k=5,
        include_metadata=True,
        include_values=False,
    )

    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    chunks = []
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else (getattr(m, "metadata", None) or {})
        chunks.append(
            {
                "chunk_id": m.get("id") if isinstance(m, dict) else getattr(m, "id", ""),
                "page_number": meta.get("page_number", 1),
                "section_title": meta.get("section_title", ""),
                "text": meta.get("text", "") or meta.get("parent_text", "") or meta.get("child_text", ""),
                "bi_encoder_score": round(float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0) or 0.0), 3),
            }
        )

    if not chunks:
        return {
            "chat_response": f"No indexed chunks found for **{doc_name}**. Try re-ingesting the PDF in Step 1.",
            "route_taken": "pdf_qa",
        }

    context_block = "\n".join(
        f"- [Page {c['page_number']} | {c['section_title']}]: {c['text'][:280]}"
        for c in chunks
    )

    llm = get_fast_groq_llm(
        user_groq_key=state.get("user_groq_key"),
        user_gemini_key=state.get("user_gemini_key"),
    )
    resp = llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are the Syllabus Gap Audit AI assistant.\n"
                    f"ACTIVE PDF: '{doc_name}' ({page_count} pages).\n"
                    "Answer concisely using ONLY the retrieved excerpts below and cite exact page numbers (e.g., Page 12).\n"
                    "End with a 1-sentence follow-up inviting the user to paste a YouTube lecture URL to audit missed topics.\n\n"
                    f"=== EXCERPTS ===\n{context_block}"
                )
            ),
            HumanMessage(content=query),
        ]
    )

    answer = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {
        "chat_response": answer,
        "retrieved_evidence": chunks[:3],
        "cited_chunks": chunks[:3],
        "route_taken": "pdf_qa",
    }
