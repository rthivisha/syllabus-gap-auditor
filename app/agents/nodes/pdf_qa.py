from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import AuditGraphState
from app.gateway.client import get_fast_groq_llm
from app.services.db.repository import get_document
from app.services.job_tracker import set_job_stage
from app.services.retrieval.embedding import embed_texts
from app.services.vector.pinecone_store import get_pinecone_index

_OVERVIEW_KEYWORDS = (
    "topic",
    "topics",
    "about",
    "contain",
    "contains",
    "summary",
    "summarize",
    "overview",
    "chapter",
    "chapters",
    "outline",
    "table of contents",
    "what is this pdf",
    "what does this pdf",
)


def _is_overview_query(query: str) -> bool:
    q = query.lower().strip()
    return any(kw in q for kw in _OVERVIEW_KEYWORDS)


def pdf_qa_node(state: AuditGraphState) -> Dict[str, Any]:
    """Grounded PDF Q&A: asymmetric BGE query embedding, TOC filtering, full parent context, and section outline."""
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

    doc_meta = get_document(doc_id) or {}
    if page_count == "unknown" and doc_meta.get("page_count"):
        page_count = doc_meta["page_count"]
    indexed_chunks_count = doc_meta.get("chunk_count", "unknown")
    sections_list: List[str] = doc_meta.get("sections") or []

    is_overview = _is_overview_query(query)
    # Use asymmetric query embedding (input_type="query") instead of passage encoding
    q_vec = embed_texts([query], is_query=True, batch_size=1)[0]

    # 1. Detect if the user is explicitly asking about one of the indexed sections (e.g., "What are the key concepts in 'ABSTRACT'?")
    q_lower = query.lower()
    matched_section = None
    for s in sections_list:
        s_title = (s.get("section_title", "") if isinstance(s, dict) else str(s)).strip()
        if len(s_title) >= 4 and s_title.lower() in q_lower:
            # Prefer the longest matching section title
            if matched_section is None or len(s_title) > len(matched_section):
                matched_section = s_title

    raw_matches = []
    if matched_section:
        sec_res = get_pinecone_index().query(
            vector=q_vec,
            filter={
                "doc_id": {"$eq": doc_id},
                "section_title": {"$eq": matched_section},
            },
            top_k=6,
            include_metadata=True,
            include_values=False,
        )
        sec_matches = sec_res.get("matches", []) if isinstance(sec_res, dict) else getattr(sec_res, "matches", [])
        raw_matches.extend(sec_matches)

    res = get_pinecone_index().query(
        vector=q_vec,
        filter={"doc_id": {"$eq": doc_id}},
        top_k=12 if is_overview else 8,
        include_metadata=True,
        include_values=False,
    )
    general_matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    raw_matches.extend(general_matches)

    chunks: List[Dict[str, Any]] = []
    seen_pages_sections = set()

    for m in raw_matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else (getattr(m, "metadata", None) or {})
        sec_title = str(meta.get("section_title") or "").strip()
        full_text = str(
            meta.get("parent_text")
            or meta.get("text")
            or meta.get("child_text")
            or ""
        ).strip()

        if not full_text:
            continue

        # Skip TOC / dot-leader stub chunks so real chapter content surfaces first
        if "table of contents" in sec_title.lower() or full_text.count("....") >= 3:
            continue

        dedup_key = (meta.get("page_number", 1), sec_title, full_text[:80])
        if dedup_key in seen_pages_sections:
            continue
        seen_pages_sections.add(dedup_key)

        chunks.append(
            {
                "chunk_id": m.get("id") if isinstance(m, dict) else getattr(m, "id", ""),
                "page_number": meta.get("page_number", 1),
                "section_title": sec_title,
                "text": full_text,
                "bi_encoder_score": round(
                    float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0) or 0.0),
                    3,
                ),
            }
        )

    if not chunks:
        return {
            "chat_response": f"No substantive indexed chunks found for **{doc_name}**. Try re-ingesting the PDF in Step 1.",
            "route_taken": "pdf_qa",
        }

    selected_chunks = chunks[:6] if is_overview else chunks[:5]
    # Pass up to 1,400 chars per chunk (instead of 280) so full paragraphs, tables, and OCR blocks are readable
    context_block = "\n\n".join(
        f"[Page {c['page_number']} | {c['section_title']}]\n{c['text'][:1400]}"
        for c in selected_chunks
    )

    outline_block = ""
    if sections_list:
        raw_titles = [
            s.get("section_title", "") if isinstance(s, dict) else str(s)
            for s in sections_list
        ]
        clean_outline = [
            t.strip()
            for t in raw_titles
            if t and t.strip() and "table of contents" not in t.strip().lower()
        ][:25]
        if clean_outline:
            outline_block = f"INDEXED DOCUMENT SECTIONS: {', '.join(clean_outline)}\n"

    llm = get_fast_groq_llm(
        user_groq_key=state.get("user_groq_key"),
        user_gemini_key=state.get("user_gemini_key"),
    )
    resp = llm.invoke(
        [
            SystemMessage(
                content=(
                    f"You are the Syllabus Gap Audit AI assistant.\n"
                    f"ACTIVE PDF: '{doc_name}' ({page_count} pages, {indexed_chunks_count} indexed chunks).\n"
                    f"{outline_block}"
                    "Answer clearly and thoroughly using ONLY the retrieved excerpts and indexed sections below, citing exact page numbers (e.g., Page 12).\n"
                    "If the user asks what topics or chapters the PDF covers, synthesize a clear structured overview from the indexed sections and excerpts.\n"
                    "End with a 1-sentence follow-up inviting the user to paste a YouTube lecture URL to audit missed topics.\n\n"
                    f"=== RETRIEVED EXCERPTS ===\n{context_block}"
                )
            ),
            HumanMessage(content=query),
        ]
    )

    answer = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {
        "chat_response": answer,
        "retrieved_evidence": selected_chunks[:3],
        "cited_chunks": selected_chunks[:3],
        "route_taken": "pdf_qa",
    }
