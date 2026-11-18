import os
import time
from typing import Any, Callable, Dict, List, Optional
from app.services.retrieval.pinecone_service import PineconeService, get_pinecone_index


def upsert_document_chunks(
    doc_id: str,
    filename: str,
    chunks: List[Dict[str, Any]],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """
    High-speed Pinecone indexer built for both small syllabi and 100MB+ textbooks:
      - Automatically scales chunk window (1,500 chars for <80 pages, 2,800 chars for 80+ pages)
      - Enables full multi-core PyTorch CPU threading + max_seq_length=256
      - Upserts in batches of 100 vectors
    """
    if not chunks:
        return 0

    t0 = time.perf_counter()
    index = get_pinecone_index()

    # Scale chunk size dynamically for huge textbooks (e.g., 91MB / 400+ pages)
    max_chunk_chars = 2800 if len(chunks) > 80 else 1500

    merged_chunks: List[Dict[str, Any]] = []
    for ch in chunks:
        txt = (ch.get("text") or "").strip()
        if not txt:
            continue
        if (
            merged_chunks
            and len(merged_chunks[-1]["text"]) + len(txt) <= max_chunk_chars
        ):
            merged_chunks[-1]["text"] += "\n" + txt
        else:
            merged_chunks.append(
                {
                    "section_title": ch.get("section_title") or f"Page {ch.get('page_number', 1)}",
                    "chapter": ch.get("chapter", ""),
                    "page_number": int(ch.get("page_number") or 1),
                    "text": txt[:max_chunk_chars],
                }
            )

    if progress_cb:
        progress_cb(f"Embedding {len(merged_chunks)} consolidated sections...")

    import torch
    torch.set_num_threads(max(4, os.cpu_count() or 4))

    from app.services.vector.embeddings import get_embedding_model
    model = get_embedding_model()
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = 256

    texts = [c["text"] for c in merged_chunks]
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    if progress_cb:
        progress_cb(f"Upserting {len(merged_chunks)} vectors into Pinecone...")

    vectors = []
    for idx, (ch, vec) in enumerate(zip(merged_chunks, embeddings)):
        vectors.append(
            {
                "id": f"{doc_id}-c{idx:04d}",
                "values": vec.tolist() if hasattr(vec, "tolist") else list(vec),
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "doc_name": filename,
                    "record_type": "child",
                    "section_title": ch["section_title"][:200],
                    "chapter": str(ch["chapter"])[:120],
                    "page_number": ch["page_number"],
                    "text": ch["text"][:2200],
                },
            }
        )

    for start in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[start : start + 100])

    print(f"[Pinecone Fast Index] Indexed {len(vectors)} chunks for '{filename}' in {time.perf_counter() - t0:.2f}s")
    return len(vectors)


def fetch_scoped_chunks_with_vectors(
    doc_id: str,
    scoped_section_titles: Optional[Any] = None,
    top_k: int = 200,
) -> List[Dict[str, Any]]:
    """
    Safe Pinecone fetcher for syllabus gap auditing:
      - Always queries by doc_id to retrieve book vectors
      - If scoped_section_titles is provided and matches are found, returns those scoped sections
      - Never returns 0 sections if the document exists in Pinecone
    """
    if not doc_id:
        return []

    index = get_pinecone_index()
    flt: Dict[str, Any] = {"doc_id": {"$eq": doc_id}}

    # Non-zero probe vector
    query_vector = [0.0] * 768
    query_vector[0] = 1.0

    res = index.query(
        vector=query_vector,
        filter=flt,
        top_k=min(max(top_k, 200), 1000),
        include_metadata=True,
        include_values=True,
    )
    raw_matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    matches = list(raw_matches or [])

    if scoped_section_titles:
        targets = [str(t).lower().strip() for t in scoped_section_titles if str(t).strip()]
        if targets:
            filtered = [
                m for m in matches
                if any(
                    t in str((m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})) or {}).get("section_title", "").lower()
                    or t in str((m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})) or {}).get("chapter", "").lower()
                    or t in str((m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})) or {}).get("text", "")[:200].lower()
                    for t in targets
                )
            ]
            # Never return 0 sections if the document exists in Pinecone!
            if filtered:
                matches = filtered

    chunks: List[Dict[str, Any]] = []
    for match in matches:
        md = (match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", {})) or {}
        values = match.get("values") if isinstance(match, dict) else getattr(match, "values", None)
        if values is None:
            continue
        mid = match.get("id", "") if isinstance(match, dict) else getattr(match, "id", "")
        score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
        chunks.append(
            {
                "chunk_id": mid,
                "doc_id": md.get("doc_id", doc_id),
                "doc_name": md.get("doc_name") or md.get("filename", ""),
                "section_title": md.get("section_title", ""),
                "chapter": md.get("chapter", ""),
                "page_number": int(md.get("page_number") or 1),
                "text": md.get("parent_text")
                or md.get("child_text")
                or md.get("text", ""),
                "score": round(float(score or 0.0), 4),
                "values": values,
            }
        )

    chunks.sort(
        key=lambda chunk: (
            chunk["page_number"],
            chunk["section_title"],
            chunk["chunk_id"],
        )
    )
    return chunks


__all__ = [
    "PineconeService",
    "fetch_scoped_chunks_with_vectors",
    "get_pinecone_index",
    "upsert_document_chunks",
]
