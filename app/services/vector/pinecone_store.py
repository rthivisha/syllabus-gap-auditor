import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional
from pinecone import Pinecone
from app.config import get_settings
from app.services.retrieval.embedding import embed_texts


@lru_cache(maxsize=1)
def get_pinecone_index():
    settings = get_settings()
    api_key = os.getenv("PINECONE_API_KEY") or settings.PINECONE_API_KEY
    index_name = os.getenv("PINECONE_INDEX_NAME") or settings.PINECONE_INDEX_NAME
    return Pinecone(api_key=api_key).Index(index_name)


def upsert_document_chunks(
    doc_id: str,
    filename: str,
    chunks: List[Dict[str, Any]],
    max_batch_bytes: int = 1_400_000,  # Stay safely under Pinecone's 2MB hard limit
    max_batch_vectors: int = 50,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> int:
    """Embeds chunks in parallel 96-text batches and upserts in <1.4MB byte-bounded batches."""
    if not chunks:
        return 0

    texts = [
        str(c.get("text") or c.get("child_text") or c.get("parent_text") or "").strip()
        for c in chunks
    ]
    valid_pairs = [(c, t) for c, t in zip(chunks, texts) if t]
    if not valid_pairs:
        return 0

    clean_chunks, clean_texts = zip(*valid_pairs)

    # 1. Parallel 96-batch hosted embedding via Pinecone Inference API (no torch)
    vectors = embed_texts(list(clean_texts), is_query=False, batch_size=96)

    # 2. Build Pinecone records & pack into byte-safe (<1.4MB) batches
    upsert_batches: List[List[Dict[str, Any]]] = []
    current_batch: List[Dict[str, Any]] = []
    current_bytes = 0
    total_records = 0

    for idx, (chunk, text, vec) in enumerate(zip(clean_chunks, clean_texts, vectors)):
        record = {
            "id": f"{doc_id}_p{chunk.get('page_number', idx + 1)}_{idx}",
            "values": vec,
            "metadata": {
                "doc_id": doc_id,
                "doc_name": filename,
                "filename": filename,
                "section_title": str(chunk.get("section_title") or "Overview")[:120],
                "chapter": str(chunk.get("chapter") or chunk.get("section_title") or "Overview")[:120],
                "page_number": int(chunk.get("page_number") or (idx + 1)),
                "text": text[:2800],
                "parent_text": str(chunk.get("parent_text") or text)[:2800],
            },
        }
        rec_bytes = len(json.dumps(record).encode("utf-8"))

        if current_batch and (
            current_bytes + rec_bytes > max_batch_bytes
            or len(current_batch) >= max_batch_vectors
        ):
            upsert_batches.append(current_batch)
            current_batch = []
            current_bytes = 0

        current_batch.append(record)
        current_bytes += rec_bytes
        total_records += 1

    if current_batch:
        upsert_batches.append(current_batch)

    # 3. Parallel Pinecone upsert across byte-safe batches
    index = get_pinecone_index()
    if len(upsert_batches) == 1:
        index.upsert(vectors=upsert_batches[0])
    else:
        with ThreadPoolExecutor(max_workers=min(5, len(upsert_batches))) as pool:
            list(pool.map(lambda b: index.upsert(vectors=b), upsert_batches))

    return total_records


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
    "fetch_scoped_chunks_with_vectors",
    "get_pinecone_index",
    "upsert_document_chunks",
]
