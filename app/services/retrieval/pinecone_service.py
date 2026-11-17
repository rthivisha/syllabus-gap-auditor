import logging
from typing import Any, Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec
from app.config import get_settings
from app.services.retrieval.embedding import embed_texts
from app.services.retrieval.ranking_service import hybrid_rerank_candidates

logger = logging.getLogger(__name__)


class PineconeService:
    def __init__(self):
        self.settings = get_settings()
        self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
        self._ensure_index()
        self.index = self.pc.Index(self.settings.PINECONE_INDEX_NAME)

    def _ensure_index(self) -> None:
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.settings.PINECONE_INDEX_NAME not in existing:
            self.pc.create_index(
                name=self.settings.PINECONE_INDEX_NAME,
                dimension=self.settings.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.PINECONE_CLOUD,
                    region=self.settings.PINECONE_REGION,
                ),
            )

    def upsert_document_vectors(
        self,
        doc_id: str,
        doc_name: str,
        child_chunks: List[Dict[str, Any]],
        section_summaries: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upserts both:
          1. Granular child chunks (record_type='child') carrying parent_text in metadata.
          2. Section summary/proposition records (record_type='section_summary') for global coverage checks.
        """
        records: List[Dict[str, Any]] = []

        # 1. Embed & package Child Chunks
        child_texts = [c["child_text"] for c in child_chunks]
        child_vecs = embed_texts(child_texts, is_query=False)
        for c, vec in zip(child_chunks, child_vecs):
            records.append(
                {
                    "id": c["chunk_id"],
                    "values": vec,
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "record_type": "child",
                        "section_title": c["section_title"],
                        "page_number": int(c["page_number"]),
                        "parent_id": c["parent_id"],
                        "child_text": c["child_text"][:1500],
                        "parent_text": c["parent_text"][:3500],
                    },
                }
            )

        # 2. Embed & package Section Summaries (for Map-Reduce Coverage Inventory)
        sec_texts = [f"{s['section_title']}: {s['summary']}" for s in section_summaries]
        sec_vecs = embed_texts(sec_texts, is_query=False)
        for s, vec in zip(section_summaries, sec_vecs):
            records.append(
                {
                    "id": f"{doc_id}-sec-{s['section_idx']}",
                    "values": vec,
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "record_type": "section_summary",
                        "section_title": s["section_title"],
                        "page_number": int(s["start_page"]),
                        "child_text": f"{s['section_title']}: {s['summary']}"[:1500],
                        "parent_text": s["summary"][:3500],
                    },
                }
            )

        if records:
            self.index.delete(filter={"doc_id": {"$eq": doc_id}})

        for i in range(0, len(records), batch_size):
            self.index.upsert(vectors=records[i : i + batch_size])

        return len(records)

    def fetch_section_inventory(
        self,
        doc_id: str,
        section_titles: Optional[List[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """Pulls section-level summaries for the user's target scope (specific chapter or entire book)."""
        flt: Dict[str, Any] = {
            "doc_id": {"$eq": doc_id},
            "record_type": {"$eq": "section_summary"},
        }
        if section_titles:
            flt["section_title"] = {"$in": section_titles}
        if page_range:
            flt["page_number"] = {"$gte": page_range[0], "$lte": page_range[1]}

        q_vec = embed_texts(
            ["table of contents main topics concepts learning objectives"],
            is_query=True,
        )[0]
        res = self.index.query(
            vector=q_vec,
            top_k=top_k,
            include_metadata=True,
            include_values=True,
            filter=flt,
        )
        items = []
        for m in res.get("matches", []):
            md = m["metadata"]
            items.append(
                {
                    "id": m["id"],
                    "section_title": md.get("section_title", "General"),
                    "page_number": int(md.get("page_number", 1)),
                    "summary": md.get("parent_text", ""),
                    "score": round(float(m["score"]), 4),
                    "values": m.get("values"),
                }
            )
        items.sort(key=lambda x: x["page_number"])
        return items

    def fetch_scoped_chunks_with_vectors(
        self,
        doc_id: str,
        scoped_section_titles: Optional[set[str]] = None,
        top_k: int = 120,
    ) -> List[Dict[str, Any]]:
        """Fetch scoped child chunks and their stored vectors in one Pinecone query."""
        if not doc_id:
            return []
        if not 1 <= top_k <= 1000:
            raise ValueError("top_k must be between 1 and 1000.")

        section_titles = sorted(
            title.strip()
            for title in scoped_section_titles or set()
            if title and title.strip()
        )
        filter_by_doc: Dict[str, Any] = {
            "doc_id": {"$eq": doc_id},
            "record_type": {"$eq": "child"},
        }
        query_filter = dict(filter_by_doc)
        if section_titles:
            query_filter["section_title"] = {"$in": section_titles}

        search_text = (
            "textbook concepts and mechanisms in: " + "; ".join(section_titles)
            if section_titles
            else "textbook topics, concepts, mechanisms, and learning objectives"
        )
        query_vector = embed_texts([search_text], is_query=True)[0]
        response = self.index.query(
            vector=query_vector,
            filter=query_filter,
            top_k=top_k,
            include_metadata=True,
            include_values=True,
        )
        matches = response.get("matches", [])

        if not matches and section_titles:
            fallback_response = self.index.query(
                vector=query_vector,
                filter=filter_by_doc,
                top_k=min(max(top_k, 250), 1000),
                include_metadata=True,
                include_values=True,
            )
            all_matches = fallback_response.get("matches", [])
            lowered_targets = [title.lower() for title in section_titles]
            matches = []
            for match in all_matches:
                metadata = match.get("metadata") or {}
                section_title = str(metadata.get("section_title") or "").lower()
                chapter = str(metadata.get("chapter") or "").lower()
                if any(
                    target in section_title or target in chapter
                    for target in lowered_targets
                ):
                    matches.append(match)
            if not matches:
                logger.warning(
                    "No Pinecone child chunks matched the requested sections for doc_id %s.",
                    doc_id,
                )

        chunks: List[Dict[str, Any]] = []
        for match in matches:
            metadata = match.get("metadata") or {}
            values = match.get("values")
            if values is None:
                raise RuntimeError(
                    "Pinecone did not return the stored embedding for chunk "
                    f"{match.get('id', 'unknown')}."
                )
            chunks.append(
                {
                    "chunk_id": match.get("id", ""),
                    "doc_id": metadata.get("doc_id", doc_id),
                    "doc_name": metadata.get("doc_name", ""),
                    "section_title": metadata.get("section_title", ""),
                    "chapter": metadata.get("chapter", ""),
                    "page_number": int(metadata.get("page_number") or 1),
                    "text": metadata.get("parent_text")
                    or metadata.get("child_text")
                    or metadata.get("text", ""),
                    "score": round(float(match.get("score", 0.0)), 4),
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

    def search_and_rerank(
        self,
        doc_id: str,
        query: str,
        section_titles: Optional[List[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
        top_k_dense: int = 12,
        top_n_final: int = 4,
    ) -> List[Dict[str, Any]]:
        """Two-stage Hybrid Dense + BM25 + FlashRank retrieval returning full Parent contexts."""
        flt: Dict[str, Any] = {
            "doc_id": {"$eq": doc_id},
            "record_type": {"$eq": "child"},
        }
        if section_titles:
            flt["section_title"] = {"$in": section_titles}
        if page_range:
            flt["page_number"] = {"$gte": page_range[0], "$lte": page_range[1]}

        q_vec = embed_texts([query], is_query=True)[0]
        res = self.index.query(
            vector=q_vec, top_k=top_k_dense, include_metadata=True, filter=flt
        )

        candidates = []
        for m in res.get("matches", []):
            md = m["metadata"]
            candidates.append(
                {
                    "chunk_id": m["id"],
                    "doc_id": md.get("doc_id", doc_id),
                    "doc_name": md.get("doc_name", ""),
                    "section_title": md.get("section_title", "General"),
                    "page_number": int(md.get("page_number", 1)),
                    "parent_id": md.get("parent_id", ""),
                    "child_text": md.get("child_text", ""),
                    "parent_text": md.get("parent_text", ""),
                    "dense_score": round(float(m["score"]), 4),
                }
            )

        return hybrid_rerank_candidates(
            query=query, candidates=candidates, top_n=top_n_final
        )
