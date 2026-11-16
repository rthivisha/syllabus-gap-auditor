import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from flashrank import RerankRequest

from app.agents.state import AuditGraphState
from app.config import get_settings
from app.services.db.repository import log_audit_event
from app.services.retrieval.embedding import embed_texts
from app.services.retrieval.pinecone_service import PineconeService
from app.services.retrieval.ranking_service import get_flashranker

logger = logging.getLogger(__name__)


def _cosine_sim(v1: List[float], v2: List[float]) -> float:
    return float(sum(a * b for a, b in zip(v1, v2)))


def _load_or_compute_window_vecs(
    video_id: str, video_windows: List[Dict[str, Any]]
) -> tuple[List[List[float]], float]:
    """Load embeddings for unchanged video windows or compute and persist them."""
    if not video_windows:
        return [], 0.0

    settings = get_settings()
    cache_file = settings.PROCESSED_VIDEOS_DIR / f"video_{video_id}.json"
    window_texts = [window["combined_text"] for window in video_windows]
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "model": settings.EMBEDDING_MODEL,
                "texts": window_texts,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    cache_payload: Dict[str, Any] = {}
    cache_writable = True
    if cache_file.exists():
        try:
            loaded = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cache_payload = loaded
            else:
                logger.warning("Ignoring non-object video cache at %s", cache_file)
                cache_writable = False
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Unable to read video embedding cache %s: %s", cache_file, exc
            )
            cache_writable = False

    cached_vecs = cache_payload.get("window_embeddings")
    if (
        cache_payload.get("window_embeddings_key") == cache_key
        and isinstance(cached_vecs, list)
        and len(cached_vecs) == len(video_windows)
        and all(
            isinstance(vector, list)
            and len(vector) == settings.EMBEDDING_DIM
            and all(isinstance(value, (int, float)) for value in vector)
            for vector in cached_vecs
        )
    ):
        return [[float(value) for value in vector] for vector in cached_vecs], 0.0

    t0 = time.perf_counter()
    vecs = embed_texts(window_texts, is_query=False)
    elapsed = round(time.perf_counter() - t0, 2)

    if cache_writable:
        cache_payload["video_id"] = video_id
        cache_payload["windows"] = video_windows
        cache_payload["window_embeddings"] = vecs
        cache_payload["window_embeddings_key"] = cache_key
        try:
            cache_file.write_text(
                json.dumps(cache_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(
                "Unable to persist video embedding cache %s: %s", cache_file, exc
            )

    return vecs, elapsed


def _fetch_section_parent_chunk(
    index: Any,
    doc_id: str,
    section: Dict[str, Any],
    query_vector: List[float],
    fallback_doc_name: str,
) -> Dict[str, Any]:
    """Fetch the best child chunk with a section-specific query vector."""
    response = index.query(
        vector=query_vector,
        top_k=8,
        include_metadata=True,
        filter={
            "doc_id": {"$eq": doc_id},
            "record_type": {"$eq": "child"},
            "section_title": {"$eq": section["section_title"]},
        },
    )
    matches = response.get("matches", [])
    for match in matches:
        metadata = match.get("metadata") or {}
        if metadata:
            return {
                "chunk_id": match.get("id", section["id"]),
                "doc_name": metadata.get("doc_name") or fallback_doc_name,
                "page_number": int(
                    metadata.get("page_number") or section["page_number"]
                ),
                "parent_text": metadata.get("parent_text")
                or metadata.get("child_text")
                or section["summary"],
            }

    return {
        "chunk_id": section["id"],
        "doc_name": fallback_doc_name,
        "page_number": section["page_number"],
        "parent_text": section["summary"],
    }


def retrieve_syllabus_evidence_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Retrieve scoped sections, compare them to video windows, and verify alignment
    with FlashRank. Section-specific Pinecone queries run concurrently.
    """
    t_step = time.perf_counter()
    doc_id = state["doc_id"]
    video_id = state.get("video_id", "")
    scoped_titles = state.get("scoped_section_titles") or None
    video_windows = state.get("video_windows", [])

    pc = PineconeService()
    t0 = time.perf_counter()
    sections = pc.fetch_section_inventory(
        doc_id=doc_id, section_titles=scoped_titles, top_k=60
    )
    t_inventory = round(time.perf_counter() - t0, 2)

    if not sections:
        return {"retrieved_sections": [], "candidate_gaps": [], "cited_chunks": []}

    window_vecs, t_window_embedding = _load_or_compute_window_vecs(
        video_id, video_windows
    )
    t0 = time.perf_counter()
    section_queries = [
        f"{section['section_title']}: {section['summary'][:350]}"
        for section in sections
    ]
    section_vecs = embed_texts(section_queries, is_query=True)
    t_embedding = round(t_window_embedding + time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    section_matches: List[Dict[str, Any]] = []
    for section, section_vec in zip(sections, section_vecs):
        scored_windows = [
            (_cosine_sim(section_vec, window_vec), idx, window)
            for idx, (window, window_vec) in enumerate(zip(video_windows, window_vecs))
        ]
        scored_windows.sort(key=lambda item: item[0], reverse=True)

        best_sim = scored_windows[0][0] if scored_windows else 0.0
        best_window = scored_windows[0][2] if scored_windows else None
        top_windows = scored_windows[:3]
        section_matches.append(
            {
                "section": section,
                "best_sim": best_sim,
                "best_window": best_window,
                "top_windows": top_windows,
                "query_vector": section_vec,
            }
        )
    t_biencoder = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()

    def fetch_parent_chunk(match: Dict[str, Any]) -> Dict[str, Any]:
        return _fetch_section_parent_chunk(
            index=pc.index,
            doc_id=doc_id,
            section=match["section"],
            query_vector=match["query_vector"],
            fallback_doc_name=state.get("doc_name", ""),
        )

    with ThreadPoolExecutor(max_workers=min(8, len(section_matches))) as executor:
        parent_chunks = list(executor.map(fetch_parent_chunk, section_matches))
    t_pinecone_fetch = round(time.perf_counter() - t0, 2)

    ranker = get_flashranker()
    candidate_gaps: List[Dict[str, Any]] = []
    cited_chunks: List[Dict[str, Any]] = []
    retrieval_audit_trail: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    for match, parent_chunk in zip(section_matches, parent_chunks):
        section = match["section"]
        best_sim = match["best_sim"]
        best_window = match["best_window"]
        top_windows = match["top_windows"]
        rerank_score = 0.0

        if top_windows:
            passages = [
                {"id": idx, "text": window["combined_text"][:600]}
                for _, idx, window in top_windows
            ]
            rerank_results = ranker.rerank(
                RerankRequest(
                    query=f"{section['section_title']}: {section['summary'][:180]}",
                    passages=passages,
                )
            )
            if rerank_results:
                top_result = rerank_results[0]
                rerank_score = round(float(top_result["score"]), 4)
                matched_id = str(top_result.get("id", ""))
                best_window = next(
                    (
                        window
                        for _, idx, window in top_windows
                        if str(idx) == matched_id
                    ),
                    best_window,
                )

        retrieval_audit_trail.append(
            {
                "section": section["section_title"],
                "chunk_id": parent_chunk["chunk_id"],
                "bi_encoder_dense_score": round(best_sim, 4),
                "flashrank_rerank_score": rerank_score,
                "closest_timestamp": best_window["timestamp"] if best_window else "",
            }
        )
        cited_chunks.append(
            {
                "chunk_id": parent_chunk["chunk_id"],
                "doc_name": parent_chunk["doc_name"],
                "section_title": section["section_title"],
                "page_number": parent_chunk["page_number"],
                "score": round(best_sim, 4),
                "bi_encoder_score": round(best_sim, 4),
                "rerank_score": rerank_score,
                "snippet": parent_chunk["parent_text"][:600],
            }
        )

        if best_sim < 0.84 or rerank_score < 0.35:
            candidate_gaps.append(
                {
                    "section_title": section["section_title"],
                    "page_number": section["page_number"],
                    "chunk_id": parent_chunk["chunk_id"],
                    "parent_text": parent_chunk["parent_text"],
                    "best_video_similarity": round(best_sim, 4),
                    "flashrank_score": rerank_score,
                    "closest_timestamp": best_window["timestamp"] if best_window else "",
                    "closest_video_text": (
                        best_window["combined_text"][:450] if best_window else ""
                    ),
                }
            )

    t_flashrank = round(time.perf_counter() - t0, 2)
    step_timings = dict(state.get("step_timings") or {})
    step_timings["pinecone_inventory_sec"] = t_inventory
    step_timings["bge_embedding_sec"] = t_embedding
    step_timings["pinecone_child_fetch_sec"] = t_pinecone_fetch
    step_timings["bi_encoder_sec"] = t_biencoder
    step_timings["flashrank_sec"] = t_flashrank
    step_timings["bi_encoder_plus_flashrank_sec"] = round(
        t_biencoder + t_flashrank, 2
    )
    step_timings["total_retrieval_sec"] = round(time.perf_counter() - t_step, 2)

    log_audit_event(
        action="retrieval_and_rerank_verified",
        user_id=state.get("user_id", "default_user"),
        detail={
            "doc_id": doc_id,
            "timings": step_timings,
            "bi_encoder_vs_reranker": retrieval_audit_trail,
        },
    )

    return {
        "retrieved_sections": sections,
        "candidate_gaps": candidate_gaps,
        "cited_chunks": cited_chunks[:15],
        "step_timings": step_timings,
    }
