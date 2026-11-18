import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from app.agents.state import AuditGraphState
from app.config import get_settings
from app.services.db.repository import log_audit_event
from app.services.retrieval.embedding import batch_encode_texts
from app.services.retrieval.pinecone_service import PineconeService
from app.services.job_tracker import set_job_stage
from app.services.retrieval.ranking_service import rerank_pairs

logger = logging.getLogger(__name__)


def _load_or_compute_window_vecs(
    video_id: str, video_windows: List[Dict[str, Any]]
) -> tuple[np.ndarray, float]:
    """Load cached window embeddings or encode the complete video in one batch."""
    if not video_windows:
        return np.empty((0, get_settings().EMBEDDING_DIM), dtype=np.float32), 0.0

    settings = get_settings()
    cache_file = settings.PROCESSED_VIDEOS_DIR / f"video_{video_id}.json"
    window_texts = [str(window.get("combined_text") or "") for window in video_windows]
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
    video_key = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:16]
    vector_cache = (
        settings.PROCESSED_VIDEOS_DIR
        / f"video_{video_key}_{cache_key[:20]}.npy"
    )

    if vector_cache.exists():
        try:
            vectors = np.load(vector_cache, allow_pickle=False)
            if (
                vectors.shape == (len(video_windows), settings.EMBEDDING_DIM)
                and np.isfinite(vectors).all()
            ):
                return np.asarray(vectors, dtype=np.float32), 0.0
            logger.warning("Ignoring invalid video embedding cache at %s", vector_cache)
        except (OSError, ValueError, EOFError) as exc:
            logger.warning(
                "Unable to read video embedding cache %s: %s", vector_cache, exc
            )

    cache_payload: Dict[str, Any] = {}
    if cache_file.exists():
        try:
            loaded = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cache_payload = loaded
            else:
                logger.warning("Ignoring non-object video cache at %s", cache_file)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "Unable to read video embedding cache %s: %s", cache_file, exc
            )

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
        vectors = np.asarray(cached_vecs, dtype=np.float32)
        if np.isfinite(vectors).all():
            _save_window_vecs(vector_cache, vectors)
            return vectors, 0.0

    t0 = time.perf_counter()
    vectors = batch_encode_texts(window_texts, batch_size=64)
    elapsed = round(time.perf_counter() - t0, 2)

    if vectors.shape != (len(video_windows), settings.EMBEDDING_DIM):
        raise ValueError(
            "Video-window embedding output has an unexpected shape: "
            f"{vectors.shape}; expected {(len(video_windows), settings.EMBEDDING_DIM)}."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Video-window embeddings contain non-finite values.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    _save_window_vecs(vector_cache, vectors)
    return vectors, elapsed


def _save_window_vecs(cache_file: Path, vectors: np.ndarray) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_file.parent,
            prefix=f"{cache_file.stem}_",
            suffix=".npy",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.save(handle, vectors, allow_pickle=False)
        os.replace(temporary_path, cache_file)
    except OSError as exc:
        logger.warning("Unable to persist video embedding cache %s: %s", cache_file, exc)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "Unable to clean up temporary embedding cache %s: %s",
                    temporary_path,
                    exc,
                )


def _select_gap_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Keep only the top 6 lowest-similarity sections (actual gap candidates) for the LLM auditor
    low_sim_sections = [
        s for s in candidates
        if s.get("flashrank_score", s.get("rerank_score", s.get("best_video_similarity", 1.0))) < 0.68
    ]
    if not low_sim_sections:
        low_sim_sections = sorted(
            candidates,
            key=lambda x: x.get("flashrank_score", x.get("rerank_score", x.get("best_video_similarity", 1.0))),
        )[:6]
    else:
        low_sim_sections = sorted(
            low_sim_sections,
            key=lambda x: x.get("flashrank_score", x.get("rerank_score", x.get("best_video_similarity", 1.0))),
        )[:6]
    return low_sim_sections


def retrieve_syllabus_evidence_node(state: AuditGraphState) -> Dict[str, Any]:
    """
    Compare Pinecone-stored scoped chunk vectors against batched video vectors.
    FlashRank only reranks the top two windows for borderline sections.
    """
    set_job_stage(
        state.get("session_id", ""),
        "2/4 Retrieving and reranking scoped textbook sections...",
    )
    t_step = time.perf_counter()
    doc_id = state.get("doc_id")
    if not doc_id:
        raise ValueError("A reference document is required for syllabus retrieval.")
    video_id = state.get("video_id", "")
    target_scope = str(state.get("target_scope", "")).lower().strip()
    raw_scoped = state.get("scoped_section_titles") or []
    if not raw_scoped or any(p in target_scope for p in ("whole", "entire", "full", "all")):
        scoped_titles = None
    else:
        scoped_titles = set(raw_scoped)
    video_windows = state.get("video_windows", [])

    pc = PineconeService()
    t0 = time.perf_counter()
    chunks = pc.fetch_scoped_chunks_with_vectors(
        doc_id=doc_id,
        scoped_section_titles=scoped_titles,
        top_k=200,
    )
    t_chunk_fetch = round(time.perf_counter() - t0, 2)

    if not chunks:
        return {
            "retrieved_sections": [],
            "retrieved_evidence": [],
            "candidate_gaps": [],
            "cited_chunks": [],
            "retrieval_audit_trail": [],
        }

    window_vecs, t_window_embedding = _load_or_compute_window_vecs(
        video_id, video_windows
    )
    t0 = time.perf_counter()
    chunks_by_section: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        section_title = str(chunk.get("section_title") or "")
        if not section_title:
            section_title = f"Page {chunk.get('page_number', 1)}"
        chunks_by_section.setdefault(section_title, []).append(chunk)

    section_records: List[Dict[str, Any]] = []
    section_vectors: List[np.ndarray] = []
    section_matches: List[Dict[str, Any]] = []
    for section_title, section_chunks in chunks_by_section.items():
        try:
            chunk_vecs = np.asarray(
                [chunk["values"] for chunk in section_chunks],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Pinecone returned invalid stored chunk embeddings for {section_title}."
            ) from exc
        if (
            chunk_vecs.ndim != 2
            or chunk_vecs.shape[1] != get_settings().EMBEDDING_DIM
            or not np.isfinite(chunk_vecs).all()
        ):
            raise RuntimeError(
                f"Pinecone returned invalid stored chunk embeddings for {section_title}."
            )
        chunk_norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
        chunk_vecs = chunk_vecs / np.maximum(chunk_norms, 1e-12)
        section_vec = np.mean(chunk_vecs, axis=0)
        section_vec /= max(float(np.linalg.norm(section_vec)), 1e-12)
        representative_idx = int(np.argmax(chunk_vecs @ section_vec))
        representative = section_chunks[representative_idx]
        unique_texts = list(
            dict.fromkeys(
                str(chunk.get("text") or "").strip()
                for chunk in section_chunks
                if str(chunk.get("text") or "").strip()
            )
        )
        section = {
            "id": representative["chunk_id"],
            "chunk_id": representative["chunk_id"],
            "section_title": section_title,
            "page_number": int(representative.get("page_number") or 1),
            "summary": "\n".join(unique_texts)[:3500],
            "score": float(representative.get("score") or 0.0),
        }
        parent_chunk = {
            "chunk_id": representative["chunk_id"],
            "doc_name": representative.get("doc_name", ""),
            "page_number": section["page_number"],
            "parent_text": str(representative.get("text") or ""),
        }
        section_records.append(section)
        section_matches.append(
            {
                "section": section,
                "parent_chunk": parent_chunk,
            }
        )
        section_vectors.append(section_vec)

    section_matrix = np.stack(section_vectors, axis=0)
    sim_matrix = window_vecs @ section_matrix.T
    borderline_pairs: List[Tuple[int, int]] = []
    for section_idx, match in enumerate(section_matches):
        if video_windows:
            section_scores = sim_matrix[:, section_idx]
            best_window_idx = int(np.argmax(section_scores))
            best_sim = float(section_scores[best_window_idx])
            best_window = video_windows[best_window_idx]
            # Only run FlashRank on truly ambiguous scores (0.48..0.64)
            if 0.48 <= best_sim <= 0.64:
                top_window_indices = np.argsort(section_scores)[-2:][::-1].tolist()
                borderline_pairs.extend(
                    (section_idx, window_idx) for window_idx in top_window_indices
                )
        else:
            best_sim = 0.0
            best_window = None
        match.update(
            {
                "best_sim": best_sim,
                "best_window": best_window,
                "rerank_score": best_sim,
            }
        )
    t_biencoder = round(time.perf_counter() - t0, 2)

    candidate_gaps: List[Dict[str, Any]] = []
    cited_chunks: List[Dict[str, Any]] = []
    retrieval_audit_trail: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    if borderline_pairs:
        rerank_inputs = [
            (
                f"{section_matches[section_idx]['section']['section_title']}: "
                f"{section_matches[section_idx]['section']['summary'][:180]}",
                str(video_windows[window_idx].get("combined_text") or "")[:600],
            )
            for section_idx, window_idx in borderline_pairs
        ]
        rerank_scores = rerank_pairs(rerank_inputs)
        if len(rerank_scores) != len(borderline_pairs):
            raise RuntimeError("FlashRank returned an incomplete set of pair scores.")
        best_scores: Dict[int, Tuple[float, int]] = {}
        for (section_idx, window_idx), score in zip(
            borderline_pairs, rerank_scores
        ):
            previous = best_scores.get(section_idx)
            if previous is None or score > previous[0]:
                best_scores[section_idx] = (score, window_idx)
        for section_idx, (score, window_idx) in best_scores.items():
            section_matches[section_idx].update(
                {
                    "rerank_score": round(float(score), 4),
                    "best_window": video_windows[window_idx],
                }
            )

    for match in section_matches:
        section = match["section"]
        parent_chunk = match["parent_chunk"]
        best_sim = match["best_sim"]
        best_window = match["best_window"]
        rerank_score = match["rerank_score"]

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

        candidate_gaps.append(
            {
                "section_title": section["section_title"],
                "page_number": section["page_number"],
                "chunk_id": parent_chunk["chunk_id"],
                "parent_text": parent_chunk["parent_text"],
                "text": parent_chunk["parent_text"][:650],
                "best_video_similarity": round(best_sim, 4),
                "bi_encoder_score": round(best_sim, 4),
                "flashrank_score": rerank_score,
                "rerank_score": rerank_score,
                "closest_timestamp": best_window["timestamp"] if best_window else "",
                "closest_video_text": (
                    best_window["combined_text"][:450] if best_window else ""
                ),
                "closest_video_window": best_window,
                "matched_window": best_window,
            }
        )

    candidate_gaps = _select_gap_candidates(candidate_gaps)

    t_flashrank = round(time.perf_counter() - t0, 2)
    step_timings = dict(state.get("step_timings") or {})
    step_timings["pinecone_chunk_fetch_sec"] = t_chunk_fetch
    step_timings["bge_embedding_sec"] = t_window_embedding
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
        "retrieved_sections": section_records,
        "retrieved_evidence": section_records,
        "candidate_gaps": candidate_gaps,
        "cited_chunks": cited_chunks[:15],
        "retrieval_audit_trail": retrieval_audit_trail,
        "step_timings": step_timings,
    }
