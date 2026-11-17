import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi
from app.config import get_settings


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def get_flashranker() -> Ranker:
    settings = get_settings()
    return Ranker(model_name=settings.RERANKER_MODEL, cache_dir="cache/flashrank")


def rerank_pairs(pairs: List[Tuple[str, str]]) -> List[float]:
    """Rerank query/passage pairs, batching passages that share a query."""
    if not pairs:
        return []

    groups: Dict[str, List[Tuple[int, str]]] = {}
    for pair_idx, (query, passage) in enumerate(pairs):
        groups.setdefault(query, []).append((pair_idx, passage))

    scores = [0.0] * len(pairs)
    ranker = get_flashranker()
    for query, indexed_passages in groups.items():
        response = ranker.rerank(
            RerankRequest(
                query=query,
                passages=[
                    {"id": pair_idx, "text": passage}
                    for pair_idx, passage in indexed_passages
                ],
            )
        )
        for result in response:
            scores[int(result["id"])] = float(result["score"])
    return scores


def hybrid_rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 4,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Two-Stage Hybrid Reranker:
      1. Reciprocal Rank Fusion (RRF) blending Dense Pinecone rank + BM25 lexical rank.
      2. FlashRank Cross-Encoder (ms-marco-MiniLM-L-12-v2) scoring query-passage pairs.
    """
    if not candidates:
        return []

    # 1. BM25 Lexical Ranking over candidate child texts
    tokenized_corpus = [_tokenize(c.get("child_text", "")) for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))

    bm25_order = sorted(
        range(len(candidates)), key=lambda i: bm25_scores[i], reverse=True
    )
    bm25_rank_map = {idx: rank + 1 for rank, idx in enumerate(bm25_order)}

    # 2. Reciprocal Rank Fusion (Dense rank is incoming order 1..N)
    fused: List[Dict[str, Any]] = []
    for dense_idx, item in enumerate(candidates):
        dense_rank = dense_idx + 1
        lexical_rank = bm25_rank_map[dense_idx]
        rrf_score = (1.0 / (rrf_k + dense_rank)) + (1.0 / (rrf_k + lexical_rank))
        enriched = {
            **item,
            "rrf_score": round(rrf_score, 5),
            "bm25_score": round(float(bm25_scores[dense_idx]), 3),
        }
        fused.append(enriched)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    shortlist = fused[: max(top_n * 3, 10)]

    # 3. FlashRank Cross-Encoder Reranking
    passages = [
        {"id": i, "text": item.get("child_text", ""), "meta": item}
        for i, item in enumerate(shortlist)
    ]
    ranker = get_flashranker()
    reranked_raw = ranker.rerank(RerankRequest(query=query, passages=passages))

    results: List[Dict[str, Any]] = []
    for r in reranked_raw[:top_n]:
        meta = dict(r["meta"])
        meta["rerank_score"] = round(float(r["score"]), 4)
        results.append(meta)

    return results
