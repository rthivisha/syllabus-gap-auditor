"""
Re-export FlashRank ranker for vector retrieval services.
"""
from app.services.retrieval.ranking_service import get_flashranker, rerank_pairs

get_flashrank_ranker = get_flashranker

__all__ = ["get_flashrank_ranker", "get_flashranker", "rerank_pairs"]
