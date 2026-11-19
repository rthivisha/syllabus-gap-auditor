import os
from functools import lru_cache
from typing import List

import numpy as np
from pinecone import Pinecone
from app.config import get_settings


class _PineconeRemoteEmbedder:
    """768-dim hosted embedder via Pinecone Inference API (uses existing PINECONE_API_KEY, zero local RAM)."""

    def __init__(self, api_key: str, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self.pc = Pinecone(api_key=api_key)
        # Map HF model name to Pinecone Inference model ID
        self.model = "bge-base-en-v1.5" if "bge-base" in model_name else model_name

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        input_type: str = "passage",
    ) -> np.ndarray:
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                res = self.pc.inference.embed(
                    model=self.model,
                    inputs=batch,
                    parameters={"input_type": input_type, "truncate": "END"},
                )
            except Exception:
                res = self.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=batch,
                    parameters={"input_type": input_type, "truncate": "END", "dimension": 768},
                )
            for item in res.data:
                vec = np.asarray(item["values"], dtype=np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 1e-12:
                        vec = vec / norm
                all_vecs.append(vec.tolist())
        return np.asarray(all_vecs, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder():
    settings = get_settings()
    pc_key = os.getenv("PINECONE_API_KEY") or settings.PINECONE_API_KEY
    if pc_key:
        return _PineconeRemoteEmbedder(api_key=pc_key, model_name=settings.EMBEDDING_MODEL)

    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except ImportError:
        return _PineconeRemoteEmbedder(api_key=pc_key or "", model_name=settings.EMBEDDING_MODEL)


def embed_texts(
    texts: List[str], is_query: bool = False, batch_size: int = 32
) -> List[List[float]]:
    if not texts:
        return []
    embedder = get_embedder()
    if isinstance(embedder, _PineconeRemoteEmbedder):
        vectors = embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            input_type="query" if is_query else "passage",
        )
        return vectors.tolist()

    formatted = (
        [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        if is_query
        else texts
    )
    vectors = embedder.encode(
        formatted,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def batch_encode_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode unprefixed video-window text in one normalized batch."""
    settings = get_settings()
    if not texts:
        return np.zeros((0, settings.EMBEDDING_DIM), dtype=np.float32)

    vectors = get_embedder().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)
