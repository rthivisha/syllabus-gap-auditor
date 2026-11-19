import os
from functools import lru_cache
from typing import List

import numpy as np
import requests
from app.config import get_settings

HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-base-en-v1.5/pipeline/feature-extraction"


class _HFRemoteEmbedder:
    """Lightweight 768-dim BGE-base client via Hugging Face Inference API (zero local PyTorch RAM)."""

    def __init__(self, model_name: str, api_token: str | None = None) -> None:
        self.model_name = model_name
        self.url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        self.headers = {"Content-Type": "application/json"}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        all_vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = requests.post(
                self.url,
                headers=self.headers,
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            # Handle both 2D [batch, 768] and 3D token-pooled [batch, seq, 768] responses
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=1)
            if normalize_embeddings and len(arr) > 0:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr = arr / np.maximum(norms, 1e-12)
            all_vecs.extend(arr.tolist())
        return np.asarray(all_vecs, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder():
    settings = get_settings()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # Use lightweight remote API when running in cloud or when HF_TOKEN is set
    if hf_token or os.getenv("USE_REMOTE_EMBEDDINGS", "1") == "1":
        return _HFRemoteEmbedder(settings.EMBEDDING_MODEL, api_token=hf_token)

    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except ImportError:
        return _HFRemoteEmbedder(settings.EMBEDDING_MODEL, api_token=hf_token)


def embed_texts(
    texts: List[str], is_query: bool = False, batch_size: int = 32
) -> List[List[float]]:
    if not texts:
        return []
    embedder = get_embedder()
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
