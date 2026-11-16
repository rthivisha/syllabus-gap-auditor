from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(
    texts: List[str], is_query: bool = False, batch_size: int = 64
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
