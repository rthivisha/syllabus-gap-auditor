from app.services.retrieval.embedding import (
    batch_encode_texts,
    embed_texts,
    get_embedder,
)

get_embedding_model = get_embedder

__all__ = ["batch_encode_texts", "embed_texts", "get_embedder", "get_embedding_model"]
