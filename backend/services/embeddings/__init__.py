"""Embedding service providers."""

from backend.services.embeddings.base import EmbeddingProvider
from backend.services.embeddings.sentence_transformer import SentenceTransformerEmbeddings

__all__ = ["EmbeddingProvider", "SentenceTransformerEmbeddings"]
