"""Retrieval service providers — Dense, BM25, and Hybrid."""

from backend.services.retrieval.base import Retriever, RetrievalResult
from backend.services.retrieval.dense import DenseRetriever
from backend.services.retrieval.bm25 import BM25Retriever
from backend.services.retrieval.hybrid import HybridRetriever
from backend.services.retrieval.vector_store import VectorStore

__all__ = [
    "Retriever",
    "RetrievalResult",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "VectorStore",
]
