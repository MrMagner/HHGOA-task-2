"""Abstract base class for reranking providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.services.retrieval.base import RetrievalResult


class Reranker(ABC):
    """Abstract base class for reranking providers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results based on query relevance.

        Args:
            query: The original query.
            results: List of retrieval results to rerank.
            top_k: Number of top results to return.

        Returns:
            Reranked list of RetrievalResult.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the reranking model."""
        ...
