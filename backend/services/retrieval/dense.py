"""Dense retrieval using Qdrant vector similarity search."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.services.retrieval.base import Retriever, RetrievalResult
from backend.services.retrieval.vector_store import VectorStore
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class DenseRetriever(Retriever):
    """Dense retrieval using vector similarity search.

    Uses pre-computed embeddings stored in Qdrant to find
    semantically similar documents to the query.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    @property
    def method_name(self) -> str:
        return "dense"

    def is_ready(self) -> bool:
        return self._vector_store.collection_exists() and self._vector_store.count() > 0

    def search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Search using dense vector similarity.

        Args:
            query: The text query (unused if query_embedding is provided).
            query_embedding: Pre-computed query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by cosine similarity.

        Raises:
            ValueError: If no query_embedding is provided.
        """
        if query_embedding is None:
            raise ValueError("Dense retrieval requires a query_embedding")

        raw_results = self._vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
        )

        results = [
            RetrievalResult(
                chunk_id=str(r["id"]),
                document_id=str(r.get("metadata", {}).get("document_id", r["id"])),
                text=r["text"],
                score=r["score"],
                metadata=r.get("metadata", {}),
                retrieval_method="dense",
            )
            for r in raw_results
        ]

        logger.info(
            "dense_retrieval_complete",
            query_length=len(query),
            results_count=len(results),
            top_score=results[0].score if results else 0.0,
        )

        return results
