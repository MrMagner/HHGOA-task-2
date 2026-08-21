"""Abstract base class for retrieval providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RetrievalResult:
    """A single result from a retrieval query."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_method: str = ""


class Retriever(ABC):
    """Abstract base class for document retrieval."""

    @abstractmethod
    def search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Search for relevant documents.

        Args:
            query: The text query.
            query_embedding: Pre-computed query embedding (for dense retrieval).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by relevance score (descending).
        """
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Name of this retrieval method."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the retriever is initialized and ready."""
        ...
