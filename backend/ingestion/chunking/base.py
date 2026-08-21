"""Abstract base class for document chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single chunk of text extracted from a document."""
    id: str
    text: str
    document_id: str = ""
    chunk_index: int = 0
    chunk_strategy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split())


class Chunker(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, document_id: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            text: The full document text.
            document_id: Unique document identifier.
            metadata: Additional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of this chunking strategy."""
        ...
