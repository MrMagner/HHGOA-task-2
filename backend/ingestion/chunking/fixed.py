"""Fixed-size character-based chunking strategy."""

from __future__ import annotations

from typing import Any

from backend.ingestion.chunking.base import Chunker, Chunk


class FixedChunker(Chunker):
    """Splits text into fixed-size chunks with overlap.

    Simple but effective. Works well when document structure
    is unknown or irregular.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk(self, text: str, document_id: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into fixed-size character chunks.

        Args:
            text: Full document text.
            document_id: Document identifier.
            metadata: Additional metadata.

        Returns:
            List of fixed-size chunks.
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        id=f"{document_id}_chunk_{chunk_idx}",
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        chunk_strategy="fixed",
                        metadata={**metadata, "strategy": "fixed"},
                    )
                )
                chunk_idx += 1

            start = end - self._overlap
            if start >= len(text):
                break

        return chunks
