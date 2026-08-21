"""Metadata-aware chunking strategy.

Preserves document metadata (query, passage info) from the MSMARCO-XI
dataset structure and creates chunks that retain this context.
"""

from __future__ import annotations

from typing import Any

from backend.ingestion.chunking.base import Chunker, Chunk
from backend.ingestion.chunking.sentence import SentenceChunker


class MetadataAwareChunker(Chunker):
    """Metadata-aware chunker optimized for MSMARCO-XI dataset.

    Preserves document-level metadata (query association, passage type)
    and delegates text splitting to the sentence chunker.
    Each chunk retains its source document context.
    """

    def __init__(self, max_chunk_size: int = 512) -> None:
        self._max_chunk_size = max_chunk_size
        self._sentence_chunker = SentenceChunker(max_chunk_size, 0)

    @property
    def strategy_name(self) -> str:
        return "metadata_aware"

    def chunk(self, text: str, document_id: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk text while preserving metadata context.

        Args:
            text: Full document/passage text.
            document_id: Document identifier.
            metadata: Metadata including query, language, passage_type, etc.

        Returns:
            List of chunks with enriched metadata.
        """
        if not text.strip():
            return []

        metadata = metadata or {}

        # For short passages that fit in a single chunk
        if len(text) <= self._max_chunk_size:
            return [
                Chunk(
                    id=f"{document_id}_chunk_0",
                    text=text.strip(),
                    document_id=document_id,
                    chunk_index=0,
                    chunk_strategy="metadata_aware",
                    metadata={
                        **metadata,
                        "strategy": "metadata_aware",
                        "is_complete_passage": True,
                    },
                )
            ]

        # Delegate to sentence chunker for longer texts
        base_chunks = self._sentence_chunker.chunk(text, document_id, metadata)

        # Enrich with metadata-aware information
        enriched = []
        for chunk in base_chunks:
            chunk.chunk_strategy = "metadata_aware"
            chunk.metadata["strategy"] = "metadata_aware"
            chunk.metadata["is_complete_passage"] = False
            chunk.metadata["total_chunks"] = len(base_chunks)
            chunk.id = f"{document_id}_ma_chunk_{chunk.chunk_index}"
            enriched.append(chunk)

        return enriched
