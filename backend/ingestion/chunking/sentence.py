"""Sentence-aware chunking strategy.

Groups sentences into chunks respecting sentence boundaries,
producing more semantically coherent chunks than fixed-size splitting.
"""

from __future__ import annotations

import re
from typing import Any

from backend.ingestion.chunking.base import Chunker, Chunk


# Simple sentence splitting regex
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex heuristics."""
    sentences = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


class SentenceChunker(Chunker):
    """Sentence-aware chunking that respects sentence boundaries.

    Groups sentences into chunks up to the target size,
    ensuring no sentence is split across chunks.
    Overlap is achieved by repeating trailing sentences.
    """

    def __init__(self, max_chunk_size: int = 512, overlap_sentences: int = 1) -> None:
        self._max_chunk_size = max_chunk_size
        self._overlap_sentences = overlap_sentences

    @property
    def strategy_name(self) -> str:
        return "sentence"

    def chunk(self, text: str, document_id: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into sentence-aware chunks.

        Args:
            text: Full document text.
            document_id: Document identifier.
            metadata: Additional metadata.

        Returns:
            List of sentence-boundary-respecting chunks.
        """
        if not text.strip():
            return []

        metadata = metadata or {}
        sentences = _split_sentences(text)

        if not sentences:
            return [
                Chunk(
                    id=f"{document_id}_chunk_0",
                    text=text.strip(),
                    document_id=document_id,
                    chunk_index=0,
                    chunk_strategy="sentence",
                    metadata={**metadata, "strategy": "sentence"},
                )
            ]

        chunks = []
        current_sentences: list[str] = []
        current_length = 0
        chunk_idx = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds max_chunk_size and we have content
            if current_length + sentence_len > self._max_chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        id=f"{document_id}_chunk_{chunk_idx}",
                        text=chunk_text,
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        chunk_strategy="sentence",
                        metadata={**metadata, "strategy": "sentence", "sentences": len(current_sentences)},
                    )
                )
                chunk_idx += 1

                # Calculate overlap: keep last N sentences
                if self._overlap_sentences > 0:
                    overlap_sentences = current_sentences[-self._overlap_sentences:]
                    current_sentences = overlap_sentences
                    current_length = sum(len(s) for s in overlap_sentences) + max(0, len(overlap_sentences)-1)
                else:
                    current_sentences = []
                    current_length = 0

            current_sentences.append(sentence)
            # +1 for space between sentences if not first
            current_length += sentence_len + (1 if current_sentences else 0)

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    id=f"{document_id}_chunk_{chunk_idx}",
                    text=chunk_text,
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    chunk_strategy="sentence",
                    metadata={**metadata, "strategy": "sentence", "sentences": len(current_sentences)},
                )
            )

        return chunks
