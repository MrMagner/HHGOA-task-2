"""BM25 lexical retrieval using rank_bm25.

Provides TF-IDF-based lexical matching as a complement to
dense semantic retrieval in the hybrid pipeline.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from backend.services.retrieval.base import Retriever, RetrievalResult
from backend.utils.logging import get_logger

logger = get_logger(__name__)


import string

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    # Remove standard punctuation but keep all unicode letters/marks
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return [t for t in text.split() if len(t) > 0]


class BM25Retriever(Retriever):
    """BM25 lexical retrieval using rank_bm25 library.

    Builds an in-memory BM25 index from document texts.
    Lightweight and requires no external infrastructure.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._documents: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []

    @property
    def method_name(self) -> str:
        return "bm25"

    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._documents) > 0

    def build_index(
        self,
        ids: list[str],
        texts: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build the BM25 index from document texts.

        Args:
            ids: List of document IDs.
            texts: List of document texts.
            metadata_list: Optional list of metadata dicts.
        """
        if metadata_list is None:
            metadata_list = [{}] * len(ids)

        self._tokenized_corpus = [_tokenize(text) for text in texts]
        self._documents = [
            {"id": doc_id, "text": text, "metadata": meta}
            for doc_id, text, meta in zip(ids, texts, metadata_list)
        ]

        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(
            "bm25_index_built",
            documents=len(self._documents),
            avg_tokens=round(
                sum(len(t) for t in self._tokenized_corpus) / max(len(self._tokenized_corpus), 1),
                1,
            ),
        )

    def search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Search using BM25 lexical matching.

        Args:
            query: The text query.
            query_embedding: Unused for BM25.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by BM25 score.

        Raises:
            RuntimeError: If the BM25 index hasn't been built yet.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue

            doc = self._documents[idx]
            results.append(
                RetrievalResult(
                    chunk_id=str(doc["id"]),
                    document_id=str(doc.get("metadata", {}).get("query_id", doc["id"])),
                    text=doc["text"],
                    score=score,
                    metadata=doc.get("metadata", {}),
                    retrieval_method="bm25",
                )
            )

        logger.info(
            "bm25_retrieval_complete",
            query_tokens=len(tokenized_query),
            results_count=len(results),
            top_score=results[0].score if results else 0.0,
        )

        return results
