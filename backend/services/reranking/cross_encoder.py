"""Cross-encoder reranker using sentence-transformers.

Provides high-quality reranking by scoring each (query, document)
pair with a cross-encoder model. More accurate than bi-encoder
similarity but significantly slower.
"""

from __future__ import annotations
from typing import Any

from backend.services.reranking.base import Reranker
from backend.services.retrieval.base import RetrievalResult
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers.

    Disabled by default due to latency impact (~100-200ms per query).
    Use for offline evaluation or when latency budget allows.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info("loading_cross_encoder", model=self._model_name)
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install it or disable RERANK_ENABLED."
                )
            self._model = CrossEncoder(self._model_name)
            logger.info("cross_encoder_loaded", model=self._model_name)
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder scoring.

        Args:
            query: The original query text.
            results: Retrieval results to rerank.
            top_k: Number of top results after reranking.

        Returns:
            Reranked list of RetrievalResult.
        """
        if not results:
            return []

        model = self._get_model()

        # Create (query, document) pairs for scoring
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)

        # Attach cross-encoder scores and sort
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                RetrievalResult(
                    id=result.id if hasattr(result, "id") else getattr(result, "chunk_id", ""),
                    chunk_id=getattr(result, "chunk_id", ""),
                    document_id=getattr(result, "document_id", ""),
                    text=result.text,
                    score=float(score),
                    metadata={**result.metadata, "original_score": result.score},
                    retrieval_method=f"reranked_{result.retrieval_method}",
                )
            )

        logger.info(
            "reranking_complete",
            input_count=len(results),
            output_count=len(reranked),
            top_score=reranked[0].score if reranked else 0.0,
        )

        return reranked
