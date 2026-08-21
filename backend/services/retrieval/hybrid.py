"""Hybrid retrieval combining Dense and BM25 using Reciprocal Rank Fusion.

RRF combines results from multiple retrieval methods by computing
a fused score based on rank positions, rather than raw scores.
This gives robust fusion even when score scales differ.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from backend.services.retrieval.base import Retriever, RetrievalResult
from backend.services.retrieval.dense import DenseRetriever
from backend.services.retrieval.bm25 import BM25Retriever
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# RRF constant — controls rank smoothing (standard value is 60)
RRF_K = 60


class HybridRetriever(Retriever):
    """Hybrid retrieval using Reciprocal Rank Fusion (RRF).

    Combines results from dense (semantic) and BM25 (lexical) retrieval
    methods to get the best of both worlds.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight

    @property
    def method_name(self) -> str:
        return "hybrid"

    def is_ready(self) -> bool:
        return self._dense.is_ready() and self._bm25.is_ready()

    def search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Perform hybrid retrieval with RRF fusion.

        Retrieves from both dense and BM25, then fuses results
        using Reciprocal Rank Fusion.

        Args:
            query: The text query.
            query_embedding: Pre-computed query embedding for dense retrieval.
            top_k: Number of final fused results to return.

        Returns:
            List of RetrievalResult sorted by fused RRF score.
        """
        # Retrieve more candidates from each method for better fusion
        candidate_k = min(top_k * 3, 50)

        # Dense retrieval
        dense_results: list[RetrievalResult] = []
        if self._dense.is_ready() and query_embedding is not None:
            try:
                dense_results = self._dense.search(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=candidate_k,
                )
            except Exception as e:
                logger.warning("dense_retrieval_failed", error=str(e))

        # BM25 retrieval
        bm25_results: list[RetrievalResult] = []
        if not self._bm25.is_ready():
            logger.info("lazy_loading_bm25")
            try:
                from backend.config.settings import get_settings
                settings = get_settings()
                vector_store = self._dense._vector_store
                scroll_res = vector_store._client.scroll(
                    collection_name=settings.qdrant_collection,
                    limit=settings.dataset_max_samples or 2000,
                    with_payload=True,
                )
                points = scroll_res[0]
                if points:
                    ids = [str(p.id) for p in points]
                    texts = [str(p.payload.get("text", "")) for p in points]
                    # self._bm25.build_index(ids, texts)
                    del points, scroll_res, ids, texts
                    import gc; gc.collect()
            except Exception as e:
                logger.warning("bm25_lazy_load_failed", error=str(e))
                
        if self._bm25.is_ready():
            try:
                bm25_results = self._bm25.search(
                    query=query,
                    top_k=candidate_k,
                )
            except Exception as e:
                logger.warning("bm25_retrieval_failed", error=str(e))

        # If only one method has results, return those
        if not dense_results and not bm25_results:
            return []
        if not dense_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return dense_results[:top_k]

        # Apply Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            top_k=top_k,
        )

        logger.info(
            "hybrid_retrieval_complete",
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
            fused_count=len(fused),
            top_score=fused[0].score if fused else 0.0,
        )

        return fused

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Apply Reciprocal Rank Fusion to combine results.

        RRF Score = sum(weight / (k + rank)) for each method

        Args:
            dense_results: Results from dense retrieval.
            bm25_results: Results from BM25 retrieval.
            top_k: Number of fused results to return.

        Returns:
            Fused results sorted by RRF score.
        """
        # Accumulate RRF scores by document ID
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, RetrievalResult] = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            rrf_scores[result.chunk_id] += self._dense_weight / (RRF_K + rank + 1)
            if result.chunk_id not in doc_map:
                doc_map[result.chunk_id] = result

        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            rrf_scores[result.chunk_id] += self._bm25_weight / (RRF_K + rank + 1)
            if result.chunk_id not in doc_map:
                doc_map[result.chunk_id] = result

        # Sort by RRF score and take top-k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        fused_results = []
        for doc_id in sorted_ids:
            original = doc_map[doc_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=doc_id,
                    document_id=original.document_id,
                    text=original.text,
                    score=round(rrf_scores[doc_id], 6),
                    metadata=original.metadata,
                    retrieval_method="hybrid",
                )
            )

        return fused_results
