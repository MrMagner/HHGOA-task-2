"""Sentence-Transformers embedding provider.

Uses the sentence-transformers library to generate dense embeddings
locally on CPU using lightweight models like all-MiniLM-L6-v2.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.services.embeddings.base import EmbeddingProvider
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local sentence-transformer embedding provider.

    Optimized for CPU with batch processing support and
    optional caching of the loaded model.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            logger.info(
                "loading_embedding_model",
                model=self._model_name,
                device=self._device,
            )
            self._model = SentenceTransformer(self._model_name, device=self._device)
            logger.info(
                "embedding_model_loaded",
                model=self._model_name,
                dimension=self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using sentence-transformers.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text.

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        model = self._get_model()
        embedding = model.encode(
            query,
            normalize_embeddings=True,
        )
        return np.asarray(embedding)

    @property
    def dimension(self) -> int:
        """Get embedding dimension from the loaded model."""
        model = self._get_model()
        dim = model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384

    @property
    def model_name(self) -> str:
        return self._model_name
