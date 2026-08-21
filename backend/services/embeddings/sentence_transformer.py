"""Sentence-Transformers embedding provider (using fastembed).

Uses the fastembed library to generate dense embeddings
locally on CPU using ONNX runtime for significantly lower memory footprint.
"""

from __future__ import annotations

import numpy as np

from backend.services.embeddings.base import EmbeddingProvider
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """Local sentence-transformer embedding provider (FastEmbed backed).

    Optimized for CPU with batch processing support and
    lazy caching of the loaded model to save RAM.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None

    def _get_model(self):
        """Lazy-load the fastembed model."""
        if self._model is None:
            logger.info(
                "loading_embedding_model",
                model=self._model_name,
                provider="fastembed"
            )
            from fastembed import TextEmbedding
            self._model = TextEmbedding(self._model_name)
            logger.info(
                "embedding_model_loaded",
                model=self._model_name,
                dimension=self.dimension,
            )
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using fastembed.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        model = self._get_model()
        # fastembed returns a generator, convert to list then array
        embeddings = list(model.embed(texts, batch_size=self._batch_size))
        return np.asarray(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text.

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        model = self._get_model()
        # fastembed expects an iterable of strings
        embedding = list(model.embed([query]))[0]
        return np.asarray(embedding)

    @property
    def dimension(self) -> int:
        """Get embedding dimension from the loaded model, default 384."""
        return 384

    @property
    def model_name(self) -> str:
        return self._model_name
