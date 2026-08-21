"""Qdrant vector store abstraction.

Provides a clean interface over qdrant-client for creating collections,
upserting vectors, and performing similarity search. Supports both
local (in-memory/on-disk) and remote Qdrant instances.
"""

from __future__ import annotations

from typing import Any, Optional
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Qdrant vector store wrapper.

    Supports local mode (no Docker needed) and remote Qdrant instances.
    Uses disk persistence in local mode for reusable indexes.
    """

    def __init__(
        self,
        collection_name: str = "msmarco_xi",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_path: str = "./qdrant_data",
        embedding_dimension: int = 384,
    ) -> None:
        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension

        if url:
            logger.info("connecting_qdrant_remote", url=url)
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            logger.info("connecting_qdrant_local", path=local_path)
            self._client = QdrantClient(path=local_path)

    @property
    def client(self) -> QdrantClient:
        """Get the raw Qdrant client."""
        return self._client

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    def collection_exists(self) -> bool:
        """Check if the collection already exists."""
        try:
            collections = self._client.get_collections().collections
            return any(c.name == self._collection_name for c in collections)
        except Exception:
            return False

    def create_collection(self, recreate: bool = False) -> None:
        """Create the vector collection.

        Args:
            recreate: If True, delete and recreate existing collection.
        """
        if recreate and self.collection_exists():
            logger.warning("recreating_collection", collection=self._collection_name)
            self._client.delete_collection(self._collection_name)

        if not self.collection_exists():
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "collection_created",
                collection=self._collection_name,
                dimension=self._embedding_dimension,
            )
        else:
            logger.info("collection_exists", collection=self._collection_name)

    def upsert_batch(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
        batch_size: int = 100,
    ) -> int:
        """Upsert a batch of vectors into the collection.

        Args:
            ids: List of unique string IDs.
            embeddings: numpy array of shape (n, dim).
            texts: List of document texts.
            metadata_list: Optional list of metadata dicts.
            batch_size: Number of points per upsert batch.

        Returns:
            Total number of points upserted.
        """
        if metadata_list is None:
            metadata_list = [{}] * len(ids)

        total_upserted = 0

        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            points = []

            for j in range(i, batch_end):
                payload = {
                    "text": texts[j],
                    **metadata_list[j],
                }
                
                # Convert string ID to deterministic UUID for Qdrant if needed
                point_id = ids[j]
                try:
                    uuid.UUID(str(point_id))
                except ValueError:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id)))
                    
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embeddings[j].tolist(),
                        payload=payload,
                    )
                )

            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            total_upserted += len(points)

            if (i // batch_size) % 10 == 0:
                logger.info(
                    "upsert_progress",
                    upserted=total_upserted,
                    total=len(ids),
                )

        logger.info("upsert_complete", total=total_upserted)
        return total_upserted

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.

        Returns:
            List of dicts with id, score, text, and metadata.
        """
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
        )

        output = []
        for point in results.points:
            payload = point.payload or {}
            output.append({
                "id": str(point.id),
                "score": point.score,
                "text": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"},
            })

        return output

    def count(self) -> int:
        """Get the number of points in the collection."""
        try:
            info = self._client.get_collection(self._collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def close(self) -> None:
        """Close the Qdrant client connection."""
        try:
            self._client.close()
        except Exception:
            pass
