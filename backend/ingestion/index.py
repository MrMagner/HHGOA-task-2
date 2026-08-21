"""Document embedding and indexing pipeline.

Takes processed chunks, generates embeddings using the provided model,
and indexes them into the vector store.
"""

from __future__ import annotations

from typing import Any

from backend.ingestion.chunking.base import Chunk
from backend.services.embeddings.base import EmbeddingProvider
from backend.services.retrieval.vector_store import VectorStore
from backend.services.retrieval.bm25 import BM25Retriever
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def build_index(
    chunks: list[Chunk],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    bm25_retriever: BM25Retriever | None = None,
    batch_size: int = 64,
) -> None:
    """Build the search index from processed chunks.

    Embeds texts and upserts them to the vector store.
    Optionally builds the BM25 index in parallel.

    Args:
        chunks: List of processed Chunk objects.
        embedding_provider: Provider to generate dense embeddings.
        vector_store: Target vector store for dense retrieval.
        bm25_retriever: Optional BM25 retriever to populate.
        batch_size: Batch size for embedding and upsertion.
    """
    total_chunks = len(chunks)
    logger.info("starting_indexing", total_chunks=total_chunks)
    
    if total_chunks == 0:
        logger.warning("no_chunks_to_index")
        return

    # 1. Initialize Vector Store
    vector_store.create_collection(recreate=True)
    
    # 2. Extract data from chunks
    ids = [chunk.id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    metadata_list = [chunk.metadata for chunk in chunks]
    
    # 3. Generate Embeddings and Upsert to Vector Store
    # We do this in batches to manage memory
    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        batch_texts = texts[i:batch_end]
        batch_ids = ids[i:batch_end]
        batch_metadata = metadata_list[i:batch_end]
        
        # Generate embeddings
        embeddings = embedding_provider.embed_texts(batch_texts)
        
        # Upsert
        vector_store.upsert_batch(
            ids=batch_ids,
            embeddings=embeddings,
            texts=batch_texts,
            metadata_list=batch_metadata,
            batch_size=batch_size,
        )
        
    logger.info("vector_indexing_complete", points_in_store=vector_store.count())
    
    # 4. Build BM25 Index (if provided)
    if bm25_retriever is not None:
        logger.info("building_bm25_index")
        bm25_retriever.build_index(ids=ids, texts=texts, metadata_list=metadata_list)
        logger.info("bm25_indexing_complete")
        
    logger.info("all_indexing_complete")
