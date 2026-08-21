"""Document processing pipeline.

Reads raw downloaded documents, applies the selected chunking strategy,
and prepares chunks for embedding and indexing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.ingestion.chunking.base import Chunker, Chunk
from backend.utils.logging import get_logger

logger = get_logger(__name__)


def process_documents(
    input_file: Path,
    chunker: Chunker,
    max_chunks_per_doc: int = 20,
) -> list[Chunk]:
    """Process raw documents into chunks.

    Args:
        input_file: Path to JSONL file containing raw documents.
        chunker: The chunking strategy to apply.
        max_chunks_per_doc: Maximum chunks to extract per document (to prevent blowup).

    Returns:
        List of processed Chunk objects.
    """
    logger.info(
        "processing_documents",
        file=str(input_file),
        chunker=chunker.strategy_name,
    )
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    all_chunks = []
    doc_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
                doc_id = str(record.get("id", f"doc_{doc_count}"))
                text = record.get("text", "")
                metadata = record.get("metadata", {})
                
                if not text:
                    skipped_count += 1
                    continue
                    
                # Apply chunking strategy
                doc_chunks = chunker.chunk(text=text, document_id=doc_id, metadata=metadata)
                
                # Enforce limit per document to prevent massive index blowup from anomalous docs
                if len(doc_chunks) > max_chunks_per_doc:
                    logger.debug(
                        "truncating_chunks",
                        doc_id=doc_id,
                        original=len(doc_chunks),
                        kept=max_chunks_per_doc,
                    )
                    doc_chunks = doc_chunks[:max_chunks_per_doc]
                    
                all_chunks.extend(doc_chunks)
                doc_count += 1
                
            except json.JSONDecodeError:
                skipped_count += 1
                
    logger.info(
        "processing_complete",
        documents_processed=doc_count,
        chunks_generated=len(all_chunks),
        documents_skipped=skipped_count,
        avg_chunks_per_doc=round(len(all_chunks) / max(doc_count, 1), 2),
    )
    
    return all_chunks
