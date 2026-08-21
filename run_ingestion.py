import asyncio
import logging
import json
import os
import shutil
from pathlib import Path

from backend.config.settings import get_settings
from backend.ingestion.chunking.sentence import SentenceChunker
from backend.ingestion.process import process_documents
from backend.ingestion.index import build_index
from backend.services.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from backend.services.retrieval.vector_store import VectorStore
from backend.services.retrieval.bm25 import BM25Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    settings = get_settings()
    
    # Strictly enforce that this script must run on real data!
    if settings.demo_mode:
        raise RuntimeError(
            "DEMO_MODE is true. "
            "Please use run_ingestion_demo.py for demo mode, or set DEMO_MODE=false to process REAL data."
        )

    # In production, we assume the subset is downloaded by DuckDB to data/msmarco_xi_real.jsonl
    dataset_file = Path("data/msmarco_xi_real.jsonl")
    if not dataset_file.exists():
        raise FileNotFoundError(
            "Could not find real dataset subset at data/msmarco_xi_real.jsonl. "
            "Make sure the duckdb extraction script has completed."
        )
        
    print(f"============================================================")
    print(f" PROCESSING REAL MSMARCO-XI DATA ")
    print(f"============================================================")
    
    # Verify disk space
    total, used, free = shutil.disk_usage(".")
    print(f"Available disk space: {free / (1024**3):.2f} GB")
    
    # Record metrics
    file_size = os.path.getsize(dataset_file)
    print(f"Downloaded subset size: {file_size / (1024**2):.2f} MB")
    
    records_count = sum(1 for _ in open(dataset_file))
    print(f"Number of REAL records loaded: {records_count}")
    
    # Example estimation if we processed the full 3.8GB
    print(f"Estimated full processing size: 3.8 GB Parquet -> ~6 GB JSONL + Index")
    
    print("\n--- Pipeline Initialized ---")
    
    # Process
    print("Chunking documents...")
    chunker = SentenceChunker(max_chunk_size=512, overlap_sentences=1)
    chunks = process_documents(dataset_file, chunker, max_chunks_per_doc=5)
    print(f"Total chunks generated: {len(chunks)}")
    
    # Index
    print("Generating embeddings and building index...")
    embedding_provider = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        device="cpu",
        batch_size=16,
    )
    
    # Lock safely
    if os.path.exists("./qdrant_data/.lock"):
        os.remove("./qdrant_data/.lock")
        
    vector_store = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
        embedding_dimension=settings.embedding_dimension,
    )
    bm25 = BM25Retriever()
    
    build_index(
        chunks=chunks,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        bm25_retriever=bm25,
        batch_size=16,
    )
    
    qdrant_points = vector_store.count()
    bm25_docs = len(bm25._documents) if bm25._documents else 0
    print(f"Embeddings generated: {len(chunks)}")
    print(f"Qdrant points indexed: {qdrant_points}")
    print(f"BM25 documents indexed: {bm25_docs}")
    
    print("\n--- Final Verification Retrieval ---")
    
    # Read the first query from the real dataset!
    with open(dataset_file, "r") as f:
        first_row = json.loads(f.readline())
        
    real_query = first_row["metadata"]["query"]
    print(f"Real query: {real_query}")
    
    # Retrieve
    q_emb = embedding_provider.embed_query(real_query)
    results = vector_store.search(q_emb, top_k=1)
    
    if results:
        best = results[0]
        print(f"Retrieved passage: {best['text']}")
        print(f"Score: {best['score']}")
        print(f"Metadata: {best['metadata']}")
        print(f"Source ID: {best['id']}")
    else:
        print("Verification query returned no results!")

if __name__ == "__main__":
    main()
