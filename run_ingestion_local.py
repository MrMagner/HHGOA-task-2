import asyncio
from pathlib import Path
from backend.config.settings import get_settings
from backend.ingestion.chunking.sentence import SentenceChunker
from backend.ingestion.process import process_documents
from backend.ingestion.index import build_index
from backend.services.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from backend.services.retrieval.vector_store import VectorStore
from backend.services.retrieval.bm25 import BM25Retriever
import logging
import json

logging.basicConfig(level=logging.INFO)

def main():
    settings = get_settings()
    
    # We load duckdb extracted data which might not be perfectly mapped
    dataset_file = Path("data/msmarco_xi_real.jsonl")
    
    # First re-map the duckdb parquet schema to our chunker schema
    # The duckdb output gives rows directly from parquet. Let's see what columns it has and map them.
    mapped_file = Path("data/real_msmarco_subset.jsonl")
    with open(dataset_file, "r") as f_in, open(mapped_file, "w") as f_out:
        for i, line in enumerate(f_in):
            row = json.loads(line)
            # Find the best column for text
            text = row.get("passage") or row.get("text") or row.get("content") or ""
            query = row.get("query", "")
            doc_id = str(row.get("passage_id") or row.get("id") or i)
            
            record = {
                "id": doc_id,
                "text": text,
                "metadata": {
                    "query": query,
                    "language": "hi",
                }
            }
            f_out.write(json.dumps(record) + "\n")
    
    print("Processing...")
    chunker = SentenceChunker(max_chunk_size=512, overlap_sentences=1)
    chunks = process_documents(mapped_file, chunker, max_chunks_per_doc=5)
    
    print("Indexing...")
    embedding_provider = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        device="cpu",
        batch_size=16,
    )
    vector_store = VectorStore(
        collection_name=settings.qdrant_collection,
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
    
    print(f"Indexed {vector_store.count()} vectors successfully in Qdrant!")
    
    # Test a simple query to verify
    print("Testing retrieval...")
    q_emb = embedding_provider.embed_query("test query")
    results = vector_store.search(q_emb, top_k=1)
    if results:
        print(f"Verification query successful, found {len(results)} results.")
        print("Top result:")
        print(results[0])
    else:
        print("Verification query returned no results.")

if __name__ == "__main__":
    main()
