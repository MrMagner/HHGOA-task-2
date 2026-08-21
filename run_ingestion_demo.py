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
import os

logging.basicConfig(level=logging.INFO)

def main():
    settings = get_settings()
    
    os.makedirs("data", exist_ok=True)
    dataset_file = Path("data/msmarco_xi_mock.jsonl")
    
    # We create a mock slice since HF datasets API returns 500 and the 4GB file takes 30 mins to download.
    mock_data = [
        {
            "query_id": "1",
            "query": "What is the capital of France?",
            "passage_id": "p1",
            "passage": "Paris is the capital and most populous city of France.",
            "language": "hi"
        },
        {
            "query_id": "2",
            "query": "How deep is the Mariana Trench?",
            "passage_id": "p2",
            "passage": "The Mariana Trench is the deepest oceanic trench on Earth. It reaches a maximum-known depth of about 10,984 meters.",
            "language": "hi"
        },
        {
            "query_id": "3",
            "query": "What is RAG in AI?",
            "passage_id": "p3",
            "passage": "Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external knowledge base to ground large language models.",
            "language": "hi"
        }
    ]
    
    with open(dataset_file, "w") as f_out:
        for row in mock_data:
            # Map it exactly like download.py does
            record = {
                "id": str(row["passage_id"]),
                "text": row["passage"],
                "metadata": {
                    "query": row["query"],
                    "query_id": str(row["query_id"]),
                    "language": row["language"],
                }
            }
            f_out.write(json.dumps(record) + "\n")
            
    print("Processing...")
    chunker = SentenceChunker(max_chunk_size=512, overlap_sentences=1)
    chunks = process_documents(dataset_file, chunker, max_chunks_per_doc=5)
    
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
    
    print("Testing retrieval...")
    q_emb = embedding_provider.embed_query("What is the capital of France?")
    results = vector_store.search(q_emb, top_k=1)
    if results:
        print(f"Verification query successful, found {len(results)} results.")
        print("Top result:")
        print(json.dumps(results[0], indent=2))
    else:
        print("Verification query returned no results.")

if __name__ == "__main__":
    main()
