import asyncio
import json
import random
from backend.config.settings import get_settings
from backend.services.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from backend.services.retrieval.vector_store import VectorStore
from backend.services.retrieval.dense import DenseRetriever
from backend.services.retrieval.bm25 import BM25Retriever
from backend.services.retrieval.hybrid import HybridRetriever

async def main():
    # 1. Load data
    data_file = "data/msmarco_xi_real.jsonl"
    records = []
    with open(data_file, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
            
    # Sample 5 queries
    random.seed(42)
    sample_records = random.sample(records, 5)
    
    print("=== Sampled Queries ===")
    for r in sample_records:
        print(f"ID: {r['id']}")
        print(f"Query: {r['metadata']['query']}")
        print(f"Passage: {r['text'][:100]}...")
        print("-" * 40)
        
    # 2. Setup Retrievers
    settings = get_settings()
    vector_store = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
        embedding_dimension=settings.embedding_dimension,
    )
    
    embedding = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    
    dense = DenseRetriever(vector_store=vector_store)
    bm25 = BM25Retriever()
    
    # Load all documents into BM25
    print("Loading BM25 index from Qdrant...")
    scroll_res = vector_store._client.scroll(
        collection_name=settings.qdrant_collection,
        limit=2000,
        with_payload=True,
    )
    points = scroll_res[0]
    ids = [str(p.id) for p in points]
    texts = [str(p.payload.get("text", "")) for p in points]
    bm25.build_index(ids, texts)
    
    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        dense_weight=settings.retrieval_dense_weight,
        bm25_weight=settings.retrieval_bm25_weight,
    )
    
    print("\n=== Retrieval Test ===")
    # Add a known problematic query
    test_queries = sample_records + [{"id": "manual", "metadata": {"query": "What is information recovery?"}, "text": ""}]
    
    for r in test_queries:
        query = r["metadata"]["query"]
        expected_id = str(r["id"])
        
        print(f"\nQuery: '{query}'")
        print(f"Expected ID: {expected_id}")
        
        query_emb = embedding.embed_query(query)
        
        # Test Dense
        dense_res = dense.search(query, query_emb, top_k=3)
        print("  Dense Top-3 IDs:", [res.document_id for res in dense_res], "| Scores:", [round(res.score, 3) for res in dense_res])
        if dense_res:
            print(f"    Top result snippet: {dense_res[0].text[:50]}")
            
        # Test BM25
        bm25_res = bm25.search(query, query_emb, top_k=3)
        print("  BM25 Top-3 IDs:", [res.document_id for res in bm25_res], "| Scores:", [round(res.score, 3) for res in bm25_res])
        if bm25_res:
            print(f"    Top result snippet: {bm25_res[0].text[:50]}")
            
        # Test Hybrid
        hybrid_res = hybrid.search(query, query_emb, top_k=3)
        print("  Hybrid Top-3 IDs:", [res.document_id for res in hybrid_res], "| Scores:", [round(res.score, 3) for res in hybrid_res])
        if hybrid_res:
            print(f"    Top result snippet: {hybrid_res[0].text[:50]}")

if __name__ == "__main__":
    asyncio.run(main())
