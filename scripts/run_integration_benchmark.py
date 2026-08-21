import asyncio
import json
from pathlib import Path

from backend.config.settings import get_settings
from backend.services.stt.demo import DemoSTT
from backend.services.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from backend.services.retrieval.vector_store import VectorStore
from backend.services.retrieval.dense import DenseRetriever
from backend.services.retrieval.bm25 import BM25Retriever
from backend.services.retrieval.hybrid import HybridRetriever
from backend.services.generation.demo import DemoLLM
from backend.services.guardrails.offtopic import OffTopicDetector
from backend.pipeline.orchestrator import VoiceRAGPipeline
from backend.evaluation.benchmark import run_latency_benchmark

async def main():
    print("Loading settings...")
    settings = get_settings()
    
    print("Loading 100 queries from MSMARCO-XI real dataset...")
    queries = []
    with open("data/msmarco_xi_real.jsonl", "r") as f:
        for line in f:
            record = json.loads(line)
            queries.append(record["metadata"]["query"])
            if len(queries) >= 100:
                break
    
    print(f"Loaded {len(queries)} queries. Initializing pipeline...")
    
    # Init STT and LLM (Demo mode for benchmarking without keys)
    stt_provider = DemoSTT()
    llm_provider = DemoLLM()
    
    # Init Embeddings
    embedding_provider = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    
    # Init Retrieval
    vector_store = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
        embedding_dimension=settings.embedding_dimension,
    )
    dense_retriever = DenseRetriever(vector_store=vector_store)
    bm25_retriever = BM25Retriever()
    
    # Re-build BM25 index because it's in-memory
    print("Re-building BM25 index from Qdrant data (in-memory)...")
    # Fetch all points from Qdrant
    points, offset = vector_store.client.scroll(
        collection_name=vector_store.collection_name, 
        limit=1000,
        with_payload=True
    )
    ids = [str(p.id) for p in points]
    texts = [p.payload.get("text", "") for p in points]
    bm25_retriever.build_index(ids=ids, texts=texts)
    
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_weight=settings.retrieval_dense_weight,
        bm25_weight=settings.retrieval_bm25_weight,
    )
    
    pipeline = VoiceRAGPipeline(
        stt_provider=stt_provider,
        embedding_provider=embedding_provider,
        retriever=retriever,
        llm_provider=llm_provider,
        offtopic_detector=OffTopicDetector(threshold=settings.offtopic_threshold),
    )
    
    results_dir = Path("results")
    print(f"Running benchmark with {len(queries)} queries...")
    metrics = await run_latency_benchmark(
        pipeline=pipeline,
        queries=queries,
        results_dir=results_dir,
        num_runs_per_query=1,
        top_k=5,
        rerank=False,
    )
    
    print("Benchmark complete!")
    print(json.dumps(metrics["percentiles"], indent=2))
    
if __name__ == "__main__":
    asyncio.run(main())
