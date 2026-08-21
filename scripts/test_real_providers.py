import asyncio
import json
import time
import os

from backend.config.settings import get_settings
from backend.services.stt.sarvam import SarvamSTT
from backend.services.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from backend.services.retrieval.vector_store import VectorStore
from backend.services.retrieval.dense import DenseRetriever
from backend.services.retrieval.bm25 import BM25Retriever
from backend.services.retrieval.hybrid import HybridRetriever
from backend.services.generation.groq_provider import GroqProvider
from backend.services.guardrails.offtopic import OffTopicDetector
from backend.pipeline.orchestrator import VoiceRAGPipeline

async def main():
    print("Loading settings...")
    settings = get_settings()
    
    # 1. Verify Configuration
    assert settings.demo_mode is False, "demo_mode should be False"
    assert settings.stt_provider == "sarvam", "stt_provider should be sarvam"
    assert settings.llm_provider == "groq", "llm_provider should be groq"
    assert settings.sarvam_api_key, "sarvam_api_key is missing"
    assert settings.groq_api_key, "groq_api_key is missing"
    print("Configuration verification passed (Keys detected without exposing).")
    
    # Init STT and LLM
    print("\nInitializing Sarvam STT and Groq LLM...")
    stt_provider = SarvamSTT(api_key=settings.sarvam_api_key, api_url=settings.sarvam_api_url, language=settings.sarvam_language)
    llm_provider = GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model, api_url=settings.groq_api_url)
    
    assert stt_provider.is_available(), "Sarvam STT is not available"
    assert llm_provider.is_available(), "Groq LLM is not available"
    
    # Init Embeddings
    print("Initializing Embeddings & Retrievers...")
    embedding_provider = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    
    vector_store = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
        embedding_dimension=settings.embedding_dimension,
    )
    dense_retriever = DenseRetriever(vector_store=vector_store)
    bm25_retriever = BM25Retriever()
    
    # Re-build BM25 index from Qdrant data (in-memory)
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
    
    # Test Text Query
    text_query = "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?"
    print(f"\n--- Testing Real Text Query: '{text_query}' ---")
    start = time.perf_counter()
    response = await pipeline.process_text_query(query=text_query, request_id="text_1", top_k=3, rerank=False)
    total_time = (time.perf_counter() - start) * 1000
    print(f"Answer: {response.answer}")
    print(f"Latency: Retrieval={response.latency.retrieval_ms}ms, Generation={response.latency.generation_ms}ms, Total={response.latency.total_ms}ms")
    
    # Test Voice Query
    print(f"\n--- Testing Real Voice Query ---")
    with open("test_hindi.mp3", "rb") as f:
        audio_data = f.read()
        
    start = time.perf_counter()
    response_voice = await pipeline.process_voice_query(
        audio_data=audio_data, 
        content_type="audio/mp3", 
        request_id="voice_1", 
        language="hi-IN",
        top_k=3,
        rerank=False
    )
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Transcript (Sarvam): {response_voice.transcript}")
    print(f"Answer (Groq): {response_voice.answer}")
    print(f"Latency: STT={response_voice.latency.stt_ms}ms, Retrieval={response_voice.latency.retrieval_ms}ms, Generation={response_voice.latency.generation_ms}ms, Total={response_voice.latency.total_ms}ms")

if __name__ == "__main__":
    asyncio.run(main())
