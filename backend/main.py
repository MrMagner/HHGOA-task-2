"""FastAPI application entrypoint for Voice-RAG system."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.config.settings import get_settings, STTProvider, LLMProvider
from backend.utils.logging import setup_logging, get_logger
from backend.api.middleware import setup_middleware, setup_exception_handlers
from backend.api.routes import router

from backend.services.stt.sarvam import SarvamSTT
from backend.services.stt.elevenlabs import ElevenLabsSTT
from backend.services.stt.demo import DemoSTT
from backend.services.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from backend.services.retrieval.vector_store import VectorStore
from backend.services.retrieval.dense import DenseRetriever
from backend.services.retrieval.bm25 import BM25Retriever
from backend.services.retrieval.hybrid import HybridRetriever
from backend.services.reranking.cross_encoder import CrossEncoderReranker
from backend.services.generation.groq_provider import GroqProvider
from backend.services.generation.openai_provider import OpenAIProvider
from backend.services.generation.demo import DemoLLM
from backend.services.guardrails.offtopic import OffTopicDetector
from backend.pipeline.orchestrator import VoiceRAGPipeline

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to setup and teardown resources."""
    settings = get_settings()
    
    # 1. Logging
    setup_logging(log_level=settings.log_level, debug=settings.debug)
    logger.info("application_startup", version=settings.app_version)
    
    # Attach settings to app state
    app.state.settings = settings
    
    # 2. Vector Store
    vector_store = VectorStore(
        collection_name=settings.qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
        embedding_dimension=settings.embedding_dimension,
    )
    if not vector_store.collection_exists():
        vector_store.create_collection()
    app.state.vector_store = vector_store
    
    # 3. STT Provider
    stt_provider = None
    if settings.stt_provider == STTProvider.DEMO or settings.demo_mode:
        stt_provider = DemoSTT()
    elif settings.stt_provider == STTProvider.SARVAM:
        stt_provider = SarvamSTT(
            api_key=settings.sarvam_api_key,
            api_url=settings.sarvam_api_url,
            language=settings.sarvam_language,
        )
    elif settings.stt_provider == STTProvider.ELEVENLABS:
        stt_provider = ElevenLabsSTT(
            api_key=settings.elevenlabs_api_key,
            api_url=settings.elevenlabs_api_url,
        )
    
    # 4. LLM Provider
    llm_provider = None
    if settings.llm_provider == LLMProvider.DEMO or settings.demo_mode:
        llm_provider = DemoLLM()
    elif settings.llm_provider == LLMProvider.GROQ:
        llm_provider = GroqProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            api_url=settings.groq_api_url,
        )
    elif settings.llm_provider == LLMProvider.OPENAI:
        llm_provider = OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            api_url=settings.openai_api_url,
        )

    # 5. Embeddings & Retrieval
    embedding_provider = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    logger.info("warming_up_embedding_model")
    embedding_provider.embed_query("warmup")

    
    dense_retriever = DenseRetriever(vector_store=vector_store)
    bm25_retriever = BM25Retriever()
    
    # Load documents into BM25 index
    logger.info("loading_bm25_from_qdrant")
    try:
        scroll_res = vector_store._client.scroll(
            collection_name=settings.qdrant_collection,
            limit=settings.dataset_max_samples or 2000,
            with_payload=True,
        )
        points = scroll_res[0]
        if points:
            ids = [str(p.id) for p in points]
            texts = [str(p.payload.get("text", "")) for p in points]
            bm25_retriever.build_index(ids, texts)
    except Exception as e:
        logger.warning("bm25_load_failed", error=str(e))
    
    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        dense_weight=settings.retrieval_dense_weight,
        bm25_weight=settings.retrieval_bm25_weight,
    )
    
    # 6. Reranking
    reranker = None
    if settings.rerank_enabled:
        reranker = CrossEncoderReranker(model_name=settings.rerank_model)
        
    # 7. Pipeline Orchestrator
    pipeline = VoiceRAGPipeline(
        stt_provider=stt_provider,  # type: ignore
        embedding_provider=embedding_provider,
        retriever=retriever,
        llm_provider=llm_provider,  # type: ignore
        reranker=reranker,
        offtopic_detector=OffTopicDetector(threshold=settings.offtopic_threshold),
    )
    app.state.pipeline = pipeline
    
    logger.info("application_ready")
    
    yield  # Run application
    
    # Teardown
    logger.info("application_shutdown")
    vector_store.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    
    # Middleware & Exception Handlers
    setup_middleware(app, cors_origins=settings.cors_origins_list)
    setup_exception_handlers(app)
    
    # Routes
    app.include_router(router)
    
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
