"""API Routes for the Voice-RAG system."""

from __future__ import annotations

import os
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from backend.api.schemas import (
    TextQueryRequest,
    RAGResponse,
    HealthResponse,
)
from backend.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def get_pipeline(request: Request):
    """Dependency to get the initialized VoiceRAGPipeline from app state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


def get_vector_store(request: Request):
    """Dependency to get the initialized VectorStore from app state."""
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return store


def get_settings(request: Request):
    """Dependency to get app settings."""
    return request.app.state.settings


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Check system health and component status."""
    pipeline = getattr(request.app.state, "pipeline", None)
    vector_store = getattr(request.app.state, "vector_store", None)
    
    services = {
        "pipeline": pipeline is not None,
        "vector_store": vector_store is not None and vector_store.collection_exists(),
        "stt": pipeline._stt.is_available() if pipeline else False,
        "llm": pipeline._llm.is_available() if pipeline else False,
    }
    
    status = "healthy" if all(services.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        version=request.app.state.settings.app_version,
        services=services,
    )


@router.get("/api/v1/config")
async def get_config(request: Request):
    """Get public configuration."""
    settings = request.app.state.settings
    return {
        "demo_mode": settings.demo_mode,
        "app_name": settings.app_name,
        "version": settings.app_version,
    }


@router.post("/api/v1/query", response_model=RAGResponse)
async def query_text(
    request: Request,
    payload: TextQueryRequest,
    pipeline=Depends(get_pipeline),
):
    """Process a text query through the RAG pipeline."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        response = await pipeline.process_text_query(
            query=payload.query,
            request_id=request_id,
            top_k=payload.top_k,
            rerank=payload.rerank,
        )
        
        return response
        
    except Exception as e:
        logger.error("text_query_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/query/voice", response_model=RAGResponse)
async def query_voice(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form("hi-IN"),
    top_k: int = Form(5),
    include_sources: bool = Form(True),
    rerank: bool = Form(False),
    pipeline=Depends(get_pipeline),
):
    """Process a voice query through the STT + RAG pipeline."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        # Read audio file
        audio_bytes = await audio.read()
        content_type = audio.content_type or "audio/wav"
        
        response = await pipeline.process_voice_query(
            audio_data=audio_bytes,
            content_type=content_type,
            request_id=request_id,
            language=language,
            top_k=top_k,
            rerank=rerank,
        )
        
        if not include_sources:
            response.sources = []
            
        return response
        
    except Exception as e:
        logger.error("voice_query_failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/admin/index")
async def index_dataset(
    request: Request,
    max_samples: int | None = None,
    settings=Depends(get_settings),
    vector_store=Depends(get_vector_store),
):
    """Admin endpoint to trigger indexing of the MSMARCO-XI dataset.
    
    Warning: This is a heavy background operation. In a production app,
    this should use Celery or similar background task queues.
    For this demo, we run it synchronously (or block).
    """
    from backend.ingestion.download import download_dataset
    from backend.ingestion.chunking.sentence import SentenceChunker
    from backend.ingestion.process import process_documents
    from backend.ingestion.index import build_index
    
    # We use dependencies from the pipeline instead of recreating
    pipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    try:
        # Download
        output_dir = settings.data_path
        dataset_file = download_dataset(
            dataset_name=settings.dataset_name,
            split=settings.dataset_split,
            language=settings.dataset_language,
            output_dir=output_dir,
            max_samples=max_samples or settings.dataset_max_samples,
        )
        
        # Process & Chunk
        chunker = SentenceChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        chunks = process_documents(
            input_file=dataset_file,
            chunker=chunker,
            max_chunks_per_doc=settings.max_chunks_per_doc,
        )
        
        # Index
        build_index(
            chunks=chunks,
            embedding_provider=pipeline._embedding,
            vector_store=vector_store,
            bm25_retriever=pipeline._retriever._bm25 if hasattr(pipeline._retriever, "_bm25") else None,
            batch_size=settings.embedding_batch_size,
        )
        
        return {
            "status": "completed",
            "total_documents": vector_store.count(),
            "collection_name": vector_store.collection_name,
            "embedding_model": pipeline._embedding.model_name,
        }
        
    except Exception as e:
        logger.error("indexing_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
