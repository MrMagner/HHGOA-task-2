"""Pydantic schemas for API requests and responses."""
from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum

class QuerySource(str, Enum):
    VOICE = "voice"
    TEXT = "text"

class TextQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=False)
    strategy: str = Field(default="hybrid")

class SourceDocument(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)

class LatencyBreakdown(BaseModel):
    stt_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    guardrails_ms: float = 0.0
    total_ms: float = 0.0

class RAGResponse(BaseModel):
    request_id: str
    query: str
    source: QuerySource
    transcript: str | None = None
    answer: str
    grounded: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sources: list[SourceDocument] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    refusal: bool = False
    refusal_reason: str | None = None
    is_demo: bool = False

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, bool] = Field(default_factory=dict)

class ConfigResponse(BaseModel):
    app_name: str
    version: str
    demo_mode: bool
    stt_provider: str
    llm_provider: str
    embedding_model: str
    chunking_strategy: str
    retrieval_strategy: str
    rerank_enabled: bool
    collection_name: str
    top_k: int

class ErrorResponse(BaseModel):
    error: str
    request_id: str | None = None
    detail: str | None = None
