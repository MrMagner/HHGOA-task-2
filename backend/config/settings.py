"""Centralized Pydantic Settings for the Voice-RAG system.

All configuration is driven by environment variables, with sensible defaults.
Uses pydantic-settings to load from .env files and environment.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class STTProvider(str, Enum):
    """Supported speech-to-text providers."""
    SARVAM = "sarvam"
    ELEVENLABS = "elevenlabs"
    DEMO = "demo"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    OPENAI = "openai"
    DEMO = "demo"


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""
    FIXED = "fixed"
    SENTENCE = "sentence"
    METADATA_AWARE = "metadata_aware"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Voice-RAG"
    app_version: str = "1.0.0"
    debug: bool = False
    demo_mode: bool = False
    log_level: str = "INFO"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # --- STT ---
    stt_provider: STTProvider = STTProvider.SARVAM
    sarvam_api_key: str = ""
    sarvam_api_url: str = "https://api.sarvam.ai/speech-to-text-translate"
    sarvam_language: str = "hi-IN"
    elevenlabs_api_key: str = ""
    elevenlabs_api_url: str = "https://api.elevenlabs.io/v1/speech-to-text"

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.GROQ
    groq_api_key: str = ""
    groq_model: str = "groq/compound"
    groq_api_url: str = "https://api.groq.com"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_api_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: int = 30

    # --- Embeddings & Vector Store ---
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    embedding_device: str = "cpu"

    # --- Qdrant ---
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "msmarco_xi"
    qdrant_local_path: str = "./qdrant_data"

    # --- Retrieval ---
    retrieval_top_k: int = 10
    retrieval_dense_weight: float = 0.6
    retrieval_bm25_weight: float = 0.4
    context_top_k: int = 5
    min_relevance_score: float = 0.3

    # --- Reranking ---
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5

    # --- Chunking ---
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_chunks_per_doc: int = 20

    # --- Guardrails ---
    grounding_threshold: float = 0.5
    safety_enabled: bool = False
    offtopic_threshold: float = 0.0

    # --- Audio ---
    max_audio_size_mb: int = 25
    allowed_audio_formats: str = "wav,mp3,webm,ogg,m4a,flac"

    # --- Caching ---
    enable_query_cache: bool = True
    query_cache_ttl: int = 3600
    enable_embedding_cache: bool = True

    # --- Rate Limiting ---
    rate_limit_rpm: int = 60

    # --- Dataset ---
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_split: str = "train"
    dataset_language: str = "en"
    dataset_max_samples: int = 10000
    data_dir: str = "./data"

    # --- Benchmark ---
    benchmark_queries: int = 50

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins string into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_audio_formats_list(self) -> list[str]:
        """Parse allowed audio formats into a list."""
        return [f.strip() for f in self.allowed_audio_formats.split(",") if f.strip()]

    @property
    def data_path(self) -> Path:
        """Get the data directory as a Path object."""
        return Path(self.data_dir)

    @property
    def qdrant_storage_path(self) -> Path:
        """Get the Qdrant local storage path."""
        return Path(self.qdrant_local_path)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    def is_stt_available(self) -> bool:
        """Check if STT provider has required credentials."""
        if self.stt_provider == STTProvider.DEMO:
            return True
        if self.stt_provider == STTProvider.SARVAM:
            return bool(self.sarvam_api_key)
        if self.stt_provider == STTProvider.ELEVENLABS:
            return bool(self.elevenlabs_api_key)
        return False

    def is_llm_available(self) -> bool:
        """Check if LLM provider has required credentials."""
        if self.llm_provider == LLMProvider.DEMO:
            return True
        if self.llm_provider == LLMProvider.GROQ:
            return bool(self.groq_api_key)
        if self.llm_provider == LLMProvider.OPENAI:
            return bool(self.openai_api_key)
        return False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
