"""Voice-RAG Pipeline Orchestrator.

Ties together all the individual services (STT, Embeddings, Retrieval,
Generation, Guardrails) into a cohesive end-to-end execution pipeline.
"""

from __future__ import annotations

import time

from backend.api.schemas import (
    RAGResponse,
    LatencyBreakdown,
    QuerySource,
    SourceDocument,
)
from backend.services.stt.base import STTProvider
from backend.services.embeddings.base import EmbeddingProvider
from backend.services.retrieval.base import Retriever, RetrievalResult
from backend.services.reranking.base import Reranker
from backend.services.generation.base import LLMProvider
from backend.services.generation.prompts import build_rag_prompt, build_system_prompt
from backend.services.guardrails.safety import check_input_safety
from backend.services.guardrails.grounding import validate_grounding
from backend.services.guardrails.offtopic import OffTopicDetector
from backend.services.guardrails.injection import check_prompt_injection
from backend.services.generation.base import GenerationResult
from backend.utils.logging import get_logger
from backend.utils.timing import StageTimer

logger = get_logger(__name__)


class VoiceRAGPipeline:
    """Orchestrates the full Voice-RAG execution flow."""

    def __init__(
        self,
        stt_provider: STTProvider,
        embedding_provider: EmbeddingProvider,
        retriever: Retriever,
        llm_provider: LLMProvider,
        reranker: Reranker | None = None,
        offtopic_detector: OffTopicDetector | None = None,
    ) -> None:
        self._stt = stt_provider
        self._embedding = embedding_provider
        self._retriever = retriever
        self._llm = llm_provider
        self._reranker = reranker
        
        # Guardrails
        self._offtopic = offtopic_detector or OffTopicDetector()

    async def process_text_query(
        self,
        query: str,
        request_id: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> RAGResponse:
        """Process a text-based RAG query."""
        timer = StageTimer()
        start_time = time.perf_counter()
        
        logger.info("processing_text_query", request_id=request_id, query=query)

        try:
            # 1. Guardrails: Pre-processing checks
            with timer.stage("guardrails_pre"):
                is_safe, safety_warnings = check_input_safety(query)
                is_clean, injection_patterns = check_prompt_injection(query)
                is_on_topic, offtopic_score = self._offtopic.check(query)
                
            if not is_safe or not is_clean:
                return self._build_error_response(
                    request_id=request_id,
                    query=query,
                    source=QuerySource.TEXT,
                    error_msg="Query violates safety or security guidelines.",
                    timer=timer,
                    start_time=start_time,
                    refusal=True,
                )
                
            if not is_on_topic:
                return self._build_error_response(
                    request_id=request_id,
                    query=query,
                    source=QuerySource.TEXT,
                    error_msg="I can only answer questions related to the dataset domain.",
                    timer=timer,
                    start_time=start_time,
                    refusal=True,
                )

            # 2. Embedding
            with timer.stage("embedding"):
                query_embedding = self._embedding.embed_query(query)

            # 3. Retrieval
            with timer.stage("retrieval"):
                results = self._retriever.search(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k * 2 if (rerank and self._reranker) else top_k,
                )
                
            if not results:
                return self._build_error_response(
                    request_id=request_id,
                    query=query,
                    source=QuerySource.TEXT,
                    error_msg="No relevant documents found.",
                    timer=timer,
                    start_time=start_time,
                )

            # 4. Reranking (optional)
            if rerank and self._reranker:
                with timer.stage("reranking"):
                    results = self._reranker.rerank(query, results, top_k=top_k)
            else:
                results = results[:top_k]

            # 5. Generation
            with timer.stage("generation"):
                system_prompt = build_system_prompt()
                user_prompt = build_rag_prompt(query, results)
                
                llm_response = await self._llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

            # 6. Guardrails: Post-processing (Grounding)
            with timer.stage("guardrails_post"):
                gen_result = GenerationResult(answer=llm_response.text)
                is_grounded, grounding_score, _ = validate_grounding(gen_result, results)

            total_ms = (time.perf_counter() - start_time) * 1000
            summary = timer.summary()["stages"]
            
            latency = LatencyBreakdown(
                embedding_ms=summary.get("embedding", 0),
                retrieval_ms=summary.get("retrieval", 0),
                reranking_ms=summary.get("reranking", 0),
                generation_ms=summary.get("generation", 0),
                guardrails_ms=(summary.get("guardrails_pre", 0) + summary.get("guardrails_post", 0)),
                total_ms=round(total_ms, 2),
            )
            
            sources = [
                SourceDocument(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    text=r.text,
                    score=r.score,
                    metadata=r.metadata,
                ) for r in results
            ]
            
            return RAGResponse(
                request_id=request_id,
                query=query,
                source=QuerySource.TEXT,
                answer=llm_response.text,
                grounded=is_grounded,
                confidence=min(max(grounding_score, 0.0), 1.0),
                sources=sources,
                latency=latency,
                is_demo=(self._llm.provider_name == "demo"),
            )

        except Exception as e:
            logger.error("pipeline_error", request_id=request_id, error=str(e))
            raise

    async def process_voice_query(
        self,
        audio_data: bytes,
        content_type: str,
        request_id: str,
        language: str = "hi-IN",
        top_k: int = 5,
        rerank: bool = False,
    ) -> RAGResponse:
        """Process a voice-based RAG query."""
        timer = StageTimer()
        start_time = time.perf_counter()
        
        logger.info("processing_voice_query", request_id=request_id, size_bytes=len(audio_data))

        try:
            with timer.stage("stt"):
                stt_result = await self._stt.transcribe(audio_data, content_type)
                
            query = stt_result.text
            
            if not query:
                return self._build_error_response(
                    request_id=request_id,
                    query="",
                    source=QuerySource.VOICE,
                    error_msg="Could not transcribe audio.",
                    timer=timer,
                    start_time=start_time,
                )

            # Route to standard text processing
            text_response = await self.process_text_query(
                query=query,
                request_id=request_id,
                top_k=top_k,
                rerank=rerank,
            )
            
            # Adjust response for voice context
            text_response.source = QuerySource.VOICE
            text_response.transcript = query
            text_response.latency.stt_ms = summary = timer.summary()["stages"].get("stt", 0.0)
            text_response.latency.total_ms = round((time.perf_counter() - start_time) * 1000, 2)
            
            return text_response
            
        except Exception as e:
            logger.error("voice_pipeline_error", request_id=request_id, error=str(e))
            raise

    def _build_error_response(
        self,
        request_id: str,
        query: str,
        source: QuerySource,
        error_msg: str,
        timer: StageTimer,
        start_time: float,
        refusal: bool = False,
    ) -> RAGResponse:
        """Helper to build an error RAGResponse."""
        total_ms = (time.perf_counter() - start_time) * 1000
        summary = timer.summary()["stages"]
        
        latency = LatencyBreakdown(
            stt_ms=summary.get("stt", 0),
            embedding_ms=summary.get("embedding", 0),
            retrieval_ms=summary.get("retrieval", 0),
            reranking_ms=summary.get("reranking", 0),
            generation_ms=summary.get("generation", 0),
            guardrails_ms=(summary.get("guardrails_pre", 0) + summary.get("guardrails_post", 0)),
            total_ms=round(total_ms, 2),
        )
        
        return RAGResponse(
            request_id=request_id,
            query=query,
            source=source,
            answer=error_msg,
            grounded=False,
            confidence=0.0,
            sources=[],
            latency=latency,
            refusal=refusal,
            refusal_reason=error_msg if refusal else None,
        )
