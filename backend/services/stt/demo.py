"""Demo/mock STT provider for testing without API keys."""

from __future__ import annotations

import time

from backend.services.stt.base import STTProvider, STTResult
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Sample queries for demo mode
DEMO_QUERIES = [
    "What is information retrieval?",
    "How does passage ranking work in search engines?",
    "Explain the concept of relevance in document retrieval",
    "What are the key challenges in multilingual search?",
    "How do neural models improve search quality?",
]


class DemoSTT(STTProvider):
    """Mock STT provider that returns predefined demo queries.

    Cycles through a list of sample queries to simulate
    voice input without requiring an actual STT API key.
    """

    def __init__(self) -> None:
        self._query_index = 0

    @property
    def provider_name(self) -> str:
        return "demo"

    def is_available(self) -> bool:
        return True

    async def transcribe(self, audio_data: bytes, content_type: str = "audio/wav") -> STTResult:
        """Return a predefined demo query.

        Args:
            audio_data: Ignored in demo mode.
            content_type: Ignored in demo mode.

        Returns:
            STTResult with a demo query text.
        """
        start = time.perf_counter()

        # Cycle through demo queries
        query = DEMO_QUERIES[self._query_index % len(DEMO_QUERIES)]
        self._query_index += 1

        duration_ms = (time.perf_counter() - start) * 1000

        logger.info("stt_demo_transcription", query=query, index=self._query_index)

        return STTResult(
            text=query,
            language="en",
            confidence=1.0,
            duration_ms=round(duration_ms, 2),
            provider="demo",
        )
