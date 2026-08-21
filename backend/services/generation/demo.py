"""Demo/mock LLM provider for testing without API keys."""

from __future__ import annotations

import time

from backend.services.generation.base import LLMProvider, LLMResponse
from backend.utils.logging import get_logger

logger = get_logger(__name__)


DEMO_RESPONSES = {
    "default": (
        "Based on the provided context, I can offer the following information:\n\n"
        "The retrieved documents suggest that the topic involves information retrieval "
        "and search systems. Key points include:\n\n"
        "1. Modern search systems use a combination of lexical and semantic matching.\n"
        "2. Passage ranking helps identify the most relevant segments of text.\n"
        "3. Neural models have significantly improved search quality over traditional methods.\n\n"
        "Please note: This is a demo response. Configure an LLM provider (Groq or OpenAI) "
        "for real AI-generated answers."
    ),
}


class DemoLLM(LLMProvider):
    """Mock LLM provider that returns predefined responses.

    Useful for testing the full pipeline without an LLM API key.
    """

    @property
    def provider_name(self) -> str:
        return "demo"

    def is_available(self) -> bool:
        return True

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return a predefined demo response.

        Args:
            system_prompt: Ignored in demo mode.
            user_prompt: Ignored in demo mode.
            temperature: Ignored in demo mode.
            max_tokens: Ignored in demo mode.

        Returns:
            LLMResponse with demo text.
        """
        start = time.perf_counter()

        text = DEMO_RESPONSES["default"]
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info("llm_demo_generation", response_length=len(text))

        return LLMResponse(
            text=text,
            model="demo",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            duration_ms=round(duration_ms, 2),
            provider="demo",
            finish_reason="stop",
        )
