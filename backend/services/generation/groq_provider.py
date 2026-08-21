"""Groq LLM provider.

Groq provides extremely fast inference for open-source models
via their LPU (Language Processing Unit) hardware.
Primary LLM provider due to speed and free tier availability.
"""

from __future__ import annotations

import time

from groq import AsyncGroq

from backend.services.generation.base import LLMProvider, LLMResponse
from backend.utils.logging import get_logger
from backend.utils.retry import with_retry

logger = get_logger(__name__)


class GroqProvider(LLMProvider):
    """Groq LLM provider for fast inference."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        api_url: str = "https://api.groq.com/openai/v1",
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._api_url = api_url
        self._timeout = timeout
        self._client: AsyncGroq | None = None

    def _get_client(self) -> AsyncGroq:
        """Lazy-initialize the Groq async client."""
        if self._client is None:
            self._client = AsyncGroq(
                api_key=self._api_key,
                base_url=self._api_url,
                timeout=self._timeout,
            )
        return self._client

    @property
    def provider_name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        return bool(self._api_key)

    @with_retry(max_attempts=2, min_wait=1.0, max_wait=5.0, retry_on=(Exception,))
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response using Groq's API.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query with context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMResponse with generated text and metadata.
        """
        client = self._get_client()
        start = time.perf_counter()

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            duration_ms = (time.perf_counter() - start) * 1000

            text = response.choices[0].message.content or ""
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            logger.info(
                "llm_generation_complete",
                provider="groq",
                model=self._model,
                duration_ms=round(duration_ms, 2),
                tokens=usage.get("total_tokens", 0),
            )

            return LLMResponse(
                text=text,
                model=self._model,
                usage=usage,
                duration_ms=round(duration_ms, 2),
                provider="groq",
                finish_reason=response.choices[0].finish_reason or "",
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("llm_error", provider="groq", error=str(e), duration_ms=round(duration_ms, 2))
            raise
