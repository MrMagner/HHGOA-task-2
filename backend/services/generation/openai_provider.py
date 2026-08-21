"""OpenAI-compatible LLM provider.

Supports OpenAI's API and any compatible endpoint (Azure, local models, etc.).
"""

from __future__ import annotations

import time

from openai import AsyncOpenAI

from backend.services.generation.base import LLMProvider, LLMResponse
from backend.utils.logging import get_logger
from backend.utils.retry import with_retry

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        api_url: str = "https://api.openai.com/v1",
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._api_url = api_url
        self._timeout = timeout
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialize the OpenAI async client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._api_url,
                timeout=self._timeout,
            )
        return self._client

    @property
    def provider_name(self) -> str:
        return "openai"

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
        """Generate a response using OpenAI's API.

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
                provider="openai",
                model=self._model,
                duration_ms=round(duration_ms, 2),
                tokens=usage.get("total_tokens", 0),
            )

            return LLMResponse(
                text=text,
                model=self._model,
                usage=usage,
                duration_ms=round(duration_ms, 2),
                provider="openai",
                finish_reason=response.choices[0].finish_reason or "",
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("llm_error", provider="openai", error=str(e), duration_ms=round(duration_ms, 2))
            raise
