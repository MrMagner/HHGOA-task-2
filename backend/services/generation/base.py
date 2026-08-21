"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM generation call."""
    text: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    provider: str = ""
    finish_reason: str = ""

@dataclass
class GenerationResult:
    """Generation result as expected by tests."""
    answer: str
    grounded: bool = False
    confidence: float = 0.0
    sources_used: list[str] = field(default_factory=list)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query with context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMResponse with the generated text and metadata.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of this LLM provider."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        ...
