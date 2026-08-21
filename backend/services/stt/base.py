"""Abstract base class for STT providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class STTResult:
    """Result from speech-to-text transcription."""
    text: str
    language: str = ""
    confidence: float = 0.0
    duration_ms: float = 0.0
    provider: str = ""


class STTProvider(ABC):
    """Abstract base class for speech-to-text providers."""

    @abstractmethod
    async def transcribe(self, audio_data: bytes, content_type: str = "audio/wav") -> STTResult:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes.
            content_type: MIME type of the audio data.

        Returns:
            STTResult with transcribed text and metadata.

        Raises:
            STTError: If transcription fails.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of this STT provider."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        ...


class STTError(Exception):
    """Raised when speech-to-text transcription fails."""

    def __init__(self, message: str, provider: str = "", status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
