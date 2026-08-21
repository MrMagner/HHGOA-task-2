"""Sarvam AI STT provider.

Sarvam AI provides multilingual speech-to-text with strong support
for Indian languages (Hindi, Tamil, Bengali, etc.).
"""

from __future__ import annotations

import time

import httpx

from backend.services.stt.base import STTProvider, STTResult, STTError
from backend.utils.logging import get_logger
from backend.utils.retry import with_retry

logger = get_logger(__name__)


class SarvamSTT(STTProvider):
    """Sarvam AI speech-to-text provider."""

    def __init__(self, api_key: str, api_url: str, language: str = "hi-IN") -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._language = language

    @property
    def provider_name(self) -> str:
        return "sarvam"

    def is_available(self) -> bool:
        return bool(self._api_key)

    @with_retry(max_attempts=2, min_wait=0.5, max_wait=5.0, retry_on=(httpx.HTTPError,))
    async def transcribe(self, audio_data: bytes, content_type: str = "audio/wav") -> STTResult:
        """Transcribe audio using Sarvam AI API.

        Args:
            audio_data: Raw audio bytes.
            content_type: MIME type of the audio data.

        Returns:
            STTResult with transcribed text.

        Raises:
            STTError: If API call fails.
        """
        start = time.perf_counter()

        # Determine file extension from content type
        ext_map = {
            "audio/wav": "audio.wav",
            "audio/mpeg": "audio.mp3",
            "audio/mp3": "audio.mp3",
            "audio/webm": "audio.webm",
            "audio/ogg": "audio.ogg",
            "audio/m4a": "audio.m4a",
            "audio/flac": "audio.flac",
        }
        filename = ext_map.get(content_type, "audio.wav")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self._api_url,
                    headers={"api-subscription-key": self._api_key},
                    files={"file": (filename, audio_data, content_type)},
                    data={
                        "language_code": self._language,
                        "model": "saaras:v2.5",
                        "with_timestamps": "false",
                    },
                )

            duration_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                raise STTError(
                    f"Sarvam API returned {response.status_code}: {response.text}",
                    provider="sarvam",
                    status_code=response.status_code,
                )

            data = response.json()
            transcript = data.get("transcript", "")

            if not transcript:
                raise STTError("Empty transcript from Sarvam API", provider="sarvam")

            logger.info(
                "stt_transcription_complete",
                provider="sarvam",
                language=self._language,
                duration_ms=round(duration_ms, 2),
                text_length=len(transcript),
            )

            return STTResult(
                text=transcript,
                language=self._language,
                confidence=data.get("confidence", 0.0),
                duration_ms=round(duration_ms, 2),
                provider="sarvam",
            )

        except httpx.HTTPError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("stt_error", provider="sarvam", error=str(e), duration_ms=round(duration_ms, 2))
            raise STTError(f"Sarvam API connection error: {e}", provider="sarvam") from e
