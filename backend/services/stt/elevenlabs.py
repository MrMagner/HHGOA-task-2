"""ElevenLabs STT provider."""

from __future__ import annotations

import time

import httpx

from backend.services.stt.base import STTProvider, STTResult, STTError
from backend.utils.logging import get_logger
from backend.utils.retry import with_retry

logger = get_logger(__name__)


class ElevenLabsSTT(STTProvider):
    """ElevenLabs speech-to-text provider."""

    def __init__(self, api_key: str, api_url: str) -> None:
        self._api_key = api_key
        self._api_url = api_url

    @property
    def provider_name(self) -> str:
        return "elevenlabs"

    def is_available(self) -> bool:
        return bool(self._api_key)

    @with_retry(max_attempts=2, min_wait=0.5, max_wait=5.0, retry_on=(httpx.HTTPError,))
    async def transcribe(self, audio_data: bytes, content_type: str = "audio/wav") -> STTResult:
        """Transcribe audio using ElevenLabs API.

        Args:
            audio_data: Raw audio bytes.
            content_type: MIME type of the audio data.

        Returns:
            STTResult with transcribed text.

        Raises:
            STTError: If API call fails.
        """
        start = time.perf_counter()

        ext_map = {
            "audio/wav": "audio.wav",
            "audio/mpeg": "audio.mp3",
            "audio/mp3": "audio.mp3",
            "audio/webm": "audio.webm",
        }
        filename = ext_map.get(content_type, "audio.wav")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self._api_url,
                    headers={"xi-api-key": self._api_key},
                    files={"audio": (filename, audio_data, content_type)},
                )

            duration_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                raise STTError(
                    f"ElevenLabs API returned {response.status_code}: {response.text}",
                    provider="elevenlabs",
                    status_code=response.status_code,
                )

            data = response.json()
            transcript = data.get("text", "")

            if not transcript:
                raise STTError("Empty transcript from ElevenLabs API", provider="elevenlabs")

            logger.info(
                "stt_transcription_complete",
                provider="elevenlabs",
                duration_ms=round(duration_ms, 2),
                text_length=len(transcript),
            )

            return STTResult(
                text=transcript,
                language=data.get("language_code", "en"),
                confidence=data.get("confidence", 0.0),
                duration_ms=round(duration_ms, 2),
                provider="elevenlabs",
            )

        except httpx.HTTPError as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("stt_error", provider="elevenlabs", error=str(e), duration_ms=round(duration_ms, 2))
            raise STTError(f"ElevenLabs API connection error: {e}", provider="elevenlabs") from e
