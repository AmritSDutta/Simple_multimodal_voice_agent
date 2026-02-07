"""
Open AI speech service implementation.
"""

import logging
import os
import tempfile

from openai import OpenAI

from src.flow_agent.speech.interface import SpeechService

logger = logging.getLogger(__name__)


class OpenAiSpeechService(SpeechService):
    """
    OpenAI implementation of SpeechService interface.

    Uses OpenAI's Whisper API for STT and TTS API for text-to-speech.
    Configuration is loaded from environment variables and settings.
    """

    def __init__(
        self,
        api_key: str | None = None,
        stt_model: str = "whisper-1",
        tts_model: str = "gpt-4o-mini-tts",
        language: str = "en-IN",
        speaker: str = "coral",
        tts_pace: float = 1.1,
        tts_sample_rate: int = 22050,
    ):
        """
        Initialize OpenAI speech service.
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")

        self._stt_model = stt_model
        self._tts_model = tts_model
        self._language = language
        self._speaker = speaker
        self._tts_pace = tts_pace
        self._tts_sample_rate = tts_sample_rate

        # Lazy initialization of client
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Get or create OpenAI client instance."""
        if self._client is None:
            self._client = OpenAI()
        return self._client

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "openai"

    async def speech_to_text(
        self, audio_bytes: bytes, file_extension: str = ".webm"
    ) -> str | None:
        """
        Convert audio to text using OpenAI Whisper API.
        """
        temp_file_path = None
        try:
            logger.info("Starting OpenAI Whisper STT transcription")

            # Save audio to temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            # Open file and transcribe
            with open(temp_file_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self._stt_model,
                    file=audio_file,
                    language=self._language[:2],  # Extract "en" from "en-IN"
                )

            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

            logger.info(f"STT transcription successful: {response.text[:100]}...")
            return response.text

        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}", exc_info=True)
            # Clean up temp file on error
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            return None

    async def text_to_speech(self, text: str) -> list | None:
        """
        Convert text to audio using OpenAI TTS API.
        """
        try:
            logger.info(f"Starting OpenAI TTS for text: {text[:100]}...")

            response = self.client.audio.speech.create(
                model=self._tts_model,
                voice=self._speaker,
                input=text,
            )

            # Convert bytes to base64 string (for consistency with SarvamAI)
            import base64

            audio_base64 = base64.b64encode(response.content).decode("utf-8")

            logger.info("TTS conversion successful")
            return [audio_base64]  # Return as list for interface consistency

        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}", exc_info=True)
            return None

