import base64
import logging
import os

from google import genai
from google.genai import types, Client

from src.flow_agent.speech.interface import SpeechService

logger = logging.getLogger(__name__)


class GenAiSpeechService(SpeechService):
    """
    Google GenAI implementation of SpeechService interface.
    """

    def __init__(
        self,
        api_key: str | None = None,
        stt_model: str = "gemini-3-flash-preview",
        tts_model: str = "gemini-2.5-flash-preview-tts",
        language: str = "en-IN",
        speaker: str = "Kore",
        tts_pace: float = 1.1,
        tts_sample_rate: int = 24000,
    ):
        """
        Initialize GenAI speech service.
        """
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        self._stt_model = stt_model
        self._tts_model = tts_model
        self._language = language
        self._speaker = speaker
        self._tts_pace = tts_pace
        self._tts_sample_rate = tts_sample_rate

        # Lazy initialization of client
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        """Get or create GenAI client instance."""
        if self._client is None:
            self._client = genai.Client()
        return self._client

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "gemini"

    async def speech_to_text(
        self, audio_bytes: bytes, file_extension: str = ".webm"
    ) -> str | None:
        """
        Convert audio to text using Google GenAI STT.
        """
        try:
            logger.info("Starting Google GenAI STT transcription")

            # Determine MIME type from file extension
            mime_type = "audio/webm" if file_extension == ".webm" else "audio/mp3"

            # Use the audio_bytes parameter directly with GenAI client
            response = self.client.models.generate_content(
                model=self._stt_model,
                contents=[  # type: ignore[arg-type]
                    "Transcribe this audio clip",
                    types.Part.from_bytes(
                        data=audio_bytes,
                        mime_type=mime_type,
                    ),
                ],
            )

            if response.text is None:
                logger.error("STT response text is None")
                return None

            logger.info(f"STT transcription successful: {response.text[:100]}...")
            return response.text

        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}", exc_info=True)
            return None

    async def text_to_speech(self, text: str) -> list | None:
        """
        Convert text to audio using Google GenAI TTS.
        """
        try:
            logger.info(f"Starting Google GenAI TTS for text: {text[:100]}...")

            response = self.client.models.generate_content(
                model=self._tts_model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self._speaker,
                            )
                        )
                    ),
                ),
            )

            # Extract PCM data from response with proper type checking
            if (
                response.candidates is None
                or len(response.candidates) == 0
                or response.candidates[0].content is None
                or response.candidates[0].content.parts is None
                or len(response.candidates[0].content.parts) == 0
                or response.candidates[0].content.parts[0].inline_data is None
                or response.candidates[0].content.parts[0].inline_data.data is None
            ):
                logger.error("TTS response structure is invalid or missing data")
                return None

            data: bytes = response.candidates[0].content.parts[0].inline_data.data

            # Convert to base64 string (for consistency with SarvamAI/OpenAI)
            audio_base64 = base64.b64encode(data).decode("utf-8")

            logger.info("TTS conversion successful")
            return [audio_base64]  # Return as list for interface consistency

        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}", exc_info=True)
            return None
