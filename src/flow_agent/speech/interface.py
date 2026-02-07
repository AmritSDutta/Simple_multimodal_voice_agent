"""Abstract base class for speech services."""

from abc import ABC, abstractmethod


class SpeechService(ABC):
    """Abstract base class for speech services.

    Provides a common interface for speech-to-text (STT) and text-to-speech (TTS)
    functionality across different providers (SarvamAI, OpenAI Whisper, Google GenAI, etc.).
    """

    @abstractmethod
    async def speech_to_text(
        self, audio_bytes: bytes, file_extension: str = ".webm"
    ) -> str | None:
        """Convert audio to text.

        Args:
            audio_bytes: Raw audio data as bytes
            file_extension: File extension for the audio format (default: ".webm")

        Returns:
            Transcribed text as string, or None on failure

        Raises:
            Exception: Provider-specific exceptions (API errors, network issues, etc.)
        """
        pass

    @abstractmethod
    async def text_to_speech(self, text: str) -> list | None:
        """Convert text to audio.

        Args:
            text: Text content to convert to speech

        Returns:
            List of audio data (base64 strings or URLs), or None on failure

        Raises:
            Exception: Provider-specific exceptions (API errors, network issues, etc.)
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name.

        Returns:
            Provider identifier (e.g., 'sarvam', 'whisper', 'genai')
        """
        pass
