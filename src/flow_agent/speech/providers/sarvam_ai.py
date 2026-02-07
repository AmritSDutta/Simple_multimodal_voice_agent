"""SarvamAI speech service implementation.

This module provides STT (Speech-to-Text) and TTS (Text-to-Speech)
functionality using the SarvamAI API.
"""

import json
import logging
import os
import tempfile

from sarvamai import JobStatusV1Response, SarvamAI
from sarvamai import TextToSpeechResponse

from src.flow_agent.speech.interface import SpeechService

logger = logging.getLogger(__name__)


class SarvamAiSpeechService(SpeechService):
    """SarvamAI implementation of SpeechService interface.

    Uses SarvamAI's job-based API for STT and direct API for TTS.
    Configuration is loaded from environment variables and settings.
    """

    def __init__(
        self,
        api_key: str | None = None,
        stt_model: str = "saaras:v3",
        tts_model: str = "bulbul:v3",
        language: str = "en-IN",
        speaker: str = "shubh",
        tts_pace: float = 1.1,
        tts_sample_rate: int = 22050,
    ):
        """Initialize SarvamAI speech service.

        Args:
            api_key: SarvamAI API subscription key (from env if None)
            stt_model: STT model name (default: "saaras:v3")
            tts_model: TTS model name (default: "bulbul:v3")
            language: Language code (default: "en-IN")
            speaker: Speaker ID for TTS (default: "shubh")
            tts_pace: Speech pace/speed (default: 1.1)
            tts_sample_rate: Audio sample rate (default: 22050)
        """
        self._api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError("SARVAM_API_KEY environment variable must be set")

        self._stt_model = stt_model
        self._tts_model = tts_model
        self._language = language
        self._speaker = speaker
        self._tts_pace = tts_pace
        self._tts_sample_rate = tts_sample_rate

        # Lazy initialization of client
        self._client: SarvamAI | None = None

    @property
    def client(self) -> SarvamAI:
        """Get or create SarvamAI client instance."""
        if self._client is None:
            self._client = SarvamAI(api_subscription_key=self._api_key)
        return self._client

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "sarvam"

    async def speech_to_text(
        self, audio_bytes: bytes, file_extension: str = ".webm"
    ) -> str | None:
        """Convert audio to text using SarvamAI STT job-based API.

        Args:
            audio_bytes: Raw audio data as bytes
            file_extension: File extension for audio format (default: ".webm")

        Returns:
            Transcribed text, or None on failure
        """
        temp_file_path = None
        output_dir = None

        try:
            logger.info("Starting SarvamAI STT transcription")

            # Create STT job
            job = self.client.speech_to_text_job.create_job(
                language_code=self._language,
                model=self._stt_model,
                with_timestamps=False,
                with_diarization=False,
            )

            # Save audio to temp file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            # Upload audio file
            job.upload_files(file_paths=[temp_file_path])

            # Start the job
            job.start()

            # Wait for completion
            logger.info("Waiting for STT job to complete...")
            final_status: JobStatusV1Response = job.wait_until_complete()

            if job.is_failed():
                logger.error(f"STT job failed: {final_status.error_message}")
                return None

            # Download and parse transcript
            output_dir = tempfile.mkdtemp()
            job.download_outputs(output_dir=output_dir)

            transcript = self._extract_transcript_from_output(output_dir)

            # Clean up temp files
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if output_dir:
                self._cleanup_directory(output_dir)

            if not transcript:
                logger.warning("No transcript found in STT output")
                return None

            logger.info(f"STT transcription successful: {transcript[:100]}...")
            return transcript

        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}", exc_info=True)

            # Clean up on error
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            if output_dir:
                self._cleanup_directory(output_dir)

            return None

    def _extract_transcript_from_output(self, output_dir: str) -> str:
        """Extract transcript text from downloaded output files.

        Args:
            output_dir: Directory containing downloaded output files

        Returns:
            Transcript text, or empty string if not found
        """
        transcript = ""

        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    with open(json_path) as f:
                        data = json.load(f)
                        # Extract transcript from JSON structure
                        if "transcript" in data:
                            transcript = data["transcript"]
                        elif "segments" in data:
                            # Combine segments
                            transcript = " ".join(
                                [seg.get("text", "") for seg in data["segments"]]
                            )
                    if transcript:
                        break
            if transcript:
                break

        return transcript

    def _cleanup_directory(self, dir_path: str):
        """Recursively clean up a temporary directory.

        Args:
            dir_path: Path to directory to clean up
        """
        try:
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for file in files:
                    os.unlink(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(dir_path)
        except Exception as e:
            logger.warning(f"Error cleaning up directory {dir_path}: {e}")

    async def text_to_speech(self, text: str) -> list | None:
        """Convert text to audio using SarvamAI TTS.

        Args:
            text: Text content to convert to speech

        Returns:
            List of audio data (base64 strings or URLs), or None on failure
        """
        try:
            logger.info(f"Starting SarvamAI TTS for text: {text[:100]}...")

            response: TextToSpeechResponse = self.client.text_to_speech.convert(
                model=self._tts_model,
                text=text,
                target_language_code=self._language,
                speaker=self._speaker,
                pace=self._tts_pace,
                speech_sample_rate=self._tts_sample_rate,
                enable_preprocessing=True,
                temperature=0.6,
            )

            logger.debug(f"TTS response type: {type(response)}")
            logger.debug(f"TTS audios type: {type(response.audios)}")
            logger.debug(
                f"TTS first audio length: {len(response.audios[0]) if response.audios else 0}"
            )

            logger.info("TTS conversion successful")
            return response.audios

        except Exception as e:
            logger.error(f"Error in text_to_speech: {e}", exc_info=True)
            return None
