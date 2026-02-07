import logging
import os
import wave

from google import genai
from google.genai import types, Client

from src.flow_agent.speech import SpeechService


class GenAiSpeechService(SpeechService):
    """
    Gemini AI implementation of SpeechService interface.
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
        """Initialize GenAi speech service.

        Args:
            api_key: GenAi API subscription key (from env if None)
            stt_model: STT model name (default: "saaras:v3")
            tts_model: TTS model name (default: "bulbul:v3")
            language: Language code (default: "en-IN")
            speaker: Speaker ID for TTS (default: "shubh")
            tts_pace: Speech pace/speed (default: 1.1)
            tts_sample_rate: Audio sample rate (default: 22050)
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
    def client(self) -> Client | None:
        """Get or create GenAi client instance."""
        if self._client is None:
            self._client = genai.Client()
        return self._client

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "sarvam"

    async def speech_to_text(self, audio_bytes: bytes, file_extension: str = ".webm") -> str | None:
        with open('path/to/small-sample.mp3', 'rb') as f:
            audio_bytes = f.read()

        client = genai.Client()
        response = client.models.generate_content(
            model=self._stt_model,
            contents=[
                'Describe this audio clip',
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type='audio/mp3',
                )
            ]
        )
        logging.info(f'generated response: {response.text}')

        print(response.text)

    async def text_to_speech(self, text: str) -> list | None:

        client = genai.Client()

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name='Kore',
                        )
                    )
                ),
            )
        )

        data = response.candidates[0].content.parts[0].inline_data.data

        file_name = 'out.wav'
        self._wave_file(file_name, data)

        return data

    def _wave_file(self, filename, pcm, channels=1, rate=24000, sample_width=2):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
