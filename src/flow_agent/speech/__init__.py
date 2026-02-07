"""Speech service package for STT/TTS functionality.

Provides a unified interface for multiple speech service providers:
- SarvamAI (default)
- OpenAI Whisper (future)
- Google GenAI (future)

Example:
    from src.flow_agent.speech import get_speech_service

    speech = await get_speech_service()
    transcript = await speech.speech_to_text(audio_bytes)
    audio_list = await speech.text_to_speech("Hello world")
"""

from src.flow_agent.speech.factory import get_speech_service
from src.flow_agent.speech.interface import SpeechService

__all__ = ["SpeechService", "get_speech_service"]
