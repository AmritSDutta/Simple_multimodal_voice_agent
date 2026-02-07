"""Speech service provider implementations.

This package contains concrete implementations of the SpeechService interface
for various providers:
- sarvam_ai: SarvamAI STT/TTS service
"""

from src.flow_agent.speech.providers.sarvam_ai import SarvamAiSpeechService

__all__ = ["SarvamAiSpeechService"]
