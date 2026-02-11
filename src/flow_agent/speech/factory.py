"""Factory function for creating speech service instances.

This module provides a centralized factory pattern for obtaining
SpeechService implementations based on configuration.
"""

import logging

from src.flow_agent.configurations.config import settings
from src.flow_agent.speech.interface import SpeechService
from src.flow_agent.speech.providers.gen_ai import GenAiSpeechService
from src.flow_agent.speech.providers.open_ai import OpenAiSpeechService
from src.flow_agent.speech.providers.sarvam_ai import SarvamAiSpeechService

logger = logging.getLogger(__name__)


# Registry of available providers
_PROVIDERS: dict[str, type[SpeechService]] = {
    "sarvam": SarvamAiSpeechService,
    "openai": OpenAiSpeechService,
    "gemini": GenAiSpeechService,
}


async def get_speech_service(
    provider: str | None = None, **kwargs
) -> SpeechService:
    """
    Factory function to get speech service implementation.

    Raises:
        ValueError: If provider is unknown or not supported

    Example:
        speech = await get_speech_service()  # Uses default provider
        speech = await get_speech_service("sarvam")  # Explicit provider
        speech = await get_speech_service("gemini")  # GenAI provider
        speech = await get_speech_service("sarvam", language="hi-IN")
    """
    # Use provider from settings if not specified
    if provider is None:
        provider = settings.SPEECH_PROVIDER
        logger.debug(f"Using default speech provider: {provider}")

    # Normalize provider name (handle aliases)
    provider_key = _normalize_provider_name(provider)

    # Look up provider class
    provider_class = _PROVIDERS.get(provider_key)

    if provider_class is None:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"Unknown speech provider: '{provider}'. "
            f"Available providers: {available}"
        )

    # Create and return instance
    logger.info(f"Creating speech service: {provider_key}")

    # For SarvamAI, use settings-based defaults
    if provider_key == "sarvam":
        return SarvamAiSpeechService(
            stt_model=kwargs.get("stt_model", settings.SARVAM_STT_MODEL),
            tts_model=kwargs.get("tts_model", settings.SARVAM_TTS_MODEL),
            language=kwargs.get("language", settings.SARVAM_LANGUAGE),
            speaker=kwargs.get("speaker", settings.SARVAM_SPEAKER),
            tts_pace=kwargs.get("tts_pace", settings.SARVAM_TTS_PACE),
            tts_sample_rate=kwargs.get(
                "tts_sample_rate", settings.SARVAM_TTS_SAMPLE_RATE
            ),
            api_key=kwargs.get("api_key"),
        )

    # For OpenAI, use settings-based defaults
    if provider_key == "openai":
        return OpenAiSpeechService(
            stt_model=kwargs.get("stt_model", settings.OPENAI_STT_MODEL),
            tts_model=kwargs.get("tts_model", settings.OPENAI_TTS_MODEL),
            language=kwargs.get("language", settings.OPENAI_LANGUAGE),
            speaker=kwargs.get("speaker", settings.OPENAI_SPEAKER),
            tts_pace=kwargs.get("tts_pace", settings.OPENAI_TTS_PACE),
            tts_sample_rate=kwargs.get(
                "tts_sample_rate", settings.OPENAI_TTS_SAMPLE_RATE
            ),
            api_key=kwargs.get("api_key"),
        )

    # For GenAI, use settings-based defaults
    if provider_key == "gemini":
        return GenAiSpeechService(
            stt_model=kwargs.get("stt_model", settings.GENAI_STT_MODEL),
            tts_model=kwargs.get("tts_model", settings.GENAI_TTS_MODEL),
            language=kwargs.get("language", settings.GENAI_LANGUAGE),
            speaker=kwargs.get("speaker", settings.GENAI_SPEAKER),
            tts_pace=kwargs.get("tts_pace", settings.GENAI_TTS_PACE),
            tts_sample_rate=kwargs.get(
                "tts_sample_rate", settings.GENAI_TTS_SAMPLE_RATE
            ),
            api_key=kwargs.get("api_key"),
        )

    # For other providers, pass through kwargs
    return provider_class(**kwargs)


def _normalize_provider_name(provider: str) -> str:
    """
    Normalize provider name to handle aliases and case variations.
    """
    provider_lower = provider.lower().strip()

    # Handle common aliases
    aliases = {
        "sarvamai": "sarvam",
        "sarvam_ai": "sarvam",
        "zai": "sarvam",  # If Zhipu is mapped to Sarvam
        "openai": "openai",
        "open_ai": "openai",
        "genai": "gemini",
        "gen_ai": "gemini",
        "google": "gemini",
        "gemini": "gemini",
    }
    response: str = aliases.get(provider_lower, provider_lower)
    logging.info(f'selected speech service provider: {response}')
    return response


def list_available_providers() -> list[str]:
    """
    Return list of available speech service providers.
    """
    return list(_PROVIDERS.keys())
