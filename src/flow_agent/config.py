from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # circuit breaker
    MAX_TRY: int = 3
    SLEEP_IN_SECONDS: int = 1

    # MODEL choices
    GEMINI_VISION_MODEL: str = 'gemma-3-27b-it'
    OPENAI_VISION_MODEL: str = 'gpt-5-nano'
    ZHIPU_VISION_MODEL: str = 'GLM-4.6V-Flash'
    OLLAMA_VISION_MODEL: str = 'qwen3-vl:235b-instruct-cloud'

    GEMINI_PROVIDER_IDENTIFIER: str = 'gemini'
    OPENAI_PROVIDER_IDENTIFIER: str = 'openai'
    ZHIPU_PROVIDER_IDENTIFIER: str = 'zhipu'
    OLLAMA_PROVIDER_IDENTIFIER: str = 'ollama'
    FALLBACK_PROVIDER_IDENTIFIER: str = 'openai'

    PROVIDER_DISTRIBUTION: dict = {
        GEMINI_PROVIDER_IDENTIFIER: 0.7,
        OPENAI_PROVIDER_IDENTIFIER: 0.01,
        ZHIPU_PROVIDER_IDENTIFIER: 0.1,
        OLLAMA_PROVIDER_IDENTIFIER: 0.2,
    }
    OLLAMA_BASE_URL: str = "https://ollama.com"
    ZHIPU_BASE_URL: str = "https://api.z.ai/api/paas/v4/"
    OLLAMA_KEY_STRING: str = "OLLAMA_API_KEY"
    ZHIPU_KEY_STRING: str = "ZAI_API_KEY"

    REASONING_NODE_PREFERENCE: str = 'langchain'
    MODERATION_API_CHECK_REQ: bool = True
    MODERATION_MODEL: str = 'omni-moderation-latest'  # OpenAI (omni-moderation-latest) -> text + image

    # Speech Service Configuration
    SPEECH_PROVIDER: str = "sarvam"  # Default provider
    SPEECH_PROVIDER_IDENTIFIER: str = 'sarvam'

    # SarvamAI Configuration
    SARVAM_STT_MODEL: str = "saaras:v3"
    SARVAM_TTS_MODEL: str = "bulbul:v3"
    SARVAM_LANGUAGE: str = "en-IN"
    SARVAM_SPEAKER: str = "shubh"
    SARVAM_TTS_PACE: float = 1.1
    SARVAM_TTS_SAMPLE_RATE: int = 22050

    GENAI_STT_MODEL: str = "gemini-3-flash-preview"
    GENAI_TTS_MODEL: str = "gemini-2.5-flash-preview-tts"
    GENAI_LANGUAGE: str = "en-IN"
    GENAI_SPEAKER: str = "Kore"
    GENAI_TTS_PACE: float = 1.1
    GENAI_TTS_SAMPLE_RATE: int = 24000

    OPENAI_STT_MODEL: str = "whisper-1"
    OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"
    OPENAI_LANGUAGE: str = "en-IN"
    OPENAI_SPEAKER: str = "coral"
    OPENAI_TTS_PACE: float = 1.1
    OPENAI_TTS_SAMPLE_RATE: int = 24000

    ENABLE_LANGSMITH_TRACING_V2: str = "false"
    LANGSMITH_PROJECT: str = 'multimodal_voice_agent'

    MAX_IMAGES_PER_REQUEST: int = 2
    PII_CONFIDENCE_THRESHOLD: float = 0.5
    IS_PII_REDACTION_ENABLED: bool = False

    # Summarization Settings
    SUMMARY_MESSAGE_THRESHOLD: int = 3  # Trigger after N messages
    SUMMARY_PROVIDER_PREFERENCE: str = 'langchain'  # 'langchain' or 'genai'


# Global settings instance
settings = Settings()
