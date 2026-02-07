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
    SLEEP: int = 1

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
        GEMINI_PROVIDER_IDENTIFIER: 0.2,
        OPENAI_PROVIDER_IDENTIFIER: 0.1,
        ZHIPU_PROVIDER_IDENTIFIER: 0.2,
        OLLAMA_PROVIDER_IDENTIFIER: 0.5,
    }
    OLLAMA_BASE_URL: str = "https://ollama.com"
    ZHIPU_BASE_URL: str = "https://api.z.ai/api/paas/v4/"
    OLLAMA_KEY_STRING: str = "OLLAMA_API_KEY"
    ZHIPU_KEY_STRING: str = "ZAI_API_KEY"

    REASONING_NODE_PREFERENCE: str = 'langchain'

# Global settings instance
settings = Settings()
