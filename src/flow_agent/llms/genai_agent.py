from google import genai
from google.genai import types
from google.genai.chats import AsyncChat
from google.genai.client import AsyncClient

MODEL_DEFAULT = "gemini-2.5-flash"
SUMMARIZER_MODEL_DEFAULT = "gemini-2.5-flash-lite"

_GENAI_PROMPT = """
You are a helpful agent.
"""

_GENAI_SUMMARIZER_PROMPT = """
You are a helpful reasoning agent.
Try to provide appropriate response to the user query
"""

# Define safety settings for ALL categories
_safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_CIVIC_INTEGRITY",
        threshold="BLOCK_LOW_AND_ABOVE",
    ),
]

# ---------- Gemini chat singletons ----------
_global_client: AsyncClient | None = None


def _get_client() -> AsyncClient:
    global _global_client
    if _global_client is None:
        # Initialize the client once
        _global_client = genai.Client().aio
    return _global_client


async def get_summarizer_agent() -> AsyncChat:
    _llm_client: AsyncClient = _get_client()
    return _llm_client.chats.create(
        model=SUMMARIZER_MODEL_DEFAULT,
        config=types.GenerateContentConfig(
            system_instruction=_GENAI_SUMMARIZER_PROMPT,
            safety_settings=_safety_settings
        )
    )
