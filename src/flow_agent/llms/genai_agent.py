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
You are a helpful summarizer agent.
whatever text passed to you please create a concrete summary where no specific 
events or events description, dates, numbers are missed.
Priorities are always assigned on issue not on user.
what can be summarized maximum  are emotions, greetings, lengthy descriptions.
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
def _create_client() -> AsyncClient:
    return genai.Client().aio


async def get_summarizer_agent() -> AsyncChat:
    _llm_client: AsyncClient = _create_client()
    return _llm_client.chats.create(
                    model=SUMMARIZER_MODEL_DEFAULT,
                    config=types.GenerateContentConfig(
                        system_instruction=_GENAI_SUMMARIZER_PROMPT,
                        safety_settings=_safety_settings
                    )
                )
