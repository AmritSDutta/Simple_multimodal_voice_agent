import logging
import os

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.language_models import BaseChatModel
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


async def get_chat_llm(
        provider: str = 'ollama') -> BaseChatModel:
    llm: BaseChatModel | None = None
    if provider.lower() == 'openai':
        llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    if provider.lower() == 'gemini':
        llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", temperature=0)
    else:
        llm = ChatOllama(
            model='qwen3-vl:235b-instruct-cloud',
            # reasoning=True,
            base_url="https://ollama.com",
            client_kwargs={
                "headers": {"Authorization": "Bearer " + os.getenv("OLLAMA_API_KEY")},
                "timeout": 60.0  # Timeout in seconds
            }
        )
    logging.info(f"Using {provider} -  model")
    llm.bind_tools([DuckDuckGoSearchRun()])
    return llm



