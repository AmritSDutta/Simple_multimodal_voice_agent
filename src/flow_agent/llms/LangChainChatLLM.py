import logging
import os
import random
from typing import Any

from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from sarvam import SarvamChat

from src.flow_agent.config import settings

random.seed(1234)


def _get_random_provider():
    """
    Random distribution based selection.
    """
    names = list(settings.VISION_PROVIDER_DISTRIBUTION.keys())
    weights = list(settings.VISION_PROVIDER_DISTRIBUTION.values())

    choice = random.choices(names, weights=weights, k=1)[0]
    logging.info(f"Using {settings.VISION_PROVIDER_DISTRIBUTION[choice]} -  weights")
    return choice


def _get_random_summarizing_provider():
    """
    Random distribution based selection.
    """
    names = list(settings.SUMMARIZATION_PROVIDER_DISTRIBUTION.keys())
    weights = list(settings.SUMMARIZATION_PROVIDER_DISTRIBUTION.values())

    choice = random.choices(names, weights=weights, k=1)[0]
    logging.info(f"Using {settings.SUMMARIZATION_PROVIDER_DISTRIBUTION[choice]} -  weights")
    return choice


async def get_chat_llm(provider: str | None = None, is_summarizer: bool = False) -> BaseChatModel | Runnable:
    """
    A trivial LLM routing technique, random distribution with prefixed distribution.
    """
    if is_summarizer:
        llm, provider = await get_summarization_models(provider)
    else:
        llm, provider = await get_vision_models(provider)
    logging.info(f"Using {provider}, {llm.model_config} , is summarization model? {is_summarizer}")

    return llm


async def get_vision_models(provider: str | None) -> tuple[BaseChatModel | None, Any]:
    if provider is None:
        provider = _get_random_provider()

    logging.info(f"Using {provider}")
    llm: BaseChatModel | None = None
    if provider.lower() == settings.OPENAI_PROVIDER_IDENTIFIER:
        llm = ChatOpenAI(model=settings.OPENAI_VISION_MODEL, temperature=0)
    elif provider.lower() == settings.GEMINI_PROVIDER_IDENTIFIER:
        llm = ChatGoogleGenerativeAI(model=settings.GEMINI_VISION_MODEL, temperature=0)
    elif provider.lower() == settings.ZHIPU_PROVIDER_IDENTIFIER:

        llm = ChatOpenAI(  # type: ignore
            temperature=0.6,
            model=settings.ZHIPU_VISION_MODEL,
            openai_api_key=os.getenv(settings.ZHIPU_KEY_STRING),  # type: ignore[arg-type]
            openai_api_base=settings.ZHIPU_BASE_URL  # type: ignore[arg-type]
        )
    else:
        api_key = os.getenv(settings.OLLAMA_KEY_STRING)
        if api_key is None:
            raise ValueError(f"{settings.OLLAMA_KEY_STRING} environment variable not set")

        llm = ChatOllama(
            model=settings.OLLAMA_VISION_MODEL,
            # reasoning=True,
            base_url=settings.OLLAMA_BASE_URL,
            client_kwargs={
                "headers": {"Authorization": "Bearer " + api_key},
                "timeout": 60.0,  # Timeout in seconds
            },
        )
    if provider not in ['gemini', 'zhipu']:
        return llm.bind_tools([DuckDuckGoSearchRun()]), provider
    return llm, provider


async def get_summarization_models(provider: str | None) -> tuple[BaseChatModel | None, Any]:
    if provider is None:
        provider = _get_random_summarizing_provider()

    logging.info(f"Using {provider}")
    llm: BaseChatModel | None = None
    if provider.lower() == settings.OPENAI_PROVIDER_IDENTIFIER:
        llm = ChatOpenAI(model=settings.OPENAI_VISION_MODEL, temperature=0)
    elif provider.lower() == settings.GEMINI_PROVIDER_IDENTIFIER:
        llm = ChatGoogleGenerativeAI(model=settings.GEMINI_VISION_MODEL, temperature=0)
    elif provider.lower() == settings.ZHIPU_PROVIDER_IDENTIFIER:

        llm = ChatOpenAI(  # type: ignore
            temperature=0.6,
            model=settings.ZHIPU_SUMMARIZATION_MODEL,
            openai_api_key=os.getenv(settings.ZHIPU_KEY_STRING),  # type: ignore[arg-type]
            openai_api_base=settings.ZHIPU_BASE_URL  # type: ignore[arg-type]
        )
    elif provider.lower() == settings.SARVAM_PROVIDER_IDENTIFIER:
        llm = SarvamChat(reasoning_effort='low')
    else:
        api_key = os.getenv(settings.OLLAMA_KEY_STRING)
        if api_key is None:
            raise ValueError(f"{settings.OLLAMA_KEY_STRING} environment variable not set")

        llm = ChatOllama(
            model=settings.OLLAMA_SUMMARIZATION_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            client_kwargs={
                "headers": {"Authorization": "Bearer " + api_key},
                "timeout": 60.0,  # Timeout in seconds
            },
        )
    if provider not in [settings.GEMINI_PROVIDER_IDENTIFIER, settings.ZHIPU_PROVIDER_IDENTIFIER,
                        settings.SARVAM_PROVIDER_IDENTIFIER]:
        return llm.bind_tools([DuckDuckGoSearchRun()]), provider
    return llm, provider
