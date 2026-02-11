import logging
from datetime import timedelta
from typing import List, Any

from aiobreaker import CircuitBreaker, CircuitBreakerError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from tenacity import stop_after_attempt, wait_exponential, retry_if_exception_type, retry
from tenacity.asyncio import retry_if_exception

from src.flow_agent.configurations.config import settings
from src.flow_agent.llms.LangChainChatLLM import get_chat_llm

breaker = CircuitBreaker(fail_max=5, timeout_duration=timedelta(seconds=30))


async def _should_retry(exc: BaseException) -> bool:
    # Retry everything EXCEPT when circuit is open
    return not isinstance(exc, CircuitBreakerError)


async def call_llm_safely(
        llm: BaseChatModel | Runnable,
        conversation: List[BaseMessage],
        new_message: HumanMessage
) -> Any:
    """
    A trivial safe llm calls , uses retry and circuit breaker.
    """
    response = None
    full_context = conversation + [new_message]
    try:
        response = await _call_llm(llm, full_context)
        logging.info(f"response metadata: {response.response_metadata}")
        logging.info(f"usage metadata: {response.usage_metadata}")
        return response
    except Exception as e:
        logging.error(f"Trying alternative, attempts exhausted / circuit open: {e}")
        try:
            llm = await get_chat_llm(settings.FALLBACK_PROVIDER_IDENTIFIER)
            response = await llm.ainvoke(full_context)
            logging.info(f"response metadata: {response.response_metadata}")
            logging.info(f"usage metadata: {response.usage_metadata}")
            return response
        except Exception as ae:
            logging.error(f"trying alternative failed too: {ae}")
            raise ae


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)
@breaker
async def _call_llm(
        llm: BaseChatModel | Runnable,
        conversation: List[BaseMessage]
) -> Any:
    """
    trivial circuit breaker  enabled with retry and exponential backoff.
    Receives full conversation history for multi-turn context.
    """
    return await llm.ainvoke(conversation)
