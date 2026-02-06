import logging

from google.genai.chats import AsyncChat
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages, get_buffer_string
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph_api.schema import Context

from src.flow_agent.llms.genai_agent import get_summarizer_agent
from src.flow_agent.utils.state import State


async def entry_node(state: State):
    if state.get("ended_once"):
        # Mark as closed
        return {"ended_once": True, "messages": AIMessage('Use another thread for run. It is already ended')}
    return state


async def should_continue(state: State):
    """Conditional edge: check if closed"""
    if state.get("ended_once"):
        logging.info("Thread already closed, skipping execution")
        return END

    return "reasoning"  # Normal flow


async def call_summarizer_model(state: State, runtime: Runtime[Context]) -> State:
    user_message: list[BaseMessage] = state.get("messages")
    ctm = convert_to_messages(user_message)
    gbt = get_buffer_string(ctm, human_prefix="", ai_prefix="").strip()
    logging.info(gbt)
    if not user_message:
        logging.info(user_message)
        return {
            "retry_count": state["retry_count"],
            "messages": state["messages"]
        }

    agent: AsyncChat = await get_summarizer_agent()
    logging.info(f'user requirement: {gbt[:100]}')
    response = await agent.send_message(gbt)
    summary: str | None = 'not available'
    genai_res: AIMessage | None = AIMessage('did nto get it, please re ask.')
    if response and response.text:
        logging.info(f'Agent summarization response: {response.text[:100]}')
        logging.info(f'Agent token usage: {response.usage_metadata.total_token_count}')
        genai_res = AIMessage(f'Issue summary: {response.text}')
        summary = response.text

    return {
        "retry_count": 0,
        "issue": summary,
        "messages": genai_res,
        "ended_once": False,
        "final_report": genai_res
    }
