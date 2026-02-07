import base64
import logging
from asyncio import sleep
from typing import List, Any, Sequence

from google.genai import types
from google.genai.chats import AsyncChat
from google.genai.types import GenerateContentResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph_api.schema import Context

from src.flow_agent.config import settings
from src.flow_agent.llms.LangChainChatLLM import get_chat_llm
from src.flow_agent.llms.genai_agent import get_summarizer_agent
from src.flow_agent.utils.state import State

"""
    # 1. Extract the specific HumanMessage containing the multimodal data
    General structure with image:
    [
        HumanMessage(content=[
            {
                'type': 'text',
                'text': 'what is there in the image'
            },
            {
                'type': 'image',
                'data': 'iVBORw0KGgoAAA...sir35/shEAAAAASUVORK5CYII=',
                'metadata': {'filename': 'Amrit Shankar Dutta - Feature Engineering.png'},
                'source_type': 'base64',
                'mime_type': 'image/png'
             }
         ],
        additional_kwargs={},
        response_metadata={},
        id='9ffb1324-0e89-4eaf-87b8-fb4cb90aacb9')
    ]

    Assuming the latest message is index -1
"""


async def entry_node(state: State):
    messages: list[BaseMessage] = state.get("messages")
    if messages:
        human_msg = messages[-1]
        content: str | list[str | dict] = human_msg.content
        if isinstance(content, str):
            # Handle plain string - convert to expected format
            content = [{"type": "text", "text": content}]
            logging.info(f'user req: {content}')

        if isinstance(content, list):
            for item in content:
                if hasattr(item, 'get') and item.get("type") == "text":
                    logging.info(f'user req: {item.get("text")}')
                elif hasattr(item, 'get') and item.get("type") != 'text':
                    logging.info(f'found {item.get("type")} media: {item.get("metadata")}')

    if state.get("ended_once"):
        # Mark as closed
        return {
            "ended_once": True,
            "messages": AIMessage("Use another thread for run. It is already ended"),
        }
    return state


async def should_continue(state: State):
    """Conditional edge: check if closed"""
    if state.get("ended_once"):
        logging.info("Thread already closed, skipping execution")
        return END

    return "reasoning"  # Normal flow


async def call_langchain_reasoning_model(
        state: State, runtime: Runtime[Context]
) -> State:
    messages = state.get("messages")
    human_msg = messages[-1]
    content = human_msg.content
    text_prompt = "Analyze this input."
    media_b64s = []  # List of base64 strings for images
    mime_type: str | None = None

    # Extract text/media from input
    if isinstance(content, list):
        for item in content:
            if hasattr(item, 'get') and item.get("type") == "text":
                text_prompt = item.get("text", "hi")
            elif hasattr(item, 'get') and item.get("type") in ["image", "audio", "video"]:
                data = item.get("data")
                if data:
                    media_b64s.append(data)
                    mime_type = item.get("mime_type")
                    logging.info(f"Media: {mime_type}")

    # Initialize ChatOpenAI with vision model
    llm: BaseChatModel | Runnable = await get_chat_llm()
    msg_content:  Sequence[dict | str] = await prepare_llm_input(text_prompt, media_b64s)
    '''
    message_content = [{"type": "text", "text": text_prompt}]
    # Add images to content array
    if media_b64s:
        for b64_image in media_b64s[:4]:  # Limit to 4 images
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                }
            )
    '''
    multimodal_msg = HumanMessage(content=msg_content)

    response = await call_llm_safely(llm, multimodal_msg)
    return process_response(state, response, text_prompt)


async def prepare_llm_input(text_prompt: str, media_b64s: list[Any] | None) -> list[dict[str, str | None | Any]]:
    """
    prepare multimodal or text based message for llm depending on the parameter
    """
    message_content: list[str | dict] = [{"type": "text", "text": text_prompt}]
    # Add images to content array
    if media_b64s:
        for b64_image in media_b64s[:4]:  # Limit to 4 images
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                }
            )
    return message_content


async def call_llm_safely(llm:  BaseChatModel | Runnable, multimodal_msg: HumanMessage) -> Any:
    """
    A trivial circuit breaker with exponential backoff
    """
    sleep_time = settings.SLEEP
    response = None
    for i in range(settings.MAX_TRY):
        try:
            response = await llm.ainvoke([multimodal_msg])
            logging.info(f"response details: {response.response_metadata}")
            return response
        except Exception as e:
            logging.warning(f"Attempt {i + 1} failed: {e}")
            await sleep(sleep_time)
            sleep_time *= 2
            if i == settings.MAX_TRY - 1:
                logging.error(f"Attempt exhausted: {e}, trying alternative")
                try:
                    llm = await get_chat_llm(settings.FALLBACK_PROVIDER_IDENTIFIER)
                    response = await llm.ainvoke([multimodal_msg])
                    return response
                except Exception as ae:
                    logging.error(f"trying alternative failed too: {ae}")
                    raise ae

    return response


def process_response(state: State, response: AIMessage, user_input: str = "") -> State:
    summary = response.content or "No response"
    new_msg = AIMessage(content=f"Issue summary: {summary}")
    return {
        "retry_count": 0,
        "issue": user_input,
        "messages": [new_msg],
        "ended_once": False,
        "final_report": str(summary),
    }


async def call_gemini_reasoning_model(state: State, runtime: Runtime[Context]) -> State:
    messages: list[BaseMessage] = state.get("messages")
    human_msg = messages[-1]
    content = human_msg.content

    # 2. Robust Extraction (handles lists of dicts safely)
    prompt = "Analyze this input."
    image_data = None
    mime_type = None

    if isinstance(content, list):
        for item in content:
            if hasattr(item, 'get') and item.get("type") == "text":
                prompt = item.get("text", "Analyze this input.")
            elif hasattr(item, 'get') and item.get("type") == "image":
                # Correctly mapping from LangGraph 'data' key
                image_data = item.get("data")
                mime_type = item.get("mime_type")
                logging.info(f"image content detected in prompt: {mime_type}")
            else:
                if isinstance(item, dict):
                    mime_type = item.get("mime_type")
                    logging.info(
                        f"other content detected in prompt: {item.get('type')},  {mime_type}"
                    )
    else:
        # Fallback for simple string content
        logging.info(f"Content: {content[:100]}")
        prompt = content

    # 3. Instruction Priming (Gemma 3 workaround for lack of System Role)
    full_prompt = f"INSTRUCTIONS: Response user query, use image data if available.\n\nUSER QUERY: {prompt}"

    # 4. Prepare the message parts
    message_parts: List[str | types.Part] = [full_prompt]

    if image_data and mime_type:
        image_part = types.Part.from_bytes(
            data=base64.b64decode(image_data), mime_type=mime_type
        )
        message_parts.append(image_part)

    # 5. Get Agent & Send
    agent: AsyncChat = await get_summarizer_agent()
    response: GenerateContentResponse = await agent.send_message(message=message_parts)

    # 6. Process Response
    summary = response.text if response and response.text else "No output generated."

    # Usage logging in 2026 SDK
    if response.usage_metadata:
        logging.info(f"Tokens: {response.usage_metadata.total_token_count}")

    genai_res = AIMessage(content=f"Issue summary: {summary}")

    return {
        "retry_count": 0,
        "issue": summary,
        "messages": [genai_res],
        "ended_once": False,
        "final_report": str(genai_res.content)
    }
