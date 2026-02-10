import base64
import logging
from asyncio import sleep
from typing import List, Any, Sequence, Tuple

from google.genai import types
from google.genai.chats import AsyncChat
from google.genai.types import GenerateContentResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import Runnable
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from langgraph_api.schema import Context

from src.flow_agent.utils.pii_redaction import PII_Redactor
from src.flow_agent.utils.input_validation import scan_for_vulnerability
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
    pii_redactor = PII_Redactor()
    redacted_message = await pii_redactor.do_pii_redaction([messages[-1]])

    # Replace last message in state
    messages = list(messages)  # copy list
    messages[-1] = redacted_message[0]
    state["messages"] = messages

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
                    logging.info(f'found media of type: {item.get("type")}')

    state['conversation_summary'] = state.get('conversation_summary', '')
    return state


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
    msg_content: Sequence[dict | str] = await prepare_llm_input(text_prompt, media_b64s)
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

    # Pass full conversation history for multi-turn context
    response = await call_llm_safely(llm, messages, multimodal_msg)
    updated_state: State = await process_response(state, response, text_prompt)
    return updated_state


async def prepare_llm_input(text_prompt: str, media_b64s: list[Any] | None) -> list[dict[str, str | None | Any]]:
    """
    prepare multimodal or text based message for llm depending on the parameter
    """
    message_content: list[str | dict] = [{"type": "text", "text": text_prompt}]
    if media_b64s and len(media_b64s) > settings.MAX_IMAGES_PER_REQUEST:
        # Limit to 4 images
        logging.info(f"more images: {len(media_b64s)} "
                     f"were passed for analysis than supported ({settings.MAX_IMAGES_PER_REQUEST}), will be ignored")
    # Add images to content array
    if media_b64s:
        for b64_image in media_b64s[:settings.MAX_IMAGES_PER_REQUEST]:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                }
            )
    return message_content


async def call_llm_safely(
        llm: BaseChatModel | Runnable,
        conversation: List[BaseMessage],
        new_message: HumanMessage
) -> Any:
    """
    A trivial circuit breaker with exponential backoff.
    Receives full conversation history for multi-turn context.
    """
    sleep_time = settings.SLEEP_IN_SECONDS
    response = None
    full_context = conversation + [new_message]
    for i in range(settings.MAX_TRY):
        try:
            response = await llm.ainvoke(full_context)
            logging.info(f"response metadata: {response.response_metadata}")
            logging.info(f"usage metadata: {response.usage_metadata}")
            return response
        except Exception as e:
            logging.warning(f"Attempt {i + 1} failed: {e}")
            await sleep(sleep_time)
            sleep_time *= 2
            if i == settings.MAX_TRY - 1:
                logging.error(f"Attempt exhausted: {e}, trying alternative")
                try:
                    llm = await get_chat_llm(settings.FALLBACK_PROVIDER_IDENTIFIER)
                    response = await llm.ainvoke(full_context)
                    return response
                except Exception as ae:
                    logging.error(f"trying alternative failed too: {ae}")
                    raise ae

    return response


async def process_response(state: State, response: AIMessage, user_input: str = "") -> State:
    summary = response.content or "No response"
    genai_res = AIMessage(content=f"Issue summary: {summary}")

    redactor = PII_Redactor(confidence_threshold=0.5)
    final_report: List[BaseMessage] = await redactor.do_pii_redaction([genai_res])
    agent_response: str = final_report[0].content
    return {
        "input_valid": True,
        "retry_count": 0,
        "issue": user_input,
        "messages": final_report,  # add_messages reducer will append this
        "final_report": agent_response,
        'conversation_summary': state['conversation_summary']
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

    redactor = PII_Redactor(confidence_threshold=0.5)
    final_report = await redactor.do_pii_redaction([genai_res])
    agent_response = final_report[0].content

    return {
        "input_valid": True,
        "retry_count": 0,
        "issue": state.get("issue", ""),
        "messages": final_report,  # add_messages reducer will append this
        "final_report": str(agent_response),
        'conversation_summary': state['conversation_summary']
    }


async def call_input_validation(state: State, runtime: Runtime[Context]) -> dict:
    user_message: list[HumanMessage] = [msg for msg in state.get("messages") if isinstance(msg, HumanMessage)]

    is_safe: bool = await scan_for_vulnerability(user_message[-1])
    if is_safe:
        return {
            # "messages": AIMessage(f"Validated user prompt..."),
            "input_valid": True  # Flag for routing
        }
    else:
        logging.warning('Input validation failed - malicious content detected')
        return {
            "messages": AIMessage("Unsafe user prompt detected..."),
            "input_valid": False  # Flag for routing
        }


async def route_after_validation(state: State) -> str:
    return "reasoning" if state["input_valid"] else END


async def call_langchain_summarizer(state: State, runtime: Runtime[Context]) -> State:
    """Summarize AI messages using LangChain interface with vision model cohort."""
    messages = state.get("messages", [])

    # Summarize all but the last 2 messages
    messages_to_summarize = messages[:-2] if len(messages) > 2 else []

    text_messages, media_messages = await _get_retained_messages(messages_to_summarize)
    recent_messages = messages[-2:] if len(messages) > 2 else messages

    # Combine media messages with recent messages, then limit to MAX_IMAGES_PER_REQUEST
    combined_messages = media_messages + recent_messages

    # Filter to only image messages and keep only the most recent MAX_IMAGES_PER_REQUEST
    image_messages_to_keep = _filter_recent_messages_by_type(
        combined_messages,
        message_type="image",
        max_count=settings.MAX_IMAGES_PER_REQUEST
    )

    # Keep all non-image messages from combined_messages + recent images
    retained_messages = [
        msg for msg in combined_messages
        if not _has_media_type(msg, "image")
    ]
    retained_messages.extend(image_messages_to_keep)

    # text_messages already contains messages_to_summarize from _get_retained_messages()

    if text_messages:
        conversation_text = "\n".join([
            f"Message: {msg.content}"
            for msg in text_messages
        ])

        summary_prompt = f"""
        Summarize the following conversation into a concise english context (max 500 words).
        Focus on key information provided and important outcomes.
        If you think it is fact retain it as is.

        AI Responses:
        {conversation_text}
        """

        llm = await get_chat_llm(is_summarizer=True)  # Uses weighted provider selection
        response = await call_llm_safely(llm, [], HumanMessage(content=summary_prompt))
        summary = response.content if hasattr(response, 'content') else str(response)

        # Create summary message
        summary_message = AIMessage(content=f"[Previous conversation Summary] {summary}")
    else:
        summary_message = None
        summary = ""

    # Reconstruct: summary + retained messages (with limited images)
    new_messages = []
    if summary_message:
        new_messages.append(summary_message)
    new_messages.extend(retained_messages)

    logging.info(f"Summarization complete: Kept {len(image_messages_to_keep)}/{len(_extract_all_messages_by_type(combined_messages, 'image'))} images (max={settings.MAX_IMAGES_PER_REQUEST})")

    # Use Overwrite to replace the entire messages list
    return {
        "messages": Overwrite(new_messages),
        "conversation_summary": summary,
    }


async def should_summarize(state: State) -> str:
    """Check if conversation has reached message threshold."""
    message_count = len(state.get("messages", []))
    threshold = settings.SUMMARY_MESSAGE_THRESHOLD

    if message_count >= threshold:
        logging.info(f"Message count ({message_count}) >= threshold ({threshold}), triggering summarization")
        return "summarizer"

    return END  # Go directly to END, not entry


async def _get_retained_messages(messages: list[BaseMessage] | None = None) -> Tuple[
    list[BaseMessage], list[BaseMessage]]:
    """
    Separate messages into text-only and media-containing (image/audio/video) messages.

    Args:
        messages: List of BaseMessage objects to process

    Returns:
        Tuple of (text_messages, media_messages)
        - text_messages: Messages containing only text content
        - media_messages: Messages containing image, audio, or video content
    """
    if messages is None or len(messages) == 0:
        return [], []

    media_messages: List[BaseMessage] = []
    text_messages: List[BaseMessage] = []

    working_copy = messages.copy()
    for message in working_copy:
        content = message.content
        if isinstance(content, str):
            text_messages.append(message)

        if isinstance(content, list):
            for item in content:
                new_message = HumanMessage(content=[item]) \
                    if isinstance(message, HumanMessage) else AIMessage(content=[item])

                if hasattr(item, 'get') and item.get("type") == "text":
                    text_messages.append(new_message)
                elif hasattr(item, 'get') and item.get("type") in ["image", "audio", "video"]:
                    media_messages.append(new_message)

    return text_messages, media_messages


def _has_media_type(message: BaseMessage, media_type: str) -> bool:
    """
    Check if a message contains a specific media type.

    Args:
        message: BaseMessage to check
        media_type: Type to check for ("image", "audio", "video")

    Returns:
        True if message contains the specified media type
    """
    content = message.content
    if isinstance(content, list):
        return any(
            item.get("type") == media_type
            for item in content
            if hasattr(item, 'get')
        )
    return False


def _extract_all_messages_by_type(messages: list[BaseMessage], media_type: str) -> list[BaseMessage]:
    """
    Extract all messages containing a specific media type.

    Args:
        messages: List of BaseMessage objects
        media_type: Type to extract ("image", "audio", "video")

    Returns:
        List of messages containing the specified media type
    """
    return [msg for msg in messages if _has_media_type(msg, media_type)]


def _filter_recent_messages_by_type(
    messages: list[BaseMessage],
    message_type: str,
    max_count: int
) -> list[BaseMessage]:
    """
    Filter messages to keep only the most recent N messages of a specific type.

    Args:
        messages: List of BaseMessage objects (in chronological order)
        message_type: Type to filter for ("image", "audio", "video")
        max_count: Maximum number of messages to keep

    Returns:
        List of the most recent max_count messages containing the specified type
    """
    # Extract all messages of the specified type
    typed_messages = _extract_all_messages_by_type(messages, message_type)

    # Keep only the most recent max_count messages (last ones in the list)
    recent_messages = typed_messages[-max_count:] if len(typed_messages) > max_count else typed_messages

    return recent_messages
