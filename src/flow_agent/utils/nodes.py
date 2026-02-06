import base64
import logging
from typing import List

from google.genai import types
from google.genai.chats import AsyncChat
from google.genai.types import GenerateContentResponse
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages, get_buffer_string
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph_api.schema import Context

from src.flow_agent.llms.genai_agent import get_summarizer_agent
from src.flow_agent.utils.state import State


async def entry_node(state: State):
    print(state.get("messages"))
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


async def call_gemini_reasoning_model(state: State, runtime: Runtime[Context]) -> State:
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
                'data': 'iVBORw0KGgoAAAANSUhEUgAAEsAAAAuQCAYAAACqz00AAAAA...sir35/shEAAAAASUVORK5CYII=',
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
    messages: list[BaseMessage] = state.get("messages")
    human_msg = messages[-1]
    content = human_msg.content

    # 2. Robust Extraction (handles lists of dicts safely)
    prompt = "Analyze this input."
    image_data = None
    mime_type = None

    if isinstance(content, list):
        for item in content:
            if item.get('type') == 'text':
                prompt = item.get('text')
            elif item.get('type') == 'image':
                # Correctly mapping from LangGraph 'data' key
                image_data = item.get('data')
                mime_type = item.get('mime_type')
                logging.info(f"image content detected in prompt: {mime_type}")
            else:
                mime_type = item.get('mime_type')
                logging.info(f"other content detected in prompt: {item.get('type')},  {mime_type}")
    else:
        # Fallback for simple string content
        logging.info(f"Content: {content[:100]}")
        prompt = content

    # 3. Instruction Priming (Gemma 3 workaround for lack of System Role)
    full_prompt = f"INSTRUCTIONS: Response user query, use image data if available.\n\nUSER QUERY: {prompt}"

    # 4. Prepare the message parts
    message_parts: List[types.Part] = [full_prompt]

    if image_data and mime_type:
        image_part = types.Part.from_bytes(
            data=base64.b64decode(image_data),
            mime_type=mime_type
        )
        message_parts.append(image_part)

    # 5. Get Agent & Send
    agent: AsyncChat = await get_summarizer_agent()
    response = await agent.send_message(message=message_parts)

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
        "final_report": genai_res
    }
