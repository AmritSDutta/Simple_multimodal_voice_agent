from operator import add
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    retry_count: Annotated[int, add]
    messages: Annotated[list[BaseMessage], add_messages]
    issue: str
    final_report: str
    input_valid: bool
    conversation_summary: str  # Stores conversation summaries
