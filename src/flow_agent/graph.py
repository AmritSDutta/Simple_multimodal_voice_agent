from __future__ import annotations

import os

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from src.flow_agent.config import settings
from src.flow_agent.logging_config import setup_logging
from src.flow_agent.utils.nodes import (
    entry_node,
    call_gemini_reasoning_model,
    call_langchain_reasoning_model,
    call_input_validation, route_after_validation,

)
from src.flow_agent.utils.state import State

setup_logging()
os.environ["LANGSMITH_TRACING_V2"] = settings.ENABLE_LANGSMITH_TRACING_V2
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT


class Context(TypedDict):
    my_configurable_param: str


# this name is mentioned in langgraph.json
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("entry", entry_node)
    .add_node("input_validator", call_input_validation)
    .add_node("reasoning",
              call_langchain_reasoning_model if settings.REASONING_NODE_PREFERENCE == 'langchain'
              else call_gemini_reasoning_model
              )
    .add_edge(START, "entry")
    .add_edge("entry", "input_validator")
    .add_conditional_edges("input_validator", route_after_validation, {"reasoning": "reasoning", END: END})
    .add_edge("reasoning", END)
)
