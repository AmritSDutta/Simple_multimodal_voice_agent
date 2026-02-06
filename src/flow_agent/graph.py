from __future__ import annotations

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from src.flow_agent.logging_config import setup_logging
from src.flow_agent.utils.nodes import entry_node, should_continue, call_gemini_reasoning_model
from src.flow_agent.utils.state import State

setup_logging()


class Context(TypedDict):
    my_configurable_param: str


# this name is mentioned in langgraph.json
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("entry", entry_node)
    .add_node("reasoning", call_gemini_reasoning_model)
    .add_edge(START, "entry")
    .add_conditional_edges('entry', should_continue, {"reasoning": "reasoning", END: END})
    .add_edge("reasoning", END)
)
