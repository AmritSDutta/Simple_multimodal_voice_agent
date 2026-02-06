from __future__ import annotations

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from src.flow_agent.logging_config import setup_logging
from src.flow_agent.utils.nodes import call_combiner_model, call_summarizer_model, call_subtask_model, assign_workers, \
    entry_node, should_continue
from src.flow_agent.utils.state import State

setup_logging()


class Context(TypedDict):
    my_configurable_param: str


# this name is mentioned in langgraph.json
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("entry", entry_node)
    .add_node("summarizer", call_summarizer_model)
    .add_node("subtask", call_subtask_model)
    .add_node("combiner", call_combiner_model)
    .add_edge(START, "entry")
    .add_conditional_edges('entry', should_continue, {"summarizer": "summarizer", END: END})
    .add_conditional_edges("summarizer", assign_workers)  # <- fan-out here
    .add_edge("subtask", "combiner")  # fan-in after workers
    .add_edge("combiner", END)
)
