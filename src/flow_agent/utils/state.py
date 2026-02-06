import operator
from operator import add
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict, NotRequired

from src.flow_agent.data_objs.business_objs import CombinedPlan


class State(TypedDict):
    retry_count: Annotated[int, add]
    messages: Annotated[list[BaseMessage], add_messages]
    issue: str
    sub_issues_decision:  tuple[str, ...]
    sub_issue: NotRequired[str]
    completed_sub_issues_decision: Annotated[
        list, operator.add
    ]
    final_report: CombinedPlan
    ended_once: bool
