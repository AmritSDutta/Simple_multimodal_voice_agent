import logging
from typing import get_args

from agents import Runner
from google.genai.chats import AsyncChat
from langchain_core.messages import AIMessage, BaseMessage, convert_to_messages, get_buffer_string
from langgraph.config import get_config
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Command, Send, interrupt
from langgraph_api.schema import Context

from src.flow_agent.data_objs.business_objs import DecisionOutput, CombinedPlan, DecisionID, DecisionContext
from src.flow_agent.llms.genai_agent import get_summarizer_agent
from src.flow_agent.llms.sub_task_agent import get_sub_task_agent_instance
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

    return "summarizer"  # Normal flow


async def call_summarizer_model(state: State, runtime: Runtime[Context]) -> Command:
    user_message: list[BaseMessage] = state.get("messages")
    ctm = convert_to_messages(user_message)
    gbt = get_buffer_string(ctm, human_prefix="", ai_prefix="").strip()
    logging.info(gbt)
    if not user_message:
        logging.info(user_message)
        return Command(update={"retry_count": state["retry_count"], "messages": state["messages"]}, goto=END)

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

    return Command(update={
        "issue": summary,
        "messages": genai_res,
        'sub_issues_decision': get_args(DecisionID),
        'completed_sub_issues_decision': [],
    }, goto='combiner')


async def call_combiner_model(state: State, runtime: Runtime[Context]) -> Command:
    issue_summary: str = state['issue']
    decisions: list[DecisionOutput] = state['completed_sub_issues_decision']
    final_output: CombinedPlan = (CombinedPlan
                                  .assemble_from_evaluators(decisions,
                                                            additional_summary=issue_summary if issue_summary else ''))
    cfg = get_config()
    _thread_id = cfg.get("configurable", {}).get("thread_id", '')
    final_output.thread_identifier = _thread_id
    logging.info(final_output)

    if not final_output:
        return Command(update={"messages": state["messages"]}, goto=END)

    output_dump = final_output.model_dump_json(indent=4)
    logging.info(output_dump)

    return Command(update={
        "ended_once": True,
        "final_report": final_output,
        "messages": AIMessage(f'{output_dump}')
    }, goto=END)


async def call_subtask_model(state: State, runtime: Runtime[Context]):
    """
       This will be called for each decision type needed.
       Worker: evaluate a single decision_id and append DecisionOutput.
       Expects state["decision_id"] injected via Send().
    """
    cfg = get_config()
    _thread_id = cfg.get("configurable", {}).get("thread_id", '')

    sub_issue: str = state["sub_issue"]
    logging.info(f'[Sub-task] {sub_issue} executing ... ')

    """
    if sub_issue == "approval_required":
        approval: bool = _interrupt_bool()
        return {
            "completed_sub_issues_decision": [
                DecisionOutput(
                    decision_id=sub_issue,
                    decision=approval,
                    confidence=1.0,
                    thread_identifier=_thread_id
                )
            ]
        }
    """

    agent = await get_sub_task_agent_instance()
    decision_ctx = DecisionContext(
        context=state["issue"],
        decision_id=sub_issue,
    )
    result = await Runner.run(
        starting_agent=agent,
        input=f"Evaluate decision_id={sub_issue} for issue: {state['issue']}",
        context=decision_ctx,
    )

    logging.info(f'[subtask] {sub_issue}: input token- {result.context_wrapper.usage.input_tokens},'
                 f'output token- {result.context_wrapper.usage.output_tokens}')
    output_dump = result.final_output.model_dump_json(indent=4)
    logging.info(f'[Sub-task] {sub_issue}: {output_dump[:150]}')
    return {
        "messages": AIMessage(f'f"Evaluated needs : {sub_issue}'),
        "completed_sub_issues_decision": [result.final_output]
    }


async def assign_workers(state: State, runtime: Runtime[Context]):
    """Assign a worker to each type of decision"""
    sub_issues = state["sub_issues_decision"]

    if not sub_issues:
        return "combiner"

    return [
        Send("subtask", {**state, "sub_issue": s})
        for s in sub_issues
    ]


def _interrupt_bool(prompt: str = "is issue required elevated approval ?") -> bool:
    value = interrupt(prompt)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)

    raise ValueError(f"Invalid approval value from interrupt: {value!r}")
