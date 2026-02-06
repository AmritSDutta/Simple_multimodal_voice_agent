from asyncio import Lock

from agents import Agent, RunContextWrapper

from src.flow_agent.data_objs.business_objs import DecisionContext, DecisionOutput, DECISION_TRIGGERS

_combiner_agent_instance = None
_auditor_agent_instance = None

_lock = Lock()


def dynamic_instructions(
        context: RunContextWrapper[DecisionContext], agent: Agent[DecisionContext]
) -> str:
    return f"""
    as a agent of {context.context.decision_id}. 
    You are an automated decision evaluator.

    Input:
    - decision_id: {context.context.decision_id}
    - context: unstructured text containing events, logs, symptoms, actions, or user reports.
    
    Task:
    1. Read and interpret the context.
    2. Based solely on the meaning of the decision_id, determine if action is required:
       - {DECISION_TRIGGERS.get(context.context.decision_id)}
    3. Return:
       - decision: true if action is warranted, false otherwise
       - confidence: 0.0–1.0 expressing certainty
       - model: name of the model producing the output
       - notes: concise reasoning (optional) , must be maximum 10 words.
       - latency_ms: leave empty
    
    Output JSON strictly in the following structure:
    
    {{
      "decision_id": "{context.context.decision_id}",
      "decision": true or false,
      "confidence": 0.0,
      "model": "gpt-5-nano",
      "notes": "short rationale",
      "latency_ms": null
    }}
    optimize output token usage without compromising on quality of output.
    Help them with their questions.
    """


async def get_sub_task_agent_instance():
    return Agent[DecisionContext](
        model='gpt-5-nano',
        name="Issue_evaluator",
        instructions=dynamic_instructions,
        output_type=DecisionOutput
    )
