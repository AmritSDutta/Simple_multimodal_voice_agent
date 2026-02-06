import asyncio
import json
import logging
import time

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.flow_agent.data_objs.business_objs import CombinedPlan
from src.flow_agent.graph import graph as raw_graph

pytestmark = pytest.mark.skip(reason="E2E Google GenAI tests disabled in CI")

MESSAGE_1 = """
Incident Notification #2345: Incident Notification
To Whom It May Concern,
This email serves as a notification of a recent incident that has taken place. 
While I am in the process of gathering all the necessary details,
I wanted to bring this matter to your attention as soon as possible.
I assure you that I am taking the necessary steps to determine the root cause of the incident 
and will implement measures to prevent similar incidents from occurring in the future.
I would appreciate your understanding and cooperation as I work to provide a full and complete report.
Thank you
"""

MESSAGE_2 = """
subject: repeated VPN disconnects and SSO failures.

Hi team,

Please take a look at this — ticket INC-448291 has been getting worse.  
User is really frustrated at this point. They’re on priority P3, but the impact is growing.

Full description:  
Over the last 24 hours, their VPN drops almost every hour. On top of that, the SSO token refresh keeps failing, 
which means they can’t log into Jira or Confluence at all. They’re basically blocked from doing any internal work. 
They’ve attached logs from today hoping it helps (log_2025_11_20.zip).

Requested action from the user: *“Please stabilize my access — I can’t keep getting kicked out like this.”*
Feels like this might need a VPN profile reset + SSO session restart, but I’ll leave it to the automation to decide.
Thanks.
"""


# @pytest.mark.skip(reason="Requires Google GenAI credentials")
@pytest.mark.asyncio(loop_scope="session")
async def test_graph_1():
    test_graph = raw_graph.compile()

    t0 = time.perf_counter()
    result = await asyncio.wait_for(test_graph.ainvoke(
        {"retry_count": 0, "messages": [HumanMessage(content=MESSAGE_1)]}, config={"configurable": {"thread_id": "1"}}
    ), timeout=90)
    elapsed = time.perf_counter() - t0
    print(f"graph latency: {elapsed:.3f}s")

    message = result["messages"][-1]
    if message and isinstance(message, AIMessage):
        logging.info(message.content)
        text: str = message.content

        # 1
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]

        # 2
        data = json.loads(json_str)

        # 3
        plan = CombinedPlan.model_validate(data)
        assert plan.confidence < 1.0
        assert plan.generated_summary is not None
        assert len(plan.task_specific_notes) > 0
    else:
        pytest.fail("AIMessage not emitted")


# @pytest.mark.skip(reason="Requires Google GenAI credentials")
@pytest.mark.asyncio(loop_scope="session")
async def test_graph_2():
    test_graph = raw_graph.compile()
    t0 = time.perf_counter()
    result = await asyncio.wait_for(
        test_graph.ainvoke(
            {"retry_count": 0, "messages": [HumanMessage(content=MESSAGE_2)]},
            config={"configurable": {"thread_id": "2"}}
        )
        , timeout=300)
    elapsed = time.perf_counter() - t0
    print(f"graph latency: {elapsed:.3f}s")

    logging.info('graph invoked')
    message = result["messages"][-1]
    assert message
    assert isinstance(message, AIMessage)
    text: str = message.content

    # 1
    start = text.index("{")
    end = text.rindex("}") + 1
    json_str = text[start:end]

    # 2
    data = json.loads(json_str)

    # 3
    plan = CombinedPlan.model_validate(data)
    assert plan.confidence < 1.0
    assert plan.generated_summary is not None
    assert len(plan.task_specific_notes) > 0
    assert plan.reset_vpn_profile is True
    assert plan.restart_sso_session is True
