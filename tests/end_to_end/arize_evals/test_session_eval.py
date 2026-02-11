"""Session-level evaluation tests for multi-turn text conversations using Phoenix llm_classify."""
import os
from unittest.mock import patch

import pandas as pd
import pytest
from phoenix.evals import OpenAIModel, llm_classify, GoogleGenAIModel
from langchain_core.messages import HumanMessage

from src.flow_agent.graph import graph

# ---- Session Evaluation Prompts ----

SESSION_CORRECTNESS_PROMPT = """
You are an expert assistant evaluating the **correctness and quality** of an AI agent's responses during a multi-turn conversation.

A session consists of multiple interactions between a user and an AI agent. You will be given:
1. The user's messages (inputs), in order.
2. The AI agent's responses (outputs), in order.

A correct and high-quality session should:
- Provide factually accurate information
- Address the user's questions directly and helpfully
- Maintain context and coherence across turns
- Avoid hallucinations or incorrect reasoning
- may include references

##
User Inputs:
{user_inputs}

Agent Outputs:
{outputs}
##

Based on the above, evaluate the session **only for correctness and response quality**.

Respond with a single word: `correct` or `incorrect`.

- Respond with `correct` if the AI agent consistently provides accurate and helpful answers.
- Respond with `incorrect` if the AI agent gives factually wrong, misleading, or incoherent explanations.
"""

SESSION_GOAL_ACHIEVEMENT_PROMPT = """
You are an AI assistant evaluating whether the AI agent successfully helped the user achieve their goals during a multi-turn conversation.

A session consists of multiple interactions between a user and an AI agent. You will be given:
1. The user's messages (inputs), in chronological order.
2. The AI agent's responses (outputs), in chronological order.

To determine if the user's goals were achieved, consider:
- Whether the AI agent addressed the user's questions and requests directly
- Whether the explanations provided resolved the user's doubts or problems
- Whether the user's inputs indicate understanding or closure by the end
- Whether the conversation logically progressed toward completing the user's objectives

##
User Inputs:
{user_inputs}

Agent Outputs:
{outputs}
##

Evaluate the session and respond with a single word: `achieved` or `not_achieved`.

- Respond with `achieved` if the session successfully met the user's goals and resolved their questions.
- Respond with `not_achieved` if the session left the user's questions unanswered or goals unmet.
"""


# ---- Test Multi-Turn Text Conversation ----


@pytest.mark.asyncio
async def test_multi_turn_session_correctness(custom_settings_with_gemma_3_12b):
    """Test a multi-turn text conversation session for correctness using Phoenix llm_classify."""

    # Configure settings for this test
    with patch("src.flow_agent.graph.settings", custom_settings_with_gemma_3_12b), \
            patch("src.flow_agent.utils.nodes.settings", custom_settings_with_gemma_3_12b), \
            patch("src.flow_agent.llms.LangChainChatLLM.settings", custom_settings_with_gemma_3_12b):

        compiled_graph = graph.compile()

        # Simulate a multi-turn conversation about Python programming
        conversation_turns = [
            "What is a list comprehension in Python?",
            "Can you show me an example with numbers?",
            "How do I add a condition to filter items?",
        ]

        # Build up conversation with context
        messages = []
        user_inputs = []
        agent_outputs = []

        for user_message in conversation_turns:
            # Add user message to conversation history
            messages.append(HumanMessage(content=user_message))

            # Invoke the graph with full conversation history
            result = await compiled_graph.ainvoke(
                {"messages": messages},
                config={
                    "configurable": {
                        "my_configurable_param": "test-value",
                    }
                },
            )

            assert result is not None

            # Extract the agent's response
            agent_response = result["messages"][-1].content

            # Collect for session evaluation
            user_inputs.append(user_message)
            agent_outputs.append(agent_response)

        # ---- Create session dataframe for Phoenix llm_classify ----
        sessions_df = pd.DataFrame([{
            "user_inputs": user_inputs,
            "outputs": agent_outputs,
        }])

        # ---- Configure the evaluation model ----
        model = GoogleGenAIModel(
            model="gemma-3-27b-it",
        )

        # ---- Run Session Correctness Evaluation ----
        rails = ["correct", "incorrect"]
        eval_results_correctness = llm_classify(
            data=sessions_df,
            template=SESSION_CORRECTNESS_PROMPT,
            model=model,
            rails=rails,
            provide_explanation=True,
            verbose=False,
        )

        # ---- Assert on correctness evaluation ----
        correctness_label = eval_results_correctness.iloc[0]["label"].lower()
        correctness_explanation = eval_results_correctness.iloc[0].get("explanation", "")

        # Print for debugging
        print(f"\n=== Session Correctness Evaluation ===")
        print(f"Label: {correctness_label}")
        print(f"Explanation: {correctness_explanation}")
        print(f"======================================\n")

        assert correctness_label == "correct", (
            f"Session correctness evaluation failed: {correctness_label}\n"
            f"Explanation: {correctness_explanation}"
        )


@pytest.mark.asyncio
async def test_multi_turn_session_goal_achievement(custom_settings_with_gemma_3_12b):
    """Test a multi-turn text conversation session for goal achievement using Phoenix llm_classify."""

    # Configure settings for this test
    with patch("src.flow_agent.graph.settings", custom_settings_with_gemma_3_12b), \
            patch("src.flow_agent.utils.nodes.settings", custom_settings_with_gemma_3_12b), \
            patch("src.flow_agent.llms.LangChainChatLLM.settings", custom_settings_with_gemma_3_12b):

        compiled_graph = graph.compile()

        # Simulate a multi-turn conversation with a clear goal
        conversation_turns = [
            "I need to understand the difference between list and tuple in Python",
            "Which one is immutable?",
            "Can you give me use cases for each?",
        ]

        # Build up conversation with context
        messages = []
        user_inputs = []
        agent_outputs = []

        for user_message in conversation_turns:
            # Add user message to conversation history
            messages.append(HumanMessage(content=user_message))

            result = await compiled_graph.ainvoke(
                {"messages": messages},
                config={
                    "configurable": {
                        "my_configurable_param": "test-value",
                    }
                },
            )

            assert result is not None
            agent_response = result["messages"][-1].content

            user_inputs.append(user_message)
            agent_outputs.append(agent_response)

        # ---- Create session dataframe ----
        sessions_df = pd.DataFrame([{
            "user_inputs": user_inputs,
            "outputs": agent_outputs,
        }])

        # ---- Configure the evaluation model ----

        model = GoogleGenAIModel(
            model="gemma-3-27b-it",
        )

        # ---- Run Goal Achievement Evaluation ----
        rails = ["achieved", "not_achieved"]
        eval_results_goal = llm_classify(
            data=sessions_df,
            template=SESSION_GOAL_ACHIEVEMENT_PROMPT,
            model=model,
            rails=rails,
            provide_explanation=True,
            verbose=False,
        )

        # ---- Assert on goal achievement evaluation ----
        goal_label = eval_results_goal.iloc[0]["label"].lower()
        goal_explanation = eval_results_goal.iloc[0].get("explanation", "")

        # Print for debugging
        print(f"\n=== Goal Achievement Evaluation ===")
        print(f"Label: {goal_label}")
        print(f"Explanation: {goal_explanation}")
        print("====================================\n")

        assert goal_label == "achieved", (
            f"Goal achievement evaluation failed: {goal_label}\n"
            f"Explanation: {goal_explanation}"
        )


@pytest.mark.asyncio
async def test_single_turn_correctness(custom_settings_with_gemma_3_12b):
    """Test a single-turn conversation (simpler case) for correctness."""

    with patch("src.flow_agent.graph.settings", custom_settings_with_gemma_3_12b), \
         patch("src.flow_agent.utils.nodes.settings", custom_settings_with_gemma_3_12b), \
         patch("src.flow_agent.llms.LangChainChatLLM.settings", custom_settings_with_gemma_3_12b):
        compiled_graph = graph.compile()

        # Single question
        user_message = "What is the capital of France?"
        message = HumanMessage(content=user_message)

        result = await compiled_graph.ainvoke(
            {"messages": [message]},
            config={
                "configurable": {
                    "my_configurable_param": "test-value",
                }
            },
        )

        assert result is not None
        agent_response = result["messages"][-1].content

        # Create session dataframe (single item lists)
        sessions_df = pd.DataFrame([{
            "user_inputs": [user_message],
            "outputs": [agent_response],
        }])

        # Run evaluation
        judge_sarvam = OpenAIModel(
            model="sarvam-m",
            base_url="https://api.sarvam.ai/v1",
            default_headers={
                "api-subscription-key": os.getenv('SARVAM_API_KEY'),
            },
            temperature=0.0,
            max_tokens=2048,
        )

        rails = ["correct", "incorrect"]

        eval_results = llm_classify(
            data=sessions_df,
            template=SESSION_CORRECTNESS_PROMPT,
            model=judge_sarvam,
            rails=rails,
            provide_explanation=True,
            verbose=False,
        )

        label = eval_results.iloc[0]["label"].lower()
        explanation = eval_results.iloc[0].get("explanation", "")

        print(f"\n=== Single-Turn Correctness ===")
        print(f"Label: {label}")
        print(f"Explanation: {explanation}")
        print("================================\n")

        assert label == "correct", f"Single-turn evaluation failed: {explanation}"
