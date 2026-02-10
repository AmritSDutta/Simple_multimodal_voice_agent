"""Test multi-turn conversation memory"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import add_messages


@pytest.mark.asyncio
async def test_call_llm_safely_signature():
    """Verify that call_llm_safely accepts conversation history + new message"""
    from src.flow_agent.utils.nodes import call_llm_safely
    import inspect

    sig = inspect.signature(call_llm_safely)
    params = list(sig.parameters.keys())

    assert "llm" in params, "call_llm_safely should have 'llm' parameter"
    assert "conversation" in params, "call_llm_safely should have 'conversation' parameter for history"
    assert "new_message" in params, "call_llm_safely should have 'new_message' parameter"

    print("PASS: call_llm_safely has correct signature (llm, conversation, new_message)")


def test_add_messages_reducer_behavior():
    """Test how LangGraph's add_messages reducer actually works"""
    from src.flow_agent.utils.state import State

    # Simulate existing messages in state
    existing_messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]

    # New messages returned from a node
    new_messages = [AIMessage(content="How can I help?")]

    # The add_messages reducer merges them
    result = add_messages(existing_messages, new_messages)

    assert len(result) == 3, f"Expected 3 messages, got {len(result)}"
    assert result[0].content == "Hello"
    assert result[1].content == "Hi there!"
    assert result[2].content == "How can I help?"

    print("PASS: add_messages reducer appends new messages to existing history")


def test_state_schema():
    """Verify State has required fields for multi-turn"""
    from src.flow_agent.utils.state import State

    annotations = State.__annotations__

    required_fields = ["messages", "retry_count", "issue", "final_report", "input_valid", "conversation_summary"]
    for field in required_fields:
        assert field in annotations, f"State missing field: {field}"

    # Check that messages uses add_messages reducer
    from typing import get_args, get_origin
    messages_annotation = annotations["messages"]
    assert "add_messages" in str(messages_annotation), "messages should use add_messages reducer"

    print("PASS: State schema has all required fields with correct reducers")


@pytest.mark.asyncio
async def test_prepare_llm_input():
    """Test that prepare_llm_input handles multimodal content"""
    from src.flow_agent.utils.nodes import prepare_llm_input

    # Test text-only
    result = await prepare_llm_input("Hello", None)
    assert len(result) == 1
    assert result[0]["type"] == "text"

    # Test with images
    images = ["base64data1", "base64data2"]
    result = await prepare_llm_input("What's in these images?", images)
    assert len(result) == 3  # 1 text + 2 images
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "image_url"

    print("PASS: prepare_llm_input handles multimodal content correctly")


@pytest.mark.asyncio
async def test_full_conversation_flow_simulation():
    """Simulate how LangGraph handles conversation state across turns"""
    from src.flow_agent.utils.nodes import process_response, call_langchain_reasoning_model
    from src.flow_agent.utils.state import State
    from langgraph.graph import add_messages

    # Initial state (like start of a conversation)
    state: State = {
        "messages": [],
        "retry_count": 0,
        "issue": "",
        "final_report": "",
        "input_valid": True,
        "conversation_summary": "",
    }

    # Turn 1: User sends first message
    user_msg_1 = HumanMessage(content="What is Python?")
    # Simulate entry_node adding this to state
    state["messages"] = add_messages(state["messages"], [user_msg_1])

    assert len(state["messages"]) == 1, "Should have 1 message after user input"

    # Turn 1: Reasoning node returns response
    mock_response_1 = AIMessage(content="Python is a programming language.")
    node_update_1 = await process_response(state, mock_response_1, "What is Python?")

    # LangGraph merges the update using add_messages reducer
    state["messages"] = add_messages(state["messages"], node_update_1["messages"])
    state["final_report"] = node_update_1["final_report"]

    assert len(state["messages"]) == 2, f"Should have 2 messages after turn 1, got {len(state['messages'])}"
    print(f"After turn 1: {len(state['messages'])} messages")

    # Turn 2: User sends follow-up (same thread)
    user_msg_2 = HumanMessage(content="Is it easy to learn?")
    state["messages"] = add_messages(state["messages"], [user_msg_2])

    assert len(state["messages"]) == 3, f"Should have 3 messages before turn 2 reasoning, got {len(state['messages'])}"

    # Turn 2: Reasoning node returns response
    mock_response_2 = AIMessage(content="Yes, Python is considered beginner-friendly.")
    node_update_2 = await process_response(state, mock_response_2, "Is it easy to learn?")

    # LangGraph merges the update using add_messages reducer
    state["messages"] = add_messages(state["messages"], node_update_2["messages"])
    state["final_report"] = node_update_2["final_report"]

    assert len(state["messages"]) == 4, f"Should have 4 messages after turn 2, got {len(state['messages'])}"
    print(f"After turn 2: {len(state['messages'])} messages")
    print("PASS: Full conversation flow maintains history correctly")
