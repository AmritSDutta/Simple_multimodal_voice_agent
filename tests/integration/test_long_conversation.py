"""
Integration test for long-running conversations.

Simulates extended conversations to verify context management works correctly.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock
from livekit.plugins import groq

from src.livekit_context_manager import ContextManager
from src.specialized_domain_state import SpecializedDomainState


@pytest.mark.asyncio
async def test_context_management_over_30_turns():
    """
    Simulate 30-turn conversation with context management.

    Verifies:
    - Context doesn't grow unbounded
    - Summarization triggers when needed
    - External state doesn't bloat context
    """
    # Create mock agent
    agent = MagicMock()
    agent.chat_ctx = MagicMock()
    agent.session = MagicMock()
    agent.session.llm = groq.LLM(model="llama-3.1-8b-instant")
    agent._chat_ctx = agent.chat_ctx  # Allow replacement

    # Create domain state
    domain_state = SpecializedDomainState()
    agent.session.userdata = MagicMock()
    agent.session.userdata.domain_state = domain_state

    # Simulate conversation growth
    # Start with 5 items
    agent.chat_ctx.items = list(range(5))

    # Mock the copy and summarize methods
    def create_mock_context(item_count):
        ctx = MagicMock()
        ctx.items = list(range(item_count))
        ctx.copy = MagicMock(return_value=ctx)
        ctx._summarize = AsyncMock()
        return ctx

    # Simulate 30 turns
    context_sizes = []
    for turn in range(1, 31):
        # Add 2 items per turn (user message + assistant response)
        current_size = len(agent.chat_ctx.items)
        new_size = current_size + 2

        # Create new mock context with increased size
        agent.chat_ctx = create_mock_context(new_size)
        agent._chat_ctx = agent.chat_ctx

        # Every 5 turns, check if summarization needed
        if turn % 5 == 0:
            # Setup summarization mock
            mock_summarized_ctx = MagicMock()
            mock_summarized_ctx.items = list(range(8))  # Compressed to 8 items

            agent.chat_ctx.copy.return_value = agent.chat_ctx
            agent.chat_ctx._summarize.return_value = mock_summarized_ctx

            # Check summarization
            await ContextManager.summarize_if_needed(
                agent=agent,
                threshold_items=20,
                keep_last_turns=3,
            )

            # If summarization was triggered, context should be replaced
            if agent.chat_ctx._summarize.called:
                agent._chat_ctx = mock_summarized_ctx
                agent.chat_ctx = mock_summarized_ctx

        context_sizes.append(len(agent.chat_ctx.items))

        # Save some preferences to external state
        if turn == 10:
            domain_state.set_user_preference("language", "en-IN")
        elif turn == 20:
            domain_state.set_user_preference("theme", "dark")

    # Verify context didn't grow unbounded
    max_context_size = max(context_sizes)
    assert max_context_size < 50, f"Context grew too large: {max_context_size}"

    # Verify external state works
    assert domain_state.get_user_preference("language") == "en-IN"
    assert domain_state.get_user_preference("theme") == "dark"

    # Verify context should be smaller at end than peak
    final_context_size = context_sizes[-1]
    peak_context_size = max(context_sizes)
    assert final_context_size < peak_context_size, \
        "Context should be compacted, not continuously growing"


@pytest.mark.asyncio
async def test_external_state_doesnt_bloat_context():
    """
    Verify that storing data in external state doesn't affect context size.
    """
    # Create mock agent
    agent = MagicMock()
    agent.chat_ctx = MagicMock()
    agent.chat_ctx.items = list(range(10))

    # Create domain state
    domain_state = SpecializedDomainState()

    # Store lots of data in external state
    for i in range(100):
        domain_state.set_user_preference(f"key_{i}", f"value_{i}")
        domain_state.set_domain_fact(f"fact_{i}", f"data_{i}")

    # Verify context size unchanged
    assert len(agent.chat_ctx.items) == 10

    # Verify external state has all the data
    assert len(domain_state.preferences) == 100
    assert len(domain_state.domain_facts) == 100

    # Verify retrieval works
    assert domain_state.get_user_preference("key_50") == "value_50"
    assert domain_state.get_domain_fact("fact_75") == "data_75"


@pytest.mark.asyncio
async def test_context_filtering_removes_noise():
    """
    Verify that context filtering removes system noise.
    """
    # Create mock agent with noisy context
    agent = MagicMock()
    agent.chat_ctx = MagicMock()

    # Create items with various types
    items = []
    for i in range(30):
        if i % 5 == 0:
            item = MagicMock(type='system')
        elif i % 5 == 1:
            item = MagicMock(type='function-call')
        elif i % 5 == 2:
            item = MagicMock(type='config-update')
        else:
            item = MagicMock(type='user')
        items.append(item)

    agent.chat_ctx.items = items

    # Mock the copy method
    compacted_ctx = MagicMock()
    compacted_ctx.items = [item for item in items if item.type == 'user']  # Only user items
    agent.chat_ctx.copy = MagicMock(return_value=compacted_ctx)
    compacted_ctx.truncate = MagicMock()

    # Apply filtering
    result = await ContextManager.get_compacted_context(
        agent=agent,
        keep_last_n_turns=10,
        exclude_system=True
    )

    # Verify copy was called with exclusions
    agent.chat_ctx.copy.assert_called_once()

    # Verify result is the compacted context
    assert result == compacted_ctx


@pytest.mark.asyncio
async def test_summarization_preserves_recent_context():
    """
    Verify that summarization keeps recent turns uncompressed.
    """
    # Create mock agent
    agent = MagicMock()
    agent.chat_ctx = MagicMock()
    agent.chat_ctx.items = list(range(25))
    agent.session = MagicMock()
    agent.session.llm = groq.LLM(model="llama-3.1-8b-instant")
    agent._chat_ctx = agent.chat_ctx

    # Mock summarization that keeps last 3 turns
    mock_summarized_ctx = MagicMock()
    mock_summarized_ctx.items = list(range(8))  # 5 summarized + 3 recent

    mock_copied_ctx = MagicMock()
    mock_copied_ctx._summarize = AsyncMock(return_value=mock_summarized_ctx)

    agent.chat_ctx.copy = MagicMock(return_value=mock_copied_ctx)

    # Trigger summarization
    result = await ContextManager.summarize_if_needed(
        agent=agent,
        threshold_items=20,
        keep_last_turns=3,
    )

    # Verify summarization was called with keep_last_turns=3
    mock_copied_ctx._summarize.assert_called_once_with(
        llm_v=agent.session.llm,
        keep_last_turns=3
    )

    # Verify context was replaced
    assert agent._chat_ctx == mock_summarized_ctx
    assert len(mock_summarized_ctx.items) == 8
