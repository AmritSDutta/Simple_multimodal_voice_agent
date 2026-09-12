"""
Unit tests for LiveKit context management.

Tests context filtering, summarization triggers, and external state.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock
from livekit.agents import llm
from livekit.plugins import groq

from src.livekit_context_manager import ContextManager
from src.specialized_domain_state import SpecializedDomainState


@pytest.fixture
def mock_agent():
    """Create a mock agent with chat context."""
    agent = MagicMock()
    agent.chat_ctx = MagicMock()
    agent.session = MagicMock()
    agent.session.llm = groq.LLM(model="llama-3.1-8b-instant")
    return agent


@pytest.fixture
def domain_state():
    """Create a domain state instance."""
    return SpecializedDomainState()


class TestContextFiltering:
    """Test context filtering and compaction."""

    @pytest.mark.asyncio
    async def test_context_filtering_reduces_noise(self, mock_agent):
        """Verify context filtering removes system messages."""
        # Setup mock chat context with various items
        mock_items = [
            MagicMock(type='system'),  # Should be excluded
            MagicMock(type='user'),
            MagicMock(type='assistant'),
            MagicMock(type='function-call'),  # Should be excluded
            MagicMock(type='user'),
        ]

        mock_agent.chat_ctx.items = mock_items
        mock_agent.chat_ctx.copy = MagicMock(return_value=mock_agent.chat_ctx)
        mock_agent.chat_ctx.truncate = MagicMock()

        # Call get_compacted_context
        compacted_ctx = await ContextManager.get_compacted_context(
            agent=mock_agent,
            keep_last_n_turns=10,
            exclude_system=True
        )

        # Verify copy was called with correct exclusions
        mock_agent.chat_ctx.copy.assert_called_once_with(
            exclude_function_call=True,
            exclude_instructions=True,
            exclude_config_update=True,
            exclude_handoff=True,
            exclude_empty_message=True,
        )

    @pytest.mark.asyncio
    async def test_context_truncation_when_needed(self, mock_agent):
        """Verify context truncates when exceeding threshold."""
        # Setup mock with more items than threshold
        mock_items = list(range(25))  # 25 items
        mock_agent.chat_ctx.items = mock_items
        mock_agent.chat_ctx.copy = MagicMock(return_value=mock_agent.chat_ctx)
        mock_agent.chat_ctx.truncate = MagicMock()

        # Call get_compacted_context with threshold of 10
        await ContextManager.get_compacted_context(
            agent=mock_agent,
            keep_last_n_turns=10,
            exclude_system=True
        )

        # Verify truncation was called
        mock_agent.chat_ctx.truncate.assert_called_once_with(max_items=10)

    @pytest.mark.asyncio
    async def test_context_no_truncation_under_threshold(self, mock_agent):
        """Verify no truncation when under threshold."""
        # Setup mock with fewer items than threshold
        mock_items = list(range(5))  # 5 items
        mock_agent.chat_ctx.items = mock_items
        mock_agent.chat_ctx.copy = MagicMock(return_value=mock_agent.chat_ctx)
        mock_agent.chat_ctx.truncate = MagicMock()

        # Call get_compacted_context with threshold of 10
        await ContextManager.get_compacted_context(
            agent=mock_agent,
            keep_last_n_turns=10,
            exclude_system=True
        )

        # Verify truncation was NOT called
        mock_agent.chat_ctx.truncate.assert_not_called()


class TestSummarization:
    """Test periodic summarization."""

    @pytest.mark.asyncio
    async def test_no_summarization_under_threshold(self, mock_agent):
        """Verify no summarization when under threshold."""
        # Setup mock with 10 items (under threshold of 20)
        mock_agent.chat_ctx.items = list(range(10))

        # Call summarize_if_needed
        result = await ContextManager.summarize_if_needed(
            agent=mock_agent,
            threshold_items=20,
        )

        # Verify original context returned
        assert result == mock_agent.chat_ctx

    @pytest.mark.asyncio
    async def test_summarization_trigger_at_threshold(self, mock_agent):
        """Verify summarization triggers when exceeding threshold."""
        # Setup mock with 25 items (exceeds threshold of 20)
        mock_agent.chat_ctx.items = list(range(25))

        # Mock the copy and summarize methods
        mock_copied_ctx = MagicMock()
        mock_summarized_ctx = MagicMock()
        mock_summarized_ctx.items = list(range(5))  # Summarized to 5 items

        mock_agent.chat_ctx.copy = MagicMock(return_value=mock_copied_ctx)
        mock_copied_ctx._summarize = AsyncMock(return_value=mock_summarized_ctx)

        # Call summarize_if_needed
        result = await ContextManager.summarize_if_needed(
            agent=mock_agent,
            threshold_items=20,
            keep_last_turns=3,
        )

        # Verify summarization was called
        mock_copied_ctx._summarize.assert_called_once_with(
            llm_v=mock_agent.session.llm,
            keep_last_turns=3
        )

        # Verify agent context was replaced
        assert mock_agent._chat_ctx == mock_summarized_ctx

    @pytest.mark.asyncio
    async def test_summarization_failure_handling(self, mock_agent):
        """Verify graceful failure when summarization errors."""
        # Setup mock with 25 items
        mock_agent.chat_ctx.items = list(range(25))

        # Mock the copy to raise exception
        mock_copied_ctx = MagicMock()
        mock_agent.chat_ctx.copy = MagicMock(return_value=mock_copied_ctx)
        mock_copied_ctx._summarize = AsyncMock(side_effect=Exception("API error"))

        # Call summarize_if_needed - should not raise
        result = await ContextManager.summarize_if_needed(
            agent=mock_agent,
            threshold_items=20,
        )

        # Verify original context returned on error
        assert result == mock_agent.chat_ctx


class TestContextStats:
    """Test context statistics generation."""

    def test_get_context_stats(self, mock_agent):
        """Verify context stats are calculated correctly."""
        # Setup mock items with different types
        item1 = MagicMock(type='system')
        item2 = MagicMock(type='user')
        item3 = MagicMock(type='function-call')
        item4 = MagicMock(type='user')

        mock_agent.chat_ctx.items = [item1, item2, item3, item4]

        # Get stats
        stats = ContextManager.get_context_stats(mock_agent)

        # Verify stats
        assert stats['total_items'] == 4
        assert stats['has_system_messages'] == True
        assert stats['has_function_calls'] == True
        assert 'MagicMock' in stats['type_breakdown']


class TestExternalState:
    """Test external state management."""

    def test_set_and_get_preference(self, domain_state):
        """Verify preferences can be stored and retrieved."""
        # Set preference
        domain_state.set_user_preference("language", "en-IN")

        # Get preference
        value = domain_state.get_user_preference("language")

        assert value == "en-IN"

    def test_get_nonexistent_preference(self, domain_state):
        """Verify getting nonexistent preference returns None."""
        value = domain_state.get_user_preference("nonexistent")
        assert value is None

    def test_add_and_get_task_result(self, domain_state):
        """Verify task results can be stored and retrieved."""
        # Add task result
        domain_state.add_task_result("data_analysis", {"rows": 100})

        # Get task history
        history = domain_state.get_task_history()

        assert len(history) == 1
        assert history[0]["task"] == "data_analysis"
        assert history[0]["result"]["rows"] == 100

    def test_filter_task_history_by_name(self, domain_state):
        """Verify task history can be filtered by task name."""
        # Add multiple tasks
        domain_state.add_task_result("task1", {"data": "a"})
        domain_state.add_task_result("task2", {"data": "b"})
        domain_state.add_task_result("task1", {"data": "c"})

        # Filter by task name
        task1_history = domain_state.get_task_history("task1")

        assert len(task1_history) == 2
        assert all(t["task"] == "task1" for t in task1_history)

    def test_domain_facts(self, domain_state):
        """Verify domain facts can be stored and retrieved."""
        # Set domain fact
        domain_state.set_domain_fact("user_expert_level", "advanced")

        # Get domain fact
        value = domain_state.get_domain_fact("user_expert_level")

        assert value == "advanced"

    def test_user_profile(self, domain_state):
        """Verify user profile can be updated."""
        # Update profile
        domain_state.update_user_profile({"name": "Alice", "role": "analyst"})

        # Verify profile
        assert domain_state.user_profile["name"] == "Alice"
        assert domain_state.user_profile["role"] == "analyst"

    def test_session_duration(self, domain_state):
        """Verify session duration is calculated."""
        import time

        # Wait a bit
        time.sleep(0.1)

        # Get duration
        duration = domain_state.get_session_duration()

        assert duration >= 0.1

    def test_to_summary(self, domain_state):
        """Verify domain state can be summarized."""
        # Add some data
        domain_state.set_user_preference("language", "en-IN")
        domain_state.update_user_profile({"name": "Alice"})
        domain_state.set_domain_fact("fact1", "value1")

        # Get summary
        summary = domain_state.to_summary()

        assert "language=en-IN" in summary
        assert "User Profile:" in summary
        assert "Domain Facts Learned:" in summary

    def test_empty_domain_state_summary(self, domain_state):
        """Verify summary for empty state."""
        summary = domain_state.to_summary()
        assert summary == "No domain state accumulated yet."
