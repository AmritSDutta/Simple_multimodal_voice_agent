"""Unit tests for circuit_breaker_llm module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.flow_agent.utils.circuit_breaker_llm import call_llm_safely


class TestCallLLMSafely:
    """Test suite for call_llm_safely function."""

    @pytest.mark.asyncio
    async def test_call_llm_safely_returns_response_on_success(self):
        """
        Test that call_llm_safely returns response when primary LLM succeeds.

        This test verifies the happy path where the LLM call succeeds on first
        attempt, covering the main code path and metadata logging.
        """
        # Arrange
        mock_llm = AsyncMock()
        conversation = [HumanMessage(content="Hello")]

        # Create mock response with required metadata attributes
        mock_response = MagicMock()
        mock_response.response_metadata = {"model": "test-model", "finish_reason": "STOP"}
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 20}
        mock_llm.ainvoke.return_value = mock_response

        new_message = HumanMessage(content="What is the capital of France?")

        # Act
        with patch("src.flow_agent.utils.circuit_breaker_llm._call_llm") as mock_call_llm:
            mock_call_llm.return_value = mock_response

            result = await call_llm_safely(mock_llm, conversation, new_message)

        # Assert
        assert result is not None
        assert result.response_metadata == {"model": "test-model", "finish_reason": "STOP"}
        assert result.usage_metadata == {"input_tokens": 10, "output_tokens": 20}
        # Verify _call_llm was invoked with full context (conversation + new_message)
        expected_full_context = conversation + [new_message]
        mock_call_llm.assert_called_once_with(mock_llm, expected_full_context)
