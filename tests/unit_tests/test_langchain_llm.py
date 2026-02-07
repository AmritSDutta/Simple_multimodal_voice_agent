"""Unit tests for LangChainChatLLM module."""

from unittest.mock import patch

import pytest
from src.flow_agent.config import settings
from src.flow_agent.llms.LangChainChatLLM import _get_random_provider


class TestGetRandomProvider:
    """Test suite for _get_random_provider function."""

    @patch("src.flow_agent.llms.LangChainChatLLM.random.choices")
    def test_returns_ollama_provider(self, mock_choices):
        """Test that 'ollama' is returned when random.choices selects it."""
        mock_choices.return_value = ["ollama"]

        result = _get_random_provider()

        assert result == "ollama"

    @patch("src.flow_agent.llms.LangChainChatLLM.random.choices")
    def test_returns_zhipu_provider(self, mock_choices):
        """Test that 'zhipu' is returned when random.choices selects it."""
        mock_choices.return_value = ["zhipu"]

        result = _get_random_provider()

        assert result == "zhipu"

    @patch("src.flow_agent.llms.LangChainChatLLM.random.choices")
    def test_returns_gemini_provider(self, mock_choices):
        """Test that 'gemini' is returned when random.choices selects it."""
        mock_choices.return_value = ["gemini"]

        result = _get_random_provider()

        assert result == "gemini"

    @patch("src.flow_agent.llms.LangChainChatLLM.random.choices")
    def test_returns_openai_provider(self, mock_choices):
        """Test that 'openai' is returned when random.choices selects it."""
        mock_choices.return_value = ["openai"]

        result = _get_random_provider()

        assert result == "openai"

    @patch("src.flow_agent.llms.LangChainChatLLM.random.choices")
    def test_random_choices_called_with_correct_parameters(self, mock_choices):
        """Test that random.choices is called with correct names, weights, and k=1."""
        mock_choices.return_value = ["ollama"]

        _get_random_provider()

        expected_names = list(settings.PROVIDER_DISTRIBUTION.keys())
        expected_weights = list(settings.PROVIDER_DISTRIBUTION.values())
        mock_choices.assert_called_once_with(
            expected_names, weights=expected_weights, k=1
        )

    @pytest.mark.parametrize(
        "provider", ["ollama", "zhipu", "gemini", "openai"]
    )
    @patch("src.flow_agent.llms.LangChainChatLLM.random.choices")
    def test_each_provider_can_be_selected(self, mock_choices, provider):
        """Parameterized test to verify all providers can be selected."""
        mock_choices.return_value = [provider]

        result = _get_random_provider()

        assert result == provider

    def test_always_returns_valid_provider(self):
        """Property-based test: result must always be one of the valid providers."""
        valid_providers = set(settings.PROVIDER_DISTRIBUTION.keys())

        for _ in range(100):
            result = _get_random_provider()
            assert result in valid_providers, f"Invalid provider returned: {result}"

    @pytest.mark.slow
    def test_distribution_matches_weights(self):
        """
        Statistical test: Run many times and verify distribution matches expected weights.

        This is a slower test that verifies the weighted random selection is working
        correctly. Marked as @pytest.mark.slow so it can be skipped during rapid
        development runs.
        """
        import collections

        num_runs = 1000
        results = [_get_random_provider() for _ in range(num_runs)]
        counts = collections.Counter(results)

        # Check each provider is within expected range (±5% tolerance)
        tolerance = 0.05
        for provider, expected_weight in settings.PROVIDER_DISTRIBUTION.items():
            actual_ratio = counts[provider] / num_runs
            delta = abs(actual_ratio - expected_weight)

            assert (
                delta < tolerance
            ), f"{provider}: expected {expected_weight:.2f}, got {actual_ratio:.2f} (delta={delta:.3f})"

    @pytest.mark.slow
    def test_all_providers_appear_in_distribution(self):
        """
        Verify that all providers appear at least once in a large sample.

        This tests that no provider is unreachable due to configuration errors.
        """
        import collections

        num_runs = 500
        results = [_get_random_provider() for _ in range(num_runs)]
        counts = collections.Counter(results)

        all_providers = set(settings.PROVIDER_DISTRIBUTION.keys())
        selected_providers = set(counts.keys())

        assert (
            selected_providers == all_providers
        ), f"Missing providers: {all_providers - selected_providers}"
