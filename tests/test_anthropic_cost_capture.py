"""Tests for cost capture in NormalizedChatAnthropic.invoke() (Phase 6, D-03/D-04/D-05).

Tests verify that:
1. AnthropicClient passes cost_* kwargs to NormalizedChatAnthropic
2. invoke() calls record_cost_from_response with correct parameters
3. Cost tracking failures never block the pipeline (D-05)
4. Existing test_anthropic_effort.py still passes (backward compatibility)
"""

import pytest
from unittest import mock

from tradingagents.llm_clients import anthropic_client as mod


def _capture_invoke_calls(monkeypatch):
    """Capture calls to record_cost_from_response during invoke()."""
    captured = {"invocations": []}

    original_record_cost = mod.record_cost_from_response

    def mock_record_cost(response, **kwargs):
        captured["invocations"].append({"response": response, "kwargs": kwargs})
        return original_record_cost(response, **kwargs)

    monkeypatch.setattr(
        "tradingagents.llm_clients.anthropic_client.record_cost_from_response",
        mock_record_cost,
    )
    return captured


def _capture_llm_kwargs(monkeypatch):
    """Capture kwargs passed to NormalizedChatAnthropic constructor."""
    captured = {"kwargs": {}}

    original_init = mod.NormalizedChatAnthropic.__init__

    def mock_init(self, **kwargs):
        captured["kwargs"] = kwargs
        # Call original __init__
        original_init(self, **kwargs)

    monkeypatch.setattr(
        mod.NormalizedChatAnthropic,
        "__init__",
        mock_init,
    )
    return captured


@pytest.mark.unit
class TestAnthropicClientCostFields:
    """Verify cost_* fields are passed to NormalizedChatAnthropic."""

    def test_with_layer_ticker_trade_date(self, monkeypatch):
        """AnthropicClient passes cost_layer/cost_ticker/cost_trade_date to NormalizedChatAnthropic."""
        captured = {}

        # Mock NormalizedChatAnthropic to capture constructor kwargs
        def mock_normalized(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        monkeypatch.setattr(mod, "NormalizedChatAnthropic", mock_normalized)

        mod.AnthropicClient(
            model="claude-haiku-4-5",
            layer="auditor",
            ticker="AAPL",
            trade_date="2026-08-05",
            api_key="x",
        ).get_llm()

        assert captured["cost_layer"] == "auditor"
        assert captured["cost_ticker"] == "AAPL"
        assert captured["cost_trade_date"] == "2026-08-05"

    def test_without_layer_ticker_trade_date(self, monkeypatch):
        """AnthropicClient uses defaults when layer/ticker/trade_date not provided."""
        captured = {}

        def mock_normalized(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        monkeypatch.setattr(mod, "NormalizedChatAnthropic", mock_normalized)

        mod.AnthropicClient(
            model="claude-haiku-4-5",
            api_key="x",
        ).get_llm()

        assert captured["cost_layer"] == "unclassified"
        assert captured["cost_ticker"] == ""
        assert captured["cost_trade_date"] == ""

    def test_partial_cost_fields(self, monkeypatch):
        """AnthropicClient correctly handles partial cost field specification."""
        captured = {}

        def mock_normalized(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        monkeypatch.setattr(mod, "NormalizedChatAnthropic", mock_normalized)

        mod.AnthropicClient(
            model="claude-sonnet-5",
            ticker="TSLA",
            api_key="x",
        ).get_llm()

        assert captured["cost_layer"] == "unclassified"
        assert captured["cost_ticker"] == "TSLA"
        assert captured["cost_trade_date"] == ""


@pytest.mark.unit
class TestInvokeCostCapture:
    """Verify invoke() calls record_cost_from_response."""

    def test_invoke_calls_record_cost_from_response(self, monkeypatch):
        """NormalizedChatAnthropic.invoke() must call record_cost_from_response."""
        # Create a real NormalizedChatAnthropic instance with mocked parent
        with mock.patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic.invoke") as mock_super_invoke:
            # Mock the parent invoke to return a response-like object
            mock_response = mock.MagicMock()
            mock_response.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "input_token_details": {"cache_read": 0, "cache_creation": 0},
            }
            mock_response.content = "Test response"
            mock_super_invoke.return_value = mock_response

            # Mock record_cost_from_response to track calls
            with mock.patch(
                "tradingagents.llm_clients.anthropic_client.record_cost_from_response"
            ) as mock_record_cost:
                mock_record_cost.return_value = None  # Cost recording succeeded

                # Create instance and call invoke
                llm = mod.NormalizedChatAnthropic(
                    model="claude-sonnet-5",
                    cost_layer="auditor",
                    cost_ticker="AAPL",
                    cost_trade_date="2026-08-05",
                    api_key="x",
                )
                result = llm.invoke("test input")

                # Verify record_cost_from_response was called
                assert mock_record_cost.called
                call_args = mock_record_cost.call_args
                assert call_args[1]["layer"] == "auditor"
                assert call_args[1]["model"] == "claude-sonnet-5"
                assert call_args[1]["ticker"] == "AAPL"
                assert call_args[1]["trade_date"] == "2026-08-05"

    def test_invoke_returns_normalized_even_on_cost_error(self, monkeypatch):
        """invoke() must return normalized response even if cost recording fails."""
        with mock.patch("tradingagents.llm_clients.anthropic_client.ChatAnthropic.invoke") as mock_super_invoke:
            mock_response = mock.MagicMock()
            mock_response.content = "Test response"
            mock_response.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "input_token_details": {"cache_read": 0, "cache_creation": 0},
            }
            mock_super_invoke.return_value = mock_response

            # Mock record_cost_from_response to raise an exception
            with mock.patch(
                "tradingagents.llm_clients.anthropic_client.record_cost_from_response"
            ) as mock_record_cost:
                mock_record_cost.side_effect = RuntimeError("Cost recording failed")

                llm = mod.NormalizedChatAnthropic(
                    model="claude-sonnet-5",
                    cost_layer="auditor",
                    api_key="x",
                )

                # Should not raise, should return normalized response
                result = llm.invoke("test input")
                assert result is not None


@pytest.mark.unit
class TestBackwardCompatibility:
    """Verify new cost_* fields don't break existing kwargs handling."""

    def test_existing_kwargs_still_forwarded(self, monkeypatch):
        """Cost fields must not interfere with existing passthrough kwargs."""
        captured = {}

        def mock_normalized(**kwargs):
            captured.update(kwargs)
            return mock.MagicMock()

        monkeypatch.setattr(mod, "NormalizedChatAnthropic", mock_normalized)

        mod.AnthropicClient(
            model="claude-haiku-4-5",
            layer="research",
            api_key="placeholder",
            max_tokens=1024,
            timeout=30,
        ).get_llm()

        # Verify existing kwargs are still there
        assert captured["api_key"] == "placeholder"
        assert captured["max_tokens"] == 1024
        assert captured["timeout"] == 30
        # And new cost fields are too
        assert captured["cost_layer"] == "research"


@pytest.mark.unit
class TestDefaultConfigKeys:
    """Verify budget configuration keys are in DEFAULT_CONFIG."""

    def test_annual_budget_target_in_config(self):
        """DEFAULT_CONFIG must have annual_budget_target_usd = 630.0."""
        from tradingagents.default_config import DEFAULT_CONFIG

        assert "annual_budget_target_usd" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["annual_budget_target_usd"] == 630.0

    def test_budget_alert_threshold_in_config(self):
        """DEFAULT_CONFIG must have budget_alert_threshold_pct = 0.80."""
        from tradingagents.default_config import DEFAULT_CONFIG

        assert "budget_alert_threshold_pct" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["budget_alert_threshold_pct"] == 0.80
