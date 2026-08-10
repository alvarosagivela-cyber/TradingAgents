"""Unit tests for cost-tracking layer/ticker/trade_date tagging at call sites (Plan 06-04).

This test suite verifies that each real LLM call site (Auditor, Conservative, Balanced, Aggressive,
and the shared graph-level clients) passes layer/ticker/trade_date kwargs through to create_llm_client(),
enabling transparent cost attribution without breaking isolation.
"""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.auditors.momentum_auditor import create_auditor_llm_node
from tradingagents.agents.risk_mgmt.conservative_perspective import create_conservative_perspective
from tradingagents.agents.risk_mgmt.balanced_perspective import create_balanced_perspective
from tradingagents.agents.risk_mgmt.aggressive_perspective import create_aggressive_perspective
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
class TestAuditorCostTagging:
    """Verify Auditor LLM node passes layer/ticker/trade_date to create_llm_client()."""

    def test_auditor_tags_layer_ticker_trade_date(self):
        """Auditor's create_llm_client call includes layer='auditor', ticker, and trade_date."""
        llm_node = create_auditor_llm_node()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-08-05",
            "auditor_phase1_status": "pass",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": 2.3,
            "thesis_verdict": "Buy",
        }

        with patch("tradingagents.agents.auditors.momentum_auditor.create_llm_client") as mock_client_factory:
            # Mock LLM client
            mock_client = MagicMock()
            mock_llm = MagicMock()
            mock_client.get_llm.return_value = mock_llm
            mock_client_factory.return_value = mock_client

            with patch("tradingagents.agents.auditors.momentum_auditor.bind_structured") as mock_bind:
                mock_structured_llm = MagicMock()
                mock_verdict = MagicMock()
                mock_verdict.verdict.value = "HOLD"
                mock_verdict.reasoning = "Test"
                mock_verdict.refutation_criterion = "Test"
                mock_verdict.data_points = {"retorno_12_1": 0.1834, "z_score": 2.3}
                mock_structured_llm.invoke.return_value = mock_verdict
                mock_bind.return_value = mock_structured_llm

                # Invoke the node
                llm_node(state)

                # Verify create_llm_client was called with correct tags
                mock_client_factory.assert_called_once()
                call_kwargs = mock_client_factory.call_args[1]
                assert call_kwargs.get("layer") == "auditor", "Auditor should tag with layer='auditor'"
                assert call_kwargs.get("ticker") == "AAPL", "Auditor should pass ticker='AAPL'"
                assert call_kwargs.get("trade_date") == "2026-08-05", "Auditor should pass trade_date"


@pytest.mark.unit
class TestConservativeCostTagging:
    """Verify Conservative perspective passes layer/ticker/trade_date to create_llm_client()."""

    def test_conservative_tags_layer_ticker_trade_date(self):
        """Conservative's create_llm_client call includes layer='conservative', ticker, and trade_date."""
        conservative_node = create_conservative_perspective()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-08-05",
            "proposed_side": "Buy",
            "portfolio_total_value": 100000.0,
            "existing_position_value": 0.0,
            "proposed_notional_usd": 2000.0,
            "risk_concentration_pct": 0.02,
        }

        with patch("tradingagents.agents.risk_mgmt.conservative_perspective.create_llm_client") as mock_client_factory:
            # Mock LLM client
            mock_client = MagicMock()
            mock_llm = MagicMock()
            mock_client.get_llm.return_value = mock_llm
            mock_client_factory.return_value = mock_client

            with patch("tradingagents.agents.risk_mgmt.conservative_perspective.bind_structured") as mock_bind:
                mock_structured_llm = MagicMock()
                mock_verdict = MagicMock()
                mock_verdict.model_dump.return_value = {
                    "verdict": "APPROVE",
                    "confidence": 0.85,
                    "reasoning": "Test",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                }
                mock_structured_llm.invoke.return_value = mock_verdict
                mock_bind.return_value = mock_structured_llm

                # Invoke the node
                conservative_node(state)

                # Verify create_llm_client was called with correct tags
                mock_client_factory.assert_called_once()
                call_kwargs = mock_client_factory.call_args[1]
                assert call_kwargs.get("layer") == "conservative", "Conservative should tag with layer='conservative'"
                assert call_kwargs.get("ticker") == "AAPL", "Conservative should pass ticker='AAPL'"
                assert call_kwargs.get("trade_date") == "2026-08-05", "Conservative should pass trade_date"


@pytest.mark.unit
class TestBalancedCostTagging:
    """Verify Balanced perspective passes layer/ticker/trade_date to create_llm_client()."""

    def test_balanced_tags_layer_ticker_trade_date(self):
        """Balanced's create_llm_client call includes layer='balanced', ticker, and trade_date."""
        balanced_node = create_balanced_perspective()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-08-05",
            "proposed_side": "Buy",
            "portfolio_total_value": 100000.0,
            "existing_position_value": 0.0,
            "proposed_notional_usd": 2000.0,
            "risk_concentration_pct": 0.02,
        }

        with patch("tradingagents.agents.risk_mgmt.balanced_perspective.create_llm_client") as mock_client_factory:
            # Mock LLM client
            mock_client = MagicMock()
            mock_llm = MagicMock()
            mock_client.get_llm.return_value = mock_llm
            mock_client_factory.return_value = mock_client

            with patch("tradingagents.agents.risk_mgmt.balanced_perspective.bind_structured") as mock_bind:
                mock_structured_llm = MagicMock()
                mock_verdict = MagicMock()
                mock_verdict.model_dump.return_value = {
                    "verdict": "APPROVE",
                    "confidence": 0.80,
                    "reasoning": "Test",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                }
                mock_structured_llm.invoke.return_value = mock_verdict
                mock_bind.return_value = mock_structured_llm

                # Invoke the node
                balanced_node(state)

                # Verify create_llm_client was called with correct tags
                mock_client_factory.assert_called_once()
                call_kwargs = mock_client_factory.call_args[1]
                assert call_kwargs.get("layer") == "balanced", "Balanced should tag with layer='balanced'"
                assert call_kwargs.get("ticker") == "AAPL", "Balanced should pass ticker='AAPL'"
                assert call_kwargs.get("trade_date") == "2026-08-05", "Balanced should pass trade_date"


@pytest.mark.unit
class TestAggressiveCostTagging:
    """Verify Aggressive perspective passes layer/ticker/trade_date to create_llm_client()."""

    def test_aggressive_tags_layer_ticker_trade_date(self):
        """Aggressive's create_llm_client call includes layer='aggressive', ticker, and trade_date."""
        aggressive_node = create_aggressive_perspective()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-08-05",
            "proposed_side": "Buy",
            "portfolio_total_value": 100000.0,
            "existing_position_value": 0.0,
            "proposed_notional_usd": 2000.0,
            "risk_concentration_pct": 0.02,
        }

        with patch("tradingagents.agents.risk_mgmt.aggressive_perspective.create_llm_client") as mock_client_factory:
            # Mock LLM client
            mock_client = MagicMock()
            mock_llm = MagicMock()
            mock_client.get_llm.return_value = mock_llm
            mock_client_factory.return_value = mock_client

            with patch("tradingagents.agents.risk_mgmt.aggressive_perspective.bind_structured") as mock_bind:
                mock_structured_llm = MagicMock()
                mock_verdict = MagicMock()
                mock_verdict.model_dump.return_value = {
                    "verdict": "APPROVE",
                    "confidence": 0.75,
                    "reasoning": "Test",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                }
                mock_structured_llm.invoke.return_value = mock_verdict
                mock_bind.return_value = mock_structured_llm

                # Invoke the node
                aggressive_node(state)

                # Verify create_llm_client was called with correct tags
                mock_client_factory.assert_called_once()
                call_kwargs = mock_client_factory.call_args[1]
                assert call_kwargs.get("layer") == "aggressive", "Aggressive should tag with layer='aggressive'"
                assert call_kwargs.get("ticker") == "AAPL", "Aggressive should pass ticker='AAPL'"
                assert call_kwargs.get("trade_date") == "2026-08-05", "Aggressive should pass trade_date"


@pytest.mark.unit
class TestSharedClientLayerTagging:
    """Verify graph-level shared clients are tagged with 'deep_thinking_shared' and 'quick_thinking_shared'."""

    def test_create_llm_clients_tags_shared_clients(self):
        """_create_llm_clients() tags deep and quick clients with _shared layer labels."""
        # Use _bare_graph pattern from test_llm_max_retries.py
        graph = object.__new__(TradingAgentsGraph)
        graph.config = {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-5.5",
            "quick_think_llm": "gpt-5.4-mini",
            "backend_url": None,
        }

        with patch("tradingagents.graph.trading_graph.create_llm_client") as mock_client_factory:
            # Mock LLM client
            mock_client = MagicMock()
            mock_llm = MagicMock()
            mock_client.get_llm.return_value = mock_llm
            mock_client_factory.return_value = mock_client

            # Call _create_llm_clients
            deep_client, quick_client = graph._create_llm_clients({})

            # Verify create_llm_client was called exactly twice
            assert mock_client_factory.call_count == 2, "Should call create_llm_client exactly twice"

            # Extract the two calls
            calls = mock_client_factory.call_args_list
            deep_call_kwargs = calls[0][1]
            quick_call_kwargs = calls[1][1]

            # Verify layer tags
            assert deep_call_kwargs.get("layer") == "deep_thinking_shared", (
                "Deep client should be tagged with layer='deep_thinking_shared'"
            )
            assert quick_call_kwargs.get("layer") == "quick_thinking_shared", (
                "Quick client should be tagged with layer='quick_thinking_shared'"
            )
