"""Unit tests for the portfolio snapshot node (RISK-01: real Alpaca account data, not LLM-estimated).

Tests enforce that:
1. Portfolio snapshot computes portfolio_total_value, existing_position_value, proposed_notional_usd, and risk_concentration_pct
   from the real Alpaca paper account (never LLM-estimated)
2. Missing position for the ticker returns existing_position_value=0.0 (no exception)
3. Alpaca account fetch failure returns the fail-safe sentinel (portfolio_snapshot_status='fail',
   risk_concentration_pct=1.0) to trigger a deterministic veto in the Risk Squad aggregator
4. proposed_side is correctly derived via parse_trader_action from the trader_investment_plan
"""

from unittest.mock import MagicMock, patch
import pytest
from tradingagents.trading.portfolio_snapshot import create_portfolio_snapshot_node


@pytest.mark.unit
class TestPortfolioSnapshot:

    def test_happy_path_with_account_and_no_position(self):
        """Happy path: account has equity, no existing position for the ticker."""
        state = {
            "company_of_interest": "AAPL",
            "trader_investment_plan": "Analysis complete. FINAL TRANSACTION PROPOSAL: **BUY**",
            "messages": [],
        }

        # Mock Alpaca client
        mock_account = MagicMock()
        mock_account.equity = "100000.00"

        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_account
        mock_client.get_open_position.side_effect = Exception("404: No position")

        with patch("tradingagents.trading.portfolio_snapshot.create_alpaca_client", return_value=mock_client):
            node = create_portfolio_snapshot_node()
            result = node(state)

        assert result["portfolio_snapshot_status"] == "pass"
        assert result["portfolio_total_value"] == 100000.0
        assert result["existing_position_value"] == 0.0
        assert result["proposed_notional_usd"] == 2000.0  # 0.02 * 100000
        assert result["risk_concentration_pct"] == 0.02
        assert result["proposed_side"] == "Buy"

    def test_happy_path_with_existing_position(self):
        """Happy path: account has equity, existing position exists."""
        state = {
            "company_of_interest": "AAPL",
            "trader_investment_plan": "Analysis complete. FINAL TRANSACTION PROPOSAL: **SELL**",
            "messages": [],
        }

        # Mock Alpaca client
        mock_account = MagicMock()
        mock_account.equity = "100000.00"

        mock_position = MagicMock()
        mock_position.market_value = "5000.00"

        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_account
        mock_client.get_open_position.return_value = mock_position

        with patch("tradingagents.trading.portfolio_snapshot.create_alpaca_client", return_value=mock_client):
            node = create_portfolio_snapshot_node()
            result = node(state)

        assert result["portfolio_snapshot_status"] == "pass"
        assert result["portfolio_total_value"] == 100000.0
        assert result["existing_position_value"] == 5000.0
        assert result["proposed_notional_usd"] == 2000.0
        assert result["risk_concentration_pct"] == 0.07  # (5000 + 2000) / 100000
        assert result["proposed_side"] == "Sell"

    def test_account_fetch_failure_returns_fail_safe(self):
        """Account fetch failure should return fail-safe sentinel (status='fail', concentration=1.0)."""
        state = {
            "company_of_interest": "AAPL",
            "trader_investment_plan": "Analysis complete. FINAL TRANSACTION PROPOSAL: **HOLD**",
            "messages": [],
        }

        # Mock Alpaca client that fails on account fetch
        mock_client = MagicMock()
        mock_client.get_account.side_effect = Exception("Network error")

        with patch("tradingagents.trading.portfolio_snapshot.create_alpaca_client", return_value=mock_client):
            node = create_portfolio_snapshot_node()
            result = node(state)

        assert result["portfolio_snapshot_status"] == "fail"
        assert result["portfolio_total_value"] == 0.0
        assert result["existing_position_value"] == 0.0
        assert result["proposed_notional_usd"] == 0.0
        assert result["risk_concentration_pct"] == 1.0  # Fail-safe sentinel
        assert result["proposed_side"] == "Hold"  # Still correctly parsed, even on failure

    def test_missing_position_treated_as_zero(self):
        """Missing position (any exception from get_open_position) should be treated as 0.0, not crashing."""
        state = {
            "company_of_interest": "AAPL",
            "trader_investment_plan": "Analysis complete. FINAL TRANSACTION PROPOSAL: **BUY**",
            "messages": [],
        }

        # Mock Alpaca client
        mock_account = MagicMock()
        mock_account.equity = "50000.00"

        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_account
        mock_client.get_open_position.side_effect = Exception("404: Position not found")

        with patch("tradingagents.trading.portfolio_snapshot.create_alpaca_client", return_value=mock_client):
            node = create_portfolio_snapshot_node()
            result = node(state)

        assert result["portfolio_snapshot_status"] == "pass"  # Node did not crash
        assert result["existing_position_value"] == 0.0

    def test_proposed_side_parsed_correctly_buy(self):
        """proposed_side should be parsed from trader_investment_plan via parse_trader_action."""
        state = {
            "company_of_interest": "AAPL",
            "trader_investment_plan": "Some reasoning. FINAL TRANSACTION PROPOSAL: **BUY**",
            "messages": [],
        }

        mock_account = MagicMock()
        mock_account.equity = "100000.00"

        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_account
        mock_client.get_open_position.side_effect = Exception()

        with patch("tradingagents.trading.portfolio_snapshot.create_alpaca_client", return_value=mock_client):
            node = create_portfolio_snapshot_node()
            result = node(state)

        assert result["proposed_side"] == "Buy"

    def test_proposed_side_parsed_correctly_hold(self):
        """proposed_side should be Hold when FINAL TRANSACTION PROPOSAL says HOLD."""
        state = {
            "company_of_interest": "AAPL",
            "trader_investment_plan": "Some reasoning. FINAL TRANSACTION PROPOSAL: **HOLD**",
            "messages": [],
        }

        mock_account = MagicMock()
        mock_account.equity = "100000.00"

        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_account
        mock_client.get_open_position.side_effect = Exception()

        with patch("tradingagents.trading.portfolio_snapshot.create_alpaca_client", return_value=mock_client):
            node = create_portfolio_snapshot_node()
            result = node(state)

        assert result["proposed_side"] == "Hold"
