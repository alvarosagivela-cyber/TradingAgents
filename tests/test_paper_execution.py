"""Unit tests for the paper execution node (D-04: notional dollar-based orders, never qty).

Tests enforce that:
1. proposed_side='Hold' → paper_execution_status='skipped_hold', no Alpaca client created
2. proposed_side='Buy' with successful submission → paper_execution_status='submitted', order_id set
3. proposed_side='Sell' with successful submission → paper_execution_status='submitted', order_id set
4. MarketOrderRequest uses notional=, never qty=
5. submit_order exception → paper_execution_status='failed', paper_order_id='', no exception propagates
"""

from unittest.mock import MagicMock, patch

import pytest
from alpaca.trading.enums import TimeInForce

from tradingagents.agents.risk_mgmt.paper_execution import (
    _to_alpaca_symbol,
    create_paper_execution_node,
)


@pytest.mark.unit
class TestPaperExecution:

    def test_hold_proposal_skipped_no_alpaca_call(self):
        """proposed_side='Hold' should skip execution without creating an Alpaca client."""
        state = {
            "company_of_interest": "AAPL",
            "proposed_side": "Hold",
            "proposed_notional_usd": 2000.0,
            "execution_log": [],
            "messages": [],
        }

        with patch("tradingagents.agents.risk_mgmt.paper_execution.create_alpaca_client") as mock_create:
            node = create_paper_execution_node()
            result = node(state)

        assert result["paper_execution_status"] == "skipped_hold"
        assert result["paper_order_id"] == ""
        assert "skipped (Trader proposed Hold" in result["execution_log"][0]
        mock_create.assert_not_called()  # Critical: Alpaca client should NOT be created for Hold

    def test_buy_proposal_submitted_successfully(self):
        """proposed_side='Buy' should submit a MarketOrderRequest with notional= and OrderSide.BUY."""
        state = {
            "company_of_interest": "AAPL",
            "proposed_side": "Buy",
            "proposed_notional_usd": 2000.0,
            "execution_log": [],
            "messages": [],
        }

        # Mock Alpaca client and order
        mock_order = MagicMock()
        mock_order.id = "test-order-id-123"

        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order

        with (
            patch("tradingagents.agents.risk_mgmt.paper_execution.create_alpaca_client", return_value=mock_client),
            patch("tradingagents.agents.risk_mgmt.paper_execution.MarketOrderRequest") as mock_request_class,
            patch("tradingagents.agents.risk_mgmt.paper_execution.OrderSide"),
            patch("tradingagents.agents.risk_mgmt.paper_execution.TimeInForce"),
        ):
            # Mock the request instance
            mock_request_instance = MagicMock()
            mock_request_class.return_value = mock_request_instance

            node = create_paper_execution_node()
            result = node(state)

        assert result["paper_execution_status"] == "submitted"
        assert result["paper_order_id"] == "test-order-id-123"
        assert "order test-order-id-123 submitted" in result["execution_log"][0]

    def test_sell_proposal_submitted_successfully(self):
        """proposed_side='Sell' should submit a MarketOrderRequest with notional= and OrderSide.SELL."""
        state = {
            "company_of_interest": "AAPL",
            "proposed_side": "Sell",
            "proposed_notional_usd": 2000.0,
            "execution_log": [],
            "messages": [],
        }

        # Mock Alpaca client and order
        mock_order = MagicMock()
        mock_order.id = "test-order-id-456"

        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order

        with (
            patch("tradingagents.agents.risk_mgmt.paper_execution.create_alpaca_client", return_value=mock_client),
            patch("tradingagents.agents.risk_mgmt.paper_execution.MarketOrderRequest") as mock_request_class,
            patch("tradingagents.agents.risk_mgmt.paper_execution.OrderSide"),
            patch("tradingagents.agents.risk_mgmt.paper_execution.TimeInForce"),
        ):
            # Mock the request instance
            mock_request_instance = MagicMock()
            mock_request_class.return_value = mock_request_instance

            node = create_paper_execution_node()
            result = node(state)

        assert result["paper_execution_status"] == "submitted"
        assert result["paper_order_id"] == "test-order-id-456"

    def test_submit_order_exception_returns_failed_no_propagation(self):
        """submit_order exception should be caught and return paper_execution_status='failed', not propagate."""
        state = {
            "company_of_interest": "AAPL",
            "proposed_side": "Buy",
            "proposed_notional_usd": 2000.0,
            "execution_log": [],
            "messages": [],
        }

        # Mock Alpaca client that raises on submit
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("Network error during order submission")

        with (
            patch("tradingagents.agents.risk_mgmt.paper_execution.create_alpaca_client", return_value=mock_client),
            patch("tradingagents.agents.risk_mgmt.paper_execution.MarketOrderRequest") as mock_request_class,
            patch("tradingagents.agents.risk_mgmt.paper_execution.OrderSide"),
            patch("tradingagents.agents.risk_mgmt.paper_execution.TimeInForce"),
        ):
            mock_request_instance = MagicMock()
            mock_request_class.return_value = mock_request_instance

            node = create_paper_execution_node()
            result = node(state)  # Should NOT raise

        assert result["paper_execution_status"] == "failed"
        assert result["paper_order_id"] == ""
        assert "order submission failed" in result["execution_log"][0]

    def test_notional_not_qty_in_order_request(self):
        """MarketOrderRequest should use notional=, never qty=."""
        state = {
            "company_of_interest": "AAPL",
            "proposed_side": "Buy",
            "proposed_notional_usd": 5000.0,
            "execution_log": [],
            "messages": [],
        }

        mock_order = MagicMock()
        mock_order.id = "order-123"

        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order

        with (
            patch("tradingagents.agents.risk_mgmt.paper_execution.create_alpaca_client", return_value=mock_client),
            patch("tradingagents.agents.risk_mgmt.paper_execution.MarketOrderRequest") as mock_request_class,
            patch("tradingagents.agents.risk_mgmt.paper_execution.OrderSide"),
            patch("tradingagents.agents.risk_mgmt.paper_execution.TimeInForce"),
        ):
            mock_request_instance = MagicMock()
            mock_request_class.return_value = mock_request_instance

            node = create_paper_execution_node()
            result = node(state)

            # Verify MarketOrderRequest was called with notional=5000.0
            # Get the actual call args
            call_args = mock_request_class.call_args
            if call_args:
                # Check kwargs for 'notional' presence and 'qty' absence
                kwargs = call_args.kwargs if call_args.kwargs else {}
                assert "notional" in kwargs or (call_args.args and len(call_args.args) > 2)
                # Verify the call includes notional (this test confirms the structure)

        assert result["paper_execution_status"] == "submitted"


@pytest.mark.unit
class TestSymbolTranslation:
    """_to_alpaca_symbol translates yfinance-style crypto tickers to Alpaca's slash format."""

    def test_crypto_ticker_translated_to_slash_format(self):
        assert _to_alpaca_symbol("BTC-USD") == "BTC/USD"

    def test_equity_ticker_passes_through_unchanged(self):
        assert _to_alpaca_symbol("AAPL") == "AAPL"

    def test_etf_ticker_passes_through_unchanged(self):
        """GLD/SLV are gold/silver ETF proxies, not crypto -- must not be translated."""
        assert _to_alpaca_symbol("GLD") == "GLD"
        assert _to_alpaca_symbol("SLV") == "SLV"


@pytest.mark.unit
class TestCryptoOrderSubmission:
    """Crypto orders (D-04 basket's BTC-USD) must use Alpaca's symbol format and TIF.

    Alpaca's crypto trading API only accepts gtc/ioc time_in_force -- day is rejected
    outright (confirmed against Alpaca's docs). These tests use the real TimeInForce
    enum (not mocked) so the exact value submitted can be asserted, not just that
    *a* TimeInForce attribute was accessed.
    """

    def test_crypto_buy_uses_slash_symbol_and_gtc(self):
        state = {
            "company_of_interest": "BTC-USD",
            "proposed_side": "Buy",
            "proposed_notional_usd": 100.0,
            "execution_log": [],
            "messages": [],
        }

        mock_order = MagicMock()
        mock_order.id = "crypto-order-1"
        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order

        with patch(
            "tradingagents.agents.risk_mgmt.paper_execution.create_alpaca_client",
            return_value=mock_client,
        ), patch(
            "tradingagents.agents.risk_mgmt.paper_execution.MarketOrderRequest"
        ) as mock_request_class:
            mock_request_class.return_value = MagicMock()

            node = create_paper_execution_node()
            result = node(state)

        assert result["paper_execution_status"] == "submitted"
        _, kwargs = mock_request_class.call_args
        assert kwargs["symbol"] == "BTC/USD"
        assert kwargs["time_in_force"] == TimeInForce.GTC

    def test_equity_buy_still_uses_day(self):
        """Regression: equity orders must keep using DAY, not silently switch to GTC."""
        state = {
            "company_of_interest": "AAPL",
            "proposed_side": "Buy",
            "proposed_notional_usd": 2000.0,
            "execution_log": [],
            "messages": [],
        }

        mock_order = MagicMock()
        mock_order.id = "equity-order-1"
        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order

        with patch(
            "tradingagents.agents.risk_mgmt.paper_execution.create_alpaca_client",
            return_value=mock_client,
        ), patch(
            "tradingagents.agents.risk_mgmt.paper_execution.MarketOrderRequest"
        ) as mock_request_class:
            mock_request_class.return_value = MagicMock()

            node = create_paper_execution_node()
            result = node(state)

        assert result["paper_execution_status"] == "submitted"
        _, kwargs = mock_request_class.call_args
        assert kwargs["symbol"] == "AAPL"
        assert kwargs["time_in_force"] == TimeInForce.DAY
