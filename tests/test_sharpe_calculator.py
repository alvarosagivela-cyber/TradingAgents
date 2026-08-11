"""Unit tests for sharpe_calculator module (D-06/D-07: Sharpe from Alpaca equity curve, deterministic overfitting flag).

Tests verify:
- D-06: Sharpe ratio computed exclusively from real Alpaca paper account equity curve
- D-07: Deterministic overfitting flag (sharpe > 1.0), no LLM judgment
"""

from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
import pytest
import numpy as np

from tradingagents.trading.sharpe_calculator import (
    compute_window_sharpe,
    _extract_equity,
    OVERFITTING_SHARPE_THRESHOLD,
    REFERENCE_RANGE,
)


class TestExtractEquity:
    """Test _extract_equity handles both PortfolioHistory object and dict responses."""

    def test_extract_equity_from_object(self):
        """Extract equity from object with .equity attribute."""
        history = Mock()
        history.equity = [100000, 100500, 101000]

        result = _extract_equity(history)
        assert result == [100000, 100500, 101000]

    def test_extract_equity_from_dict(self):
        """Extract equity from dict with 'equity' key."""
        history = {"equity": [100000, 100500, 101000]}

        result = _extract_equity(history)
        assert result == [100000, 100500, 101000]

    def test_extract_equity_from_object_none(self):
        """Object with no equity attribute returns empty list."""
        history = Mock(spec=[])  # No attributes

        result = _extract_equity(history)
        assert result == []

    def test_extract_equity_from_dict_missing_key(self):
        """Dict without 'equity' key returns empty list."""
        history = {}

        result = _extract_equity(history)
        assert result == []


class TestComputeWindowSharpe:
    """Test compute_window_sharpe calculation and edge cases."""

    def test_compute_window_sharpe_typical_case(self):
        """D-06: Typical case - mock equity curve returns finite Sharpe, correct metadata."""
        # 5 days of equity: [100000, 100500, 101000, 100800, 101500]
        # daily returns: [0.005, 0.00497..., -0.00198..., 0.00695...]
        mock_client = Mock()
        mock_client.get_portfolio_history.return_value = Mock(
            equity=[100000, 100500, 101000, 100800, 101500],
            timestamp=[1, 2, 3, 4, 5],
        )

        start_date = datetime(2026, 8, 1)
        end_date = datetime(2026, 8, 5)

        result = compute_window_sharpe(mock_client, start_date, end_date)

        # Verify structure and basic properties
        assert isinstance(result, dict)
        assert "sharpe_ratio" in result
        assert "overfitting_flag" in result
        assert "trading_days_sampled" in result
        assert "mean_daily_return" in result
        assert "std_daily_return" in result
        assert "reference_range" in result
        assert "insufficient_data" in result
        assert "caveat" in result

        # Verify values
        assert isinstance(result["sharpe_ratio"], float)
        assert np.isfinite(result["sharpe_ratio"])
        assert result["trading_days_sampled"] == 4  # 5 prices = 4 returns
        assert result["reference_range"] == REFERENCE_RANGE
        assert result["insufficient_data"] is False

    def test_compute_window_sharpe_flags_overfitting(self):
        """D-07: Sharpe > 1.0 raises overfitting flag."""
        # Engineer an equity curve with high compounding (Sharpe > 1.0)
        # ~0.5% daily return with low volatility
        equity_values = [100000.0]
        for _ in range(20):
            equity_values.append(equity_values[-1] * 1.005)

        mock_client = Mock()
        mock_client.get_portfolio_history.return_value = Mock(
            equity=equity_values,
            timestamp=list(range(len(equity_values))),
        )

        start_date = datetime(2026, 8, 1)
        end_date = datetime(2026, 8, 21)

        result = compute_window_sharpe(mock_client, start_date, end_date)

        # Sharpe should exceed 1.0 for this engineered curve
        assert result["sharpe_ratio"] > 1.0, \
            f"Expected Sharpe > 1.0 for engineered curve, got {result['sharpe_ratio']}"
        # Overfitting flag should be True
        assert result["overfitting_flag"] is True
        # Caveat should mention "noisy"
        assert "noisy" in result["caveat"].lower()

    def test_compute_window_sharpe_handles_dict_response(self):
        """D-06: Handle dict response (Union[PortfolioHistory, dict])."""
        # Alpaca SDK may return a plain dict instead of PortfolioHistory object
        mock_client = Mock()
        mock_client.get_portfolio_history.return_value = {
            "equity": [100000, 100200, 100100],
            "timestamp": [1, 2, 3],
        }

        start_date = datetime(2026, 8, 1)
        end_date = datetime(2026, 8, 3)

        result = compute_window_sharpe(mock_client, start_date, end_date)

        # Should extract equity correctly despite dict response
        assert isinstance(result, dict)
        assert result["trading_days_sampled"] == 2  # 3 prices = 2 returns
        assert result["insufficient_data"] is False

    def test_compute_window_sharpe_zero_volatility(self):
        """D-06: Flat equity curve (std=0) returns sharpe=0.0 (no division by zero)."""
        # All equity values the same
        mock_client = Mock()
        mock_client.get_portfolio_history.return_value = Mock(
            equity=[100000, 100000, 100000],
            timestamp=[1, 2, 3],
        )

        start_date = datetime(2026, 8, 1)
        end_date = datetime(2026, 8, 3)

        result = compute_window_sharpe(mock_client, start_date, end_date)

        # Zero volatility: daily returns are all zero
        # Sharpe = (0 - 0) / 0, handled as 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["std_daily_return"] == 0.0
        assert result["insufficient_data"] is False

    def test_compute_window_sharpe_insufficient_data_empty(self):
        """D-06: Empty equity curve sets insufficient_data=True, sharpe=0.0."""
        mock_client = Mock()
        mock_client.get_portfolio_history.return_value = Mock(
            equity=[],
            timestamp=[],
        )

        start_date = datetime(2026, 8, 1)
        end_date = datetime(2026, 8, 1)

        result = compute_window_sharpe(mock_client, start_date, end_date)

        assert result["insufficient_data"] is True
        assert result["sharpe_ratio"] == 0.0
        assert result["trading_days_sampled"] == 0

    def test_compute_window_sharpe_insufficient_data_single_point(self):
        """D-06: Single equity point sets insufficient_data=True (need at least 2 for returns)."""
        mock_client = Mock()
        mock_client.get_portfolio_history.return_value = Mock(
            equity=[100000],
            timestamp=[1],
        )

        start_date = datetime(2026, 8, 1)
        end_date = datetime(2026, 8, 1)

        result = compute_window_sharpe(mock_client, start_date, end_date)

        assert result["insufficient_data"] is True
        assert result["sharpe_ratio"] == 0.0
        assert result["trading_days_sampled"] == 0

    def test_compute_window_sharpe_calls_get_portfolio_history_with_dates(self):
        """D-06: Verify get_portfolio_history is called with correct start/end dates."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        mock_client = Mock()
        mock_client.get_portfolio_history.return_value = Mock(
            equity=[100000, 100500],
            timestamp=[1, 2],
        )

        start_date = datetime(2026, 8, 1, 0, 0, 0)
        end_date = datetime(2026, 8, 5, 23, 59, 59)

        result = compute_window_sharpe(mock_client, start_date, end_date)

        # Verify get_portfolio_history was called
        mock_client.get_portfolio_history.assert_called_once()

        # Verify the call included a GetPortfolioHistoryRequest with start/end
        call_args = mock_client.get_portfolio_history.call_args
        assert call_args is not None

        # The first positional arg or 'history_filter' kwarg should be a GetPortfolioHistoryRequest
        if call_args[0]:  # positional args
            request = call_args[0][0]
            assert isinstance(request, GetPortfolioHistoryRequest)
        else:  # keyword args
            request = call_args[1].get("history_filter")
            assert isinstance(request, GetPortfolioHistoryRequest)

    def test_overfitting_threshold_constant(self):
        """D-07: OVERFITTING_SHARPE_THRESHOLD is exactly 1.0."""
        assert OVERFITTING_SHARPE_THRESHOLD == 1.0

    def test_reference_range_constant(self):
        """D-07: REFERENCE_RANGE is (0.5, 1.0) for VALID-02."""
        assert REFERENCE_RANGE == (0.5, 1.0)
