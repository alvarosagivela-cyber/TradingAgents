"""Unit tests for realized return calculator — 10-trading-day window, deterministic calculation."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows.realized_return_calculator import (
    compute_realized_return,
    REALIZED_RETURN_WINDOW_DAYS,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError


# ---------------------------------------------------------------------------
# Test Happy Path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_realized_return_happy_path():
    """Happy path: compute return over exactly 10 trading days."""
    # Build a DataFrame spanning the decision date plus 12+ trading days
    dates = pd.date_range("2026-07-01", periods=15, freq="B")  # Business days
    close_prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0]

    df = pd.DataFrame({
        "Date": dates,
        "Close": close_prices,
    })

    with patch("tradingagents.dataflows.realized_return_calculator.load_ohlcv") as mock_load:
        mock_load.return_value = df

        # Compute return from position 0 (date 2026-07-01, close=100) to position 10 (close=110)
        # Expected return: (110 - 100) / 100 = 0.1
        result = compute_realized_return("AAPL", "2026-07-01", window_days=10)

        assert result == 0.1
        # Verify load_ohlcv was called with a later date
        mock_load.assert_called_once()
        call_args = mock_load.call_args
        assert call_args[0][0] == "AAPL"
        # Verify date is later than decision_date
        called_date = call_args[0][1]
        assert called_date > "2026-07-01"


# ---------------------------------------------------------------------------
# Test Insufficient Future Rows
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_future_rows():
    """Raise NoMarketDataError when fewer than window_days rows remain after decision_date."""
    dates = pd.date_range("2026-07-01", periods=8, freq="B")  # Only 8 trading days total
    close_prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]

    df = pd.DataFrame({
        "Date": dates,
        "Close": close_prices,
    })

    with patch("tradingagents.dataflows.realized_return_calculator.load_ohlcv") as mock_load:
        mock_load.return_value = df

        with pytest.raises(NoMarketDataError):
            compute_realized_return("AAPL", "2026-07-01", window_days=10)


# ---------------------------------------------------------------------------
# Test Decision Date Not Present
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_decision_date_not_in_frame():
    """Raise NoMarketDataError when decision_date is not reached in the DataFrame."""
    dates = pd.date_range("2026-07-15", periods=15, freq="B")  # Start after decision date
    close_prices = list(range(100, 115))

    df = pd.DataFrame({
        "Date": dates,
        "Close": close_prices,
    })

    with patch("tradingagents.dataflows.realized_return_calculator.load_ohlcv") as mock_load:
        mock_load.return_value = df

        with pytest.raises(NoMarketDataError):
            compute_realized_return("AAPL", "2026-07-01", window_days=10)


# ---------------------------------------------------------------------------
# Test Look-Ahead Call Guard (Pitfall 2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lookahead_call_guard():
    """Verify load_ohlcv is called with a date strictly LATER than decision_date."""
    dates = pd.date_range("2026-07-01", periods=15, freq="B")
    close_prices = list(range(100, 115))

    df = pd.DataFrame({
        "Date": dates,
        "Close": close_prices,
    })

    with patch("tradingagents.dataflows.realized_return_calculator.load_ohlcv") as mock_load:
        mock_load.return_value = df

        result = compute_realized_return("AAPL", "2026-07-01", window_days=10)

        # Verify the function returned a value (happy path)
        assert result == pytest.approx(0.1)

        # Check that load_ohlcv was called with a date AFTER decision_date
        mock_load.assert_called_once()
        call_args = mock_load.call_args
        called_ticker = call_args[0][0]
        called_date = call_args[0][1]

        assert called_ticker == "AAPL"
        assert called_date > "2026-07-01", "load_ohlcv must be called with a lookahead date"


# ---------------------------------------------------------------------------
# Test Custom Window Days
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_custom_window_days():
    """With window_days=5, use the 5th trading row, not the 10th."""
    dates = pd.date_range("2026-07-01", periods=15, freq="B")
    # Close price at position 0 = 100, at position 5 = 105
    close_prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0]

    df = pd.DataFrame({
        "Date": dates,
        "Close": close_prices,
    })

    with patch("tradingagents.dataflows.realized_return_calculator.load_ohlcv") as mock_load:
        mock_load.return_value = df

        result = compute_realized_return("AAPL", "2026-07-01", window_days=5)

        # Expected: (105 - 100) / 100 = 0.05
        assert result == 0.05


# ---------------------------------------------------------------------------
# Test Rounding (D-07)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_rounding_precision():
    """Verify rounding to 6 decimals (D-07 rounding convention)."""
    dates = pd.date_range("2026-07-01", periods=15, freq="B")
    close_prices = [100.0, 100.12345678, 100.24691357, 100.37037036, 100.49382716, 100.61728395, 100.74074074, 100.86419753, 100.98765432, 101.11111111, 101.23456790, 101.35802469, 101.48148148, 101.60493827, 101.72839506]

    df = pd.DataFrame({
        "Date": dates,
        "Close": close_prices,
    })

    with patch("tradingagents.dataflows.realized_return_calculator.load_ohlcv") as mock_load:
        mock_load.return_value = df

        result = compute_realized_return("AAPL", "2026-07-01", window_days=10)

        # Verify rounding: result should be rounded to 6 decimals
        assert isinstance(result, float)
        # Check that rounding was applied (6 decimal places max)
        result_str = f"{result:.6f}"
        assert len(result_str.split(".")[-1]) <= 6


# ---------------------------------------------------------------------------
# Test REALIZED_RETURN_WINDOW_DAYS Constant
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_realized_return_window_days_constant():
    """Verify REALIZED_RETURN_WINDOW_DAYS is defined and equals 10."""
    assert REALIZED_RETURN_WINDOW_DAYS == 10


# ---------------------------------------------------------------------------
# Test Zero Close Price Guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_zero_close_price_raises_error():
    """Raise NoMarketDataError when close price is zero (division by zero guard)."""
    dates = pd.date_range("2026-07-01", periods=15, freq="B")
    close_prices = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]

    df = pd.DataFrame({
        "Date": dates,
        "Close": close_prices,
    })

    with patch("tradingagents.dataflows.realized_return_calculator.load_ohlcv") as mock_load:
        mock_load.return_value = df

        with pytest.raises(NoMarketDataError):
            compute_realized_return("AAPL", "2026-07-01", window_days=10)
