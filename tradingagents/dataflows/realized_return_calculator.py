"""Deterministic realized return calculator using actual trading-day row positions.

This module computes the realized return over a fixed number of trading days
(default 10) by row position in the OHLCV DataFrame, not by calendar-day or
weekday heuristics that would miscount around market holidays (D-02).

Key design:
- Single call to load_ohlcv with a lookahead date (~30 calendar days beyond decision_date)
- Row-position lookup to find the decision_date and window endpoint
- 6-decimal rounding to match D-07 precision convention
"""

from __future__ import annotations

import logging

import pandas as pd

from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.dataflows.symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)

# Phase 5: Default window for realized return measurement (D-02)
REALIZED_RETURN_WINDOW_DAYS = 10


def compute_realized_return(
    ticker: str, decision_date: str, window_days: int = REALIZED_RETURN_WINDOW_DAYS
) -> float:
    """Compute realized return over a fixed number of trading days.

    Fetches OHLCV data in a single call with a lookahead date (~30 calendar days
    beyond decision_date) to ensure the DataFrame includes enough future rows for
    the window measurement. Then locates the decision_date by row position and
    computes return to the row at position (decision_row_index + window_days).

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL")
        decision_date: Date the decision was made (YYYY-MM-DD format)
        window_days: Number of trading days to measure return over (default 10)

    Returns:
        Realized return as a float, rounded to 6 decimals (D-07 convention)
        Example: 0.1 for a 10% return, -0.03 for -3%

    Raises:
        NoMarketDataError: If decision_date is not found in the data, or if
            there are fewer than window_days trading rows remaining after the
            decision_date, or if the close price at decision_date is zero
    """
    # Compute a lookahead date (30 calendar days is comfortably more than 10 trading days
    # even across multiple holidays)
    lookahead_date = (pd.to_datetime(decision_date) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    # Fetch OHLCV data in a single call
    data = load_ohlcv(ticker, lookahead_date)

    # Normalize the Date column to datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Find rows on or after the decision_date
    decision_date_dt = pd.to_datetime(decision_date)
    matching_rows = data[data["Date"] >= decision_date_dt]

    if matching_rows.empty:
        raise NoMarketDataError(
            ticker,
            None,
            f"No OHLCV on or after {decision_date}",
        )

    # Verify the first matching row is close to decision_date (same day or within 1 trading day)
    # This catches cases where the data simply doesn't include the decision_date
    actual_date = pd.to_datetime(matching_rows.iloc[0]["Date"])
    days_diff = (actual_date - decision_date_dt).days

    # Allow 0 days (exact match) or up to 2 days difference to account for weekends
    if days_diff > 2:
        raise NoMarketDataError(
            ticker,
            None,
            f"Decision date {decision_date} not found; earliest available data is {actual_date.strftime('%Y-%m-%d')}",
        )

    # Get the row index position of the matched row
    decision_row_pos = data.index.get_loc(matching_rows.index[0])

    # Check if we have enough rows for the window
    if decision_row_pos + window_days >= len(data):
        available = len(data) - decision_row_pos - 1
        raise NoMarketDataError(
            ticker,
            None,
            f"Insufficient trading days elapsed since {decision_date}: need {window_days} more rows, only {available} available",
        )

    # Get close prices at start and end of window
    close_start = float(data["Close"].iloc[decision_row_pos])
    close_end = float(data["Close"].iloc[decision_row_pos + window_days])

    # Guard against division by zero
    if close_start == 0:
        raise NoMarketDataError(
            ticker,
            None,
            f"Close price is zero at {decision_date}",
        )

    # Compute return and round to 6 decimals (D-07 convention)
    realized_return = (close_end - close_start) / close_start
    return round(realized_return, 6)
