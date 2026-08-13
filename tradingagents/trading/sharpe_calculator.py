"""Sharpe ratio calculator from Alpaca portfolio history (D-06/D-07).

D-06: The Sharpe ratio is computed exclusively from the real Alpaca paper account's
equity curve via TradingClient.get_portfolio_history() — never reconstructed from
internal Research/Auditor/Risk JSONL logs.

D-07: The overfitting flag is a deterministic Python comparison (sharpe_ratio > 1.0) —
no LLM is invoked anywhere in this module. This implements VALID-02 and is consistent
with D-11 discipline (never let an LLM interpret something Python can decide with a
fixed rule).

Key design choices:
- Handles both PortfolioHistory objects and dict responses (Union return type from SDK)
- Zero-volatility edge case returns sharpe=0.0 (no division by zero)
- Insufficient data (<2 points) sets insufficient_data=True
- Caveat text explicitly warns about small-sample overfitting risk
"""

import logging
from datetime import datetime

import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

logger = logging.getLogger(__name__)

# D-07: Deterministic overfitting flag threshold
OVERFITTING_SHARPE_THRESHOLD: float = 1.0

# VALID-02: Reference range for interpreting Sharpe ratio
REFERENCE_RANGE: tuple[float, float] = (0.5, 1.0)

# Standard annualization factor (252 trading days per year)
TRADING_DAYS_PER_YEAR: int = 252


def _extract_equity(history: object | dict) -> list[float]:
    """Extract equity array from Alpaca response (handles both object and dict types).

    The Alpaca SDK's get_portfolio_history() returns Union[PortfolioHistory, dict].
    This function defensively handles both shapes.

    Args:
        history: Alpaca response (could be PortfolioHistory object or dict)

    Returns:
        List of equity floats, or empty list if extraction fails
    """
    # Try dict access first (Union typing suggests dict is possible)
    if isinstance(history, dict):
        equity = history.get("equity")
    else:
        # Try object attribute access
        equity = getattr(history, "equity", None)

    if equity is None:
        return []

    # A brand-new paper account can have None entries for days before funding.
    # Drop them rather than let a downstream np.array(dtype=float) raise.
    return [value for value in equity if value is not None]


def compute_window_sharpe(
    client: TradingClient,
    start_date: datetime,
    end_date: datetime,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute annualized Sharpe ratio from Alpaca paper account equity curve (D-06/D-07).

    Retrieves the daily portfolio equity from Alpaca's /v2/portfolio/history endpoint,
    computes daily returns, and calculates the annualized Sharpe ratio. Flags potential
    overfitting when Sharpe > 1.0 (D-07).

    Args:
        client: Alpaca TradingClient (paper=True)
        start_date: Window start (datetime)
        end_date: Window end (datetime)
        risk_free_rate: Daily risk-free rate for Sharpe calculation (default 0.0 for paper trading)

    Returns:
        Dict with keys:
        - sharpe_ratio: Annualized Sharpe ratio (float)
        - overfitting_flag: True if sharpe_ratio > OVERFITTING_SHARPE_THRESHOLD (bool)
        - trading_days_sampled: Number of daily returns computed (int)
        - mean_daily_return: Mean of daily returns (float)
        - std_daily_return: Standard deviation of daily returns (float)
        - reference_range: (0.5, 1.0) for VALID-02 context (tuple)
        - insufficient_data: True if < 2 equity points (bool)
        - caveat: Explanatory text about small-sample noise and overfitting risk (str)
    """
    try:
        # D-06: Fetch historical equity from Alpaca paper account
        history_filter = GetPortfolioHistoryRequest(
            start=start_date,
            end=end_date,
            timeframe="1D",  # Daily data
        )
        history = client.get_portfolio_history(history_filter=history_filter)

        # Defensively extract equity array (handles Union[PortfolioHistory, dict])
        equity = _extract_equity(history)

        # Edge case: insufficient data (need at least 2 points to compute 1 return)
        if len(equity) < 2:
            return {
                "sharpe_ratio": 0.0,
                "overfitting_flag": False,
                "trading_days_sampled": 0,
                "mean_daily_return": 0.0,
                "std_daily_return": 0.0,
                "reference_range": REFERENCE_RANGE,
                "insufficient_data": True,
                "caveat": "Insufficient equity-curve data points to compute a Sharpe ratio for this window.",
            }

        # Compute daily returns from equity curve
        equity_array = np.array(equity, dtype=float)
        daily_returns = np.diff(equity_array) / equity_array[:-1]

        # Compute Sharpe statistics
        mean_return = float(np.mean(daily_returns))
        std_return = float(np.std(daily_returns))

        # Handle zero-volatility case (flat equity curve)
        if std_return > 0:
            sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(TRADING_DAYS_PER_YEAR)
        else:
            sharpe = 0.0

        # D-07: Deterministic overfitting flag (sharpe > 1.0 always triggers)
        overfitting_flag = bool(sharpe > OVERFITTING_SHARPE_THRESHOLD)

        # Verbatim caveat text per 07-RESEARCH.md Key Decision #3
        caveat = (
            "Sharpe ratio from a 20-30 trading day sample is inherently noisy; "
            "values > 1.0 are suspicious but not impossible in small windows. "
            "This flag is deterministic (> 1.0 always triggers) and does not prove overfitting; "
            "interpretation requires domain judgment."
        )

        return {
            "sharpe_ratio": sharpe,
            "overfitting_flag": overfitting_flag,
            "trading_days_sampled": len(daily_returns),
            "mean_daily_return": mean_return,
            "std_daily_return": std_return,
            "reference_range": REFERENCE_RANGE,
            "insufficient_data": False,
            "caveat": caveat,
        }

    except Exception as exc:
        logger.exception("Failed to compute window Sharpe: %s", exc)
        # Return fail-safe defaults on any error
        return {
            "sharpe_ratio": 0.0,
            "overfitting_flag": False,
            "trading_days_sampled": 0,
            "mean_daily_return": 0.0,
            "std_daily_return": 0.0,
            "reference_range": REFERENCE_RANGE,
            "insufficient_data": True,
            "caveat": f"Sharpe calculation failed: {str(exc)}",
        }
