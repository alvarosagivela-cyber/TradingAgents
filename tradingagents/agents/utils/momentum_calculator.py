"""Deterministic momentum calculation using the Jegadeesh-Titman 12-1 formula.

This module computes the 12-month buy-and-hold return (skipping the most recent month)
and derives z-score confidence metrics for research theses. All calculations are pure
Python, deterministic (bit-identical across runs with identical input), and designed
to produce reliable, reproducible values that can be validated post-LLM.

Window units differ by asset class, both aiming at the same "12 months, skip the
most recent month" semantic:
- Equities/ETFs trade ~252 days/year (weekends and holidays closed), so the window
  is expressed in trading-day row counts (252/21).
- Crypto trades every calendar day (confirmed empirically: yfinance returns one bar
  per calendar day, weekends included), so 252 row-based "trading days" would only
  span ~8.3 calendar months, not 12. The window is instead expressed directly in
  calendar days (365/30), which for a 24/7 asset's gap-free daily series is the
  same as a row count.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.dataflows.symbol_utils import NoMarketDataError, crypto_base

logger = logging.getLogger(__name__)

# Equities/ETFs: trading-day row counts (~252 trading days/year).
EQUITY_LOOKBACK_DAYS = 252
EQUITY_SKIP_DAYS = 21
EQUITY_MIN_HISTORY = EQUITY_LOOKBACK_DAYS + EQUITY_SKIP_DAYS + 1  # 274

# Crypto: calendar-day row counts (~365 calendar days/year, 24/7 trading).
CRYPTO_LOOKBACK_DAYS = 365
CRYPTO_SKIP_DAYS = 30
CRYPTO_MIN_HISTORY = CRYPTO_LOOKBACK_DAYS + CRYPTO_SKIP_DAYS + 1  # 396


class MomentumMetrics(NamedTuple):
    """Deterministic output of momentum calculation.

    Fields:
        retorno_12_1: 12-month momentum, skip-month formula (D-07)
        z_score: Z-score of retorno_12_1 vs 5-year rolling distribution (D-15)
        confidence_level: high|medium|low, derived from abs(z_score) — never LLM-generated
        lookback_days: Number of trading days in the OHLCV frame used
        valid: False if insufficient history or data fetch failed (fail-open, D-10)
    """

    retorno_12_1: float
    z_score: float
    confidence_level: str
    lookback_days: int
    valid: bool


def compute_momentum(ticker: str, as_of_date: str) -> MomentumMetrics:
    """Compute Jegadeesh-Titman 12-1 momentum for a ticker as of a given date.

    Implements the skip-month momentum formula exactly (D-07):
    - Fetch 5-year OHLCV ending at as_of_date
    - Skip the most recent month (21 trading days for equities, 30 calendar days
      for 24/7 crypto -- see module docstring)
    - Calculate return from ~12 months ago (252 trading days / 365 calendar days)
      to skip_days ago
    - Derive z-score from rolling 12-1 distribution over the full 5-year window

    Fail-open on any data fetch error: returns valid=False, retorno_12_1=0.0.

    Args:
        ticker: Stock ticker symbol
        as_of_date: Date string (YYYY-MM-DD) for the calculation date

    Returns:
        MomentumMetrics with retorno_12_1, z_score, confidence_level, and valid flag
    """
    try:
        is_crypto = crypto_base(ticker) is not None
        lookback_days = CRYPTO_LOOKBACK_DAYS if is_crypto else EQUITY_LOOKBACK_DAYS
        skip_days = CRYPTO_SKIP_DAYS if is_crypto else EQUITY_SKIP_DAYS
        min_history = CRYPTO_MIN_HISTORY if is_crypto else EQUITY_MIN_HISTORY

        # Load OHLCV data (5-year lookback, already cleaned and ascending-by-Date)
        data = load_ohlcv(ticker, as_of_date)

        # Ensure enough history: need lookback_days for the 12-month leg + (skip_days + 1)
        # for the skip month (e.g. equities: 252 + 22 = 274 rows needed)
        n = len(data)
        if n < min_history:
            logger.warning(
                f"Insufficient OHLCV history for {ticker}: {n} rows < {min_history} required "
                f"(12-1 skip-month formula, {'crypto/calendar-day' if is_crypto else 'equity/trading-day'} window)"
            )
            return MomentumMetrics(
                retorno_12_1=0.0,
                z_score=0.0,
                confidence_level="low",
                lookback_days=n,
                valid=False,
            )

        # Compute retorno_12_1 (Jegadeesh-Titman skip-month formula)
        # Data is sorted ascending by Date, so:
        # - Most recent: data.iloc[-1] (index n-1)
        # - skip_days ago: data.iloc[-(skip_days+1)]
        # - lookback_days ago: data.iloc[-(lookback_days+1)]
        # Return = (close_1m_ago - close_12m_ago) / close_12m_ago
        idx_1m = n - (skip_days + 1)
        idx_12m = n - (lookback_days + 1)

        close_1m_ago = data["Close"].iloc[idx_1m]
        close_12m_ago = data["Close"].iloc[idx_12m]

        if close_12m_ago == 0:
            logger.warning(f"Zero close price 12 months ago for {ticker}")
            return MomentumMetrics(
                retorno_12_1=0.0,
                z_score=0.0,
                confidence_level="low",
                lookback_days=n,
                valid=False,
            )

        retorno_12_1 = (close_1m_ago - close_12m_ago) / close_12m_ago
        # float() is required here: round() on a numpy.float64 (what pandas .iloc
        # scalars actually are, despite the MomentumMetrics.retorno_12_1: float
        # annotation) returns another numpy.float64, not a native Python float.
        # An un-cast numpy.float64 flowing into graph state fails msgpack
        # serialization at checkpoint time (SqliteSaver), which only surfaces on
        # a real checkpointed run, not the mocked/dict-state unit tests.
        retorno_12_1 = float(round(retorno_12_1, 6))  # 6 decimals as per D-07

        # Compute z-score from rolling 12-1 distribution (D-15)
        # Build rolling returns: shift(skip_days+1) to skip month, shift(lookback_days+1)
        # for the 12-month lookback. This matches the idx_1m/idx_12m formula above.
        rolling = (
            (data["Close"].shift(skip_days + 1) - data["Close"].shift(lookback_days + 1))
            / data["Close"].shift(lookback_days + 1)
        )
        rolling = rolling.dropna()

        # If fewer than 30 rolling observations, z-score is undefined
        if len(rolling) < 30:
            logger.debug(
                f"Insufficient rolling observations for {ticker}: {len(rolling)} < 30"
            )
            z_score = 0.0
        else:
            mean = rolling.mean()
            std = rolling.std()
            z_score = 0.0 if std == 0 else (retorno_12_1 - mean) / std
            z_score = float(round(z_score, 4))  # 4 decimals as per D-15; see float() note above

        # Derive confidence level from z-score magnitude (D-15)
        confidence_level = derive_confidence_level(z_score)

        return MomentumMetrics(
            retorno_12_1=retorno_12_1,
            z_score=z_score,
            confidence_level=confidence_level,
            lookback_days=n,
            valid=True,
        )

    except NoMarketDataError as exc:
        logger.warning(f"No market data for {ticker}: {exc}")
        return MomentumMetrics(
            retorno_12_1=0.0,
            z_score=0.0,
            confidence_level="low",
            lookback_days=0,
            valid=False,
        )
    except Exception as exc:
        logger.exception(f"Unexpected error computing momentum for {ticker}: {exc}")
        return MomentumMetrics(
            retorno_12_1=0.0,
            z_score=0.0,
            confidence_level="low",
            lookback_days=0,
            valid=False,
        )


def derive_confidence_level(z_score: float) -> str:
    """Derive confidence level from z-score magnitude (D-15).

    Thresholds (exact, from `02-CONTEXT.md`):
        high: abs(z_score) >= 2.0
        medium: 0.5 <= abs(z_score) < 2.0
        low: abs(z_score) < 0.5

    This function replaces the LLM's self-reported confidence with a deterministic,
    objective metric (D-15, D-18).

    Args:
        z_score: The computed z-score

    Returns:
        "high", "medium", or "low"
    """
    abs_z = abs(z_score)
    if abs_z >= 2.0:
        return "high"
    elif abs_z >= 0.5:
        return "medium"
    else:
        return "low"
