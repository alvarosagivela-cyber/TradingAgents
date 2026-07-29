"""Unit tests for momentum_calculator — determinism, formula verification, fail-open paths."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.utils.momentum_calculator import (
    compute_momentum,
    derive_confidence_level,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_frame(close_prices: list[float], n_rows: int | None = None) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame for testing.

    Args:
        close_prices: List of close prices (or shorter list if n_rows is specified)
        n_rows: If specified, pad close_prices to this length with repeated last value

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume columns, sorted ascending
    """
    if n_rows and len(close_prices) < n_rows:
        # Pad with the last price
        close_prices = close_prices + [close_prices[-1]] * (n_rows - len(close_prices))

    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=len(close_prices), freq="D"),
        "Open": close_prices,
        "High": [p * 1.02 for p in close_prices],
        "Low": [p * 0.98 for p in close_prices],
        "Close": close_prices,
        "Volume": [1_000_000] * len(close_prices),
    })
    return df.sort_values("Date", ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMomentumCalculatorDeterminism:
    """Test that momentum calculation is deterministic (bit-identical across runs)."""

    def test_compute_momentum_determinism(self):
        """Calling compute_momentum twice with identical input returns identical output (D-10)."""
        # Create a frame with 273+ rows of realistic prices
        close_prices = [100.0 + i * 0.5 for i in range(300)]
        frame = _make_ohlcv_frame(close_prices)

        with patch(
            "tradingagents.agents.utils.momentum_calculator.load_ohlcv"
        ) as mock_load:
            mock_load.return_value = frame.copy()

            result1 = compute_momentum("AAPL", "2020-11-01")
            result2 = compute_momentum("AAPL", "2020-11-01")

            # All fields should be identical (bit-level equality for floats)
            assert result1.retorno_12_1 == result2.retorno_12_1
            assert result1.z_score == result2.z_score
            assert result1.confidence_level == result2.confidence_level
            assert result1.valid == result2.valid


@pytest.mark.unit
class TestJegadeeshTitmanFormula:
    """Test that the Jegadeesh-Titman skip-month formula is implemented correctly."""

    def test_compute_momentum_skips_most_recent_month(self):
        """The skip-month formula excludes the most recent 21 rows (D-07)."""
        # Create a frame where close prices are flat except for a spike in the last 21 rows
        base_price = 100.0
        prices = [base_price] * (300 - 21) + [base_price * 2.0] * 21  # Spike in last month
        frame = _make_ohlcv_frame(prices)

        with patch(
            "tradingagents.agents.utils.momentum_calculator.load_ohlcv"
        ) as mock_load:
            mock_load.return_value = frame

            result = compute_momentum("AAPL", "2020-11-01")

            # Since prices are flat except for the last 21 rows (which are skipped),
            # the retorno_12_1 should be ~0 (base_price to base_price, 21 rows ago to 252 rows ago)
            assert result.valid is True
            # The return should be (100 - 100) / 100 = 0.0, not 2.0 (which would include the spike)
            assert abs(result.retorno_12_1) < 0.01  # Should be near zero, not 2.0


@pytest.mark.unit
class TestFailOpen:
    """Test that momentum calculation fails gracefully (fail-open) on data errors."""

    def test_compute_momentum_insufficient_history_fails_open(self):
        """Frame with fewer than 274 rows returns valid=False (D-10)."""
        frame = _make_ohlcv_frame([100.0] * 200)  # Only 200 rows

        with patch(
            "tradingagents.agents.utils.momentum_calculator.load_ohlcv"
        ) as mock_load:
            mock_load.return_value = frame

            result = compute_momentum("TICKER", "2020-11-01")

            assert result.valid is False
            assert result.retorno_12_1 == 0.0
            assert result.lookback_days == 200

    def test_compute_momentum_no_market_data_fails_open(self):
        """NoMarketDataError from load_ohlcv is caught and returns valid=False (D-10)."""
        with patch(
            "tradingagents.agents.utils.momentum_calculator.load_ohlcv"
        ) as mock_load:
            mock_load.side_effect = NoMarketDataError(
                symbol="INVALID",
                canonical="INVALID",
                detail="No data available",
            )

            result = compute_momentum("INVALID", "2020-11-01")

            assert result.valid is False
            assert result.retorno_12_1 == 0.0
            assert result.z_score == 0.0
            assert result.confidence_level == "low"


@pytest.mark.unit
class TestConfidenceLevelThresholds:
    """Test derive_confidence_level thresholds (D-15)."""

    @pytest.mark.parametrize("z_score,expected_level", [
        (2.5, "high"),
        (2.0, "high"),  # Boundary: >= 2.0
        (1.99, "medium"),
        (1.0, "medium"),
        (0.5, "medium"),  # Boundary: >= 0.5
        (0.49, "low"),
        (0.3, "low"),
        (0.0, "low"),
        (-0.5, "medium"),
        (-2.0, "high"),  # Abs value matters
        (-2.5, "high"),
    ])
    def test_derive_confidence_level_thresholds(self, z_score, expected_level):
        """Verify exact threshold boundaries for confidence level derivation (D-15)."""
        level = derive_confidence_level(z_score)
        assert level == expected_level, (
            f"z_score={z_score} should yield {expected_level}, got {level}"
        )


@pytest.mark.unit
class TestZScoreCalculation:
    """Test z-score derivation from rolling distribution."""

    def test_compute_momentum_z_score_bounds(self):
        """Z-score should be reasonable (typically between -3 and +3)."""
        # Create prices with trending return
        prices = [100.0 + i * 0.3 for i in range(300)]
        frame = _make_ohlcv_frame(prices)

        with patch(
            "tradingagents.agents.utils.momentum_calculator.load_ohlcv"
        ) as mock_load:
            mock_load.return_value = frame

            result = compute_momentum("AAPL", "2020-11-01")

            assert result.valid is True
            # Z-score should be reasonable (not inf, not nan)
            assert -10 < result.z_score < 10, f"Unexpected z-score: {result.z_score}"

    def test_compute_momentum_insufficient_rolling_returns_z_score_zero(self):
        """If fewer than 30 rolling observations, z_score defaults to 0.0."""
        # Create a small frame (>= 274 for valid retorno, but < 30 rolling values after dropna)
        frame = _make_ohlcv_frame([100.0] * 280)  # Exactly 280 rows (just enough for retorno, < 30 rolling after shift/dropna)

        with patch(
            "tradingagents.agents.utils.momentum_calculator.load_ohlcv"
        ) as mock_load:
            mock_load.return_value = frame

            result = compute_momentum("TICKER", "2020-11-01")

            # Should still be valid (has enough for retorno_12_1) but z_score is undefined
            assert result.valid is True
            assert result.z_score == 0.0
