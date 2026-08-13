"""Unit tests for Phase 7 daily runner (scripts/phase7_daily_runner.py).

Tests verify:
- D-08: Retry-once-then-skip logic (exactly 2 attempts, skip on both failures)
- D-03: Default wiring to PHASE7_TICKER_BASKET
- D-01: checkpoint_enabled is forced to True
- Exit code is always 0 for expected skip behavior
"""

from __future__ import annotations

import unittest
from unittest import mock

import pytest

import scripts.phase7_daily_runner as runner


@pytest.mark.unit
class TestRetryThenSkip(unittest.TestCase):
    """Test D-08 retry-once-then-skip logic."""

    def test_run_ticker_with_retry_succeeds_on_retry(self):
        """Test that retry succeeds on second attempt after first failure.

        Proves:
        - Graph is constructed exactly once (not per attempt)
        - propagate() is called exactly twice
        - time.sleep() is called exactly once (between attempts)
        - Function returns True
        """
        with mock.patch.object(runner, "TradingAgentsGraph") as mock_graph_class, \
             mock.patch.object(runner.time, "sleep") as mock_sleep:
            # Setup: first call raises, second succeeds
            mock_instance = mock.MagicMock()
            mock_instance.propagate.side_effect = [
                RuntimeError("transient API error"),
                ({"final_trade_decision": "HOLD"}, "HOLD"),
            ]
            mock_graph_class.return_value = mock_instance

            # Act
            result = runner.run_ticker_with_retry(
                "AAPL",
                "2026-08-11",
                {},
                ["market"],
            )

            # Assert
            assert result is True
            assert mock_graph_class.call_count == 1  # Graph built once
            assert mock_instance.propagate.call_count == 2  # Called twice
            assert mock_sleep.call_count == 1  # Slept once (between attempts)

    def test_run_ticker_with_retry_skips_after_exhausting_retry(self):
        """Test that ticker is skipped after both attempts fail.

        Proves:
        - propagate() is called exactly twice
        - time.sleep() is called exactly once (not after the final failure)
        - Function returns False
        """
        with mock.patch.object(runner, "TradingAgentsGraph") as mock_graph_class, \
             mock.patch.object(runner.time, "sleep") as mock_sleep:
            # Setup: both calls raise
            mock_instance = mock.MagicMock()
            mock_instance.propagate.side_effect = [
                RuntimeError("error 1"),
                RuntimeError("error 2"),
            ]
            mock_graph_class.return_value = mock_instance

            # Act
            result = runner.run_ticker_with_retry(
                "XOM",
                "2026-08-11",
                {},
                ["market"],
            )

            # Assert
            assert result is False
            assert mock_instance.propagate.call_count == 2
            assert mock_sleep.call_count == 1  # Not called before the final failure

    def test_run_ticker_with_retry_recovers_from_graph_construction_failure(self):
        """Test that a TradingAgentsGraph() constructor failure is retried, not fatal.

        Proves a transient failure while building the graph (e.g. LLM client
        init) is treated like a propagate() failure: retried once, and does
        not escape run_ticker_with_retry() to abort the rest of the day.
        """
        mock_instance = mock.MagicMock()
        mock_instance.propagate.return_value = (
            {"final_trade_decision": "HOLD"},
            "HOLD",
        )

        with mock.patch.object(
            runner,
            "TradingAgentsGraph",
            side_effect=[RuntimeError("LLM client init failed"), mock_instance],
        ) as mock_graph_class, mock.patch.object(runner.time, "sleep") as mock_sleep:
            result = runner.run_ticker_with_retry(
                "AAPL",
                "2026-08-11",
                {},
                ["market"],
            )

            assert result is True
            assert mock_graph_class.call_count == 2  # retried construction
            assert mock_instance.propagate.call_count == 1
            assert mock_sleep.call_count == 1

    def test_run_ticker_with_retry_skips_when_graph_construction_always_fails(self):
        """Test that a ticker is skipped, not fatal, if construction never succeeds."""
        with mock.patch.object(
            runner,
            "TradingAgentsGraph",
            side_effect=RuntimeError("LLM client init failed"),
        ) as mock_graph_class, mock.patch.object(runner.time, "sleep") as mock_sleep:
            result = runner.run_ticker_with_retry(
                "XOM",
                "2026-08-11",
                {},
                ["market"],
            )

            assert result is False
            assert mock_graph_class.call_count == 2
            assert mock_sleep.call_count == 1


@pytest.mark.unit
class TestMainDefaults(unittest.TestCase):
    """Test D-03 default basket wiring and CLI behavior."""

    def test_main_defaults_to_phase7_ticker_basket(self):
        """Test that main() uses PHASE7_TICKER_BASKET by default.

        Proves:
        - propagate() is called once per ticker in the basket
        - main() returns 0
        """
        with mock.patch.object(runner, "TradingAgentsGraph") as mock_graph_class, \
             mock.patch.object(runner, "summarize_costs") as mock_summarize:
            # Setup: all tickers succeed
            mock_instance = mock.MagicMock()
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "HOLD"},
                "HOLD",
            )
            mock_graph_class.return_value = mock_instance

            # Mock summarize_costs to return minimal valid dict
            mock_summarize.return_value = {
                "total_cost_usd": 0.0,
                "call_count": 0,
                "by_layer": {},
                "by_model": {},
                "projected_annual_usd": 0.0,
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80,
                "over_threshold": False,
            }

            # Act: no --tickers override
            exit_code = runner.main(["--trade-date", "2026-08-11"])

            # Assert
            assert exit_code == 0
            # Should be called once per ticker in PHASE7_TICKER_BASKET
            assert mock_instance.propagate.call_count == len(
                runner.PHASE7_TICKER_BASKET
            )

    def test_main_returns_zero_even_when_all_tickers_skip(self):
        """Test that main() returns 0 even if all tickers fail (D-08).

        Proves:
        - Exit code is 0 even when all tickers are skipped
        - The run continues to completion and budget status is printed
        """
        with mock.patch.object(runner, "TradingAgentsGraph") as mock_graph_class, \
             mock.patch.object(runner.time, "sleep"), \
             mock.patch.object(runner, "summarize_costs") as mock_summarize:
            # Setup: all propagate calls raise
            mock_instance = mock.MagicMock()
            mock_instance.propagate.side_effect = RuntimeError("always fails")
            mock_graph_class.return_value = mock_instance

            mock_summarize.return_value = {
                "total_cost_usd": 10.0,
                "call_count": 5,
                "by_layer": {},
                "by_model": {},
                "projected_annual_usd": 100.0,
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80,
                "over_threshold": False,
            }

            # Act: with a small subset to avoid long test time
            exit_code = runner.main(
                ["--trade-date", "2026-08-11", "--tickers", "AAPL,XOM"]
            )

            # Assert: exit code is still 0 (D-08)
            assert exit_code == 0


@pytest.mark.unit
class TestCheckpointForced(unittest.TestCase):
    """Test D-01 dependency: checkpoint_enabled is forced to True."""

    def test_main_forces_checkpoint_enabled(self):
        """Test that checkpoint_enabled is set to True in the config.

        Proves:
        - TradingAgentsGraph is called with config["checkpoint_enabled"] = True
        - This is true regardless of DEFAULT_CONFIG's own value
        """
        captured_configs = []

        def capture_graph_init(selected_analysts, config=None, debug=False):
            if config is not None:
                captured_configs.append(config)
            mock_instance = mock.MagicMock()
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "HOLD"},
                "HOLD",
            )
            return mock_instance

        with mock.patch.object(runner, "TradingAgentsGraph", side_effect=capture_graph_init), \
             mock.patch.object(runner, "summarize_costs") as mock_summarize:
            mock_summarize.return_value = {
                "total_cost_usd": 0.0,
                "call_count": 0,
                "by_layer": {},
                "by_model": {},
                "projected_annual_usd": 0.0,
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80,
                "over_threshold": False,
            }

            # Act: run with just one ticker to keep test fast
            runner.main(["--trade-date", "2026-08-11", "--tickers", "AAPL"])

            # Assert: at least one config was passed, and it has checkpoint_enabled=True
            assert len(captured_configs) > 0
            for cfg in captured_configs:
                assert cfg.get("checkpoint_enabled") is True
