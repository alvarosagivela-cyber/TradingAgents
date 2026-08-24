"""Unit tests for Phase 7 daily runner (scripts/phase7_daily_runner.py).

Tests verify:
- D-08: Retry-once-then-skip logic (exactly 2 attempts, skip on both failures)
- D-03: Default wiring to PHASE7_TICKER_BASKET
- D-01: checkpoint_enabled is forced to True
- Exit code is always 0 for expected skip behavior
"""

from __future__ import annotations

import os
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
             mock.patch.object(runner, "summarize_costs") as mock_summarize, \
             mock.patch.object(runner, "get_already_processed_tickers", return_value=set()):
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
             mock.patch.object(runner, "summarize_costs") as mock_summarize, \
             mock.patch.object(runner, "get_already_processed_tickers", return_value=set()):
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

        with mock.patch.object(runner, "TradingAgentsGraph", side_effect=capture_graph_init) as mock_graph_class, \
             mock.patch.object(runner, "summarize_costs") as mock_summarize, \
             mock.patch.object(runner, "get_already_processed_tickers", return_value=set()):
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


@pytest.mark.unit
class TestIdempotencyGuard(unittest.TestCase):
    """Test the idempotency guard that prevents double-processing a ticker.

    Found in production: a manual workflow_dispatch test run and that day's
    normal scheduled cron trigger landed on the same calendar day, and both
    processed all 13 tickers -- doubling real LLM spend and, worse, doubling
    every persisted decision record (which would inflate VALID-01's Auditor-
    refutation count on any day with a real mismatch verdict).
    """

    def test_get_already_processed_tickers_missing_file_returns_empty(self):
        """No auditor log yet (first-ever run) -> nothing is skipped."""
        result = runner.get_already_processed_tickers(
            "2026-08-24", "/nonexistent/path/audits.jsonl"
        )
        assert result == set()

    def test_get_already_processed_tickers_matches_only_given_date(self):
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({"ticker": "AAPL", "trade_date": "2026-08-24"}) + "\n")
            f.write(json.dumps({"ticker": "XOM", "trade_date": "2026-08-24"}) + "\n")
            f.write(json.dumps({"ticker": "JPM", "trade_date": "2026-08-23"}) + "\n")
            path = f.name

        try:
            result = runner.get_already_processed_tickers("2026-08-24", path)
            assert result == {"AAPL", "XOM"}
        finally:
            os.remove(path)

    def test_get_already_processed_tickers_fails_open_on_malformed_json(self):
        """A corrupt/unreadable log must never block a real run (D-08 philosophy)."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write("{not valid json\n")
            path = f.name

        try:
            result = runner.get_already_processed_tickers("2026-08-24", path)
            assert result == set()
        finally:
            os.remove(path)

    def test_get_already_processed_tickers_skips_only_one_bad_line_not_whole_file(self):
        """A single valid-JSON-but-non-dict line (e.g. a bare number) must not
        disable the guard for every other valid record in the file.

        record.get(...) on a non-dict raises AttributeError, not
        JSONDecodeError -- same bug class Phase 6 hardened
        read_cost_records()/decision_reconstructor against elsewhere in this
        codebase. The fix must skip only the bad line, not fail the whole read
        (which would silently re-enable double-processing for every ticker).
        """
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({"ticker": "AAPL", "trade_date": "2026-08-24"}) + "\n")
            f.write("42\n")  # valid JSON, not a dict
            f.write(json.dumps({"ticker": "XOM", "trade_date": "2026-08-24"}) + "\n")
            path = f.name

        try:
            result = runner.get_already_processed_tickers("2026-08-24", path)
            assert result == {"AAPL", "XOM"}
        finally:
            os.remove(path)

    def test_main_skips_already_processed_ticker_without_calling_propagate(self):
        """A ticker already persisted for trade_date must not be reprocessed."""
        with mock.patch.object(runner, "TradingAgentsGraph") as mock_graph_class, \
             mock.patch.object(runner, "summarize_costs") as mock_summarize, \
             mock.patch.object(
                 runner, "get_already_processed_tickers", return_value={"AAPL"}
             ):
            mock_instance = mock.MagicMock()
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "HOLD"}, "HOLD",
            )
            mock_graph_class.return_value = mock_instance
            mock_summarize.return_value = {
                "total_cost_usd": 0.0, "call_count": 0, "by_layer": {},
                "by_model": {}, "projected_annual_usd": 0.0,
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80, "over_threshold": False,
            }

            exit_code = runner.main(
                ["--trade-date", "2026-08-24", "--tickers", "AAPL,XOM"]
            )

            assert exit_code == 0
            # Only XOM should have been processed -- AAPL was already done
            assert mock_instance.propagate.call_count == 1
            mock_instance.propagate.assert_called_once_with(
                "XOM", "2026-08-24", asset_type="stock"
            )

    def test_main_force_bypasses_idempotency_guard(self):
        """--force reprocesses every ticker even if already persisted."""
        with mock.patch.object(runner, "TradingAgentsGraph") as mock_graph_class, \
             mock.patch.object(runner, "summarize_costs") as mock_summarize, \
             mock.patch.object(
                 runner, "get_already_processed_tickers"
             ) as mock_already_processed:
            mock_instance = mock.MagicMock()
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "HOLD"}, "HOLD",
            )
            mock_graph_class.return_value = mock_instance
            mock_summarize.return_value = {
                "total_cost_usd": 0.0, "call_count": 0, "by_layer": {},
                "by_model": {}, "projected_annual_usd": 0.0,
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80, "over_threshold": False,
            }

            exit_code = runner.main(
                ["--trade-date", "2026-08-24", "--tickers", "AAPL,XOM", "--force"]
            )

            assert exit_code == 0
            # --force must skip the idempotency check entirely
            mock_already_processed.assert_not_called()
            assert mock_instance.propagate.call_count == 2
