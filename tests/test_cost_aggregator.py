"""Tests for cost aggregation and budget alert (Phase 6, D-05/D-06).

Tests verify that:
1. summarize_costs() aggregates multiple cost records into running total, layer/model breakdown
2. Projected annual spend is calculated via linear extrapolation from days elapsed
3. Budget alert threshold (80% of $630 target = $504) is respected
4. Empty or missing cost logs return zeroed dict without errors
5. CLI budget report surfaces the aggregated data and prints alert banner when over threshold
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

from tradingagents.jobs.cost_aggregator import summarize_costs
from tradingagents.llm_clients.cost_tracker import CostRecord
from scripts.cost_budget_report import main as cli_main


@pytest.mark.unit
class TestCostAggregator:
    """Verify cost aggregation, projection, and budget alert logic."""

    def test_summarize_costs_basic_aggregation(self):
        """Given 3 records (one per layer), summarize_costs() returns correct totals."""
        # Create a temp JSONL with 3 records (research, auditor, conservative)
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"

            # Create 3 records: research=$0.10, auditor=$0.20, conservative=$0.30
            records = [
                CostRecord(
                    timestamp="2026-08-10T10:00:00+00:00",
                    layer="research",
                    model="claude-haiku-4-5",
                    ticker="AAPL",
                    trade_date="2026-08-10",
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.10,
                ),
                CostRecord(
                    timestamp="2026-08-10T10:05:00+00:00",
                    layer="auditor",
                    model="claude-sonnet-5",
                    ticker="AAPL",
                    trade_date="2026-08-10",
                    input_tokens=500,
                    output_tokens=200,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.20,
                ),
                CostRecord(
                    timestamp="2026-08-10T10:10:00+00:00",
                    layer="conservative",
                    model="claude-haiku-4-5",
                    ticker="AAPL",
                    trade_date="2026-08-10",
                    input_tokens=200,
                    output_tokens=100,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.30,
                ),
            ]

            # Write records to JSONL
            with open(cost_log_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record.to_json_dict()) + "\n")

            # Mock get_config to return our temp path
            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value={"cost_log_path": str(cost_log_path)}
            ):
                result = summarize_costs()

            # Verify aggregation
            assert result["total_cost_usd"] == 0.60
            assert result["call_count"] == 3
            assert result["by_layer"]["research"] == 0.10
            assert result["by_layer"]["auditor"] == 0.20
            assert result["by_layer"]["conservative"] == 0.30
            assert result["by_model"]["claude-haiku-4-5"] == 0.40  # research + conservative
            assert result["by_model"]["claude-sonnet-5"] == 0.20   # auditor

    def test_empty_cost_log(self):
        """Given empty cost log, summarize_costs() returns zeroed dict without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"
            # Create empty file
            cost_log_path.touch()

            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value={"cost_log_path": str(cost_log_path)}
            ):
                result = summarize_costs()

            assert result["total_cost_usd"] == 0.0
            assert result["call_count"] == 0
            assert result["by_layer"] == {}
            assert result["by_model"] == {}
            assert result["projected_annual_usd"] == 0.0
            assert result["over_threshold"] is False

    def test_missing_cost_log(self):
        """Given missing cost log file, summarize_costs() returns zeroed dict without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "nonexistent" / "cost.jsonl"

            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value={"cost_log_path": str(cost_log_path)}
            ):
                result = summarize_costs()

            assert result["total_cost_usd"] == 0.0
            assert result["call_count"] == 0
            assert result["by_layer"] == {}
            assert result["by_model"] == {}
            assert result["projected_annual_usd"] == 0.0
            assert result["over_threshold"] is False

    def test_projected_annual_extrapolation_math(self):
        """Given records spanning exactly 24 hours with total_cost_usd=$2.00,
        projected_annual_usd must equal 2.00 * 365 (linear extrapolation).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"

            # Use fixed timestamps to avoid timing issues during test execution
            earliest_ts = "2026-08-09T12:00:00+00:00"
            latest_ts = "2026-08-10T12:00:00+00:00"
            fixed_now = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

            records = [
                CostRecord(
                    timestamp=earliest_ts,
                    layer="research",
                    model="claude-haiku-4-5",
                    ticker="",
                    trade_date="",
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=1.50,
                ),
                CostRecord(
                    timestamp=latest_ts,
                    layer="auditor",
                    model="claude-haiku-4-5",
                    ticker="",
                    trade_date="",
                    input_tokens=200,
                    output_tokens=100,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.50,
                ),
            ]

            with open(cost_log_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record.to_json_dict()) + "\n")

            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value={"cost_log_path": str(cost_log_path)}
            ):
                with mock.patch(
                    "tradingagents.jobs.cost_aggregator.datetime"
                ) as mock_datetime:
                    # Make datetime.now return our fixed time
                    mock_datetime.now.return_value = fixed_now
                    # But allow datetime.fromisoformat to work normally
                    mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
                    # Allow timezone.utc to work
                    mock_datetime.timezone.utc = timezone.utc
                    result = summarize_costs()

            # Total cost is $2.00, elapsed time is 24 hours (1 day)
            # Extrapolation: 2.00 * (365 / 1) = 730.0
            assert result["total_cost_usd"] == 2.00
            assert result["projected_annual_usd"] == 730.0

    def test_budget_alert_threshold(self, caplog):
        """Given annual_budget_target_usd=630 and threshold_pct=0.80 (threshold=$504),
        - projected_annual_usd >= $504 sets over_threshold=True and logs warning
        - projected_annual_usd < $504 sets over_threshold=False and logs nothing
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"

            # Test case 1: exactly at threshold ($504)
            earliest_ts = "2026-08-09T12:00:00+00:00"
            fixed_now = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

            record_at_threshold = CostRecord(
                timestamp=earliest_ts,
                layer="research",
                model="claude-haiku-4-5",
                ticker="",
                trade_date="",
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=504.0 / 365.0,  # This will project to exactly $504/year
            )

            with open(cost_log_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record_at_threshold.to_json_dict()) + "\n")

            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value={
                    "cost_log_path": str(cost_log_path),
                    "annual_budget_target_usd": 630.0,
                    "budget_alert_threshold_pct": 0.80,
                }
            ):
                with mock.patch(
                    "tradingagents.jobs.cost_aggregator.datetime"
                ) as mock_datetime:
                    mock_datetime.now.return_value = fixed_now
                    mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
                    mock_datetime.timezone.utc = timezone.utc
                    with caplog.at_level(logging.WARNING):
                        result = summarize_costs()

            assert result["over_threshold"] is True
            assert "Budget alert" in caplog.text

    def test_burst_of_calls_within_an_hour_does_not_false_alarm(self, caplog):
        """A short burst of calls (e.g. one paper-trading session run in a few
        minutes) must NOT trigger over_threshold even if the naive linear
        extrapolation would cross the budget threshold — regression for the gap
        found in Phase 6 code review: the 1-hour floor on days_elapsed produces up
        to a 365*24x multiplier, which can false-alarm on the very first session
        before there's enough elapsed time to trust the annual trend.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"

            # All records within the same few minutes (well under 1 day elapsed).
            # A single $0.10 call extrapolated via the 1-hour floor projects to
            # 0.10 * 365 * 24 = $876/year -- comfortably over the $504 threshold --
            # yet this is a burst, not a trend, so the alert must stay suppressed.
            earliest_ts = "2026-08-09T12:00:00+00:00"
            fixed_now = datetime(2026, 8, 9, 12, 5, 0, tzinfo=timezone.utc)

            record = CostRecord(
                timestamp=earliest_ts,
                layer="research",
                model="claude-haiku-4-5",
                ticker="",
                trade_date="",
                input_tokens=1000,
                output_tokens=500,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.10,
            )

            with open(cost_log_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record.to_json_dict()) + "\n")

            config = {
                "cost_log_path": str(cost_log_path),
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80,
            }

            with mock.patch("tradingagents.jobs.cost_aggregator.get_config", return_value=config):
                with mock.patch("tradingagents.jobs.cost_aggregator.datetime") as mock_datetime:
                    mock_datetime.now.return_value = fixed_now
                    mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
                    mock_datetime.timezone.utc = timezone.utc
                    result = summarize_costs()

            # The naive projection is still high (confirms the burst really would
            # have crossed threshold without the gate) ...
            assert result["projected_annual_usd"] > 504.0
            # ... but the alert itself must be suppressed at this elapsed window.
            assert result["over_threshold"] is False
            assert "Budget alert" not in caplog.text

    def test_budget_alert_below_threshold(self, caplog):
        """Given projected annual spend well below 80% threshold, no warning logged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"

            # Create a record that will project to less than 80% of target
            # Target=$630, threshold=80%=$504. Create a record that projects to $200/year
            earliest_ts = "2026-08-09T12:00:00+00:00"
            fixed_now = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

            record_below_threshold = CostRecord(
                timestamp=earliest_ts,
                layer="research",
                model="claude-haiku-4-5",
                ticker="",
                trade_date="",
                input_tokens=50,
                output_tokens=25,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=200.0 / 365.0,  # This will project to exactly $200/year
            )

            with open(cost_log_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record_below_threshold.to_json_dict()) + "\n")

            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value={
                    "cost_log_path": str(cost_log_path),
                    "annual_budget_target_usd": 630.0,
                    "budget_alert_threshold_pct": 0.80,
                }
            ):
                with mock.patch(
                    "tradingagents.jobs.cost_aggregator.datetime"
                ) as mock_datetime:
                    mock_datetime.now.return_value = fixed_now
                    mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
                    mock_datetime.timezone.utc = timezone.utc
                    with caplog.at_level(logging.WARNING):
                        result = summarize_costs()

            assert result["over_threshold"] is False
            assert "Budget alert" not in caplog.text

    def test_summarize_costs_returns_config_values(self):
        """Verify that return dict includes the resolved config values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"
            cost_log_path.touch()

            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value={
                    "cost_log_path": str(cost_log_path),
                    "annual_budget_target_usd": 630.0,
                    "budget_alert_threshold_pct": 0.80,
                }
            ):
                result = summarize_costs()

            assert result["annual_budget_target_usd"] == 630.0
            assert result["budget_alert_threshold_pct"] == 0.80


@pytest.mark.unit
class TestCostBudgetReportCLI:
    """Verify CLI budget report formatting and alert banner."""

    def test_cli_budget_report_over_threshold(self, capsys):
        """Given cost log exceeding 80% threshold, CLI prints BUDGET ALERT banner and returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"

            # Create record that projects to >= 80% of $630 = $504
            earliest_ts = "2026-08-09T12:00:00+00:00"
            fixed_now = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

            record = CostRecord(
                timestamp=earliest_ts,
                layer="research",
                model="claude-haiku-4-5",
                ticker="AAPL",
                trade_date="2026-08-10",
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=520.0 / 365.0,  # Projects to $520/year
            )

            with open(cost_log_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record.to_json_dict()) + "\n")

            config_dict = {
                "cost_log_path": str(cost_log_path),
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80,
            }

            # Mock get_config in both the CLI and the aggregator module
            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value=config_dict
            ):
                with mock.patch(
                    "tradingagents.jobs.cost_aggregator.datetime"
                ) as mock_datetime:
                    mock_datetime.now.return_value = fixed_now
                    mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
                    mock_datetime.timezone.utc = timezone.utc
                    exit_code = cli_main([])

            captured = capsys.readouterr()
            assert "BUDGET ALERT" in captured.out
            assert exit_code == 0

    def test_cli_budget_report_below_threshold(self, capsys):
        """Given cost log below 80% threshold, CLI does NOT print BUDGET ALERT but shows breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"

            # Create record that projects to < 80% of $630 = $504
            earliest_ts = "2026-08-09T12:00:00+00:00"
            fixed_now = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

            record = CostRecord(
                timestamp=earliest_ts,
                layer="auditor",
                model="claude-sonnet-5",
                ticker="MSFT",
                trade_date="2026-08-10",
                input_tokens=500,
                output_tokens=200,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=100.0 / 365.0,  # Projects to $100/year
            )

            with open(cost_log_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record.to_json_dict()) + "\n")

            config_dict = {
                "cost_log_path": str(cost_log_path),
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80,
            }

            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value=config_dict
            ):
                with mock.patch(
                    "tradingagents.jobs.cost_aggregator.datetime"
                ) as mock_datetime:
                    mock_datetime.now.return_value = fixed_now
                    mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
                    mock_datetime.timezone.utc = timezone.utc
                    exit_code = cli_main([])

            captured = capsys.readouterr()
            assert "BUDGET ALERT" not in captured.out
            # Should still show the breakdown tables
            assert "auditor" in captured.out or "claude-sonnet-5" in captured.out or "100" in captured.out
            assert exit_code == 0

    def test_cli_budget_report_shows_breakdown_tables(self, capsys):
        """CLI shows per-layer and per-model breakdown tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cost_log_path = Path(tmpdir) / "cost.jsonl"

            earliest_ts = "2026-08-09T12:00:00+00:00"
            fixed_now = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)

            # Create two records from different layers/models
            records = [
                CostRecord(
                    timestamp=earliest_ts,
                    layer="research",
                    model="claude-haiku-4-5",
                    ticker="AAPL",
                    trade_date="2026-08-10",
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.15,
                ),
                CostRecord(
                    timestamp=earliest_ts,
                    layer="auditor",
                    model="claude-sonnet-5",
                    ticker="AAPL",
                    trade_date="2026-08-10",
                    input_tokens=500,
                    output_tokens=200,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.30,
                ),
            ]

            with open(cost_log_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record.to_json_dict()) + "\n")

            config_dict = {
                "cost_log_path": str(cost_log_path),
                "annual_budget_target_usd": 630.0,
                "budget_alert_threshold_pct": 0.80,
            }

            with mock.patch(
                "tradingagents.jobs.cost_aggregator.get_config",
                return_value=config_dict
            ):
                with mock.patch(
                    "tradingagents.jobs.cost_aggregator.datetime"
                ) as mock_datetime:
                    mock_datetime.now.return_value = fixed_now
                    mock_datetime.fromisoformat.side_effect = datetime.fromisoformat
                    mock_datetime.timezone.utc = timezone.utc
                    exit_code = cli_main([])

            captured = capsys.readouterr()
            # Should show call count
            assert "2" in captured.out  # call_count
            # Should show both dollar amounts
            assert "0.45" in captured.out
            assert exit_code == 0
