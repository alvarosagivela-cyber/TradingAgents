"""Tests for Auditor false-positive analysis (Phase 6, D-07).

This test module verifies the aggregation of Auditor reflections to compute
false-positive and false-negative rates for the Auditor's Buy/Sell verdicts.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tradingagents.dataflows.config import get_config, set_config
from tradingagents.jobs.auditor_false_positive_analyzer import aggregate_auditor_reflections


@pytest.fixture
def auditor_reflections_fixture(tmp_path: Path) -> Path:
    """Create a temporary auditor_reflections.jsonl with 18 records.

    Returns:
        Path to the temporary JSONL file
    """
    fixture_file = tmp_path / "auditor_reflections.jsonl"

    records = [
        # 3 Buy / negative (false positives)
        {
            "ticker": "AAPL",
            "decision_date": "2026-07-01",
            "decision_verdict": "Buy",
            "realized_return": -0.03,
            "classification": "negative",
            "lesson_text": "Buy decision resulted in a -3.00% return...",
            "created_at": "2026-07-11T10:00:00Z",
        },
        {
            "ticker": "MSFT",
            "decision_date": "2026-07-02",
            "decision_verdict": "Buy",
            "realized_return": -0.015,
            "classification": "negative",
            "lesson_text": "Buy decision resulted in a -1.50% return...",
            "created_at": "2026-07-12T10:00:00Z",
        },
        {
            "ticker": "GOOGL",
            "decision_date": "2026-07-03",
            "decision_verdict": "Buy",
            "realized_return": -0.02,
            "classification": "negative",
            "lesson_text": "Buy decision resulted in a -2.00% return...",
            "created_at": "2026-07-13T10:00:00Z",
        },
        # 7 Buy / positive or neutral
        {
            "ticker": "AMZN",
            "decision_date": "2026-07-04",
            "decision_verdict": "Buy",
            "realized_return": 0.05,
            "classification": "positive",
            "lesson_text": "Buy decision resulted in a +5.00% return...",
            "created_at": "2026-07-14T10:00:00Z",
        },
        {
            "ticker": "TSLA",
            "decision_date": "2026-07-05",
            "decision_verdict": "Buy",
            "realized_return": 0.03,
            "classification": "positive",
            "lesson_text": "Buy decision resulted in a +3.00% return...",
            "created_at": "2026-07-15T10:00:00Z",
        },
        {
            "ticker": "META",
            "decision_date": "2026-07-06",
            "decision_verdict": "Buy",
            "realized_return": 0.02,
            "classification": "positive",
            "lesson_text": "Buy decision resulted in a +2.00% return...",
            "created_at": "2026-07-16T10:00:00Z",
        },
        {
            "ticker": "NVDA",
            "decision_date": "2026-07-07",
            "decision_verdict": "Buy",
            "realized_return": 0.07,
            "classification": "positive",
            "lesson_text": "Buy decision resulted in a +7.00% return...",
            "created_at": "2026-07-17T10:00:00Z",
        },
        {
            "ticker": "JPM",
            "decision_date": "2026-07-08",
            "decision_verdict": "Buy",
            "realized_return": 0.01,
            "classification": "positive",
            "lesson_text": "Buy decision resulted in a +1.00% return...",
            "created_at": "2026-07-18T10:00:00Z",
        },
        {
            "ticker": "BAC",
            "decision_date": "2026-07-09",
            "decision_verdict": "Buy",
            "realized_return": 0.0,
            "classification": "neutral",
            "lesson_text": "Buy decision resulted in a +0.00% return...",
            "created_at": "2026-07-19T10:00:00Z",
        },
        {
            "ticker": "GS",
            "decision_date": "2026-07-10",
            "decision_verdict": "Buy",
            "realized_return": 0.005,
            "classification": "neutral",
            "lesson_text": "Buy decision resulted in a +0.50% return...",
            "created_at": "2026-07-20T10:00:00Z",
        },
        # 2 Sell / positive (Auditor said Sell but stock went up — missed opportunity)
        {
            "ticker": "XOM",
            "decision_date": "2026-07-11",
            "decision_verdict": "Sell",
            "realized_return": 0.04,
            "classification": "positive",
            "lesson_text": "Sell decision resulted in a +4.00% return...",
            "created_at": "2026-07-21T10:00:00Z",
        },
        {
            "ticker": "CVX",
            "decision_date": "2026-07-12",
            "decision_verdict": "Sell",
            "realized_return": 0.02,
            "classification": "positive",
            "lesson_text": "Sell decision resulted in a +2.00% return...",
            "created_at": "2026-07-22T10:00:00Z",
        },
        # 3 Sell / negative or neutral
        {
            "ticker": "WMT",
            "decision_date": "2026-07-13",
            "decision_verdict": "Sell",
            "realized_return": -0.02,
            "classification": "negative",
            "lesson_text": "Sell decision resulted in a -2.00% return...",
            "created_at": "2026-07-23T10:00:00Z",
        },
        {
            "ticker": "KO",
            "decision_date": "2026-07-14",
            "decision_verdict": "Sell",
            "realized_return": -0.01,
            "classification": "negative",
            "lesson_text": "Sell decision resulted in a -1.00% return...",
            "created_at": "2026-07-24T10:00:00Z",
        },
        {
            "ticker": "PEP",
            "decision_date": "2026-07-15",
            "decision_verdict": "Sell",
            "realized_return": 0.005,
            "classification": "neutral",
            "lesson_text": "Sell decision resulted in a +0.50% return...",
            "created_at": "2026-07-25T10:00:00Z",
        },
        # 5 Hold records
        {
            "ticker": "DIS",
            "decision_date": "2026-07-16",
            "decision_verdict": "Hold",
            "realized_return": 0.01,
            "classification": "positive",
            "lesson_text": "Hold decision resulted in a +1.00% return...",
            "created_at": "2026-07-26T10:00:00Z",
        },
        {
            "ticker": "MCD",
            "decision_date": "2026-07-17",
            "decision_verdict": "Hold",
            "realized_return": -0.01,
            "classification": "negative",
            "lesson_text": "Hold decision resulted in a -1.00% return...",
            "created_at": "2026-07-27T10:00:00Z",
        },
        {
            "ticker": "NFLX",
            "decision_date": "2026-07-18",
            "decision_verdict": "Hold",
            "realized_return": 0.005,
            "classification": "neutral",
            "lesson_text": "Hold decision resulted in a +0.50% return...",
            "created_at": "2026-07-28T10:00:00Z",
        },
        {
            "ticker": "PYPL",
            "decision_date": "2026-07-19",
            "decision_verdict": "Hold",
            "realized_return": 0.02,
            "classification": "positive",
            "lesson_text": "Hold decision resulted in a +2.00% return...",
            "created_at": "2026-07-29T10:00:00Z",
        },
        {
            "ticker": "COIN",
            "decision_date": "2026-07-20",
            "decision_verdict": "Hold",
            "realized_return": -0.03,
            "classification": "negative",
            "lesson_text": "Hold decision resulted in a -3.00% return...",
            "created_at": "2026-07-30T10:00:00Z",
        },
    ]

    with open(fixture_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return fixture_file


class TestAuditorFalsePositiveAnalyzer:
    """Tests for the auditor_false_positive_analyzer module."""

    def test_aggregates_reflections(self, auditor_reflections_fixture: Path):
        """Test that aggregate_auditor_reflections correctly computes rates.

        Given a fixture with 3 Buy/negative, 7 Buy/positive-or-neutral,
        2 Sell/positive, 3 Sell/negative-or-neutral, and 5 Hold records,
        verify the aggregation math.
        """
        # Configure the path
        set_config({"auditor_reflections_log_path": str(auditor_reflections_fixture)})

        result = aggregate_auditor_reflections(min_samples=10)

        # Verify Buy aggregation
        assert result["buy_total"] == 10
        assert result["buy_negative_count"] == 3
        assert result["buy_negative_pct"] == pytest.approx(0.3, abs=0.01)

        # Verify Sell aggregation
        assert result["sell_total"] == 5
        assert result["sell_positive_count"] == 2
        assert result["sell_positive_pct"] == pytest.approx(0.4, abs=0.01)

        # Verify Hold count
        assert result["hold_total"] == 5

        # Verify sufficient_samples (with min_samples=10, Sell fails: 5 < 10)
        assert result["sufficient_samples"] is False
        assert result["min_samples"] == 10

        # Verify top_false_positives (up to 5 Buy/negative, sorted by return ascending)
        assert len(result["top_false_positives"]) == 3
        # Sorted ascending (worst loss first): -0.03, -0.02, -0.015
        assert result["top_false_positives"][0]["ticker"] == "AAPL"
        assert result["top_false_positives"][0]["realized_return"] == -0.03
        assert result["top_false_positives"][1]["ticker"] == "GOOGL"
        assert result["top_false_positives"][1]["realized_return"] == -0.02
        assert result["top_false_positives"][2]["ticker"] == "MSFT"
        assert result["top_false_positives"][2]["realized_return"] == -0.015

    def test_graceful_empty_file(self, tmp_path: Path):
        """Test that an empty reflections file returns zero counts without error."""
        empty_file = tmp_path / "empty_reflections.jsonl"
        empty_file.touch()

        set_config({"auditor_reflections_log_path": str(empty_file)})

        result = aggregate_auditor_reflections(min_samples=10)

        assert result["buy_total"] == 0
        assert result["buy_negative_pct"] is None
        assert result["sell_total"] == 0
        assert result["sell_positive_pct"] is None
        assert result["hold_total"] == 0
        assert result["sufficient_samples"] is False
        assert result["top_false_positives"] == []

    def test_graceful_missing_file(self):
        """Test that a missing reflections file returns zero counts without error."""
        set_config({"auditor_reflections_log_path": "/nonexistent/path/reflections.jsonl"})

        result = aggregate_auditor_reflections(min_samples=10)

        assert result["buy_total"] == 0
        assert result["buy_negative_pct"] is None
        assert result["sell_total"] == 0
        assert result["sell_positive_pct"] is None
        assert result["hold_total"] == 0
        assert result["sufficient_samples"] is False
        assert result["top_false_positives"] == []

    def test_malformed_json_line_skipped(self, tmp_path: Path):
        """Test that malformed JSON lines are logged and skipped gracefully."""
        fixture_file = tmp_path / "reflections_with_bad_json.jsonl"

        good_record = {
            "ticker": "AAPL",
            "decision_date": "2026-07-01",
            "decision_verdict": "Buy",
            "realized_return": 0.05,
            "classification": "positive",
            "lesson_text": "Buy decision resulted in a +5.00% return...",
            "created_at": "2026-07-11T10:00:00Z",
        }

        with open(fixture_file, "w") as f:
            f.write(json.dumps(good_record) + "\n")
            f.write("{ MALFORMED JSON }\n")  # Invalid JSON
            f.write(json.dumps(good_record) + "\n")

        set_config({"auditor_reflections_log_path": str(fixture_file)})

        # Should not raise, should skip the malformed line and count the good ones
        result = aggregate_auditor_reflections(min_samples=1)

        assert result["buy_total"] == 2

    def test_missing_field_in_record_skipped(self, tmp_path: Path):
        """Test that records with missing required fields are skipped."""
        fixture_file = tmp_path / "reflections_with_bad_record.jsonl"

        good_record = {
            "ticker": "AAPL",
            "decision_date": "2026-07-01",
            "decision_verdict": "Buy",
            "realized_return": 0.05,
            "classification": "positive",
            "lesson_text": "Buy decision resulted in a +5.00% return...",
            "created_at": "2026-07-11T10:00:00Z",
        }

        incomplete_record = {
            "ticker": "MSFT",
            "decision_date": "2026-07-02",
            # Missing decision_verdict
            "realized_return": -0.02,
            "classification": "negative",
            "lesson_text": "Buy decision resulted in a -2.00% return...",
            "created_at": "2026-07-12T10:00:00Z",
        }

        with open(fixture_file, "w") as f:
            f.write(json.dumps(good_record) + "\n")
            f.write(json.dumps(incomplete_record) + "\n")
            f.write(json.dumps(good_record) + "\n")

        set_config({"auditor_reflections_log_path": str(fixture_file)})

        result = aggregate_auditor_reflections(min_samples=1)

        # Only 2 good records should be counted; the incomplete one is skipped
        assert result["buy_total"] == 2

    def test_sufficient_samples_true(self, auditor_reflections_fixture: Path):
        """Test that sufficient_samples is True when both Buy and Sell >= min_samples."""
        set_config({"auditor_reflections_log_path": str(auditor_reflections_fixture)})

        # With min_samples=1, both Buy (10) and Sell (5) are >= 1
        result = aggregate_auditor_reflections(min_samples=1)

        assert result["sufficient_samples"] is True

    def test_sufficient_samples_false_buy_low(self, auditor_reflections_fixture: Path):
        """Test that sufficient_samples is False when Buy < min_samples."""
        set_config({"auditor_reflections_log_path": str(auditor_reflections_fixture)})

        # With min_samples=50, both Buy and Sell are below threshold
        result = aggregate_auditor_reflections(min_samples=50)

        assert result["sufficient_samples"] is False

    def test_sufficient_samples_false_sell_low(self, auditor_reflections_fixture: Path):
        """Test that sufficient_samples is False when Sell < min_samples."""
        set_config({"auditor_reflections_log_path": str(auditor_reflections_fixture)})

        # With min_samples=6, Buy (10) >= 6 but Sell (5) < 6
        result = aggregate_auditor_reflections(min_samples=6)

        assert result["sufficient_samples"] is False

    def test_top_false_positives_limited_to_5(self, tmp_path: Path):
        """Test that top_false_positives is limited to 5 entries."""
        fixture_file = tmp_path / "many_false_positives.jsonl"

        records = []
        for i in range(10):
            records.append(
                {
                    "ticker": f"TICK{i}",
                    "decision_date": f"2026-07-{i+1:02d}",
                    "decision_verdict": "Buy",
                    "realized_return": -(0.01 + i * 0.01),  # -0.01, -0.02, ..., -0.10
                    "classification": "negative",
                    "lesson_text": f"Buy decision resulted in a -{0.01 + i * 0.01:.2%} return...",
                    "created_at": f"2026-07-{i+11:02d}T10:00:00Z",
                }
            )

        with open(fixture_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        set_config({"auditor_reflections_log_path": str(fixture_file)})

        result = aggregate_auditor_reflections(min_samples=1)

        # Only top 5 should be returned, sorted by return ascending (worst first)
        assert len(result["top_false_positives"]) == 5
        # Worst should be -0.10
        assert result["top_false_positives"][0]["realized_return"] == pytest.approx(-0.10)
        # Best of the top 5 should be -0.06
        assert result["top_false_positives"][4]["realized_return"] == pytest.approx(-0.06)
