"""Tests for auditor_refutation_aggregator (VALID-01, Task 1).

Verifies that count_auditor_refutations() correctly:
1. Counts mismatch entries as refutations in a JSONL audit log
2. Excludes non-matching comparison_result values
3. Filters records outside the date range
4. Handles missing log files gracefully (no raise)
5. Skips malformed JSON lines (no raise)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradingagents.dataflows.config import set_config
from tradingagents.jobs.auditor_refutation_aggregator import count_auditor_refutations


@pytest.fixture
def tmp_auditor_log(tmp_path: Path):
    """Provide a temporary directory for auditor log fixtures."""
    return tmp_path / "auditor" / "audits.jsonl"


def test_counts_mismatch_as_refutation(tmp_auditor_log: Path):
    """Test that mismatch entries are correctly counted as refutations."""
    # Create log directory
    tmp_auditor_log.parent.mkdir(parents=True, exist_ok=True)

    # Fixture: 5 records with mixed comparison_result values
    records = [
        {
            "ticker": "AAPL",
            "trade_date": "2026-08-11",
            "comparison_result": "match",
            "auditor_reasoning": "Matches Research",
        },
        {
            "ticker": "GOOGL",
            "trade_date": "2026-08-12",
            "comparison_result": "mismatch",
            "auditor_reasoning": "Refuted momentum thesis",
        },
        {
            "ticker": "MSFT",
            "trade_date": "2026-08-13",
            "comparison_result": "match",
            "auditor_reasoning": "Matches Research",
        },
        {
            "ticker": "TSLA",
            "trade_date": "2026-08-14",
            "comparison_result": "mismatch",
            "auditor_reasoning": "Data contradiction found",
        },
        {
            "ticker": "XOM",
            "trade_date": "2026-08-15",
            "comparison_result": "not_reached",
            "auditor_reasoning": "Insufficient data",
        },
    ]

    # Write to JSONL
    with open(tmp_auditor_log, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Configure to use this log
    set_config({"auditor_log_path": str(tmp_auditor_log)})

    # Test: call count_auditor_refutations
    result = count_auditor_refutations("2026-08-11", "2026-08-15")

    # Assert
    assert result["total_cycles"] == 5
    assert result["refutation_count"] == 2
    assert result["at_least_one_refutation"] is True
    assert len(result["refutations"]) == 2
    assert result["start_date"] == "2026-08-11"
    assert result["end_date"] == "2026-08-15"

    # Verify the refutations list has correct entries
    refutations = result["refutations"]
    assert refutations[0]["ticker"] == "GOOGL"
    assert refutations[0]["trade_date"] == "2026-08-12"
    assert refutations[0]["auditor_reasoning"] == "Refuted momentum thesis"
    assert refutations[1]["ticker"] == "TSLA"
    assert refutations[1]["trade_date"] == "2026-08-14"


def test_no_refutations_in_window(tmp_auditor_log: Path):
    """Test that when no mismatches exist, the result is correct."""
    tmp_auditor_log.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "ticker": "AAPL",
            "trade_date": "2026-08-11",
            "comparison_result": "match",
            "auditor_reasoning": "All good",
        },
        {
            "ticker": "GOOGL",
            "trade_date": "2026-08-12",
            "comparison_result": "match",
            "auditor_reasoning": "All good",
        },
        {
            "ticker": "MSFT",
            "trade_date": "2026-08-13",
            "comparison_result": "not_reached",
            "auditor_reasoning": "Inconclusive",
        },
    ]

    with open(tmp_auditor_log, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    set_config({"auditor_log_path": str(tmp_auditor_log)})

    result = count_auditor_refutations("2026-08-11", "2026-08-13")

    assert result["refutation_count"] == 0
    assert result["at_least_one_refutation"] is False
    assert result["refutations"] == []
    assert result["total_cycles"] == 3


def test_excludes_records_outside_date_range(tmp_auditor_log: Path):
    """Test that records outside [start_date, end_date] are excluded."""
    tmp_auditor_log.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "ticker": "AAPL",
            "trade_date": "2026-08-10",
            "comparison_result": "mismatch",
            "auditor_reasoning": "Outside window",
        },
        {
            "ticker": "GOOGL",
            "trade_date": "2026-08-11",
            "comparison_result": "mismatch",
            "auditor_reasoning": "Inside window",
        },
        {
            "ticker": "MSFT",
            "trade_date": "2026-08-15",
            "comparison_result": "match",
            "auditor_reasoning": "Inside window",
        },
        {
            "ticker": "TSLA",
            "trade_date": "2026-08-16",
            "comparison_result": "mismatch",
            "auditor_reasoning": "Outside window",
        },
    ]

    with open(tmp_auditor_log, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    set_config({"auditor_log_path": str(tmp_auditor_log)})

    result = count_auditor_refutations("2026-08-11", "2026-08-15")

    # Only records with trade_date in [2026-08-11, 2026-08-15] should be counted
    assert result["total_cycles"] == 2  # 2026-08-11 (GOOGL), 2026-08-15 (MSFT)
    assert result["refutation_count"] == 1  # Only 2026-08-11 (GOOGL) mismatch
    assert len(result["refutations"]) == 1
    assert result["refutations"][0]["ticker"] == "GOOGL"


def test_missing_log_file_returns_zero_not_raise(tmp_path: Path):
    """Test that missing log file returns zero-result dict, never raises."""
    nonexistent_log = tmp_path / "does_not_exist.jsonl"

    set_config({"auditor_log_path": str(nonexistent_log)})

    # This should not raise
    result = count_auditor_refutations("2026-08-11", "2026-08-15")

    assert result["total_cycles"] == 0
    assert result["refutation_count"] == 0
    assert result["at_least_one_refutation"] is False
    assert result["refutations"] == []
    assert result["start_date"] == "2026-08-11"
    assert result["end_date"] == "2026-08-15"


def test_malformed_json_line_skipped_not_raise(tmp_auditor_log: Path):
    """Test that malformed JSON lines are skipped without raising."""
    tmp_auditor_log.parent.mkdir(parents=True, exist_ok=True)

    # Write a mix of valid and invalid lines
    lines = [
        json.dumps(
            {
                "ticker": "AAPL",
                "trade_date": "2026-08-11",
                "comparison_result": "match",
                "auditor_reasoning": "Valid",
            }
        ),
        "this is not valid json at all {{{",  # Malformed JSON
        json.dumps(
            {
                "ticker": "GOOGL",
                "trade_date": "2026-08-12",
                "comparison_result": "mismatch",
                "auditor_reasoning": "Valid",
            }
        ),
        '"just_a_string_not_an_object"',  # Valid JSON but not an object
        json.dumps(
            {
                "ticker": "MSFT",
                "trade_date": "2026-08-13",
                "comparison_result": "match",
                "auditor_reasoning": "Valid",
            }
        ),
    ]

    with open(tmp_auditor_log, "w") as f:
        for line in lines:
            f.write(line + "\n")

    set_config({"auditor_log_path": str(tmp_auditor_log)})

    # Should not raise, should skip malformed lines
    result = count_auditor_refutations("2026-08-11", "2026-08-13")

    # Only the 3 valid dict records should be counted
    assert result["total_cycles"] == 3
    assert result["refutation_count"] == 1
    assert len(result["refutations"]) == 1
    assert result["refutations"][0]["ticker"] == "GOOGL"
