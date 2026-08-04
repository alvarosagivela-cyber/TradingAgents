"""Tests for the realized outcome processor job.

Tests validate that the job correctly reads per-layer decision logs, computes
realized outcomes, and idempotently appends reflections to isolated per-layer stores.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tradingagents.dataflows.symbol_utils import NoMarketDataError
from tradingagents.jobs.realized_outcome_processor import (
    _extract_auditor_verdict,
    _extract_research_verdict,
    _extract_risk_verdict,
    _existing_reflection_keys,
    _read_jsonl,
    process_pending_outcomes,
)
from tradingagents.agents.reflectors.reflection_schema import ReflectionRecord


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """Fixture that patches config to use temp directories."""
    from tradingagents.dataflows.config import set_config

    research_log = tmp_path / "research.jsonl"
    auditor_log = tmp_path / "auditor.jsonl"
    risk_log = tmp_path / "risk.jsonl"
    research_reflections = tmp_path / "research_reflections.jsonl"
    auditor_reflections = tmp_path / "auditor_reflections.jsonl"
    risk_reflections = tmp_path / "risk_reflections.jsonl"

    set_config({
        "research_thesis_log_path": str(research_log),
        "auditor_log_path": str(auditor_log),
        "risk_log_path": str(risk_log),
        "research_reflections_log_path": str(research_reflections),
        "auditor_reflections_log_path": str(auditor_reflections),
        "risk_reflections_log_path": str(risk_reflections),
    })

    return {
        "research_log": research_log,
        "auditor_log": auditor_log,
        "risk_log": risk_log,
        "research_reflections": research_reflections,
        "auditor_reflections": auditor_reflections,
        "risk_reflections": risk_reflections,
    }


class TestVerdictExtraction:
    """Tests for verdict extraction functions."""

    def test_extract_research_verdict(self):
        """Research verdict comes from 'verdict' field."""
        record = {"ticker": "AAPL", "trade_date": "2026-07-01", "verdict": "Buy"}
        assert _extract_research_verdict(record) == "Buy"

    def test_extract_research_verdict_none_when_missing(self):
        """Research verdict returns None when 'verdict' field is missing."""
        record = {"ticker": "AAPL"}
        assert _extract_research_verdict(record) is None

    def test_extract_auditor_verdict(self):
        """Auditor verdict comes from 'auditor_verdict' field."""
        record = {
            "ticker": "AAPL",
            "trade_date": "2026-07-01",
            "auditor_verdict": "Sell",
            "research_verdict": "Buy",
        }
        assert _extract_auditor_verdict(record) == "Sell"

    def test_extract_auditor_verdict_inconclusive_returns_none(self):
        """Auditor verdict returns None when 'auditor_verdict' is INCONCLUSIVE."""
        record = {
            "ticker": "AAPL",
            "auditor_verdict": "INCONCLUSIVE",
            "research_verdict": "Buy",
        }
        assert _extract_auditor_verdict(record) is None

    def test_extract_auditor_verdict_missing_returns_none(self):
        """Auditor verdict returns None when 'auditor_verdict' field is missing."""
        record = {"ticker": "AAPL"}
        assert _extract_auditor_verdict(record) is None

    def test_extract_risk_verdict_veto(self):
        """Risk verdict returns 'VETO' when final_veto is True."""
        record = {"ticker": "AAPL", "final_veto": True}
        assert _extract_risk_verdict(record) == "VETO"

    def test_extract_risk_verdict_approve(self):
        """Risk verdict returns 'APPROVE' when final_veto is False."""
        record = {"ticker": "AAPL", "final_veto": False}
        assert _extract_risk_verdict(record) == "APPROVE"


class TestReadJsonl:
    """Tests for JSONL reading utility."""

    def test_read_jsonl_single_line(self, tmp_path):
        """Read a single JSON line."""
        log_file = tmp_path / "test.jsonl"
        log_file.write_text('{"ticker": "AAPL", "value": 1}\n')

        records = _read_jsonl(log_file)

        assert len(records) == 1
        assert records[0]["ticker"] == "AAPL"

    def test_read_jsonl_multiple_lines(self, tmp_path):
        """Read multiple JSON lines."""
        log_file = tmp_path / "test.jsonl"
        log_file.write_text(
            '{"ticker": "AAPL", "value": 1}\n'
            '{"ticker": "MSFT", "value": 2}\n'
        )

        records = _read_jsonl(log_file)

        assert len(records) == 2
        assert records[0]["ticker"] == "AAPL"
        assert records[1]["ticker"] == "MSFT"

    def test_read_jsonl_missing_file(self, tmp_path):
        """Missing file returns empty list."""
        log_file = tmp_path / "nonexistent.jsonl"

        records = _read_jsonl(log_file)

        assert records == []

    def test_read_jsonl_skip_malformed_line(self, tmp_path):
        """Skip malformed JSON lines."""
        log_file = tmp_path / "test.jsonl"
        log_file.write_text(
            '{"ticker": "AAPL", "value": 1}\n'
            'INVALID JSON\n'
            '{"ticker": "MSFT", "value": 2}\n'
        )

        records = _read_jsonl(log_file)

        assert len(records) == 2
        assert records[0]["ticker"] == "AAPL"
        assert records[1]["ticker"] == "MSFT"


class TestExistingReflectionKeys:
    """Tests for checking existing reflection keys."""

    def test_existing_keys_single_record(self, tmp_path):
        """Extract existing (ticker, decision_date) pairs."""
        reflections_file = tmp_path / "reflections.jsonl"
        reflections_file.write_text(
            '{"ticker": "AAPL", "decision_date": "2026-07-01"}\n'
        )

        keys = _existing_reflection_keys(reflections_file)

        assert keys == {("AAPL", "2026-07-01")}

    def test_existing_keys_multiple_records(self, tmp_path):
        """Extract multiple (ticker, decision_date) pairs."""
        reflections_file = tmp_path / "reflections.jsonl"
        reflections_file.write_text(
            '{"ticker": "AAPL", "decision_date": "2026-07-01"}\n'
            '{"ticker": "MSFT", "decision_date": "2026-07-02"}\n'
            '{"ticker": "AAPL", "decision_date": "2026-07-05"}\n'
        )

        keys = _existing_reflection_keys(reflections_file)

        assert keys == {
            ("AAPL", "2026-07-01"),
            ("MSFT", "2026-07-02"),
            ("AAPL", "2026-07-05"),
        }

    def test_existing_keys_missing_file(self, tmp_path):
        """Missing reflections file returns empty set."""
        reflections_file = tmp_path / "nonexistent.jsonl"

        keys = _existing_reflection_keys(reflections_file)

        assert keys == set()


class TestResearchLayerHappyPath:
    """Test 1: Research layer, happy path."""

    def test_research_layer_writes_to_research_store_only(
        self, temp_config, monkeypatch
    ):
        """Running process_pending_outcomes on research log writes to research store only."""
        from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes

        # Mock compute_realized_return to return 0.03 for any call
        def mock_compute(ticker, decision_date, window_days=10):
            return 0.03

        monkeypatch.setattr(
            "tradingagents.jobs.realized_outcome_processor.compute_realized_return",
            mock_compute,
        )

        # Write one research decision
        temp_config["research_log"].write_text(
            json.dumps({
                "timestamp": "2026-08-04T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-01",
                "verdict": "Buy",
                "retorno_12_1": 0.18,
                "momentum_z_score": 2.3,
            })
            + "\n"
        )

        # Run the job
        result = process_pending_outcomes()

        # Verify results
        assert result["research"] == 1
        assert result["auditor"] == 0
        assert result["risk"] == 0

        # Verify only research reflections file has content
        research_reflections = temp_config["research_reflections"].read_text()
        assert research_reflections.count("\n") == 1  # One line written

        auditor_reflections = (
            temp_config["auditor_reflections"].read_text()
            if temp_config["auditor_reflections"].exists()
            else ""
        )
        assert auditor_reflections == ""

        risk_reflections = (
            temp_config["risk_reflections"].read_text()
            if temp_config["risk_reflections"].exists()
            else ""
        )
        assert risk_reflections == ""

        # Verify the reflection record content
        reflection_line = research_reflections.strip()
        reflection_dict = json.loads(reflection_line)
        assert reflection_dict["ticker"] == "AAPL"
        assert reflection_dict["decision_date"] == "2026-07-01"
        assert reflection_dict["decision_verdict"] == "Buy"
        assert reflection_dict["realized_return"] == 0.03
        assert reflection_dict["classification"] == "positive"


class TestAuditorLayerInconclusiveSkipped:
    """Test 2: Auditor layer, INCONCLUSIVE skipped."""

    def test_auditor_inconclusive_verdict_skipped(self, temp_config, monkeypatch):
        """Auditor INCONCLUSIVE records are not processed."""
        from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes

        mock_compute = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        monkeypatch.setattr(
            "tradingagents.jobs.realized_outcome_processor.compute_realized_return",
            mock_compute,
        )

        # Write one auditor decision with INCONCLUSIVE verdict
        temp_config["auditor_log"].write_text(
            json.dumps({
                "timestamp": "2026-08-04T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-01",
                "auditor_verdict": "INCONCLUSIVE",
                "research_verdict": "Buy",
                "comparison_result": "not_reached",
            })
            + "\n"
        )

        # Run the job
        result = process_pending_outcomes()

        # Verify no processing happened
        assert result["auditor"] == 0
        mock_compute.assert_not_called()
        auditor_reflections = (
            temp_config["auditor_reflections"].read_text()
            if temp_config["auditor_reflections"].exists()
            else ""
        )
        assert auditor_reflections == ""


class TestRiskLayerVerdictDerivation:
    """Test 3: Risk layer, verdict derivation."""

    def test_risk_veto_true_produces_veto_verdict(
        self, temp_config, monkeypatch
    ):
        """Risk record with final_veto=true produces VETO decision verdict."""
        from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes

        def mock_compute(ticker, decision_date, window_days=10):
            return 0.02

        monkeypatch.setattr(
            "tradingagents.jobs.realized_outcome_processor.compute_realized_return",
            mock_compute,
        )

        # Write one risk decision with final_veto=true
        temp_config["risk_log"].write_text(
            json.dumps({
                "timestamp": "2026-08-04T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-01",
                "proposed_side": "Buy",
                "final_veto": True,
                "conservative_verdict": {"verdict": "VETO"},
                "balanced_verdict": {"verdict": "APPROVE"},
                "aggressive_verdict": {"verdict": "APPROVE"},
            })
            + "\n"
        )

        # Run the job
        result = process_pending_outcomes()

        assert result["risk"] == 1

        # Verify the reflection record has VETO as decision_verdict
        risk_reflections = temp_config["risk_reflections"].read_text()
        reflection_dict = json.loads(risk_reflections.strip())
        assert reflection_dict["decision_verdict"] == "VETO"

    def test_risk_veto_false_produces_approve_verdict(
        self, temp_config, monkeypatch
    ):
        """Risk record with final_veto=false produces APPROVE decision verdict."""
        from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes

        def mock_compute(ticker, decision_date, window_days=10):
            return 0.02

        monkeypatch.setattr(
            "tradingagents.jobs.realized_outcome_processor.compute_realized_return",
            mock_compute,
        )

        # Write one risk decision with final_veto=false
        temp_config["risk_log"].write_text(
            json.dumps({
                "timestamp": "2026-08-04T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-01",
                "proposed_side": "Buy",
                "final_veto": False,
                "conservative_verdict": {"verdict": "APPROVE"},
                "balanced_verdict": {"verdict": "APPROVE"},
                "aggressive_verdict": {"verdict": "APPROVE"},
            })
            + "\n"
        )

        # Run the job
        result = process_pending_outcomes()

        assert result["risk"] == 1

        # Verify the reflection record has APPROVE as decision_verdict
        risk_reflections = temp_config["risk_reflections"].read_text()
        reflection_dict = json.loads(risk_reflections.strip())
        assert reflection_dict["decision_verdict"] == "APPROVE"


class TestIdempotency:
    """Test 4: Idempotency — no duplicate reflections on re-run."""

    def test_rerun_does_not_duplicate_reflections(
        self, temp_config, monkeypatch
    ):
        """Running process_pending_outcomes twice does not duplicate reflections."""
        from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes

        def mock_compute(ticker, decision_date, window_days=10):
            return 0.03

        monkeypatch.setattr(
            "tradingagents.jobs.realized_outcome_processor.compute_realized_return",
            mock_compute,
        )

        # Write one research decision
        temp_config["research_log"].write_text(
            json.dumps({
                "timestamp": "2026-08-04T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-01",
                "verdict": "Buy",
                "retorno_12_1": 0.18,
                "momentum_z_score": 2.3,
            })
            + "\n"
        )

        # First run
        result1 = process_pending_outcomes()
        assert result1["research"] == 1

        research_reflections_1 = temp_config["research_reflections"].read_text()
        line_count_1 = research_reflections_1.count("\n")

        # Second run (same decision log, unchanged)
        result2 = process_pending_outcomes()
        assert result2["research"] == 0

        research_reflections_2 = temp_config["research_reflections"].read_text()
        line_count_2 = research_reflections_2.count("\n")

        # Line count should not increase
        assert line_count_1 == line_count_2


class TestWindowNotElapsedSkipped:
    """Test 5: Window not yet elapsed, skip and continue."""

    def test_no_market_data_error_skipped_and_next_processed(
        self, temp_config, monkeypatch
    ):
        """NoMarketDataError is caught, logged, and next decision is still processed."""
        from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes

        def mock_compute(ticker, decision_date, window_days=10):
            if decision_date == "2026-07-01":
                raise NoMarketDataError(
                    ticker, None, f"Insufficient trading days for {ticker}"
                )
            elif decision_date == "2026-07-05":
                return 0.05
            else:
                raise ValueError("Unexpected decision_date")

        monkeypatch.setattr(
            "tradingagents.jobs.realized_outcome_processor.compute_realized_return",
            mock_compute,
        )

        # Write two research decisions
        temp_config["research_log"].write_text(
            json.dumps({
                "timestamp": "2026-08-04T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-01",
                "verdict": "Buy",
                "retorno_12_1": 0.18,
                "momentum_z_score": 2.3,
            })
            + "\n"
            + json.dumps({
                "timestamp": "2026-08-04T10:05:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-05",
                "verdict": "Hold",
                "retorno_12_1": 0.15,
                "momentum_z_score": 1.8,
            })
            + "\n"
        )

        # Run the job
        result = process_pending_outcomes()

        # Only the second decision should be processed
        assert result["research"] == 1

        research_reflections = temp_config["research_reflections"].read_text()
        lines = research_reflections.strip().split("\n")
        assert len(lines) == 1
        reflection_dict = json.loads(lines[0])
        assert reflection_dict["decision_date"] == "2026-07-05"


class TestDryRunMode:
    """Test 6: dry_run flag."""

    def test_dry_run_computes_but_does_not_write(
        self, temp_config, monkeypatch
    ):
        """dry_run=True computes results but writes nothing to reflection stores."""
        from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes

        def mock_compute(ticker, decision_date, window_days=10):
            return 0.03

        monkeypatch.setattr(
            "tradingagents.jobs.realized_outcome_processor.compute_realized_return",
            mock_compute,
        )

        # Write one research decision
        temp_config["research_log"].write_text(
            json.dumps({
                "timestamp": "2026-08-04T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-01",
                "verdict": "Buy",
                "retorno_12_1": 0.18,
                "momentum_z_score": 2.3,
            })
            + "\n"
        )

        # Run in dry_run mode
        result = process_pending_outcomes(dry_run=True)

        # Count should still reflect what WOULD have been written
        assert result["research"] == 1

        # But no bytes should be written to the store
        research_reflections = (
            temp_config["research_reflections"].read_text()
            if temp_config["research_reflections"].exists()
            else ""
        )
        assert research_reflections == ""


class TestSinceDateFilter:
    """Test 7: since_date filter."""

    def test_since_date_filters_records(self, temp_config, monkeypatch):
        """only records at-or-after since_date are processed."""
        from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes

        def mock_compute(ticker, decision_date, window_days=10):
            return 0.03

        monkeypatch.setattr(
            "tradingagents.jobs.realized_outcome_processor.compute_realized_return",
            mock_compute,
        )

        # Write two research decisions
        temp_config["research_log"].write_text(
            json.dumps({
                "timestamp": "2026-08-04T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-07-01",
                "verdict": "Buy",
                "retorno_12_1": 0.18,
                "momentum_z_score": 2.3,
            })
            + "\n"
            + json.dumps({
                "timestamp": "2026-08-04T10:05:00Z",
                "ticker": "MSFT",
                "trade_date": "2026-07-05",
                "verdict": "Hold",
                "retorno_12_1": 0.15,
                "momentum_z_score": 1.8,
            })
            + "\n"
        )

        # Run with since_date filter
        result = process_pending_outcomes(since_date="2026-07-05")

        # Only the second decision should be processed
        assert result["research"] == 1

        research_reflections = temp_config["research_reflections"].read_text()
        lines = research_reflections.strip().split("\n")
        assert len(lines) == 1
        reflection_dict = json.loads(lines[0])
        assert reflection_dict["ticker"] == "MSFT"
