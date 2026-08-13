"""Unit tests for Auditor compare node — tolerance boundaries and unconditional persistence."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.agents.auditors.thesis_comparator import (
    create_auditor_compare_node,
)


@pytest.mark.unit
class TestAuditorCompareNode:
    """Test Auditor comparison logic and tolerance boundaries (D-11)."""

    def test_comparison_match(self):
        """Matching verdicts produce comparison_result='match'."""
        compare_node = create_auditor_compare_node()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Buy",
            "thesis_verdict": "Buy",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": 2.3,
            "auditor_data_points": {"retorno_12_1": 0.1834, "z_score": 2.3},
            "auditor_phase1_status": "pass",
        }

        result = compare_node(state)

        assert result["comparison_result"] == "match"

    def test_comparison_mismatch(self):
        """Differing verdicts produce comparison_result='mismatch'."""
        compare_node = create_auditor_compare_node()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Sell",
            "thesis_verdict": "Buy",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": 2.3,
            "auditor_data_points": {"retorno_12_1": 0.1834, "z_score": 2.3},
            "auditor_phase1_status": "pass",
        }

        result = compare_node(state)

        assert result["comparison_result"] == "mismatch"

    def test_comparison_inconclusive_not_reached(self):
        """INCONCLUSIVE verdict produces comparison_result='not_reached' without touching tolerance logic."""
        compare_node = create_auditor_compare_node()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "INCONCLUSIVE",
            "thesis_verdict": "Buy",
            "auditor_retorno_12_1": 0.0,
            "auditor_z_score": 0.0,
            "auditor_data_points": {},
            "auditor_phase1_status": "fail",
        }

        result = compare_node(state)

        assert result["comparison_result"] == "not_reached"
        assert result["auditor_verified"] is False


@pytest.mark.unit
class TestAuditorToleranceBoundaries:
    """Test D-11 tolerance boundaries for auditor_verified (mirroring Research's boundaries)."""

    def test_tolerance_boundary_retorno_within(self):
        """Retorno diff of 0.00009 passes tolerance (within ±0.0001)."""
        compare_node = create_auditor_compare_node()

        base_retorno = 0.1834
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Buy",
            "thesis_verdict": "Buy",
            "auditor_retorno_12_1": base_retorno,
            "auditor_z_score": 2.3,
            "auditor_data_points": {
                "retorno_12_1": base_retorno + 0.00009,  # Just within tolerance
                "z_score": 2.3,
            },
            "auditor_phase1_status": "pass",
        }

        result = compare_node(state)

        assert result["auditor_verified"] is True, "Diff of 0.00009 should pass"

    def test_tolerance_boundary_retorno_outside(self):
        """Retorno diff of 0.00011 fails tolerance (outside ±0.0001)."""
        compare_node = create_auditor_compare_node()

        base_retorno = 0.1834
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Buy",
            "thesis_verdict": "Buy",
            "auditor_retorno_12_1": base_retorno,
            "auditor_z_score": 2.3,
            "auditor_data_points": {
                "retorno_12_1": base_retorno + 0.00011,  # Just outside tolerance
                "z_score": 2.3,
            },
            "auditor_phase1_status": "pass",
        }

        result = compare_node(state)

        assert result["auditor_verified"] is False, "Diff of 0.00011 should fail"

    def test_tolerance_boundary_z_score_within(self):
        """Z-score diff of 0.009 passes tolerance (within ±0.01)."""
        compare_node = create_auditor_compare_node()

        base_z = 2.3
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Buy",
            "thesis_verdict": "Buy",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": base_z,
            "auditor_data_points": {
                "retorno_12_1": 0.1834,
                "z_score": base_z + 0.009,  # Just within tolerance
            },
            "auditor_phase1_status": "pass",
        }

        result = compare_node(state)

        assert result["auditor_verified"] is True, "Diff of 0.009 should pass"

    def test_tolerance_boundary_z_score_outside(self):
        """Z-score diff of 0.011 fails tolerance (outside ±0.01)."""
        compare_node = create_auditor_compare_node()

        base_z = 2.3
        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Buy",
            "thesis_verdict": "Buy",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": base_z,
            "auditor_data_points": {
                "retorno_12_1": 0.1834,
                "z_score": base_z + 0.011,  # Just outside tolerance
            },
            "auditor_phase1_status": "pass",
        }

        result = compare_node(state)

        assert result["auditor_verified"] is False, "Diff of 0.011 should fail"


@pytest.mark.unit
class TestAuditorPersistence:
    """Test unconditional JSONL persistence for all comparison results (D-06)."""

    def test_match_cycle_persists(self, tmp_path):
        """Matching verdict cycle persists to JSONL with comparison_result='match'."""
        compare_node = create_auditor_compare_node()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Buy",
            "thesis_verdict": "Buy",
            "auditor_phase1_status": "pass",
            "auditor_phase1_failure_reason": "",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": 2.3,
            "auditor_confidence_level": "high",
            "auditor_reasoning": "Momentum is strong.",
            "auditor_refutation_criterion": "Price breaks 50-day MA.",
            "auditor_data_points": {"retorno_12_1": 0.1834, "z_score": 2.3},
        }

        with patch("tradingagents.agents.auditors.thesis_comparator.get_config") as mock_config:
            log_path = str(tmp_path / "audits.jsonl")
            mock_config.return_value = {"auditor_log_path": log_path}

            compare_node(state)

            # Verify JSONL record was written
            assert Path(log_path).exists(), f"JSONL log should exist at {log_path}"
            with open(log_path) as f:
                records = [json.loads(line) for line in f]

            assert len(records) == 1, "Should have exactly one record"
            record = records[0]
            assert record["comparison_result"] == "match"
            assert record["ticker"] == "AAPL"

    def test_mismatch_cycle_persists(self, tmp_path):
        """Mismatching verdict cycle ALSO persists to JSONL with comparison_result='mismatch' (D-06 regression guard)."""
        compare_node = create_auditor_compare_node()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Sell",
            "thesis_verdict": "Buy",
            "auditor_phase1_status": "pass",
            "auditor_phase1_failure_reason": "",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": 2.3,
            "auditor_confidence_level": "high",
            "auditor_reasoning": "Momentum could reverse.",
            "auditor_refutation_criterion": "Price breaks support.",
            "auditor_data_points": {"retorno_12_1": 0.1834, "z_score": 2.3},
        }

        with patch("tradingagents.agents.auditors.thesis_comparator.get_config") as mock_config:
            log_path = str(tmp_path / "audits.jsonl")
            mock_config.return_value = {"auditor_log_path": log_path}

            compare_node(state)

            # Verify JSONL record was written
            assert Path(log_path).exists(), f"JSONL log should exist at {log_path}"
            with open(log_path) as f:
                records = [json.loads(line) for line in f]

            assert len(records) == 1, "Should have exactly one record"
            record = records[0]
            assert record["comparison_result"] == "mismatch", "Mismatch MUST be persisted (D-06)"
            assert record["ticker"] == "AAPL"

    def test_not_reached_cycle_persists(self, tmp_path):
        """Inconclusive (not_reached) cycle ALSO persists to JSONL with comparison_result='not_reached' (D-06)."""
        compare_node = create_auditor_compare_node()

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
            "auditor_verdict": "INCONCLUSIVE",
            "thesis_verdict": "Buy",
            "auditor_phase1_status": "fail",
            "auditor_phase1_failure_reason": "insufficient_ohlcv_history",
            "auditor_retorno_12_1": 0.0,
            "auditor_z_score": 0.0,
            "auditor_confidence_level": "low",
            "auditor_reasoning": "Compute phase failed, LLM not invoked.",
            "auditor_refutation_criterion": "",
            "auditor_data_points": {},
        }

        with patch("tradingagents.agents.auditors.thesis_comparator.get_config") as mock_config:
            log_path = str(tmp_path / "audits.jsonl")
            mock_config.return_value = {"auditor_log_path": log_path}

            compare_node(state)

            # Verify JSONL record was written
            assert Path(log_path).exists(), f"JSONL log should exist at {log_path}"
            with open(log_path) as f:
                records = [json.loads(line) for line in f]

            assert len(records) == 1, "Should have exactly one record"
            record = records[0]
            assert record["comparison_result"] == "not_reached", "Not-reached MUST be persisted (D-06)"
            assert record["ticker"] == "AAPL"

    def test_persistence_record_structure(self, tmp_path):
        """JSONL record contains all required fields (D-06, D-13)."""
        compare_node = create_auditor_compare_node()

        state = {
            "company_of_interest": "MSFT",
            "trade_date": "2026-07-29",
            "auditor_verdict": "Hold",
            "thesis_verdict": "Buy",
            "auditor_phase1_status": "pass",
            "auditor_phase1_failure_reason": "",
            "auditor_retorno_12_1": 0.15,
            "auditor_z_score": 1.8,
            "auditor_confidence_level": "medium",
            "auditor_reasoning": "Momentum is moderate.",
            "auditor_refutation_criterion": "Needs further analysis.",
            "auditor_data_points": {"retorno_12_1": 0.15, "z_score": 1.8},
        }

        with patch("tradingagents.agents.auditors.thesis_comparator.get_config") as mock_config:
            log_path = str(tmp_path / "audits.jsonl")
            mock_config.return_value = {"auditor_log_path": log_path}

            compare_node(state)

            with open(log_path) as f:
                record = json.loads(f.readline())

            # Verify required fields are present
            required_fields = [
                "timestamp",
                "ticker",
                "trade_date",
                "phase1_status",
                "phase1_failure_reason",
                "auditor_retorno_12_1",
                "auditor_z_score",
                "auditor_verdict",
                "auditor_reasoning",
                "auditor_refutation_criterion",
                "research_verdict",
                "auditor_data_points",
                "comparison_result",
                "verified",
            ]

            for field in required_fields:
                assert field in record, f"Record must contain {field}"

            # Verify specific values
            assert record["ticker"] == "MSFT"
            assert record["auditor_retorno_12_1"] == 0.15
            assert record["verified"] is True  # Within tolerance
