"""Unit tests for decision reconstruction (Phase 6, D-02).

Tests the join of the three existing per-layer JSONL decision logs
(research, auditor, risk) by ticker+trade_date into a single readable report.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.jobs.decision_reconstructor import reconstruct_decision


@pytest.mark.unit
class TestReconstructDecision:
    """Test decision reconstruction by joining three per-layer JSONL logs."""

    @pytest.fixture
    def temp_logs(self):
        """Create temporary JSONL log files for testing."""
        tmpdir = Path(tempfile.mkdtemp())

        # Create research log
        research_log = tmpdir / "research.jsonl"
        research_records = [
            {
                "timestamp": "2026-08-01T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-08-01",
                "verdict": "Buy",
                "retorno_12_1": 0.15,
                "z_score": 2.1,
                "confidence_level": "high",
                "verified_fields": ["retorno_12_1", "z_score"],
                "reasoning": "Strong momentum signal",
                "refutation_criterion": "If momentum reverses",
                "refutation_quality": "medium",
            },
        ]
        with open(research_log, "w") as f:
            for record in research_records:
                f.write(json.dumps(record) + "\n")

        # Create auditor log
        auditor_log = tmpdir / "auditor.jsonl"
        auditor_records = [
            {
                "timestamp": "2026-08-01T10:05:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-08-01",
                "phase1_status": "pass",
                "phase1_failure_reason": None,
                "auditor_retorno_12_1": 0.15,
                "auditor_z_score": 2.1,
                "auditor_verdict": "Buy",
                "auditor_reasoning": "Confirmed momentum",
                "auditor_refutation_criterion": "If independent data diverges",
                "research_verdict": "Buy",
                "auditor_data_points": {"momentum": 2.1},
                "comparison_result": "match",
                "verified": True,
            },
        ]
        with open(auditor_log, "w") as f:
            for record in auditor_records:
                f.write(json.dumps(record) + "\n")

        # Create risk log
        risk_log = tmpdir / "risk.jsonl"
        risk_records = [
            {
                "timestamp": "2026-08-01T10:10:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-08-01",
                "proposed_side": "long",
                "portfolio_snapshot_status": "ok",
                "portfolio_total_value": 100000,
                "existing_position_value": 0,
                "proposed_notional_usd": 2000,
                "risk_concentration_pct": 0.02,
                "conservative_verdict": {
                    "verdict": "APPROVE",
                    "confidence": 0.8,
                    "reasoning": "Within limits",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                },
                "balanced_verdict": {
                    "verdict": "APPROVE",
                    "confidence": 0.9,
                    "reasoning": "Good risk/reward",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                },
                "aggressive_verdict": {
                    "verdict": "APPROVE",
                    "confidence": 0.95,
                    "reasoning": "Positive setup",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                },
                "final_veto": False,
                "concentration_verified": True,
            },
        ]
        with open(risk_log, "w") as f:
            for record in risk_records:
                f.write(json.dumps(record) + "\n")

        yield {
            "research_log": research_log,
            "auditor_log": auditor_log,
            "risk_log": risk_log,
            "tmpdir": tmpdir,
        }

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_joins_logs(self, temp_logs):
        """Joins three logs by ticker+trade_date into a single dict."""
        config = {
            "research_thesis_log_path": str(temp_logs["research_log"]),
            "auditor_log_path": str(temp_logs["auditor_log"]),
            "risk_log_path": str(temp_logs["risk_log"]),
        }

        with patch("tradingagents.jobs.decision_reconstructor.get_config", return_value=config):
            result = reconstruct_decision("AAPL", "2026-08-01")

        # Should return a dict with ticker, trade_date, and three layer keys
        assert result["ticker"] == "AAPL"
        assert result["trade_date"] == "2026-08-01"

        # All three layers should be populated
        assert result["research"] is not None
        assert result["auditor"] is not None
        assert result["risk"] is not None

        # Research record should have the expected fields
        assert result["research"]["verdict"] == "Buy"
        assert result["research"]["retorno_12_1"] == 0.15

        # Auditor record should be present
        assert result["auditor"]["auditor_verdict"] == "Buy"

        # Risk record should have veto status
        assert result["risk"]["final_veto"] is False

    def test_missing_research_layer(self, temp_logs):
        """Missing research log returns None for research key."""
        config = {
            "research_thesis_log_path": str(temp_logs["tmpdir"] / "missing_research.jsonl"),
            "auditor_log_path": str(temp_logs["auditor_log"]),
            "risk_log_path": str(temp_logs["risk_log"]),
        }

        with patch("tradingagents.jobs.decision_reconstructor.get_config", return_value=config):
            result = reconstruct_decision("AAPL", "2026-08-01")

        # Research should be None, but auditor and risk should be populated
        assert result["ticker"] == "AAPL"
        assert result["trade_date"] == "2026-08-01"
        assert result["research"] is None
        assert result["auditor"] is not None
        assert result["risk"] is not None

    def test_no_matching_record(self, temp_logs):
        """No matching record in any layer returns all None."""
        config = {
            "research_thesis_log_path": str(temp_logs["research_log"]),
            "auditor_log_path": str(temp_logs["auditor_log"]),
            "risk_log_path": str(temp_logs["risk_log"]),
        }

        with patch("tradingagents.jobs.decision_reconstructor.get_config", return_value=config):
            result = reconstruct_decision("GOOGL", "2026-08-01")

        # All layers should be None
        assert result["ticker"] == "GOOGL"
        assert result["trade_date"] == "2026-08-01"
        assert result["research"] is None
        assert result["auditor"] is None
        assert result["risk"] is None

    def test_most_recent_match_wins(self, temp_logs):
        """When log has two records matching ticker+trade_date, last one wins."""
        # Write two research records for the same ticker+trade_date
        research_log = temp_logs["research_log"]
        second_record = {
            "timestamp": "2026-08-01T10:30:00Z",
            "ticker": "AAPL",
            "trade_date": "2026-08-01",
            "verdict": "Sell",  # Different verdict
            "retorno_12_1": 0.05,
            "z_score": 0.5,
            "confidence_level": "low",
            "verified_fields": ["retorno_12_1"],
            "reasoning": "Recalculated signal",
            "refutation_criterion": "If more data available",
            "refutation_quality": "low",
        }
        with open(research_log, "a") as f:
            f.write(json.dumps(second_record) + "\n")

        config = {
            "research_thesis_log_path": str(research_log),
            "auditor_log_path": str(temp_logs["auditor_log"]),
            "risk_log_path": str(temp_logs["risk_log"]),
        }

        with patch("tradingagents.jobs.decision_reconstructor.get_config", return_value=config):
            result = reconstruct_decision("AAPL", "2026-08-01")

        # Should return the SECOND record (most recent in file order)
        assert result["research"]["verdict"] == "Sell"
        assert result["research"]["z_score"] == 0.5

    def test_malformed_line_skipped(self, temp_logs):
        """Malformed JSON lines are logged and skipped gracefully."""
        research_log = temp_logs["research_log"]

        # Append a malformed line
        with open(research_log, "a") as f:
            f.write("{ invalid json }\n")

        config = {
            "research_thesis_log_path": str(research_log),
            "auditor_log_path": str(temp_logs["auditor_log"]),
            "risk_log_path": str(temp_logs["risk_log"]),
        }

        # Should not raise, even with malformed line
        with patch("tradingagents.jobs.decision_reconstructor.get_config", return_value=config):
            result = reconstruct_decision("AAPL", "2026-08-01")

        # Should still find the first valid record
        assert result["research"] is not None
        assert result["research"]["verdict"] == "Buy"


@pytest.mark.unit
class TestReconstructDecisionCLI:
    """Test the CLI script for decision reconstruction."""

    @pytest.fixture
    def temp_logs(self):
        """Create temporary JSONL log files for CLI testing."""
        tmpdir = Path(tempfile.mkdtemp())

        # Create research log
        research_log = tmpdir / "research.jsonl"
        research_records = [
            {
                "timestamp": "2026-08-01T10:00:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-08-01",
                "verdict": "Buy",
                "retorno_12_1": 0.15,
                "z_score": 2.1,
                "confidence_level": "high",
                "verified_fields": ["retorno_12_1", "z_score"],
                "reasoning": "Strong momentum signal",
                "refutation_criterion": "If momentum reverses",
                "refutation_quality": "medium",
            },
        ]
        with open(research_log, "w") as f:
            for record in research_records:
                f.write(json.dumps(record) + "\n")

        # Create auditor log
        auditor_log = tmpdir / "auditor.jsonl"
        auditor_records = [
            {
                "timestamp": "2026-08-01T10:05:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-08-01",
                "phase1_status": "pass",
                "phase1_failure_reason": None,
                "auditor_retorno_12_1": 0.15,
                "auditor_z_score": 2.1,
                "auditor_verdict": "Buy",
                "auditor_reasoning": "Confirmed momentum",
                "auditor_refutation_criterion": "If independent data diverges",
                "research_verdict": "Buy",
                "auditor_data_points": {"momentum": 2.1},
                "comparison_result": "match",
                "verified": True,
            },
        ]
        with open(auditor_log, "w") as f:
            for record in auditor_records:
                f.write(json.dumps(record) + "\n")

        # Create risk log
        risk_log = tmpdir / "risk.jsonl"
        risk_records = [
            {
                "timestamp": "2026-08-01T10:10:00Z",
                "ticker": "AAPL",
                "trade_date": "2026-08-01",
                "proposed_side": "long",
                "portfolio_snapshot_status": "ok",
                "portfolio_total_value": 100000,
                "existing_position_value": 0,
                "proposed_notional_usd": 2000,
                "risk_concentration_pct": 0.02,
                "conservative_verdict": {
                    "verdict": "APPROVE",
                    "confidence": 0.8,
                    "reasoning": "Within limits",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                },
                "balanced_verdict": {
                    "verdict": "APPROVE",
                    "confidence": 0.9,
                    "reasoning": "Good risk/reward",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                },
                "aggressive_verdict": {
                    "verdict": "APPROVE",
                    "confidence": 0.95,
                    "reasoning": "Positive setup",
                    "risk_factors": [],
                    "cited_concentration_pct": 0.02,
                },
                "final_veto": False,
                "concentration_verified": True,
            },
        ]
        with open(risk_log, "w") as f:
            for record in risk_records:
                f.write(json.dumps(record) + "\n")

        yield {
            "research_log": research_log,
            "auditor_log": auditor_log,
            "risk_log": risk_log,
            "tmpdir": tmpdir,
        }

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_cli_with_valid_args(self, temp_logs, capsys):
        """CLI with valid ticker and date prints report and returns 0."""
        from scripts.reconstruct_decision import main
        from tradingagents.dataflows import config as config_module

        # Get the current config and update the log paths
        config = config_module.get_config()
        config["research_thesis_log_path"] = str(temp_logs["research_log"])
        config["auditor_log_path"] = str(temp_logs["auditor_log"])
        config["risk_log_path"] = str(temp_logs["risk_log"])

        # Set the config directly in the module
        config_module.set_config(config)

        # Run the CLI
        exit_code = main(["--ticker", "AAPL", "--trade-date", "2026-08-01"])

        assert exit_code == 0

        # Verify output contains all four section headers
        captured = capsys.readouterr()
        assert "RESEARCH VERDICT:" in captured.out
        assert "AUDITOR VERDICT:" in captured.out
        assert "RISK VERDICTS:" in captured.out
        assert "EXECUTION STATUS:" in captured.out

    def test_cli_with_invalid_ticker(self):
        """CLI with invalid ticker format raises SystemExit."""
        from scripts.reconstruct_decision import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--ticker", "not-a-ticker!!", "--trade-date", "2026-08-01"])

        # argparse exits with code 2 on validation failure
        assert exc_info.value.code == 2

    def test_cli_with_invalid_date(self):
        """CLI with invalid date format raises SystemExit."""
        from scripts.reconstruct_decision import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--ticker", "AAPL", "--trade-date", "not-a-date"])

        assert exc_info.value.code == 2

    def test_cli_no_matching_records(self, temp_logs, capsys):
        """CLI with no matching records returns 1."""
        from scripts.reconstruct_decision import main
        from tradingagents.dataflows import config as config_module

        # Get the current config and update the log paths
        config = config_module.get_config()
        config["research_thesis_log_path"] = str(temp_logs["research_log"])
        config["auditor_log_path"] = str(temp_logs["auditor_log"])
        config["risk_log_path"] = str(temp_logs["risk_log"])

        # Set the config directly in the module
        config_module.set_config(config)

        # Use a valid ticker format (GOOGL) but one not in the logs
        exit_code = main(["--ticker", "GOOGL", "--trade-date", "2026-08-01"])

        # Should return 1 when nothing found
        assert exit_code == 1

        # Verify output shows N/A for all sections
        captured = capsys.readouterr()
        assert "N/A" in captured.out

    def test_cli_without_args(self):
        """CLI without required args raises SystemExit."""
        from scripts.reconstruct_decision import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--ticker", "AAPL"])

        # Missing --trade-date
        assert exc_info.value.code == 2
