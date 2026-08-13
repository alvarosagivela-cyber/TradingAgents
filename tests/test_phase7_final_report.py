"""Tests for phase7_final_report.py (D-05, Task 2).

Verifies that the final report script:
1. Closes positions before computing Sharpe (D-09 ordering)
2. Can skip position closing with --skip-close-positions flag
3. Builds explicit Core Value conclusion based on Auditor refutations
4. Includes Sharpe ratio in the report with overfitting warning if needed
5. Writes report to results_dir with correct filename
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.phase7_final_report as phase7_final_report
from tradingagents.dataflows.config import set_config


@pytest.fixture
def mock_alpaca_client():
    """Provide a mocked Alpaca TradingClient."""
    return MagicMock()


@pytest.fixture
def tmp_results_dir(tmp_path: Path):
    """Provide a temporary results directory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    set_config({"results_dir": str(results_dir)})
    return results_dir


def test_report_calls_close_positions_before_sharpe(
    tmp_results_dir: Path, mock_alpaca_client: MagicMock
):
    """Test that close_all_open_positions is called before compute_window_sharpe."""
    with patch.object(phase7_final_report, "create_alpaca_client") as mock_create, \
         patch.object(phase7_final_report, "close_all_open_positions") as mock_close, \
         patch.object(phase7_final_report, "compute_window_sharpe") as mock_sharpe, \
         patch.object(phase7_final_report, "count_auditor_refutations") as mock_refute, \
         patch.object(phase7_final_report, "aggregate_auditor_reflections") as mock_agg, \
         patch.object(phase7_final_report, "summarize_costs") as mock_costs:

        mock_create.return_value = mock_alpaca_client
        mock_close.return_value = {"status": "closed", "positions_closed": 5, "error": None}
        mock_sharpe.return_value = {
            "sharpe_ratio": 0.8,
            "overfitting_flag": False,
            "reference_range": (0.5, 1.0),
            "insufficient_data": False,
            "caveat": "Normal window",
        }
        mock_refute.return_value = {
            "total_cycles": 10,
            "refutation_count": 1,
            "at_least_one_refutation": True,
            "refutations": [],
        }
        mock_agg.return_value = {"false_positives": 0}
        mock_costs.return_value = {"total_cost_usd": 10.0}

        result = phase7_final_report.main(
            ["--start-date", "2026-08-11", "--end-date", "2026-09-22"]
        )

        # Verify close_all_open_positions was called
        assert mock_close.call_count == 1

        # Verify compute_window_sharpe was called after close_all_open_positions
        assert mock_sharpe.call_count == 1

        # Verify both were called
        assert result == 0


def test_report_skip_close_positions_flag(
    tmp_results_dir: Path, mock_alpaca_client: MagicMock
):
    """Test that --skip-close-positions flag prevents closing positions."""
    with patch.object(phase7_final_report, "create_alpaca_client") as mock_create, \
         patch.object(phase7_final_report, "close_all_open_positions") as mock_close, \
         patch.object(phase7_final_report, "compute_window_sharpe") as mock_sharpe, \
         patch.object(phase7_final_report, "count_auditor_refutations") as mock_refute, \
         patch.object(phase7_final_report, "aggregate_auditor_reflections") as mock_agg, \
         patch.object(phase7_final_report, "summarize_costs") as mock_costs:

        mock_create.return_value = mock_alpaca_client
        mock_sharpe.return_value = {
            "sharpe_ratio": 0.8,
            "overfitting_flag": False,
            "reference_range": (0.5, 1.0),
            "insufficient_data": False,
            "caveat": "Normal",
        }
        mock_refute.return_value = {
            "total_cycles": 10,
            "refutation_count": 0,
            "at_least_one_refutation": False,
            "refutations": [],
        }
        mock_agg.return_value = {}
        mock_costs.return_value = {}

        result = phase7_final_report.main(
            [
                "--start-date",
                "2026-08-11",
                "--end-date",
                "2026-09-22",
                "--skip-close-positions",
            ]
        )

        # close_all_open_positions should NOT be called
        assert mock_close.call_count == 0
        assert result == 0


def test_core_value_conclusion_yes_when_refutation_found(
    tmp_results_dir: Path, mock_alpaca_client: MagicMock, capsys
):
    """Test that Core Value conclusion says YES when Auditor refuted a thesis."""
    with patch.object(phase7_final_report, "create_alpaca_client") as mock_create, \
         patch.object(phase7_final_report, "close_all_open_positions") as mock_close, \
         patch.object(phase7_final_report, "compute_window_sharpe") as mock_sharpe, \
         patch.object(phase7_final_report, "count_auditor_refutations") as mock_refute, \
         patch.object(phase7_final_report, "aggregate_auditor_reflections") as mock_agg, \
         patch.object(phase7_final_report, "summarize_costs") as mock_costs:

        mock_create.return_value = mock_alpaca_client
        mock_close.return_value = {"status": "closed", "positions_closed": 3, "error": None}
        mock_sharpe.return_value = {
            "sharpe_ratio": 0.6,
            "overfitting_flag": False,
            "reference_range": (0.5, 1.0),
            "insufficient_data": False,
            "caveat": "Normal",
        }
        mock_refute.return_value = {
            "total_cycles": 10,
            "refutation_count": 2,
            "at_least_one_refutation": True,
            "refutations": [
                {"ticker": "AAPL", "trade_date": "2026-08-15", "auditor_reasoning": "Bad thesis"},
                {"ticker": "GOOGL", "trade_date": "2026-08-20", "auditor_reasoning": "Poor thesis"},
            ],
        }
        mock_agg.return_value = {"false_positives": 0}
        mock_costs.return_value = {"total_cost_usd": 5.0}

        result = phase7_final_report.main(
            ["--start-date", "2026-08-11", "--end-date", "2026-09-22"]
        )

        captured = capsys.readouterr()
        assert "CORE VALUE ANSWER: YES" in captured.out
        assert result == 0


def test_core_value_conclusion_no_when_no_refutation(
    tmp_results_dir: Path, mock_alpaca_client: MagicMock, capsys
):
    """Test that Core Value conclusion says NO when Auditor never refuted."""
    with patch.object(phase7_final_report, "create_alpaca_client") as mock_create, \
         patch.object(phase7_final_report, "close_all_open_positions") as mock_close, \
         patch.object(phase7_final_report, "compute_window_sharpe") as mock_sharpe, \
         patch.object(phase7_final_report, "count_auditor_refutations") as mock_refute, \
         patch.object(phase7_final_report, "aggregate_auditor_reflections") as mock_agg, \
         patch.object(phase7_final_report, "summarize_costs") as mock_costs:

        mock_create.return_value = mock_alpaca_client
        mock_close.return_value = {"status": "closed", "positions_closed": 0, "error": None}
        mock_sharpe.return_value = {
            "sharpe_ratio": 0.7,
            "overfitting_flag": False,
            "reference_range": (0.5, 1.0),
            "insufficient_data": False,
            "caveat": "Normal",
        }
        mock_refute.return_value = {
            "total_cycles": 10,
            "refutation_count": 0,
            "at_least_one_refutation": False,
            "refutations": [],
        }
        mock_agg.return_value = {}
        mock_costs.return_value = {}

        result = phase7_final_report.main(
            ["--start-date", "2026-08-11", "--end-date", "2026-09-22"]
        )

        captured = capsys.readouterr()
        assert "CORE VALUE ANSWER: NO" in captured.out
        assert result == 0


def test_sharpe_overfitting_warning_in_report(
    tmp_results_dir: Path, mock_alpaca_client: MagicMock, capsys
):
    """Test that overfitting warning appears when Sharpe > 1.0."""
    with patch.object(phase7_final_report, "create_alpaca_client") as mock_create, \
         patch.object(phase7_final_report, "close_all_open_positions") as mock_close, \
         patch.object(phase7_final_report, "compute_window_sharpe") as mock_sharpe, \
         patch.object(phase7_final_report, "count_auditor_refutations") as mock_refute, \
         patch.object(phase7_final_report, "aggregate_auditor_reflections") as mock_agg, \
         patch.object(phase7_final_report, "summarize_costs") as mock_costs:

        mock_create.return_value = mock_alpaca_client
        mock_close.return_value = {"status": "closed", "positions_closed": 2, "error": None}
        mock_sharpe.return_value = {
            "sharpe_ratio": 1.5,
            "overfitting_flag": True,
            "reference_range": (0.5, 1.0),
            "insufficient_data": False,
            "caveat": "High Sharpe suggests potential overfitting",
        }
        mock_refute.return_value = {
            "total_cycles": 5,
            "refutation_count": 0,
            "at_least_one_refutation": False,
            "refutations": [],
        }
        mock_agg.return_value = {}
        mock_costs.return_value = {}

        result = phase7_final_report.main(
            ["--start-date", "2026-08-11", "--end-date", "2026-09-22"]
        )

        captured = capsys.readouterr()
        assert "1.5" in captured.out or "1.5" in captured.out.replace(",", ".")
        assert "overfitting" in captured.out.lower()
        assert result == 0


def test_report_written_to_results_dir(
    tmp_results_dir: Path, mock_alpaca_client: MagicMock
):
    """Test that report is written to results_dir with correct filename."""
    with patch.object(phase7_final_report, "create_alpaca_client") as mock_create, \
         patch.object(phase7_final_report, "close_all_open_positions") as mock_close, \
         patch.object(phase7_final_report, "compute_window_sharpe") as mock_sharpe, \
         patch.object(phase7_final_report, "count_auditor_refutations") as mock_refute, \
         patch.object(phase7_final_report, "aggregate_auditor_reflections") as mock_agg, \
         patch.object(phase7_final_report, "summarize_costs") as mock_costs:

        mock_create.return_value = mock_alpaca_client
        mock_close.return_value = {"status": "closed", "positions_closed": 1, "error": None}
        mock_sharpe.return_value = {
            "sharpe_ratio": 0.65,
            "overfitting_flag": False,
            "reference_range": (0.5, 1.0),
            "insufficient_data": False,
            "caveat": "Normal",
        }
        mock_refute.return_value = {
            "total_cycles": 15,
            "refutation_count": 1,
            "at_least_one_refutation": True,
            "refutations": [{"ticker": "XYZ", "trade_date": "2026-08-20", "auditor_reasoning": "Bad"}],
        }
        mock_agg.return_value = {"false_positives": 1, "false_negatives": 0}
        mock_costs.return_value = {"total_cost_usd": 8.5}

        result = phase7_final_report.main(
            ["--start-date", "2026-08-11", "--end-date", "2026-09-22"]
        )

        # Check that report file was created
        report_file = tmp_results_dir / "phase7_final_report_2026-09-22.md"
        assert report_file.exists()

        # Read and verify content
        content = report_file.read_text()
        assert len(content) > 0
        assert "CORE VALUE ANSWER" in content
        assert result == 0
