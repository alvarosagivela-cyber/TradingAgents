"""Unit tests for Auditor compute node — determinism, fail-open, field isolation."""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.auditors.momentum_auditor import (
    create_auditor_compute_node,
)


@pytest.mark.unit
class TestAuditorComputeNode:
    """Test Auditor compute node determinism, fail-open, and field isolation (AUDIT-01)."""

    def test_compute_node_determinism(self):
        """Calling compute node twice with identical mocked momentum returns identical results (D-10)."""
        compute_node = create_auditor_compute_node()

        # Mock compute_momentum to return a deterministic result
        mock_metrics = MagicMock()
        mock_metrics.valid = True
        mock_metrics.retorno_12_1 = 0.1834
        mock_metrics.z_score = 2.3
        mock_metrics.confidence_level = "high"

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
        }

        with patch(
            "tradingagents.agents.auditors.momentum_auditor.compute_momentum"
        ) as mock_compute:
            mock_compute.return_value = mock_metrics

            result1 = compute_node(state)
            result2 = compute_node(state)

            # All fields should be identical
            assert result1["auditor_retorno_12_1"] == result2["auditor_retorno_12_1"]
            assert result1["auditor_z_score"] == result2["auditor_z_score"]
            assert result1["auditor_phase1_status"] == result2["auditor_phase1_status"]
            assert result1["auditor_confidence_level"] == result2["auditor_confidence_level"]

    def test_compute_node_fail_open(self):
        """Insufficient history fails open: auditor_phase1_status='fail', LLM not invoked (D-02)."""
        compute_node = create_auditor_compute_node()

        # Mock compute_momentum to return invalid (insufficient data)
        mock_metrics = MagicMock()
        mock_metrics.valid = False

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
        }

        with patch(
            "tradingagents.agents.auditors.momentum_auditor.compute_momentum"
        ) as mock_compute:
            mock_compute.return_value = mock_metrics

            result = compute_node(state)

            # Must fail open with clear reason
            assert result["auditor_phase1_status"] == "fail"
            assert result["auditor_phase1_failure_reason"] == "insufficient_ohlcv_history"
            assert result["auditor_retorno_12_1"] == 0.0
            assert result["auditor_z_score"] == 0.0

    def test_compute_node_field_isolation(self):
        """Source does NOT contain forbidden Research-only fields (AUDIT-01)."""
        create_auditor_compute_node()
        source = inspect.getsource(create_auditor_compute_node)

        # Forbidden fields that should NOT appear in Auditor compute node
        forbidden_patterns = [
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            'state["thesis"',
            "state.get('thesis'",
        ]

        for pattern in forbidden_patterns:
            assert pattern not in source, f"Auditor compute node should not read {pattern}"

    def test_compute_node_independent_fetch(self):
        """Compute node calls compute_momentum independently (AUDIT-01)."""
        compute_node = create_auditor_compute_node()

        mock_metrics = MagicMock()
        mock_metrics.valid = True
        mock_metrics.retorno_12_1 = 0.1834
        mock_metrics.z_score = 2.3

        state = {
            "company_of_interest": "AAPL",
            "trade_date": "2026-07-29",
        }

        with patch(
            "tradingagents.agents.auditors.momentum_auditor.compute_momentum"
        ) as mock_compute:
            mock_compute.return_value = mock_metrics

            result = compute_node(state)

            # Verify compute_momentum was called with the right ticker and date
            mock_compute.assert_called_once_with("AAPL", "2026-07-29")
            assert result["auditor_phase1_status"] == "pass"
