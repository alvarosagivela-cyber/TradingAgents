"""Integration tests for the Auditor verification pipeline (compute -> LLM -> compare -> persist).

Tests the full chain: auditor momentum compute node, auditor LLM node, comparison node, and JSONL persistence.
All tests use mocked yfinance/load_ohlcv and LLM, so no network or API calls are made.

The Auditor pipeline is the independent verification layer: it retrieves its own OHLCV data,
runs its own momentum calculation, and invokes a separate LLM instance to produce a refutation verdict.
This pipeline deliberately does NOT read Research's computed fields or cached reports — AUDIT-01/02 isolation.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.schemas import AuditorVerdict, TraderAction
from tradingagents.agents.utils.agent_states import AgentState


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_auditor_state(
    company: str = "AAPL",
    date: str = "2026-07-29",
    thesis_verdict: str = "Buy",
    auditor_retorno_12_1: float = 0.1834,
    auditor_z_score: float = 2.3,
    auditor_phase1_status: str = "pass",
    auditor_data_points: dict | None = None,
) -> dict:
    """Build a minimal AgentState dict pre-populated with Research verdict + Auditor compute fields for testing.

    Simulates the state AFTER the Research compute node has run and set thesis_verdict,
    but BEFORE the Auditor compute/LLM/compare nodes run.
    """
    if auditor_data_points is None:
        auditor_data_points = {"retorno_12_1": auditor_retorno_12_1, "z_score": auditor_z_score}

    return {
        "company_of_interest": company,
        "trade_date": date,
        # Research fields (already computed by prior pipeline)
        "thesis_verdict": thesis_verdict,
        "retorno_12_1": 0.18,  # Research's own value
        "momentum_z_score": 2.1,  # Research's own value
        # Auditor compute phase output
        "auditor_phase1_status": auditor_phase1_status,
        "auditor_phase1_failure_reason": "" if auditor_phase1_status == "pass" else "insufficient_ohlcv_history",
        "auditor_retorno_12_1": auditor_retorno_12_1,
        "auditor_z_score": auditor_z_score,
        "auditor_confidence_level": "high" if abs(auditor_z_score) > 1.96 else "medium",
        # Auditor LLM phase (will be populated by the node)
        "auditor_verdict": "",
        "auditor_reasoning": "",
        "auditor_refutation_criterion": "",
        "auditor_data_points": auditor_data_points,
        # Auditor comparison phase (will be populated by the node)
        "auditor_verified": False,
        "comparison_result": "",
    }


def _structured_auditor_llm(captured: dict, verdict: AuditorVerdict | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real AuditorVerdict.
    """
    if verdict is None:
        verdict = AuditorVerdict(
            verdict=TraderAction.SELL,
            reasoning=(
                "The 12-month momentum of 18.34% with a z-score of 2.3 initially appears strong, "
                "but the refutation criterion shows that Research's thesis has a brittle foundation. "
                "Historical analysis reveals that at this exact z-score level, mean reversion occurs 45% of the time "
                "when the market breadth is contracting (which it is). The cited momentum is real but backward-looking; "
                "leading indicators suggest the catalyst driving this momentum has already peaked."
            ),
            refutation_criterion="Thesis fails if RSI > 70 AND price closes below the 20-day EMA within 5 trading days, signaling early reversal of the momentum driver.",
            data_points={"retorno_12_1": 0.1834, "z_score": 2.3},
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or verdict
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAuditorPipeline:

    def test_1_happy_path_match(self, tmp_path):
        """Test 1 (happy path / match): Auditor computes metrics (mocked),
        LLM returns a verdict matching Research thesis_verdict and citing data_points within tolerance
        -> comparison_result == "match", auditor_verified == True, exactly one JSONL record written with comparison_result "match".
        """
        # Imports that will fail until the Auditor modules are created
        from tradingagents.agents.auditors.momentum_auditor import (
            create_auditor_compute_node,
            create_auditor_llm_node,
        )
        from tradingagents.agents.auditors.thesis_comparator import (
            create_auditor_compare_node,
        )

        # Setup: Research verdict is BUY, Auditor will also say BUY
        state = _make_auditor_state(thesis_verdict="Buy")
        captured = {}

        matching_verdict = AuditorVerdict(
            verdict=TraderAction.BUY,  # Same as Research
            reasoning="The momentum thesis is sound. The 18.34% return over 12 months with a z-score of 2.3 is robust.",
            refutation_criterion="Thesis fails if price closes below the 50-day MA on 2x volume within 10 days.",
            data_points={"retorno_12_1": 0.1834, "z_score": 2.3},
        )
        mock_llm = _structured_auditor_llm(captured, matching_verdict)

        with patch(
            "tradingagents.agents.auditors.thesis_comparator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "auditor_log_path": str(tmp_path / "audits.jsonl")
            }
            with patch(
                "tradingagents.agents.auditors.momentum_auditor.create_llm_client"
            ) as mock_factory:
                mock_factory.return_value.get_llm.return_value = mock_llm

                # Run the auditor pipeline
                # Note: We skip the compute node in this test (pre-populate auditor_retorno_12_1/z_score in state)
                llm_node = create_auditor_llm_node()
                llm_update = llm_node(state)
                state.update(llm_update)

                compare_node = create_auditor_compare_node()
                compare_update = compare_node(state)
                state.update(compare_update)

                # Assertions
                assert state.get("auditor_verdict") == "Buy", "Auditor should say BUY"
                assert state.get("comparison_result") == "match", "Match when verdicts align"
                assert state.get("auditor_verified") is True, "Should be verified (data_points within tolerance)"
                assert (tmp_path / "audits.jsonl").exists(), "JSONL file should be created"

                # Verify JSONL content
                lines = (tmp_path / "audits.jsonl").read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) == 1, "Should have exactly one record"
                import json
                record = json.loads(lines[0])
                assert record["comparison_result"] == "match"

    def test_2_refuted_persists_d06_regression_guard(self, tmp_path):
        """Test 2 (REFUTED persists — D-06 regression guard, the single most important test):
        LLM returns a verdict differing from thesis_verdict
        -> comparison_result == "mismatch", AND the JSONL record still exists with comparison_result "mismatch"
        (asserts persistence is unconditional, not verified-gated like Research's pattern).
        """
        from tradingagents.agents.auditors.momentum_auditor import (
            create_auditor_llm_node,
        )
        from tradingagents.agents.auditors.thesis_comparator import (
            create_auditor_compare_node,
        )

        # Setup: Research says BUY, Auditor will say SELL (mismatch)
        state = _make_auditor_state(thesis_verdict="Buy")
        captured = {}

        mismatched_verdict = AuditorVerdict(
            verdict=TraderAction.SELL,  # Different from Research BUY
            reasoning="The momentum is real but the refutation criterion confirms the thesis is brittle. Mean reversion imminent.",
            refutation_criterion="Thesis fails if RSI > 70 within 5 days.",
            data_points={"retorno_12_1": 0.1834, "z_score": 2.3},
        )
        mock_llm = _structured_auditor_llm(captured, mismatched_verdict)

        with patch(
            "tradingagents.agents.auditors.thesis_comparator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "auditor_log_path": str(tmp_path / "audits.jsonl")
            }
            with patch(
                "tradingagents.agents.auditors.momentum_auditor.create_llm_client"
            ) as mock_factory:
                mock_factory.return_value.get_llm.return_value = mock_llm

                # Run LLM and comparison nodes
                llm_node = create_auditor_llm_node()
                llm_update = llm_node(state)
                state.update(llm_update)

                compare_node = create_auditor_compare_node()
                compare_update = compare_node(state)
                state.update(compare_update)

                # Assertions
                assert state.get("auditor_verdict") == "Sell", "Auditor should say SELL"
                assert state.get("comparison_result") == "mismatch", "Mismatch when verdicts differ"
                # D-06 critical: JSONL is persisted EVEN THOUGH verdicts differ
                assert (tmp_path / "audits.jsonl").exists(), "JSONL file should be created for refuted thesis (D-06)"

                # Verify JSONL content
                lines = (tmp_path / "audits.jsonl").read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) == 1, "Should have exactly one record"
                import json
                record = json.loads(lines[0])
                assert record["comparison_result"] == "mismatch", "Mismatch record persisted (D-06)"

    def test_3_fail_open_inconclusive_not_reached(self, tmp_path):
        """Test 3 (fail-open): auditor_phase1_status == "fail" (compute_momentum returns valid=False)
        -> auditor_verdict == "INCONCLUSIVE" without the LLM's with_structured_output ever being called (mock assertion),
        comparison_result == "not_reached", and this cycle is STILL persisted to the JSONL.
        """
        from tradingagents.agents.auditors.momentum_auditor import (
            create_auditor_llm_node,
        )
        from tradingagents.agents.auditors.thesis_comparator import (
            create_auditor_compare_node,
        )

        # Setup: auditor_phase1_status=fail signals insufficient OHLCV history
        state = _make_auditor_state(auditor_phase1_status="fail")
        captured = {}
        mock_llm = _structured_auditor_llm(captured)

        with patch(
            "tradingagents.agents.auditors.thesis_comparator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "auditor_log_path": str(tmp_path / "audits.jsonl")
            }
            with patch(
                "tradingagents.agents.auditors.momentum_auditor.create_llm_client"
            ) as mock_factory:
                mock_factory.return_value.get_llm.return_value = mock_llm

                # Run LLM node
                llm_node = create_auditor_llm_node()
                llm_update = llm_node(state)

                # Assertions: LLM should NOT have been called (short-circuit)
                mock_llm.with_structured_output.assert_not_called()
                assert llm_update.get("auditor_verdict") == "INCONCLUSIVE"

                state.update(llm_update)

                # Run comparison node
                compare_node = create_auditor_compare_node()
                compare_update = compare_node(state)
                state.update(compare_update)

                # Assertions
                assert state.get("auditor_verdict") == "INCONCLUSIVE", "Verdict should be INCONCLUSIVE"
                assert state.get("comparison_result") == "not_reached", "Result is not_reached when computation fails"
                assert state.get("auditor_verified") is False, "not_reached cycles are not verified"
                # D-02/D-06 critical: JSONL is persisted EVEN THOUGH we never reached the LLM
                assert (tmp_path / "audits.jsonl").exists(), "JSONL file should be created for not_reached cycle"

                # Verify JSONL content
                lines = (tmp_path / "audits.jsonl").read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) == 1, "Should have exactly one record"
                import json
                record = json.loads(lines[0])
                assert record["comparison_result"] == "not_reached"

    def test_4_numeric_hallucination_verified_false(self, tmp_path):
        """Test 4 (numeric self-hallucination):
        LLM's data_points don't match the auditor's own computed auditor_retorno_12_1/auditor_z_score within tolerance
        -> auditor_verified == False while comparison_result is still computed independently
        (verified and comparison_result are orthogonal fields, unlike Research where verified gates persistence entirely).
        """
        from tradingagents.agents.auditors.momentum_auditor import (
            create_auditor_llm_node,
        )
        from tradingagents.agents.auditors.thesis_comparator import (
            create_auditor_compare_node,
        )

        # Setup: Auditor's computed value is 0.1834, but LLM will hallucinate 0.20 (differs by >0.0001)
        state = _make_auditor_state(auditor_retorno_12_1=0.1834, thesis_verdict="Buy")
        captured = {}

        hallucinating_verdict = AuditorVerdict(
            verdict=TraderAction.BUY,  # Verdict matches Research
            reasoning="The momentum is strong and should persist.",
            refutation_criterion="Thesis fails if RSI > 70.",
            data_points={"retorno_12_1": 0.20, "z_score": 2.3},  # Hallucination! Should be 0.1834
        )
        mock_llm = _structured_auditor_llm(captured, hallucinating_verdict)

        with patch(
            "tradingagents.agents.auditors.thesis_comparator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "auditor_log_path": str(tmp_path / "audits.jsonl")
            }
            with patch(
                "tradingagents.agents.auditors.momentum_auditor.create_llm_client"
            ) as mock_factory:
                mock_factory.return_value.get_llm.return_value = mock_llm

                # Run LLM and comparison nodes
                llm_node = create_auditor_llm_node()
                llm_update = llm_node(state)
                state.update(llm_update)

                compare_node = create_auditor_compare_node()
                compare_update = compare_node(state)
                state.update(compare_update)

                # Assertions
                assert state.get("auditor_verdict") == "Buy", "Verdict still returns (no short-circuit on hallucination)"
                assert state.get("auditor_verified") is False, "verified=False due to numeric mismatch (D-11)"
                assert state.get("comparison_result") == "match", "comparison_result still computed (verdict match with Research)"
                # D-06 critical: JSONL is persisted EVEN THOUGH auditor_verified=False
                assert (tmp_path / "audits.jsonl").exists(), "JSONL file should be created for hallucination cycle"

                # Verify JSONL content
                lines = (tmp_path / "audits.jsonl").read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) == 1, "Should have exactly one record"
                import json
                record = json.loads(lines[0])
                assert record["verified"] is False, "Hallucination recorded as unverified"
                assert record["comparison_result"] == "match", "comparison_result independent of verified"
