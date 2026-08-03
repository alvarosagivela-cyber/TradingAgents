"""Integration tests for the Risk Squad pipeline (3 perspectives + deterministic aggregator).

Tests the full chain: conservative/balanced/aggressive perspective nodes and the
risk_aggregator node that computes final_veto and persists cycles.

All tests use mocked LLMs, so no network or API calls are made.

The Risk Squad pipeline is the binary veto layer: it evaluates each trade already
approved by Research+Auditor from three distinct risk appetites, with isolated
LLM instances per perspective (D-02) and deterministic veto rules (RISK-01).

IMPORTANT: This file MUST fail at import time with ModuleNotFoundError for the
risk_mgmt submodules until Task 2 implements them. The import errors below are
the expected RED state (TDD phase 1).
"""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.schemas import RiskDecision
from tradingagents.agents.utils.agent_states import AgentState


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_risk_squad_state(
    company: str = "AAPL",
    date: str = "2026-07-29",
    proposed_side: str = "Buy",
    portfolio_total_value: float = 100000.0,
    existing_position_value: float = 0.0,
    risk_concentration_pct: float = 0.02,
    portfolio_snapshot_status: str = "pass",
) -> dict:
    """Build a minimal AgentState dict pre-populated with Research/Auditor verdicts
    and portfolio context for Risk Squad testing.

    Simulates the state AFTER Research and Auditor have run, pre-populated with
    portfolio snapshot data (Alpaca paper account equity, existing positions, etc.),
    and ready for the Risk Squad to evaluate.
    """
    proposed_notional_usd = portfolio_total_value * 0.02  # 2% default sizing

    return {
        # Ticker and date context
        "company_of_interest": company,
        "trade_date": date,
        # Trader's proposed side (deterministically parsed from trader_investment_plan)
        "proposed_side": proposed_side,
        # Portfolio snapshot from Alpaca paper account
        "portfolio_snapshot_status": portfolio_snapshot_status,
        "portfolio_total_value": portfolio_total_value,
        "existing_position_value": existing_position_value,
        "proposed_notional_usd": proposed_notional_usd,
        "risk_concentration_pct": risk_concentration_pct,
        # Risk Squad perspective verdicts (will be populated by perspective nodes)
        "conservative_verdict": {},
        "balanced_verdict": {},
        "aggressive_verdict": {},
        # Risk aggregator output (will be populated by aggregator)
        "final_veto": False,
        "concentration_verified": False,
        "execution_log": [],
        # Paper execution output (set by execution layer, not Risk Squad)
        "paper_order_id": "",
        "paper_execution_status": "not_attempted",
    }


def _structured_risk_perspective_llm(
    captured: dict,
    verdict: RiskDecision | None = None,
) -> MagicMock:
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real RiskDecision.
    """
    if verdict is None:
        verdict = RiskDecision(
            verdict="APPROVE",
            confidence=0.85,
            reasoning=(
                "The concentration at 2% of portfolio is well within acceptable limits. "
                "The proposed position size aligns with risk guidelines. "
                "No sector overlap concerns noted. Recommend approval."
            ),
            risk_factors=[],
            cited_concentration_pct=0.02,
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
class TestRiskSquadPipeline:
    """Integration tests for the Risk Squad 3-perspective + aggregator pipeline."""

    def test_1_all_approve(self, tmp_path):
        """Test 1 (all-approve):
        All three perspective LLMs return verdict='APPROVE' with cited_concentration_pct
        matching state's risk_concentration_pct exactly
        -> final_veto=False, concentration_verified=True, exactly one JSONL record
        written with final_veto: false.
        """
        # Imports that will fail until the Risk Squad modules are created (RED state)
        from tradingagents.agents.risk_mgmt.conservative_perspective import (
            create_conservative_perspective,
        )
        from tradingagents.agents.risk_mgmt.balanced_perspective import (
            create_balanced_perspective,
        )
        from tradingagents.agents.risk_mgmt.aggressive_perspective import (
            create_aggressive_perspective,
        )
        from tradingagents.agents.risk_mgmt.risk_aggregator import (
            create_risk_aggregator,
        )

        state = _make_risk_squad_state()
        captured_conservative = {}
        captured_balanced = {}
        captured_aggressive = {}

        # All three perspectives return APPROVE with matching concentration
        approve_verdict = RiskDecision(
            verdict="APPROVE",
            confidence=0.85,
            reasoning="The position size and concentration are acceptable. No veto warranted.",
            risk_factors=[],
            cited_concentration_pct=0.02,
        )
        mock_conservative = _structured_risk_perspective_llm(
            captured_conservative, approve_verdict
        )
        mock_balanced = _structured_risk_perspective_llm(
            captured_balanced, approve_verdict
        )
        mock_aggressive = _structured_risk_perspective_llm(
            captured_aggressive, approve_verdict
        )

        with patch("tradingagents.agents.risk_mgmt.risk_aggregator.get_config") as mock_config:
            mock_config.return_value = {
                "risk_log_path": str(tmp_path / "risk.jsonl")
            }
            with patch(
                "tradingagents.agents.risk_mgmt.conservative_perspective.create_llm_client"
            ) as mock_factory_cons:
                with patch(
                    "tradingagents.agents.risk_mgmt.balanced_perspective.create_llm_client"
                ) as mock_factory_bal:
                    with patch(
                        "tradingagents.agents.risk_mgmt.aggressive_perspective.create_llm_client"
                    ) as mock_factory_agg:
                        mock_factory_cons.return_value.get_llm.return_value = mock_conservative
                        mock_factory_bal.return_value.get_llm.return_value = mock_balanced
                        mock_factory_agg.return_value.get_llm.return_value = mock_aggressive

                        # Run all three perspective nodes
                        cons_node = create_conservative_perspective()
                        cons_update = cons_node(state)
                        state.update(cons_update)

                        bal_node = create_balanced_perspective()
                        bal_update = bal_node(state)
                        state.update(bal_update)

                        agg_node = create_aggressive_perspective()
                        agg_update = agg_node(state)
                        state.update(agg_update)

                        # Run aggregator
                        aggregator = create_risk_aggregator()
                        agg_result = aggregator(state)
                        state.update(agg_result)

                        # Assertions
                        assert state.get("final_veto") is False, "All approve → final_veto should be False"
                        assert state.get("concentration_verified") is True, "All cite correct concentration → verified"
                        assert (tmp_path / "risk.jsonl").exists(), "JSONL file should be created"

                        # Verify JSONL content
                        import json
                        lines = (tmp_path / "risk.jsonl").read_text(encoding="utf-8").strip().split("\n")
                        assert len(lines) == 1, "Should have exactly one record"
                        record = json.loads(lines[0])
                        assert record["final_veto"] is False

    def test_2_any_veto_persists_d06_regression_guard(self, tmp_path):
        """Test 2 (any-veto — D-06 regression guard, the single most important test):
        Conservative returns verdict='VETO', balanced and aggressive return 'APPROVE'
        -> final_veto=True, AND the JSONL record still exists with final_veto: true
        (asserts persistence is unconditional, not gated on approval like Research's pattern).
        """
        from tradingagents.agents.risk_mgmt.conservative_perspective import (
            create_conservative_perspective,
        )
        from tradingagents.agents.risk_mgmt.balanced_perspective import (
            create_balanced_perspective,
        )
        from tradingagents.agents.risk_mgmt.aggressive_perspective import (
            create_aggressive_perspective,
        )
        from tradingagents.agents.risk_mgmt.risk_aggregator import (
            create_risk_aggregator,
        )

        state = _make_risk_squad_state()
        captured_conservative = {}
        captured_balanced = {}
        captured_aggressive = {}

        # Conservative vetoes, others approve
        veto_verdict = RiskDecision(
            verdict="VETO",
            confidence=0.90,
            reasoning="The concentration at 2% breaches our risk guidelines. Position too large relative to portfolio.",
            risk_factors=["concentration_breach", "sector_concentration"],
            cited_concentration_pct=0.02,
        )
        approve_verdict = RiskDecision(
            verdict="APPROVE",
            confidence=0.75,
            reasoning="The position is acceptable from this perspective.",
            risk_factors=[],
            cited_concentration_pct=0.02,
        )
        mock_conservative = _structured_risk_perspective_llm(
            captured_conservative, veto_verdict
        )
        mock_balanced = _structured_risk_perspective_llm(
            captured_balanced, approve_verdict
        )
        mock_aggressive = _structured_risk_perspective_llm(
            captured_aggressive, approve_verdict
        )

        with patch("tradingagents.agents.risk_mgmt.risk_aggregator.get_config") as mock_config:
            mock_config.return_value = {
                "risk_log_path": str(tmp_path / "risk.jsonl")
            }
            with patch(
                "tradingagents.agents.risk_mgmt.conservative_perspective.create_llm_client"
            ) as mock_factory_cons:
                with patch(
                    "tradingagents.agents.risk_mgmt.balanced_perspective.create_llm_client"
                ) as mock_factory_bal:
                    with patch(
                        "tradingagents.agents.risk_mgmt.aggressive_perspective.create_llm_client"
                    ) as mock_factory_agg:
                        mock_factory_cons.return_value.get_llm.return_value = mock_conservative
                        mock_factory_bal.return_value.get_llm.return_value = mock_balanced
                        mock_factory_agg.return_value.get_llm.return_value = mock_aggressive

                        # Run perspective nodes
                        cons_node = create_conservative_perspective()
                        cons_update = cons_node(state)
                        state.update(cons_update)

                        bal_node = create_balanced_perspective()
                        bal_update = bal_node(state)
                        state.update(bal_update)

                        agg_node = create_aggressive_perspective()
                        agg_update = agg_node(state)
                        state.update(agg_update)

                        # Run aggregator
                        aggregator = create_risk_aggregator()
                        agg_result = aggregator(state)
                        state.update(agg_result)

                        # Assertions
                        assert state.get("final_veto") is True, "Any veto → final_veto should be True"
                        assert (tmp_path / "risk.jsonl").exists(), "JSONL file should exist even on veto"

                        # Verify JSONL content (unconditional persistence)
                        import json
                        lines = (tmp_path / "risk.jsonl").read_text(encoding="utf-8").strip().split("\n")
                        assert len(lines) == 1, "Should have exactly one record"
                        record = json.loads(lines[0])
                        assert record["final_veto"] is True, "Vetoed cycle must be persisted"

    def test_3_snapshot_failure_forces_veto(self, tmp_path):
        """Test 3 (snapshot-failure forces veto — RISK-01 safety net):
        portfolio_snapshot_status='fail' in input state, but all three perspective
        LLMs return 'APPROVE' → final_veto=True anyway (aggregator's fail-safe
        override is never bypassed by LLM optimism), and the cycle is still persisted.
        """
        from tradingagents.agents.risk_mgmt.conservative_perspective import (
            create_conservative_perspective,
        )
        from tradingagents.agents.risk_mgmt.balanced_perspective import (
            create_balanced_perspective,
        )
        from tradingagents.agents.risk_mgmt.aggressive_perspective import (
            create_aggressive_perspective,
        )
        from tradingagents.agents.risk_mgmt.risk_aggregator import (
            create_risk_aggregator,
        )

        # Create state with failed snapshot, but all perspectives will try to approve
        state = _make_risk_squad_state(portfolio_snapshot_status="fail")
        captured_conservative = {}
        captured_balanced = {}
        captured_aggressive = {}

        approve_verdict = RiskDecision(
            verdict="APPROVE",
            confidence=0.80,
            reasoning="The position looks good from this perspective.",
            risk_factors=[],
            cited_concentration_pct=1.0,  # fail-safe sentinel
        )
        mock_conservative = _structured_risk_perspective_llm(
            captured_conservative, approve_verdict
        )
        mock_balanced = _structured_risk_perspective_llm(
            captured_balanced, approve_verdict
        )
        mock_aggressive = _structured_risk_perspective_llm(
            captured_aggressive, approve_verdict
        )

        with patch("tradingagents.agents.risk_mgmt.risk_aggregator.get_config") as mock_config:
            mock_config.return_value = {
                "risk_log_path": str(tmp_path / "risk.jsonl")
            }
            with patch(
                "tradingagents.agents.risk_mgmt.conservative_perspective.create_llm_client"
            ) as mock_factory_cons:
                with patch(
                    "tradingagents.agents.risk_mgmt.balanced_perspective.create_llm_client"
                ) as mock_factory_bal:
                    with patch(
                        "tradingagents.agents.risk_mgmt.aggressive_perspective.create_llm_client"
                    ) as mock_factory_agg:
                        mock_factory_cons.return_value.get_llm.return_value = mock_conservative
                        mock_factory_bal.return_value.get_llm.return_value = mock_balanced
                        mock_factory_agg.return_value.get_llm.return_value = mock_aggressive

                        # Run perspective nodes
                        cons_node = create_conservative_perspective()
                        cons_update = cons_node(state)
                        state.update(cons_update)

                        bal_node = create_balanced_perspective()
                        bal_update = bal_node(state)
                        state.update(bal_update)

                        agg_node = create_aggressive_perspective()
                        agg_update = agg_node(state)
                        state.update(agg_update)

                        # Run aggregator
                        aggregator = create_risk_aggregator()
                        agg_result = aggregator(state)
                        state.update(agg_result)

                        # Assertions: snapshot failure overrides all approvals
                        assert state.get("final_veto") is True, "Snapshot failure → final_veto must be True regardless of LLM verdicts"
                        assert state.get("portfolio_snapshot_status") == "fail"

                        # Verify persistence
                        assert (tmp_path / "risk.jsonl").exists()
                        import json
                        lines = (tmp_path / "risk.jsonl").read_text(encoding="utf-8").strip().split("\n")
                        record = json.loads(lines[0])
                        assert record["final_veto"] is True
                        assert record["portfolio_snapshot_status"] == "fail"

    def test_4_concentration_hallucination_flagged(self, tmp_path):
        """Test 4 (concentration-hallucination flagged — D-11 discipline applied a third time):
        One perspective's cited_concentration_pct differs from state's risk_concentration_pct
        by more than 0.005 (0.5pp) → concentration_verified=False in persisted record,
        while final_veto is still computed independently from verdicts alone
        (orthogonal fields, mirrors Auditor's verified/comparison_result orthogonality).
        """
        from tradingagents.agents.risk_mgmt.conservative_perspective import (
            create_conservative_perspective,
        )
        from tradingagents.agents.risk_mgmt.balanced_perspective import (
            create_balanced_perspective,
        )
        from tradingagents.agents.risk_mgmt.aggressive_perspective import (
            create_aggressive_perspective,
        )
        from tradingagents.agents.risk_mgmt.risk_aggregator import (
            create_risk_aggregator,
        )

        state = _make_risk_squad_state(risk_concentration_pct=0.02)
        captured_conservative = {}
        captured_balanced = {}
        captured_aggressive = {}

        # Conservative hallucinates concentration (0.06 vs actual 0.02, difference > 0.5pp)
        hallucinating_verdict = RiskDecision(
            verdict="APPROVE",
            confidence=0.70,
            reasoning="Position looks acceptable.",
            risk_factors=[],
            cited_concentration_pct=0.06,  # Hallucinates 6% when actual is 2%
        )
        # Others cite correctly
        correct_verdict = RiskDecision(
            verdict="APPROVE",
            confidence=0.85,
            reasoning="Position size and concentration are within limits.",
            risk_factors=[],
            cited_concentration_pct=0.02,  # Correct
        )
        mock_conservative = _structured_risk_perspective_llm(
            captured_conservative, hallucinating_verdict
        )
        mock_balanced = _structured_risk_perspective_llm(
            captured_balanced, correct_verdict
        )
        mock_aggressive = _structured_risk_perspective_llm(
            captured_aggressive, correct_verdict
        )

        with patch("tradingagents.agents.risk_mgmt.risk_aggregator.get_config") as mock_config:
            mock_config.return_value = {
                "risk_log_path": str(tmp_path / "risk.jsonl")
            }
            with patch(
                "tradingagents.agents.risk_mgmt.conservative_perspective.create_llm_client"
            ) as mock_factory_cons:
                with patch(
                    "tradingagents.agents.risk_mgmt.balanced_perspective.create_llm_client"
                ) as mock_factory_bal:
                    with patch(
                        "tradingagents.agents.risk_mgmt.aggressive_perspective.create_llm_client"
                    ) as mock_factory_agg:
                        mock_factory_cons.return_value.get_llm.return_value = mock_conservative
                        mock_factory_bal.return_value.get_llm.return_value = mock_balanced
                        mock_factory_agg.return_value.get_llm.return_value = mock_aggressive

                        # Run perspective nodes
                        cons_node = create_conservative_perspective()
                        cons_update = cons_node(state)
                        state.update(cons_update)

                        bal_node = create_balanced_perspective()
                        bal_update = bal_node(state)
                        state.update(bal_update)

                        agg_node = create_aggressive_perspective()
                        agg_update = agg_node(state)
                        state.update(agg_update)

                        # Run aggregator
                        aggregator = create_risk_aggregator()
                        agg_result = aggregator(state)
                        state.update(agg_result)

                        # Assertions
                        assert state.get("concentration_verified") is False, "Hallucination detected → not verified"
                        # But final_veto is independent: all approved, so it's False
                        assert state.get("final_veto") is False, "final_veto independent of concentration_verified"

                        # Verify persisted record captures both orthogonal fields
                        assert (tmp_path / "risk.jsonl").exists()
                        import json
                        lines = (tmp_path / "risk.jsonl").read_text(encoding="utf-8").strip().split("\n")
                        record = json.loads(lines[0])
                        assert record["concentration_verified"] is False, "Hallucination recorded"
                        assert record["final_veto"] is False, "Verdicts still determine veto independently"
