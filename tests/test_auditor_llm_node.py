"""Unit tests for Auditor LLM node — instance isolation, field isolation, asymmetric prompt, short-circuit."""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.auditors.momentum_auditor import (
    create_auditor_llm_node,
)
from tradingagents.agents.schemas import AuditorVerdict, TraderAction


@pytest.mark.unit
class TestAuditorLLMNode:
    """Test Auditor LLM node isolation, asymmetric prompt, and fallback behavior (AUDIT-02)."""

    def test_llm_node_instance_isolation(self):
        """Source contains create_llm_client call and does NOT share quick_thinking_llm (AUDIT-02, D-03)."""
        source = inspect.getsource(create_auditor_llm_node)

        # MUST have create_llm_client call (fresh instance per invocation)
        assert "create_llm_client(" in source, "Auditor LLM node must create fresh LLM instance"

        # MUST NOT read state["quick_thinking_llm"] or state.get("quick_thinking_llm")
        # (Can mention in comments/docstrings, but not in actual code)
        assert 'state["quick_thinking_llm"]' not in source, "Auditor should NOT read shared quick_thinking_llm"
        assert 'state.get("quick_thinking_llm")' not in source, "Auditor should NOT read shared quick_thinking_llm"

        # MUST NOT have raw ChatAnthropic call (should go through factory)
        assert "ChatAnthropic(" not in source, "Should use factory, not raw ChatAnthropic()"

    def test_llm_node_field_isolation(self):
        """Source does NOT contain forbidden Research-only fields (AUDIT-01)."""
        source = inspect.getsource(create_auditor_llm_node)

        # Forbidden fields that should NOT appear in Auditor LLM node
        forbidden_patterns = [
            'state.get("market_report"',
            'state["market_report"',
            'state.get("sentiment_report"',
            'state["sentiment_report"',
            'state.get("news_report"',
            'state["news_report"',
            'state.get("fundamentals_report"',
            'state["fundamentals_report"',
            'state.get("thesis"',
            'state["thesis"',
        ]

        for pattern in forbidden_patterns:
            assert pattern not in source, f"Auditor LLM node should not read {pattern}"

    def test_llm_node_asymmetric_prompt(self):
        """Source contains asymmetric refutation mandate language (D-04)."""
        source = inspect.getsource(create_auditor_llm_node)

        # Must have explicit refutation language
        assert "wrong" in source.lower() or "refut" in source.lower(), (
            "Auditor LLM node must have refutation mandate language"
        )

    def test_llm_node_short_circuit_on_phase1_fail(self):
        """If auditor_phase1_status != 'pass', short-circuit without calling LLM (D-02)."""
        llm_node = create_auditor_llm_node()

        state = {
            "company_of_interest": "AAPL",
            "auditor_phase1_status": "fail",
            "auditor_phase1_failure_reason": "insufficient_ohlcv_history",
            "thesis_verdict": "Buy",
        }

        result = llm_node(state)

        # Must return INCONCLUSIVE without invoking LLM
        assert result["auditor_verdict"] == "INCONCLUSIVE"
        assert "REJECTED" in result["auditor_reasoning"]

    def test_llm_node_structured_output_none_fallback(self):
        """If structured_output returns None, return INCONCLUSIVE (D-02)."""
        llm_node = create_auditor_llm_node()

        state = {
            "company_of_interest": "AAPL",
            "auditor_phase1_status": "pass",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": 2.3,
            "thesis_verdict": "Buy",
        }

        with patch(
            "tradingagents.agents.auditors.momentum_auditor.create_llm_client"
        ) as mock_client_factory:
            # Mock LLM client that returns None on invoke
            mock_client = MagicMock()
            mock_llm = MagicMock()
            mock_client.get_llm.return_value = mock_llm
            mock_client_factory.return_value = mock_client

            with patch(
                "tradingagents.agents.auditors.momentum_auditor.bind_structured"
            ) as mock_bind:
                # bind_structured returns a mock that returns None on invoke
                mock_structured_llm = MagicMock()
                mock_structured_llm.invoke.return_value = None
                mock_bind.return_value = mock_structured_llm

                result = llm_node(state)

                # Must return INCONCLUSIVE
                assert result["auditor_verdict"] == "INCONCLUSIVE"
                assert "REJECTED" in result["auditor_reasoning"]

    def test_llm_node_invokes_fresh_llm_on_pass(self):
        """If phase1_status='pass', invoke fresh LLM instance and return structured verdict."""
        llm_node = create_auditor_llm_node()

        state = {
            "company_of_interest": "AAPL",
            "auditor_phase1_status": "pass",
            "auditor_retorno_12_1": 0.1834,
            "auditor_z_score": 2.3,
            "thesis_verdict": "Buy",
        }

        # Create a proper AuditorVerdict mock
        mock_verdict = AuditorVerdict(
            verdict=TraderAction.HOLD,
            reasoning="The Research thesis is based on short-term momentum which could reverse. Auditor sees significant risks.",
            refutation_criterion="Price breaks 50-day MA.",
            data_points={"retorno_12_1": 0.1834, "z_score": 2.3},
        )

        with patch(
            "tradingagents.agents.auditors.momentum_auditor.create_llm_client"
        ) as mock_client_factory:
            mock_client = MagicMock()
            mock_llm = MagicMock()
            mock_client.get_llm.return_value = mock_llm
            mock_client_factory.return_value = mock_client

            with patch(
                "tradingagents.agents.auditors.momentum_auditor.bind_structured"
            ) as mock_bind:
                mock_structured_llm = MagicMock()
                mock_structured_llm.invoke.return_value = mock_verdict
                mock_bind.return_value = mock_structured_llm

                result = llm_node(state)

                # Verify create_llm_client was called (fresh instance)
                mock_client_factory.assert_called_once()

                # Verify result contains the structured verdict
                assert result["auditor_verdict"] == "Hold"
                assert "Research thesis is based on short-term momentum" in result["auditor_reasoning"]
                assert result["auditor_refutation_criterion"] == "Price breaks 50-day MA."
                assert result["auditor_data_points"]["retorno_12_1"] == 0.1834
