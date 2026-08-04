"""Unit tests for auditor with reflection injection (D-01)."""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.auditors.momentum_auditor import create_auditor_llm_node
from tradingagents.agents.reflectors.reflection_schema import ReflectionRecord


# ---------------------------------------------------------------------------
# Test Reflection Injection When Present
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reflection_injected_when_present():
    """Reflection is injected into the prompt when read_reflection returns a record."""
    # Create a fixture reflection
    reflection = ReflectionRecord(
        ticker="AAPL",
        decision_date="2026-07-15",
        decision_verdict="Sell",
        realized_return=-0.03,
        classification="negative",
        lesson_text="Prior auditor analysis revealed weakness; thesis was correctly refuted.",
        created_at="2026-07-20T00:00:00Z",
    )

    # Mock the LLM to capture the prompt
    mock_llm = MagicMock()
    captured_prompt = None

    def capture_prompt(prompt_text):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        # Return a minimal valid auditor verdict object
        verdict = MagicMock()
        verdict.verdict.value = "Sell"
        verdict.reasoning = "Test reasoning"
        verdict.refutation_criterion = "Test criterion"
        verdict.data_points = {"retorno_12_1": 0.1, "z_score": 2.0}
        return verdict

    mock_llm.invoke = capture_prompt

    # Mock bind_structured to return our mock LLM
    with patch(
        "tradingagents.agents.auditors.momentum_auditor.bind_structured",
        return_value=mock_llm,
    ):
        # Mock create_llm_client to avoid actual LLM creation
        with patch(
            "tradingagents.agents.auditors.momentum_auditor.create_llm_client",
        ):
            # Mock read_reflection_for_ticker to return the fixture
            with patch(
                "tradingagents.agents.auditors.momentum_auditor.read_reflection_for_ticker",
                return_value=reflection,
            ) as mock_read:
                # Create the LLM node and invoke it
                llm_node = create_auditor_llm_node("claude-sonnet-5")

                state = {
                    "company_of_interest": "AAPL",
                    "auditor_phase1_status": "pass",
                    "auditor_retorno_12_1": 0.1,
                    "auditor_z_score": 2.0,
                    "thesis_verdict": "Buy",
                }

                result = llm_node(state)

                # Verify read_reflection_for_ticker was called with "auditor" layer
                mock_read.assert_called_once_with("auditor", "AAPL")

                # Verify the prompt contains the reflection content
                assert captured_prompt is not None
                assert "2026-07-15" in captured_prompt
                assert "Sell" in captured_prompt
                assert "Prior auditor analysis revealed weakness" in captured_prompt


# ---------------------------------------------------------------------------
# Test No Reflection Injection When None
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_reflection_injection_when_none():
    """Reflection block is absent from prompt when read_reflection returns None."""
    # Mock the LLM to capture the prompt
    mock_llm = MagicMock()
    captured_prompt = None

    def capture_prompt(prompt_text):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        verdict = MagicMock()
        verdict.verdict.value = "Hold"
        verdict.reasoning = "Test reasoning"
        verdict.refutation_criterion = "Test criterion"
        verdict.data_points = {"retorno_12_1": 0.1, "z_score": 2.0}
        return verdict

    mock_llm.invoke = capture_prompt

    with patch(
        "tradingagents.agents.auditors.momentum_auditor.bind_structured",
        return_value=mock_llm,
    ):
        # Mock create_llm_client to avoid actual LLM creation
        with patch(
            "tradingagents.agents.auditors.momentum_auditor.create_llm_client",
        ):
            # Mock read_reflection_for_ticker to return None
            with patch(
                "tradingagents.agents.auditors.momentum_auditor.read_reflection_for_ticker",
                return_value=None,
            ):
                llm_node = create_auditor_llm_node("claude-sonnet-5")

                state = {
                    "company_of_interest": "AAPL",
                    "auditor_phase1_status": "pass",
                    "auditor_retorno_12_1": 0.1,
                    "auditor_z_score": 2.0,
                    "thesis_verdict": "Buy",
                }

                result = llm_node(state)

                # Verify the prompt does NOT contain a reflection section header
                assert captured_prompt is not None
                # The prompt should not have a reflection block (check for a common header we'd use)
                assert "Prior Auditor Analysis" not in captured_prompt or captured_prompt.count("Prior Auditor Analysis") == 0


# ---------------------------------------------------------------------------
# Test Read Happens Before Prompt Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_happens_before_prompt_construction():
    """Inspect source to verify read_reflection_for_ticker appears before prompt f-string."""
    source = inspect.getsource(create_auditor_llm_node)

    # Find positions of read_reflection_for_ticker and prompt construction
    read_pos = source.find("read_reflection_for_ticker")
    prompt_pos = source.find('prompt = f"""')

    assert read_pos != -1, "read_reflection_for_ticker should be present in source"
    assert prompt_pos != -1, "prompt construction should be present in source"
    assert (
        read_pos < prompt_pos
    ), "read_reflection_for_ticker must appear before prompt construction"


# ---------------------------------------------------------------------------
# Test INCONCLUSIVE Short-Circuit Unaffected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inconclusive_short_circuit_unaffected():
    """With auditor_phase1_status != 'pass', read_reflection_for_ticker is never called."""
    mock_llm = MagicMock()

    with patch(
        "tradingagents.agents.auditors.momentum_auditor.bind_structured",
        return_value=mock_llm,
    ):
        with patch(
            "tradingagents.agents.auditors.momentum_auditor.read_reflection_for_ticker"
        ) as mock_read:
            llm_node = create_auditor_llm_node(MagicMock())

            state = {
                "company_of_interest": "AAPL",
                "auditor_phase1_status": "fail",  # Short-circuit
                "auditor_phase1_failure_reason": "insufficient_ohlcv_history",
                "auditor_retorno_12_1": 0.0,
                "auditor_z_score": 0.0,
                "thesis_verdict": "Buy",
            }

            result = llm_node(state)

            # Verify read_reflection_for_ticker was NEVER called
            mock_read.assert_not_called()
            # Verify we got the fail-open response
            assert result["auditor_verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Test Isolation Guard (Local Source Inspection)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_isolation_guard_local():
    """Verify auditor reads 'auditor' layer only, not 'research' or 'risk'."""
    source = inspect.getsource(create_auditor_llm_node)

    # Should contain "auditor" as the layer argument
    assert '"auditor"' in source, "Should read from 'auditor' layer"

    # Should NOT contain references to research or risk layers
    assert "research_reflections" not in source, "Should not reference research_reflections"
    assert "risk_reflections" not in source, "Should not reference risk_reflections"
