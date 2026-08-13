"""Unit tests for momentum researcher with reflection injection (D-01)."""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.reflectors.reflection_schema import ReflectionRecord
from tradingagents.agents.researchers.momentum_researcher import create_momentum_llm_node

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
        lesson_text="Prior analysis showed weakness; thesis invalidated.",
        created_at="2026-07-20T00:00:00Z",
    )

    # Mock the LLM to capture the prompt
    mock_llm = MagicMock()
    captured_prompt = None

    def capture_prompt(prompt_text):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        # Return a minimal valid thesis object
        thesis = MagicMock()
        thesis.reasoning = "Test reasoning"
        thesis.verdict.value = "Buy"
        thesis.refutation_criterion = "Test criterion"
        thesis.data_points = {"retorno_12_1": 0.1, "z_score": 2.0}
        return thesis

    mock_llm.invoke = capture_prompt

    # Mock bind_structured to return our mock LLM and read_reflection_for_ticker
    # to return the fixture
    with (
        patch(
            "tradingagents.agents.researchers.momentum_researcher.bind_structured",
            return_value=mock_llm,
        ),
        patch(
            "tradingagents.agents.researchers.momentum_researcher.read_reflection_for_ticker",
            return_value=reflection,
        ) as mock_read,
    ):
        # Create the LLM node and invoke it
        llm_node = create_momentum_llm_node(MagicMock())

        state = {
            "company_of_interest": "AAPL",
            "momentum_valid": True,
            "retorno_12_1": 0.1,
            "momentum_z_score": 2.0,
            "market_report": "Market is bullish",
        }

        llm_node(state)

        # Verify read_reflection_for_ticker was called with "research" layer
        mock_read.assert_called_once_with("research", "AAPL")

        # Verify the prompt contains the reflection content
        assert captured_prompt is not None
        assert "2026-07-15" in captured_prompt
        assert "Sell" in captured_prompt
        assert "Prior analysis showed weakness" in captured_prompt


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
        thesis = MagicMock()
        thesis.reasoning = "Test reasoning"
        thesis.verdict.value = "Buy"
        thesis.refutation_criterion = "Test criterion"
        thesis.data_points = {"retorno_12_1": 0.1, "z_score": 2.0}
        return thesis

    mock_llm.invoke = capture_prompt

    # Mock bind_structured to return our mock LLM and read_reflection_for_ticker
    # to return None
    with (
        patch(
            "tradingagents.agents.researchers.momentum_researcher.bind_structured",
            return_value=mock_llm,
        ),
        patch(
            "tradingagents.agents.researchers.momentum_researcher.read_reflection_for_ticker",
            return_value=None,
        ),
    ):
        llm_node = create_momentum_llm_node(MagicMock())

        state = {
            "company_of_interest": "AAPL",
            "momentum_valid": True,
            "retorno_12_1": 0.1,
            "momentum_z_score": 2.0,
            "market_report": "Market is bullish",
        }

        llm_node(state)

        # Verify the prompt does NOT contain a reflection section header
        assert captured_prompt is not None
        # The prompt should not have a reflection block (check for a common header we'd use)
        assert "Prior Analysis" not in captured_prompt or captured_prompt.count("Prior Analysis") == 0


# ---------------------------------------------------------------------------
# Test Read Happens Before Prompt Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_happens_before_prompt_construction():
    """Inspect source to verify read_reflection_for_ticker appears before prompt f-string."""
    source = inspect.getsource(create_momentum_llm_node)

    # Find positions of read_reflection_for_ticker and prompt construction
    read_pos = source.find("read_reflection_for_ticker")
    prompt_pos = source.find('prompt = f"""')

    assert read_pos != -1, "read_reflection_for_ticker should be present in source"
    assert prompt_pos != -1, "prompt construction should be present in source"
    assert (
        read_pos < prompt_pos
    ), "read_reflection_for_ticker must appear before prompt construction"


# ---------------------------------------------------------------------------
# Test Correct Layer and Source Isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_correct_layer_source_isolation():
    """Verify function reads 'research' layer only, not 'auditor' or 'risk'."""
    source = inspect.getsource(create_momentum_llm_node)

    # Should contain "research" as the layer argument
    assert '"research"' in source, "Should read from 'research' layer"

    # Should NOT contain references to auditor or risk layers
    assert "auditor_reflections" not in source, "Should not reference auditor_reflections"
    assert "risk_reflections" not in source, "Should not reference risk_reflections"
    assert '"auditor"' not in source, "Should not use 'auditor' layer"
    assert '"risk"' not in source, "Should not use 'risk' layer"


# ---------------------------------------------------------------------------
# Test Fail-Open Unaffected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fail_open_unaffected():
    """With momentum_valid=False, reflection is never read (fail-open short-circuit)."""
    mock_llm = MagicMock()

    with patch(
        "tradingagents.agents.researchers.momentum_researcher.bind_structured",
        return_value=mock_llm,
    ), patch(
        "tradingagents.agents.researchers.momentum_researcher.read_reflection_for_ticker"
    ) as mock_read:
        llm_node = create_momentum_llm_node(MagicMock())

        state = {
            "company_of_interest": "AAPL",
            "momentum_valid": False,  # Short-circuit
            "retorno_12_1": 0.0,
            "momentum_z_score": 0.0,
            "market_report": "",
        }

        result = llm_node(state)

        # Verify read_reflection_for_ticker was NEVER called
        mock_read.assert_not_called()
        # Verify we got the fail-open response
        assert result["verified"] is False


# ---------------------------------------------------------------------------
# Test No Memory Log References
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_memory_log_references():
    """Verify momentum_researcher does not reference TradingMemoryLog machinery."""
    source = inspect.getsource(
        __import__("tradingagents.agents.researchers.momentum_researcher", fromlist=[""])
    )

    # Should NOT reference the old memory/reflection system
    assert "memory_log" not in source, "Should not reference memory_log"
    assert "TradingMemoryLog" not in source, "Should not reference TradingMemoryLog"
    assert "get_past_context" not in source, "Should not reference get_past_context"
