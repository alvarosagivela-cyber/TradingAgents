"""Unit tests for Risk Squad perspectives with reflection injection (D-01)."""

import inspect
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.reflectors.reflection_schema import ReflectionRecord
from tradingagents.agents.risk_mgmt.aggressive_perspective import (
    create_aggressive_perspective,
)
from tradingagents.agents.risk_mgmt.balanced_perspective import (
    create_balanced_perspective,
)
from tradingagents.agents.risk_mgmt.conservative_perspective import (
    create_conservative_perspective,
)

# ---------------------------------------------------------------------------
# Parameterized Test: Reflection Injection When Present (All 3 Perspectives)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "perspective_factory,perspective_name,verdict_field",
    [
        (create_conservative_perspective, "conservative", "conservative_verdict"),
        (create_balanced_perspective, "balanced", "balanced_verdict"),
        (create_aggressive_perspective, "aggressive", "aggressive_verdict"),
    ],
)
def test_reflection_injected_when_present(perspective_factory, perspective_name, verdict_field):
    """Reflection is injected into the prompt when read_reflection returns a record."""
    # Create a fixture reflection
    reflection = ReflectionRecord(
        ticker="AAPL",
        decision_date="2026-07-15",
        decision_verdict="VETO",
        realized_return=-0.05,
        classification="negative",
        lesson_text="Prior risk assessment correctly identified concentration breach.",
        created_at="2026-07-20T00:00:00Z",
    )

    # Mock the LLM to capture the prompt
    mock_llm = MagicMock()
    captured_prompt = None

    def capture_prompt(prompt_text):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        # Return a minimal valid risk decision object
        verdict = MagicMock()
        verdict.verdict = "APPROVE"
        verdict.confidence = 0.8
        verdict.reasoning = "Test reasoning"
        verdict.risk_factors = []
        verdict.cited_concentration_pct = 0.02
        verdict.model_dump = MagicMock(
            return_value={
                "verdict": "APPROVE",
                "confidence": 0.8,
                "reasoning": "Test reasoning",
                "risk_factors": [],
                "cited_concentration_pct": 0.02,
            }
        )
        return verdict

    mock_llm.invoke = capture_prompt

    # Patch the module-specific path for this perspective
    module_path = f"tradingagents.agents.risk_mgmt.{perspective_name}_perspective"

    # Mock bind_structured to return our mock LLM, create_llm_client to avoid
    # actual LLM creation, and read_reflection_for_ticker to return the fixture
    with (
        patch(f"{module_path}.bind_structured", return_value=mock_llm),
        patch(f"{module_path}.create_llm_client"),
        patch(
            f"{module_path}.read_reflection_for_ticker",
            return_value=reflection,
        ) as mock_read,
    ):
        # Create the perspective node and invoke it
        node = perspective_factory("claude-haiku-4-5")

        state = {
            "company_of_interest": "AAPL",
            "proposed_side": "Buy",
            "portfolio_total_value": 100000.0,
            "existing_position_value": 5000.0,
            "proposed_notional_usd": 10000.0,
            "risk_concentration_pct": 0.10,
        }

        node(state)

        # Verify read_reflection_for_ticker was called with "risk" layer
        mock_read.assert_called_once_with("risk", "AAPL")

        # Verify the prompt contains the reflection content
        assert captured_prompt is not None
        assert "2026-07-15" in captured_prompt
        assert "VETO" in captured_prompt
        assert "concentration breach" in captured_prompt


# ---------------------------------------------------------------------------
# Parameterized Test: No Reflection Injection When None (All 3 Perspectives)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "perspective_factory,perspective_name,verdict_field",
    [
        (create_conservative_perspective, "conservative", "conservative_verdict"),
        (create_balanced_perspective, "balanced", "balanced_verdict"),
        (create_aggressive_perspective, "aggressive", "aggressive_verdict"),
    ],
)
def test_no_reflection_injection_when_none(perspective_factory, perspective_name, verdict_field):
    """Reflection block is absent from prompt when read_reflection returns None."""
    # Mock the LLM to capture the prompt
    mock_llm = MagicMock()
    captured_prompt = None

    def capture_prompt(prompt_text):
        nonlocal captured_prompt
        captured_prompt = prompt_text
        verdict = MagicMock()
        verdict.verdict = "APPROVE"
        verdict.confidence = 0.8
        verdict.reasoning = "Test reasoning"
        verdict.risk_factors = []
        verdict.cited_concentration_pct = 0.02
        verdict.model_dump = MagicMock(
            return_value={
                "verdict": "APPROVE",
                "confidence": 0.8,
                "reasoning": "Test reasoning",
                "risk_factors": [],
                "cited_concentration_pct": 0.02,
            }
        )
        return verdict

    mock_llm.invoke = capture_prompt

    # Patch the module-specific path for this perspective
    module_path = f"tradingagents.agents.risk_mgmt.{perspective_name}_perspective"

    # Mock bind_structured to return our mock LLM, create_llm_client to avoid
    # actual LLM creation, and read_reflection_for_ticker to return None
    with (
        patch(f"{module_path}.bind_structured", return_value=mock_llm),
        patch(f"{module_path}.create_llm_client"),
        patch(
            f"{module_path}.read_reflection_for_ticker",
            return_value=None,
        ),
    ):
        node = perspective_factory("claude-haiku-4-5")

        state = {
            "company_of_interest": "AAPL",
            "proposed_side": "Buy",
            "portfolio_total_value": 100000.0,
            "existing_position_value": 5000.0,
            "proposed_notional_usd": 10000.0,
            "risk_concentration_pct": 0.10,
        }

        node(state)

        # Verify the prompt does NOT contain a reflection section header
        assert captured_prompt is not None
        # The prompt should not have a reflection block
        assert "Prior Risk Assessment" not in captured_prompt or captured_prompt.count("Prior Risk Assessment") == 0


# ---------------------------------------------------------------------------
# Test Ordering: Read Happens Before Prompt Construction (All 3 Perspectives)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "perspective_factory,perspective_name",
    [
        (create_conservative_perspective, "conservative"),
        (create_balanced_perspective, "balanced"),
        (create_aggressive_perspective, "aggressive"),
    ],
)
def test_read_happens_before_prompt_construction(perspective_factory, perspective_name):
    """Inspect source to verify read_reflection_for_ticker appears before prompt f-string."""
    source = inspect.getsource(perspective_factory)

    # Find positions of read_reflection_for_ticker and prompt construction
    read_pos = source.find("read_reflection_for_ticker")
    prompt_pos = source.find('prompt = f"""')

    assert read_pos != -1, "read_reflection_for_ticker should be present in source"
    assert prompt_pos != -1, "prompt construction should be present in source"
    assert (
        read_pos < prompt_pos
    ), "read_reflection_for_ticker must appear before prompt construction"


# ---------------------------------------------------------------------------
# Test Shared Risk Layer Store (All 3 Perspectives Use Same Layer)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "perspective_factory,perspective_name",
    [
        (create_conservative_perspective, "conservative"),
        (create_balanced_perspective, "balanced"),
        (create_aggressive_perspective, "aggressive"),
    ],
)
def test_shared_risk_layer_store(perspective_factory, perspective_name):
    """All three perspectives call read_reflection_for_ticker with the same 'risk' layer."""
    source = inspect.getsource(perspective_factory)

    # Should contain "risk" as the layer argument
    assert '"risk"' in source, f"{perspective_name} should read from 'risk' layer"


# ---------------------------------------------------------------------------
# Test Isolation Guard: No Cross-Layer References (All 3 Perspectives)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "perspective_factory,perspective_name",
    [
        (create_conservative_perspective, "conservative"),
        (create_balanced_perspective, "balanced"),
        (create_aggressive_perspective, "aggressive"),
    ],
)
def test_isolation_guard_no_cross_layer_references(perspective_factory, perspective_name):
    """Verify each perspective reads 'risk' layer only, not 'research' or 'auditor'."""
    source = inspect.getsource(perspective_factory)

    # Should NOT contain references to research or auditor layers
    assert (
        "research_reflections" not in source
    ), f"{perspective_name} should not reference research_reflections"
    assert (
        "auditor_reflections" not in source
    ), f"{perspective_name} should not reference auditor_reflections"
