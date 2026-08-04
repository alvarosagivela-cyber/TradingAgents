"""Structural tests for the D-04 reflection A/B spike script's own dispatch logic.

These tests never invoke a real node or LLM — they only check the script's own
control flow (adapter table shape, --layer selection, divergence comparison, and
the run-one-layer orchestration) with a MagicMock node substituted in. The real
run against the live Anthropic API is a separate, manually-invoked checkpoint
(Task 2 of this plan), not exercised here.
"""

from unittest.mock import MagicMock, patch

import pytest

from scripts.reflection_ab_spike import (
    ADAPTERS,
    LayerAdapter,
    _compute_divergence,
    _run_layer,
    _select_layers,
)
from tradingagents.agents.reflectors.reflection_schema import ReflectionRecord


# A dummy target this test module owns, so patch("...read_reflection_stub") is a
# real, resolvable dotted path — mirroring how each real adapter's patch_target
# points at that layer's own module-level import of read_reflection_for_ticker.
def read_reflection_stub(layer: str, ticker: str):
    raise AssertionError("read_reflection_stub should always be patched in tests")


@pytest.mark.unit
class TestAdapterTable:
    def test_adapter_table_has_exactly_three_layer_keys(self):
        assert set(ADAPTERS.keys()) == {"research", "auditor", "risk"}

    def test_each_adapter_entry_is_a_layer_adapter(self):
        for layer, adapter in ADAPTERS.items():
            assert isinstance(adapter, LayerAdapter), f"{layer} adapter has wrong type"
            assert callable(adapter.build_state)
            assert callable(adapter.build_node)
            assert callable(adapter.extract_value)
            assert isinstance(adapter.patch_target, str) and adapter.patch_target


@pytest.mark.unit
class TestSelectLayers:
    def test_layer_all_selects_all_three_layers(self):
        assert _select_layers("all") == ["research", "auditor", "risk"]

    def test_layer_research_selects_only_research(self):
        assert _select_layers("research") == ["research"]

    def test_layer_auditor_selects_only_auditor(self):
        assert _select_layers("auditor") == ["auditor"]

    def test_layer_risk_selects_only_risk(self):
        assert _select_layers("risk") == ["risk"]


@pytest.mark.unit
class TestComputeDivergence:
    def test_returns_true_when_values_differ(self):
        assert _compute_divergence("Buy", "Sell") is True

    def test_returns_false_when_values_are_equal(self):
        assert _compute_divergence("Hold", "Hold") is False

    def test_returns_true_when_risk_tuple_confidence_differs_but_verdict_matches(self):
        # Mirrors the D-04 rule: verdict unchanged but numeric confidence shifted
        # still counts as divergence, since the extracted "value" for risk is the
        # (verdict, confidence) pair.
        without_value = ("VETO", 0.8)
        with_value = ("VETO", 0.6)
        assert _compute_divergence(without_value, with_value) is True

    def test_returns_false_when_risk_tuple_is_fully_identical(self):
        without_value = ("APPROVE", 0.7)
        with_value = ("APPROVE", 0.7)
        assert _compute_divergence(without_value, with_value) is False


@pytest.mark.unit
class TestRunLayer:
    """Covers _run_layer's own orchestration with a fully mocked adapter."""

    def _make_fake_adapter(self, without_result: dict, with_result: dict) -> LayerAdapter:
        node = MagicMock(side_effect=[without_result, with_result])
        return LayerAdapter(
            build_state=MagicMock(return_value={"company_of_interest": "AAPL"}),
            build_node=MagicMock(return_value=node),
            patch_target=f"{__name__}.read_reflection_stub",
            extract_value=lambda result: result.get("verdict"),
        )

    def test_run_layer_reports_diverged_true_when_verdicts_differ(self):
        fake_adapter = self._make_fake_adapter(
            without_result={"verdict": "Buy"},
            with_result={"verdict": "Sell"},
        )
        with patch.dict(ADAPTERS, {"research": fake_adapter}):
            reflection = ReflectionRecord(
                ticker="AAPL",
                decision_date="2026-07-29",
                decision_verdict="Sell",
                realized_return=-0.05,
                classification="negative",
                lesson_text="Sell decision resulted in a loss.",
                created_at="2026-01-01T00:00:00Z",
            )
            layer, without_value, with_value, diverged = _run_layer(
                "research", "AAPL", "2026-07-29", reflection
            )

        assert layer == "research"
        assert without_value == "Buy"
        assert with_value == "Sell"
        assert diverged is True
        fake_adapter.build_state.assert_called_once_with("AAPL", "2026-07-29")
        fake_adapter.build_node.assert_called_once_with()

    def test_run_layer_reports_diverged_false_when_verdicts_match(self):
        fake_adapter = self._make_fake_adapter(
            without_result={"verdict": "Hold"},
            with_result={"verdict": "Hold"},
        )
        with patch.dict(ADAPTERS, {"research": fake_adapter}):
            reflection = ReflectionRecord(
                ticker="AAPL",
                decision_date="2026-07-29",
                decision_verdict="Sell",
                realized_return=-0.05,
                classification="negative",
                lesson_text="Sell decision resulted in a loss.",
                created_at="2026-01-01T00:00:00Z",
            )
            layer, without_value, with_value, diverged = _run_layer(
                "research", "AAPL", "2026-07-29", reflection
            )

        assert without_value == "Hold"
        assert with_value == "Hold"
        assert diverged is False
