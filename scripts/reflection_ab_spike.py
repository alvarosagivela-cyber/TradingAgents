"""D-04 reflection A/B divergence spike: prove a negative reflection changes real output.

This script runs each of the three memory-consuming layers (Research's Momentum LLM
node, the Auditor's LLM node, and the Risk Squad's Conservative perspective) TWICE
against the real Anthropic API with identical ticker/date/state — once with
``read_reflection_for_ticker`` patched to return ``None`` (no prior reflection), and
once patched to return a single synthetic, strongly-negative ``ReflectionRecord``.

For each layer the two runs' verdict (or, for the Risk layer, the (verdict, confidence)
pair — since a VETO/APPROVE binary verdict can stay identical while confidence still
shifts) is mechanically compared. This is deliberately NOT a check that the reflection
text merely appears in the prompt (05-01/05-02 mocked wiring tests already prove that) —
it is proof the reflection observably changes what each layer actually decides.

Usage:
    ANTHROPIC_API_KEY=... python scripts/reflection_ab_spike.py --layer all --ticker AAPL
    ANTHROPIC_API_KEY=... python scripts/reflection_ab_spike.py --layer research

Cost: 2 real LLM calls per requested layer (6 total for --layer all), comparable in
scale to Phase 2's D-02 temperature spike (momentum_temperature_spike.py).

Exit code: 0 if every requested layer diverged, 1 if any requested layer did NOT
diverge (this is a genuine automated pass/fail check, not merely a human-eyeballed
printout — the deep review of the finding itself is a separate human checkpoint).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from tradingagents.agents import (
    create_momentum_compute_node,
    create_momentum_llm_node,
)
from tradingagents.agents.auditors.momentum_auditor import create_auditor_llm_node
from tradingagents.agents.reflectors.reflection_schema import ReflectionRecord
from tradingagents.agents.risk_mgmt.conservative_perspective import (
    create_conservative_perspective,
)
from tradingagents.llm_clients import create_llm_client

# ---------------------------------------------------------------------------
# Per-layer adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerAdapter:
    """Everything the spike needs to drive one layer's real node through an A/B run.

    Attributes:
        build_state: Builds the minimal state dict the layer's node needs.
        build_node: Constructs the real node callable (real LLM client included).
        patch_target: Dotted-path string to that layer's own read_reflection_for_ticker
                      import, passed straight to unittest.mock.patch.
        extract_value: Pulls the comparable verdict/confidence value out of a node result.
    """

    build_state: Callable[[str, str], dict[str, Any]]
    build_node: Callable[[], Callable[[dict[str, Any]], dict[str, Any]]]
    patch_target: str
    extract_value: Callable[[dict[str, Any]], Any]


def _research_build_state(ticker: str, trade_date: str) -> dict[str, Any]:
    """Mirrors momentum_temperature_spike.py's minimal state + a real compute-node run.

    The compute node's deterministic output is computed once here and reused for
    both A/B invocations of the LLM node (it does not depend on the reflection).
    """
    base_state: dict[str, Any] = {
        "company_of_interest": ticker,
        "trade_date": trade_date,
        "market_report": (
            f"Market context for {ticker}: typical trading activity. "
            "No specific catalysts mentioned in this spike run."
        ),
    }
    compute_node = create_momentum_compute_node()
    compute_result = compute_node(base_state)
    base_state.update(compute_result)
    return base_state


def _research_build_node() -> Callable[[dict[str, Any]], dict[str, Any]]:
    llm = create_llm_client(
        "anthropic", "claude-haiku-4-5", temperature=0.2
    ).get_llm()
    return create_momentum_llm_node(llm)


def _extract_research(result: dict[str, Any]) -> Any:
    return result.get("thesis_verdict")


def _auditor_build_state(ticker: str, trade_date: str) -> dict[str, Any]:
    """Mirrors test_auditor_pipeline.py::_make_auditor_state's defaults.

    Uses a fixed thesis_verdict="Buy" (the Research verdict the Auditor refutes)
    since this spike drives the Auditor layer in isolation, per the interfaces
    block's --layer auditor standalone fallback.
    """
    return {
        "company_of_interest": ticker,
        "trade_date": trade_date,
        "thesis_verdict": "Buy",
        "retorno_12_1": 0.18,
        "momentum_z_score": 2.1,
        "auditor_phase1_status": "pass",
        "auditor_phase1_failure_reason": "",
        "auditor_retorno_12_1": 0.1834,
        "auditor_z_score": 2.3,
        "auditor_confidence_level": "high",
        "auditor_verdict": "",
        "auditor_reasoning": "",
        "auditor_refutation_criterion": "",
        "auditor_data_points": {"retorno_12_1": 0.1834, "z_score": 2.3},
        "auditor_verified": False,
        "comparison_result": "",
    }


def _auditor_build_node() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return create_auditor_llm_node()


def _extract_auditor(result: dict[str, Any]) -> Any:
    return result.get("auditor_verdict")


def _risk_build_state(ticker: str, trade_date: str) -> dict[str, Any]:
    """Conservative-perspective-shaped state with fixed, plausible portfolio numbers."""
    return {
        "company_of_interest": ticker,
        "trade_date": trade_date,
        "proposed_side": "Buy",
        "portfolio_total_value": 100000.0,
        "existing_position_value": 0.0,
        "proposed_notional_usd": 2000.0,
        "risk_concentration_pct": 0.02,
    }


def _risk_build_node() -> Callable[[dict[str, Any]], dict[str, Any]]:
    return create_conservative_perspective()


def _extract_risk(result: dict[str, Any]) -> Any:
    # (verdict, confidence): comparing the pair naturally implements the D-04 rule
    # "compare the verdict, or when verdict is unchanged, the numeric confidence field" —
    # if either element differs the pair as a whole differs.
    verdict = result.get("conservative_verdict") or {}
    return (verdict.get("verdict"), verdict.get("confidence"))


ADAPTERS: dict[str, LayerAdapter] = {
    "research": LayerAdapter(
        build_state=_research_build_state,
        build_node=_research_build_node,
        patch_target="tradingagents.agents.researchers.momentum_researcher.read_reflection_for_ticker",
        extract_value=_extract_research,
    ),
    "auditor": LayerAdapter(
        build_state=_auditor_build_state,
        build_node=_auditor_build_node,
        patch_target="tradingagents.agents.auditors.momentum_auditor.read_reflection_for_ticker",
        extract_value=_extract_auditor,
    ),
    "risk": LayerAdapter(
        build_state=_risk_build_state,
        build_node=_risk_build_node,
        patch_target="tradingagents.agents.risk_mgmt.conservative_perspective.read_reflection_for_ticker",
        extract_value=_extract_risk,
    ),
}


# ---------------------------------------------------------------------------
# Shared synthetic negative reflection
# ---------------------------------------------------------------------------


def _build_negative_reflection(ticker: str, trade_date: str) -> ReflectionRecord:
    """One synthetic, strongly-negative reflection shared across all three layers' 'with' run."""
    return ReflectionRecord(
        ticker=ticker,
        decision_date=trade_date,
        decision_verdict="Sell",
        realized_return=-0.05,
        classification="negative",
        lesson_text=(
            f"Sell decision on {ticker} resulted in a -5.00% return over the "
            "10-trading-day window, classified as NEGATIVE. The position was closed "
            "at a clear loss after the thesis failed to hold; prior momentum-based "
            "conviction proved overconfident against a swift reversal."
        ),
        created_at="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Dispatch / comparison logic (covered by the structural unit test)
# ---------------------------------------------------------------------------


def _select_layers(layer_arg: str) -> list[str]:
    """Expand --layer's CLI value into the ordered list of adapter keys to run."""
    if layer_arg == "all":
        return ["research", "auditor", "risk"]
    return [layer_arg]


def _compute_divergence(without_value: Any, with_value: Any) -> bool:
    """True if the without-reflection and with-reflection extracted values differ."""
    return without_value != with_value


def _run_layer(
    layer: str,
    ticker: str,
    trade_date: str,
    negative_reflection: ReflectionRecord,
) -> tuple[str, Any, Any, bool]:
    """Run one layer's real node twice (without reflection, then with) and compare."""
    adapter = ADAPTERS[layer]
    state = adapter.build_state(ticker, trade_date)
    node = adapter.build_node()

    with patch(adapter.patch_target, return_value=None):
        without_result = node(dict(state))
    without_value = adapter.extract_value(without_result)

    with patch(adapter.patch_target, return_value=negative_reflection):
        with_result = node(dict(state))
    with_value = adapter.extract_value(with_result)

    diverged = _compute_divergence(without_value, with_value)
    return layer, without_value, with_value, diverged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer",
        choices=["research", "auditor", "risk", "all"],
        default="all",
        help="Which layer(s) to A/B test (default: all)",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker to analyze (default: AAPL)",
    )
    parser.add_argument(
        "--trade-date",
        default="2026-07-29",
        help="Trade date, YYYY-MM-DD (default: 2026-07-29)",
    )
    args = parser.parse_args()

    layers = _select_layers(args.layer)
    negative_reflection = _build_negative_reflection(args.ticker, args.trade_date)

    print("=" * 70)
    print("D-04 Reflection A/B Divergence Spike")
    print("=" * 70)
    print(f"Ticker: {args.ticker}")
    print(f"Trade Date: {args.trade_date}")
    print(f"Layers: {layers}")
    print("=" * 70)
    print()

    results: list[tuple[str, Any, Any, bool]] = []
    for layer in layers:
        print(f"Running layer: {layer}")
        result = _run_layer(layer, args.ticker, args.trade_date, negative_reflection)
        results.append(result)
        _, without_value, with_value, diverged = result
        print(f"  without_reflection={without_value}")
        print(f"  with_reflection={with_value}")
        print(f"  diverged={diverged}")
        print()

    print("=" * 70)
    print("Divergence Summary Table (D-04 Criterion)")
    print("=" * 70)
    print(f"{'Layer':<12} {'Without':<30} {'With':<30} {'Diverged':<10}")
    print("-" * 70)
    for layer, without_value, with_value, diverged in results:
        print(f"{layer:<12} {str(without_value):<30} {str(with_value):<30} {str(diverged):<10}")
    print()

    all_diverged = all(diverged for _, _, _, diverged in results)

    print("=" * 70)
    if all_diverged:
        print("RESULT: ALL REQUESTED LAYERS DIVERGED (D-04: PASS)")
        print("Interpretation: The injected negative reflection observably changed")
        print("what each requested layer actually decided.")
    else:
        non_diverging = [layer for layer, _, _, diverged in results if not diverged]
        print(f"RESULT: NOT ALL LAYERS DIVERGED. Non-diverging layers: {non_diverging}")
        print("Interpretation: recorded honestly — this is a valid, informative")
        print("outcome, not something to re-run or tune away.")
    print("=" * 70)

    return 0 if all_diverged else 1


if __name__ == "__main__":
    sys.exit(main())
