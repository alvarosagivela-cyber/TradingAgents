"""D-02 temperature divergence spike: sample momentum LLM node at multiple temperatures.

This script tests whether the momentum LLM node generates distinct verdicts
(Buy/Hold/Sell) when sampled at different temperatures, or if it always
converges to the same conclusion regardless of temperature variation.

Success criterion (D-02): More than 1 distinct verdict across all samples.
Failure: All samples converge to the same verdict despite temperature variation.

Usage:
    ANTHROPIC_API_KEY=... python scripts/momentum_temperature_spike.py
    ANTHROPIC_API_KEY=... python scripts/momentum_temperature_spike.py --ticker XOM --temperatures 0.3,0.9,1.5 --samples-per-temp 3

The script does NOT call propagate(), keeping surface and cost tight. It exercises only:
1. Momentum compute node (once per ticker, deterministic)
2. Momentum LLM node (once per temperature per sample)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import Any

from tradingagents.agents import (
    create_momentum_compute_node,
    create_momentum_llm_node,
)
from tradingagents.llm_clients import create_llm_client


def _build_minimal_state(ticker: str, trade_date: str) -> dict[str, Any]:
    """Build a minimal AgentState sufficient for momentum nodes.

    The compute node needs: company_of_interest, trade_date
    The LLM node needs: all the compute node outputs + market_report
    """
    return {
        "company_of_interest": ticker,
        "trade_date": trade_date,
        "market_report": (
            f"Market context for {ticker}: typical trading activity. "
            "No specific catalysts mentioned in this spike run."
        ),
    }


def _run_momentum_compute(ticker: str, trade_date: str) -> dict[str, Any]:
    """Run compute node once (deterministic result, reused across all samples)."""
    compute_node = create_momentum_compute_node()
    state = _build_minimal_state(ticker, trade_date)

    result = compute_node(state)

    print(f"[Compute] ticker={ticker}, trade_date={trade_date}")
    print(f"  retorno_12_1={result.get('retorno_12_1', 'N/A')}")
    print(f"  momentum_z_score={result.get('momentum_z_score', 'N/A')}")
    print(f"  momentum_valid={result.get('momentum_valid', 'N/A')}")
    print(f"  confidence_level={result.get('confidence_level', 'N/A')}")
    print()

    return result


def _run_momentum_llm_sample(
    state: dict[str, Any],
    llm: Any,
    temperature: float,
    sample_idx: int,
) -> str | None:
    """Run LLM node once at given temperature, return verdict or None."""
    llm_node = create_momentum_llm_node(llm)

    result = llm_node(state)

    verdict = result.get("thesis_verdict", "").strip()

    return verdict if verdict else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker to analyze (default: AAPL)",
    )
    parser.add_argument(
        "--trade-date",
        default=str(date.today()),
        help="Trade date (default: today)",
    )
    parser.add_argument(
        "--temperatures",
        default="0.3,0.7,1.0",
        help="Comma-separated temperatures to sample (default: 0.3,0.7,1.0). Range: 0..1",
    )
    parser.add_argument(
        "--samples-per-temp",
        type=int,
        default=3,
        help="Number of samples per temperature (default: 3)",
    )
    args = parser.parse_args()

    ticker = args.ticker
    trade_date = args.trade_date
    temperatures = [float(t.strip()) for t in args.temperatures.split(",")]
    samples_per_temp = args.samples_per_temp

    print("=" * 70)
    print("D-02 Temperature Divergence Spike")
    print("=" * 70)
    print(f"Ticker: {ticker}")
    print(f"Trade Date: {trade_date}")
    print(f"Temperatures: {temperatures}")
    print(f"Samples per temperature: {samples_per_temp}")
    print("=" * 70)
    print()

    # Run compute once (deterministic)
    compute_result = _run_momentum_compute(ticker, trade_date)

    # Build base state with compute results
    base_state = _build_minimal_state(ticker, trade_date)
    base_state.update(compute_result)

    # Collect all verdicts: (temperature, sample_idx, verdict)
    verdicts_table: list[tuple[float, int, str | None]] = []

    print("=" * 70)
    print("Sampling LLM at different temperatures")
    print("=" * 70)
    print()

    for temp in temperatures:
        print(f"Temperature: {temp}")

        # Create LLM client with this temperature
        llm_client = create_llm_client(
            provider="anthropic",
            model="claude-haiku-4-5",
            temperature=temp,
        )
        llm = llm_client.get_llm()

        for sample_idx in range(samples_per_temp):
            # Run LLM node with fresh sample
            verdict = _run_momentum_llm_sample(base_state, llm, temp, sample_idx)
            verdicts_table.append((temp, sample_idx, verdict))

            print(f"  Sample {sample_idx}: verdict={verdict}")

        print()

    # Print verdict table
    print("=" * 70)
    print("Verdict Table")
    print("=" * 70)
    print(f"{'Temperature':<15} {'Sample':<10} {'Verdict':<20}")
    print("-" * 70)
    for temp, sample_idx, verdict in verdicts_table:
        print(f"{temp:<15.1f} {sample_idx:<10} {str(verdict):<20}")
    print()

    # Compute divergence
    all_verdicts = {v for _, _, v in verdicts_table if v is not None}
    divergence_detected = len(all_verdicts) > 1

    print("=" * 70)
    print("Divergence Analysis (D-02 Criterion)")
    print("=" * 70)
    print(f"Distinct verdicts found: {sorted(all_verdicts)}")
    print(f"Number of distinct verdicts: {len(all_verdicts)}")
    print()

    if divergence_detected:
        print("RESULT: DIVERGENCE DETECTED")
        print("Interpretation: Temperature alone produces distinct conclusions.")
        print("This suggests sufficient diversity for 3-agent escalation (D-02 spike: PASS)")
    else:
        print("RESULT: NO DIVERGENCE")
        print("Interpretation: Temperature variation does not change verdict.")
        print("This suggests temperature alone is insufficient for diversity.")
        print("Fallback to Bull/Bear + Research Manager pattern (D-03) may be needed.")

    print()
    print("=" * 70)
    print("Exit code: 0 (data-gathering run; no hard pass/fail)")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
