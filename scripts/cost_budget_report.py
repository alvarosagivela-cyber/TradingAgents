"""CLI entry point for cost tracking and budget alert reporting.

This script provides a command-line interface to the cost aggregator,
allowing manual or scheduled invocation to review LLM costs and budget status.

Usage:
    python scripts/cost_budget_report.py

The script prints a readable report showing:
- Total call count and accumulated cost
- Per-layer and per-model cost breakdown
- Projected annual spend
- Non-blocking budget alert when spending approaches the annual target (D-05/D-06)
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.jobs.cost_aggregator import summarize_costs


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] when None)

    Returns:
        Exit code (always 0 — this is a non-blocking report, D-05)
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No required arguments for basic report
    parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Aggregate costs
    result = summarize_costs()

    # Print header
    print()
    print("=" * 80)
    print("LLM Cost & Budget Status".center(80))
    print("=" * 80)
    print()

    # Print summary metrics
    print(f"Total Calls:           {result['call_count']}")
    print(f"Total Cost (USD):      ${result['total_cost_usd']:.2f}")
    print()

    # Print per-layer breakdown
    if result["by_layer"]:
        print("Cost by Layer:")
        print("-" * 80)
        for layer in sorted(result["by_layer"].keys(), key=lambda k: result["by_layer"][k], reverse=True):
            cost = result["by_layer"][layer]
            print(f"  {layer:<30} ${cost:>10.2f}")
        print()

    # Print per-model breakdown
    if result["by_model"]:
        print("Cost by Model:")
        print("-" * 80)
        for model in sorted(result["by_model"].keys(), key=lambda k: result["by_model"][k], reverse=True):
            cost = result["by_model"][model]
            print(f"  {model:<30} ${cost:>10.2f}")
        print()

    # Print budget status
    print("Budget Status:")
    print("-" * 80)
    print(f"  Projected Annual Spend: ${result['projected_annual_usd']:>10.2f}")
    print(f"  Annual Target:          ${result['annual_budget_target_usd']:>10.2f}")
    ratio_pct = (result['projected_annual_usd'] / result['annual_budget_target_usd']) * 100 if result['annual_budget_target_usd'] > 0 else 0
    print(f"  Current Usage:          {ratio_pct:>10.1f}%")
    print()

    # Print budget alert if over threshold
    if result["over_threshold"]:
        print("!" * 80)
        print("BUDGET ALERT: projected annual spend has reached "
              f"{result['budget_alert_threshold_pct']:.0%} of the "
              f"${result['annual_budget_target_usd']:.2f}/year target (D-05/D-06) — "
              "non-blocking, review before Phase 7.".center(80))
        print("!" * 80)
        print()

    print("=" * 80)
    print()

    # Always return 0 (non-blocking report, D-05)
    return 0


if __name__ == "__main__":
    sys.exit(main())
