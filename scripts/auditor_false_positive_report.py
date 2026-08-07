"""CLI for the Auditor false-positive/false-negative analysis report (Phase 6, D-07).

This script generates a report showing what percentage of the Auditor's Buy verdicts
ended with negative returns (false positives) and what percentage of Sell verdicts
ended with positive returns (missed opportunities).

The report explicitly flags when sample volume is too low to be statistically
meaningful, per RESEARCH.md Assumption A6.

Usage:
    python scripts/auditor_false_positive_report.py
    python scripts/auditor_false_positive_report.py --min-samples 10
    python scripts/auditor_false_positive_report.py --min-samples 5
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.jobs.auditor_false_positive_analyzer import aggregate_auditor_reflections


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] when None)

    Returns:
        Exit code (always 0, as this is a report tool, not a pass/fail gate)
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum Buy and Sell sample count for statistical reliability (default: 10)",
    )
    args = parser.parse_args(argv)

    # Call the analyzer
    result = aggregate_auditor_reflections(min_samples=args.min_samples)

    # Print the report with bordered sections
    border = "=" * 80
    print()
    print(border)
    print("Auditor False-Positive Analysis".center(80))
    print(border)
    print()

    # Buy verdict -> Negative outcome (false positives)
    buy_total = result["buy_total"]
    if buy_total > 0:
        buy_negative_count = result["buy_negative_count"]
        buy_negative_pct = result["buy_negative_pct"]
        print(f"BUY verdict -> NEGATIVE return: {buy_negative_count} / {buy_total} = {buy_negative_pct:,.1%}")
    else:
        print("BUY verdict -> NEGATIVE return: N/A (0 Buy verdicts recorded)")

    print()

    # Sell verdict -> Positive outcome (Auditor said Sell but stock went up)
    sell_total = result["sell_total"]
    if sell_total > 0:
        sell_positive_count = result["sell_positive_count"]
        sell_positive_pct = result["sell_positive_pct"]
        print(f"SELL verdict -> POSITIVE return: {sell_positive_count} / {sell_total} = {sell_positive_pct:,.1%}")
    else:
        print("SELL verdict -> POSITIVE return: N/A (0 Sell verdicts recorded)")

    print()

    # Hold count
    hold_total = result["hold_total"]
    print(f"HOLD verdicts: {hold_total}")

    print()
    print(border)

    # Top false positives section
    print("Top False Positives (Buy -> Negative)".center(80))
    print(border)
    print()

    top_false_positives = result["top_false_positives"]
    if top_false_positives:
        for entry in top_false_positives:
            ticker = entry["ticker"]
            decision_date = entry["decision_date"]
            realized_return = entry["realized_return"]
            print(f"  {ticker} {decision_date}: Buy -> {realized_return:+.1%}")
    else:
        print("  (none)")

    print()
    print(border)
    print()

    # Explicit insufficient-data banner (RESEARCH.md Assumption A6)
    if not result["sufficient_samples"]:
        print("!" * 80)
        print(
            f"INSUFFICIENT DATA: fewer than {args.min_samples} samples for Buy and/or Sell "
            "— these percentages are not yet statistically meaningful "
            "(revisit once Phase 7 paper-trading volume accumulates)."
        )
        print("!" * 80)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
