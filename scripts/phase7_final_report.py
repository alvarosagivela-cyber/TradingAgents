"""CLI script to generate the Phase 7 final report (D-05).

This script is run once, manually, at the end of the 4-6 week paper trading window.
It synthesizes the window's Sharpe ratio, Auditor refutation count, and Phase 6's
false-positive/cost tools into one report with an explicit Core Value conclusion.

Usage:
    python scripts/phase7_final_report.py --start-date 2026-08-11 --end-date 2026-09-22
    python scripts/phase7_final_report.py --start-date 2026-08-11 --end-date 2026-09-22 --skip-close-positions

Exit Codes:
    0: Report generated and written successfully
    1: Missing required arguments or invalid date format
    2: API error or other runtime error

Design:
- D-05: Synthesizes Phase 6's existing tools (never reimplements them)
- D-09: Closes all open positions before computing Sharpe (first step)
- VALID-01/VALID-02: Reports Auditor refutation count and Sharpe ratio explicitly
- EXEC-01: create_alpaca_client(paper=True) is the ONLY client-construction call site
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tradingagents.dataflows.config import get_config
from tradingagents.jobs.auditor_false_positive_analyzer import (
    aggregate_auditor_reflections,
)
from tradingagents.jobs.auditor_refutation_aggregator import count_auditor_refutations
from tradingagents.jobs.cost_aggregator import summarize_costs
from tradingagents.trading.alpaca_client import create_alpaca_client
from tradingagents.trading.position_cleanup import close_all_open_positions
from tradingagents.trading.sharpe_calculator import compute_window_sharpe


def _validate_date(value: str) -> str:
    r"""Validate date format: ^\d{4}-\d{2}-\d{2}$

    Expects YYYY-MM-DD format.

    Args:
        value: The date string to validate

    Returns:
        The validated date string

    Raises:
        argparse.ArgumentTypeError: If the date does not match the pattern
    """
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {value!r}. "
            "Date must be in YYYY-MM-DD format (e.g., '2026-08-11')"
        )
    return value


def _build_core_value_conclusion(
    refutation_result: dict[str, Any], sharpe_result: dict[str, Any]
) -> str:
    """Build the Core Value conclusion sentence (VALID-01/VALID-02).

    Combines two lines:
    1. CORE VALUE ANSWER based on whether Auditor refuted any theses
    2. SHARPE status based on the computed ratio and overfitting flag

    Args:
        refutation_result: Output from count_auditor_refutations()
        sharpe_result: Output from compute_window_sharpe()

    Returns:
        Two-line string with Core Value conclusion and Sharpe status
    """
    # Line 1: Core Value conclusion
    if refutation_result["at_least_one_refutation"]:
        line1 = (
            "CORE VALUE ANSWER: YES"
            f" — the Auditor refuted {refutation_result['refutation_count']} of "
            f"{refutation_result['total_cycles']} Research theses during the window "
            "(comparison_result=mismatch), providing direct evidence the independent "
            "Research/Auditor separation catches bad theses under real scrutiny."
        )
    else:
        line1 = (
            "CORE VALUE ANSWER: NO"
            f" — across {refutation_result['total_cycles']} audit cycles in the window, "
            "the Auditor never refuted a Research thesis (comparison_result=mismatch count "
            "is 0). Per the project's own Core Value definition, this means the project's "
            "purpose was not demonstrated in this window, independent of paper-trading P&L."
        )

    # Line 2: Sharpe status
    if sharpe_result.get("insufficient_data"):
        line2 = "SHARPE: insufficient equity-curve data to compute a ratio for this window."
    elif sharpe_result["overfitting_flag"]:
        sharpe_ratio = sharpe_result.get("sharpe_ratio", "N/A")
        caveat = sharpe_result.get("caveat", "")
        line2 = (
            f"SHARPE: {sharpe_ratio} ABOVE the reference range (0.5, 1.0) — "
            f"this OVERFITTING signal indicates the window may reflect random market "
            f"noise rather than genuine alpha. {caveat}"
        )
    else:
        sharpe_ratio = sharpe_result.get("sharpe_ratio", "N/A")
        ref_range = sharpe_result.get("reference_range", (0.5, 1.0))
        if ref_range[0] <= sharpe_ratio <= ref_range[1]:
            line2 = (
                f"SHARPE: {sharpe_ratio} is within the reference range "
                f"{ref_range}, indicating reasonable risk-adjusted returns without "
                f"obvious overfitting."
            )
        else:
            line2 = (
                f"SHARPE: {sharpe_ratio} is outside the reference range {ref_range}. "
                f"{sharpe_result.get('caveat', '')}"
            )

    return line1 + "\n" + line2


def main(argv: list[str] | None = None) -> int:
    """Generate the Phase 7 final report (D-05).

    Args:
        argv: Command-line arguments (for testing); if None, uses sys.argv[1:]

    Returns:
        0 on success, non-zero on error
    """
    parser = argparse.ArgumentParser(
        description="Generate Phase 7 final report: closes positions, computes Sharpe, "
        "counts Auditor refutations, and provides Core Value conclusion."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        type=_validate_date,
        help="Window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=_validate_date,
        help="Window end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--skip-close-positions",
        action="store_true",
        help="Skip closing positions (for dry-run/test use only)",
    )

    args = parser.parse_args(argv)

    # D-09: Create Alpaca client (paper=True — EXEC-01 guard, only call site for this)
    try:
        client = create_alpaca_client(paper=True)
    except ValueError as e:
        print(f"Error: Failed to create Alpaca client: {e}", file=sys.stderr)
        return 2

    # D-09: Close all open positions before Sharpe computation (must be first step)
    if not args.skip_close_positions:
        close_result = close_all_open_positions(client)
        close_status_line = (
            f"Closed {close_result['positions_closed']} position(s). "
            f"Status: {close_result['status']}"
        )
        if close_result.get("error"):
            close_status_line += f" (Error: {close_result['error']})"
    else:
        close_result = {"status": "skipped", "positions_closed": 0, "error": None}
        close_status_line = "Position closing skipped (--skip-close-positions)"

    # Parse dates for Sharpe computation
    try:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"Error: Invalid date format: {e}", file=sys.stderr)
        return 2

    # Compute Sharpe ratio (D-06/D-07)
    try:
        sharpe_result = compute_window_sharpe(client, start_dt, end_dt)
    except Exception as e:
        print(f"Error: Failed to compute Sharpe ratio: {e}", file=sys.stderr)
        return 2

    # Count Auditor refutations (VALID-01)
    try:
        refutation_result = count_auditor_refutations(args.start_date, args.end_date)
    except Exception as e:
        print(f"Error: Failed to count refutations: {e}", file=sys.stderr)
        return 2

    # Reuse Phase 6's false-positive analyzer (D-05, never reimplemented)
    try:
        false_positive_result = aggregate_auditor_reflections()
    except Exception as e:
        print(f"Warning: Failed to aggregate false positives: {e}", file=sys.stderr)
        false_positive_result = {}

    # Reuse Phase 6's cost aggregator (D-05/D-11, never reimplemented)
    try:
        cost_result = summarize_costs()
    except Exception as e:
        print(f"Warning: Failed to summarize costs: {e}", file=sys.stderr)
        cost_result = {}

    # Build the report sections
    report_sections = []

    # Section 1: Window dates
    report_sections.append(
        f"# Phase 7 Final Report: {args.start_date} to {args.end_date}"
    )
    report_sections.append(f"Generated: {datetime.now().isoformat()}")

    # Section 2: Position close status (D-09)
    report_sections.append("## Position Closure Status (D-09)")
    report_sections.append(close_status_line)

    # Section 3: Sharpe ratio (D-06/D-07)
    report_sections.append("## Sharpe Ratio Analysis (D-06/D-07)")
    report_sections.append(f"- Sharpe Ratio: {sharpe_result.get('sharpe_ratio', 'N/A')}")
    report_sections.append(f"- Reference Range: {sharpe_result.get('reference_range', 'N/A')}")
    report_sections.append(f"- Overfitting Flag: {sharpe_result.get('overfitting_flag', False)}")
    report_sections.append(f"- Insufficient Data: {sharpe_result.get('insufficient_data', False)}")
    report_sections.append(f"- Caveat: {sharpe_result.get('caveat', 'None')}")

    # Section 4: Auditor refutations (VALID-01)
    report_sections.append("## Auditor Refutation Record (VALID-01)")
    report_sections.append(f"- Total Audit Cycles: {refutation_result['total_cycles']}")
    report_sections.append(f"- Refutation Count: {refutation_result['refutation_count']}")
    report_sections.append(f"- At Least One Refutation: {refutation_result['at_least_one_refutation']}")
    if refutation_result["refutations"]:
        report_sections.append("- Refuted Theses:")
        for refutation in refutation_result["refutations"]:
            report_sections.append(
                f"  - {refutation['ticker']} ({refutation['trade_date']}): "
                f"{refutation['auditor_reasoning']}"
            )
    else:
        report_sections.append("- No refuted theses in this window")

    # Section 5: Phase 6 false-positive summary
    report_sections.append("## Auditor False Positive Analysis (Phase 6)")
    if false_positive_result:
        for key, value in false_positive_result.items():
            report_sections.append(f"- {key}: {value}")
    else:
        report_sections.append("- No data available")

    # Section 6: Cost summary (Phase 6)
    report_sections.append("## Cost & Budget Summary (Phase 6)")
    if cost_result:
        report_sections.append(f"- Total Cost (USD): {cost_result.get('total_cost_usd', 'N/A')}")
        report_sections.append(
            f"- Projected Annual (USD): {cost_result.get('projected_annual_usd', 'N/A')}"
        )
        report_sections.append(
            f"- Budget Target (USD): {cost_result.get('annual_budget_target_usd', 'N/A')}"
        )
        report_sections.append(
            f"- Over Threshold: {cost_result.get('over_threshold', False)}"
        )
    else:
        report_sections.append("- No cost data available")

    # Section 7: Core Value conclusion (VALID-01/VALID-02)
    report_sections.append("## Core Value Conclusion")
    core_value_text = _build_core_value_conclusion(refutation_result, sharpe_result)
    report_sections.append(core_value_text)

    # Assemble final report
    report_text = "\n\n".join(report_sections)

    # Print to stdout
    print(report_text)

    # Write to results_dir
    try:
        config = get_config()
        results_dir = Path(config["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        report_file = results_dir / f"phase7_final_report_{args.end_date}.md"
        report_file.write_text(report_text)
        print(f"\nReport saved to: {report_file}")
    except Exception as e:
        print(f"Error: Failed to write report to results_dir: {e}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
