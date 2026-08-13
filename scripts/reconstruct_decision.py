"""CLI entry point for reconstructing complete decision trails (Phase 6, D-02).

Given a ticker and trade-date, reconstructs the complete Research -> Auditor -> Risk
decision trail by joining the three existing separate per-layer JSONL audit logs into
one legible, human-readable report.

Usage:
    python scripts/reconstruct_decision.py --ticker AAPL --trade-date 2026-08-01
    python scripts/reconstruct_decision.py --ticker BRK.B --trade-date 2026-08-01

The script prints a 4-section report (Research Verdict, Auditor Verdict, Risk Verdicts,
Execution Status) and exits with code 0 if at least one layer has a record, 1 if none
found (a genuine automated pass/fail signal per Nyquist principle).

Exit Codes:
    0: At least one layer matched and was reconstructed
    1: No records found for the given ticker/trade-date in any of the three logs
    2: Invalid CLI arguments (argparse validation failure)
"""

from __future__ import annotations

import argparse
import re
import sys

from tradingagents.jobs.decision_reconstructor import reconstruct_decision


def _validate_ticker(value: str) -> str:
    r"""Validate ticker format: ^[A-Z]{1,5}(\.[A-Z]{1,3})?$

    Allows 1-5 uppercase letters, optionally followed by a dot and 1-3 uppercase letters
    (e.g., "AAPL", "BRK.B").

    Args:
        value: The ticker string to validate

    Returns:
        The validated ticker string

    Raises:
        argparse.ArgumentTypeError: If the ticker does not match the pattern
    """
    if not re.match(r"^[A-Z]{1,5}(\.[A-Z]{1,3})?$", value):
        raise argparse.ArgumentTypeError(
            f"Invalid ticker format: {value!r}. "
            "Ticker must be 1-5 uppercase letters, optionally followed by a dot and 1-3 uppercase letters (e.g., 'AAPL', 'BRK.B')"
        )
    return value


def _validate_trade_date(value: str) -> str:
    r"""Validate trade-date format: ^\d{4}-\d{2}-\d{2}$

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
            "Date must be in YYYY-MM-DD format (e.g., '2026-08-01')"
        )
    return value


def _format_research_section(research_record: dict | None) -> str:
    """Format the Research Verdict section of the report."""
    if research_record is None:
        return "N/A - no verified thesis record found for this ticker/trade-date"

    lines = []
    lines.append(f"  Verdict: {research_record.get('verdict', 'N/A')}")
    lines.append(f"  Reasoning: {research_record.get('reasoning', 'N/A')}")
    lines.append(f"  Refutation Criterion: {research_record.get('refutation_criterion', 'N/A')}")
    lines.append("  Data Points:")
    lines.append(f"    retorno_12_1: {research_record.get('retorno_12_1', 'N/A')}")
    lines.append(f"    z_score: {research_record.get('z_score', 'N/A')}")
    lines.append(f"    confidence_level: {research_record.get('confidence_level', 'N/A')}")
    lines.append(f"  Verified: {research_record.get('verified_fields', [])}")
    lines.append(f"  Timestamp: {research_record.get('timestamp', 'N/A')}")
    return "\n".join(lines)


def _format_auditor_section(auditor_record: dict | None) -> str:
    """Format the Auditor Verdict section of the report."""
    if auditor_record is None:
        return "N/A - no audit record found for this ticker/trade-date"

    lines = []
    lines.append(f"  Verdict: {auditor_record.get('auditor_verdict', 'N/A')}")
    lines.append(f"  Reasoning: {auditor_record.get('auditor_reasoning', 'N/A')}")
    lines.append(f"  Refutation Criterion: {auditor_record.get('auditor_refutation_criterion', 'N/A')}")
    lines.append(f"  Data Points: {auditor_record.get('auditor_data_points', {})}")
    lines.append(f"  Verified: {auditor_record.get('verified', False)}")
    lines.append(f"  Comparison Result: {auditor_record.get('comparison_result', 'N/A')}")
    lines.append(f"  Timestamp: {auditor_record.get('timestamp', 'N/A')}")
    return "\n".join(lines)


def _format_risk_section(risk_record: dict | None) -> str:
    """Format the Risk Verdicts section of the report."""
    if risk_record is None:
        return "N/A - no Risk cycle record found for this ticker/trade-date"

    lines = []
    lines.append("  Conservative Perspective:")
    conservative = risk_record.get("conservative_verdict", {})
    lines.append(f"    Verdict: {conservative.get('verdict', 'N/A')}")
    lines.append(f"    Reasoning: {conservative.get('reasoning', 'N/A')}")
    lines.append(f"    Confidence: {conservative.get('confidence', 'N/A')}")

    lines.append("  Balanced Perspective:")
    balanced = risk_record.get("balanced_verdict", {})
    lines.append(f"    Verdict: {balanced.get('verdict', 'N/A')}")
    lines.append(f"    Reasoning: {balanced.get('reasoning', 'N/A')}")
    lines.append(f"    Confidence: {balanced.get('confidence', 'N/A')}")

    lines.append("  Aggressive Perspective:")
    aggressive = risk_record.get("aggressive_verdict", {})
    lines.append(f"    Verdict: {aggressive.get('verdict', 'N/A')}")
    lines.append(f"    Reasoning: {aggressive.get('reasoning', 'N/A')}")
    lines.append(f"    Confidence: {aggressive.get('confidence', 'N/A')}")

    lines.append(f"  Timestamp: {risk_record.get('timestamp', 'N/A')}")
    return "\n".join(lines)


def _format_execution_section(risk_record: dict | None) -> str:
    """Format the Execution Status section of the report."""
    if risk_record is None:
        return "UNKNOWN - no Risk cycle record found for this ticker/trade-date"

    final_veto = risk_record.get("final_veto")
    if final_veto is True:
        return "VETOED"
    elif final_veto is False:
        return "APPROVED"
    else:
        return "UNKNOWN - final_veto status unclear"


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Optional list of command-line arguments for direct testability.
              Defaults to sys.argv[1:] when None.

    Returns:
        Exit code (0 if at least one layer matched, 1 if nothing found)
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        type=_validate_ticker,
        required=True,
        help="Stock ticker (e.g., AAPL, BRK.B)",
    )
    parser.add_argument(
        "--trade-date",
        type=_validate_trade_date,
        required=True,
        help="Trade date in YYYY-MM-DD format",
    )
    args = parser.parse_args(argv)

    # Reconstruct the decision
    result = reconstruct_decision(args.ticker, args.trade_date)

    # Print the report
    print()
    print("=" * 70)
    print("Decision Reconstruction Report")
    print("=" * 70)
    print(f"Ticker: {result['ticker']}")
    print(f"Trade Date: {result['trade_date']}")
    print("=" * 70)
    print()

    print("RESEARCH VERDICT:")
    print(_format_research_section(result["research"]))
    print()

    print("AUDITOR VERDICT:")
    print(_format_auditor_section(result["auditor"]))
    print()

    print("RISK VERDICTS:")
    print(_format_risk_section(result["risk"]))
    print()

    print("EXECUTION STATUS:")
    print(_format_execution_section(result["risk"]))
    print()
    print("=" * 70)
    print()

    # Determine exit code: 0 if at least one layer matched, 1 if all are None
    has_any_record = any(
        [result["research"], result["auditor"], result["risk"]]
    )

    return 0 if has_any_record else 1


if __name__ == "__main__":
    sys.exit(main())
