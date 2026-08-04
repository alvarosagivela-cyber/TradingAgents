"""CLI entry point for processing realized outcomes and generating reflections.

This script provides a command-line interface to the realized outcome processor job,
allowing manual or scheduled invocation to compute reflections from existing decision logs.

Usage:
    python scripts/process_realized_outcomes.py
    python scripts/process_realized_outcomes.py --dry-run
    python scripts/process_realized_outcomes.py --since-date 2026-08-01
    python scripts/process_realized_outcomes.py --dry-run --since-date 2026-08-01

The script prints a readable summary table showing how many reflections were processed
for each layer (research, auditor, risk).
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.jobs.realized_outcome_processor import process_pending_outcomes


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 on success)
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute outcomes and log results, but do not write reflections to disk",
    )
    parser.add_argument(
        "--since-date",
        type=str,
        default=None,
        help="Only process decisions from this date onward (YYYY-MM-DD format)",
    )
    args = parser.parse_args()

    # Call the job
    result = process_pending_outcomes(dry_run=args.dry_run, since_date=args.since_date)

    # Print results as a readable table
    print()
    print("=" * 70)
    print("Realized Outcome Processing Results")
    print("=" * 70)
    print(f"research: {result['research']}")
    print(f"auditor:  {result['auditor']}")
    print(f"risk:     {result['risk']}")
    print(f"TOTAL:    {sum(result.values())}")
    print("=" * 70)
    print()

    if args.dry_run:
        print("(dry-run mode: no reflections were written to disk)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
