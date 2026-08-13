"""Count Auditor refutations across a validation window (VALID-01, Phase 7 Task 1).

This module scans the existing Phase 3 auditor_log_path JSONL file and reports
whether the Auditor refuted (mismatch verdict against) a Research thesis at least
once during the window. A refutation is recorded when comparison_result=='mismatch'.

Design:
- Read-only scan over Phase 3's unconditionally-persisted audit trail (D-06)
- No new LLM invocation, no new state written
- Mirrors decision_reconstructor.py's established read-only-join pattern
- Gracefully degrades on missing/malformed files (never raises)

This directly satisfies the ROADMAP's Phase 7 success criterion #2:
"At the end of the window, there is a documented record of whether the Auditor
marked at least one thesis as bad under real scrutiny."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)


def count_auditor_refutations(start_date: str, end_date: str) -> dict[str, Any]:
    """Count Auditor refutations (mismatch verdicts) in the window (VALID-01).

    Scans the auditor_log_path JSONL file for records with comparison_result=='mismatch'
    within the [start_date, end_date] window (lexicographic ISO-date comparison).
    Returns a dict with total_cycles, refutation_count, at_least_one_refutation, and
    a list of refuted (ticker, trade_date, auditor_reasoning) entries.

    Graceful degradation:
    - Missing file → return zero-result dict immediately (no raise)
    - Malformed JSON line → log warning, skip, continue (no raise)
    - I/O error → log error, return zero-result dict (no raise)

    Args:
        start_date: Window start (YYYY-MM-DD, lexicographic comparison)
        end_date: Window end (YYYY-MM-DD, lexicographic comparison)

    Returns:
        Dict with keys:
        {
            "total_cycles": int (all records in window),
            "refutation_count": int (records with comparison_result=='mismatch'),
            "at_least_one_refutation": bool,
            "refutations": list[dict] with keys ["ticker", "trade_date", "auditor_reasoning"],
            "start_date": str,
            "end_date": str,
        }
    """
    config = get_config()
    log_path = Path(config["auditor_log_path"])

    # Graceful degradation: missing file returns zero-result dict
    if not log_path.exists():
        return {
            "total_cycles": 0,
            "refutation_count": 0,
            "at_least_one_refutation": False,
            "refutations": [],
            "start_date": start_date,
            "end_date": end_date,
        }

    total_cycles = 0
    refutations: list[dict[str, Any]] = []

    try:
        with open(log_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue

                # Parse JSON, gracefully skip malformed lines
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError(
                            f"expected a JSON object, got {type(record).__name__}"
                        )
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse line {line_num} in {log_path.name}: {e}"
                    )
                    continue

                # Check if record is within date range
                trade_date = record.get("trade_date", "")
                if not (start_date <= trade_date <= end_date):
                    continue

                # Record is in the window
                total_cycles += 1

                # Check if this is a refutation (mismatch)
                if record.get("comparison_result") == "mismatch":
                    refutations.append(
                        {
                            "ticker": record.get("ticker"),
                            "trade_date": trade_date,
                            "auditor_reasoning": record.get("auditor_reasoning"),
                        }
                    )

    except OSError as e:
        logger.error(f"Error reading {log_path}: {e}")
        return {
            "total_cycles": 0,
            "refutation_count": 0,
            "at_least_one_refutation": False,
            "refutations": [],
            "start_date": start_date,
            "end_date": end_date,
        }

    return {
        "total_cycles": total_cycles,
        "refutation_count": len(refutations),
        "at_least_one_refutation": len(refutations) > 0,
        "refutations": refutations,
        "start_date": start_date,
        "end_date": end_date,
    }
