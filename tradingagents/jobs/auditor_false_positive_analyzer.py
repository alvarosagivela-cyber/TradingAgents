"""Analyzer for Auditor false-positive and false-negative rates (Phase 6, D-07).

This module aggregates persisted auditor reflections to compute what percentage
of the Auditor's Buy verdicts ended with negative returns (false positives) and
what percentage of its Sell verdicts ended with positive returns (missed opportunities).

Design:
- Reads auditor_reflections.jsonl (already persisted by Phase 5)
- Gracefully handles missing/empty files and malformed JSON
- Returns a structured report with explicit "insufficient data" flag
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tradingagents.agents.reflectors.reflection_schema import ReflectionRecord
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)


def _read_jsonl_records(path: Path) -> list[ReflectionRecord]:
    """Read and parse reflection records from a JSONL file.

    Gracefully handles:
    - Missing files → returns []
    - Empty files → returns []
    - Malformed JSON lines → logs warning, skips line
    - Missing fields in record → logs warning, skips record

    Args:
        path: Path to the JSONL file

    Returns:
        List of successfully parsed ReflectionRecord instances
    """
    if not path.exists():
        logger.debug(f"Reflections file not found: {path}")
        return []

    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Malformed JSON at {path}:{line_num}: {e}. Skipping line."
                    )
                    continue

                try:
                    record = ReflectionRecord.from_json_dict(parsed)
                    records.append(record)
                except (KeyError, TypeError) as e:
                    logger.warning(
                        f"Invalid record at {path}:{line_num}: {e}. Skipping record."
                    )
                    continue

    except Exception as e:
        logger.error(f"Error reading reflections file {path}: {e}")
        return []

    return records


def aggregate_auditor_reflections(min_samples: int = 10) -> dict[str, Any]:
    """Aggregate auditor reflections to compute false-positive/negative rates.

    Reads all auditor reflection records and computes:
    - Percentage of Buy verdicts that realized negative returns
    - Percentage of Sell verdicts that realized positive returns
    - Flag indicating whether sample size is sufficient for statistical reliability

    Args:
        min_samples: Minimum number of Buy and Sell records required to mark
                     sufficient_samples as True (default: 10)

    Returns:
        A dict with keys:
        - buy_total: Count of Buy verdicts
        - buy_negative_count: Count of Buy verdicts with negative outcome
        - buy_negative_pct: Percentage (0.0-1.0) or None if no Buy records
        - sell_total: Count of Sell verdicts
        - sell_positive_count: Count of Sell verdicts with positive outcome
        - sell_positive_pct: Percentage (0.0-1.0) or None if no Sell records
        - hold_total: Count of Hold verdicts
        - sufficient_samples: True only if both buy_total >= min_samples AND sell_total >= min_samples
        - min_samples: The threshold used
        - top_false_positives: List of up to 5 dicts with worst Buy/negative records
                               (sorted by return ascending: worst loss first)
                               Each dict has: {"ticker", "decision_date", "realized_return"}
    """
    # Read reflections from config
    path = Path(get_config()["auditor_reflections_log_path"])
    records = _read_jsonl_records(path)

    # Filter by verdict
    buy_records = [r for r in records if r.decision_verdict == "Buy"]
    sell_records = [r for r in records if r.decision_verdict == "Sell"]
    hold_total = sum(1 for r in records if r.decision_verdict == "Hold")

    # Compute false positives (Buy -> negative)
    buy_negative = [r for r in buy_records if r.classification == "negative"]
    buy_negative_pct = (
        len(buy_negative) / len(buy_records) if buy_records else None
    )

    # Compute false negatives / missed opportunities (Sell -> positive)
    sell_positive = [r for r in sell_records if r.classification == "positive"]
    sell_positive_pct = (
        len(sell_positive) / len(sell_records) if sell_records else None
    )

    # Determine if sample size is sufficient
    sufficient_samples = len(buy_records) >= min_samples and len(sell_records) >= min_samples

    # Top false positives: worst Buy/negative records (up to 5, sorted by return ascending)
    buy_negative_sorted = sorted(buy_negative, key=lambda r: r.realized_return)
    top_false_positives = [
        {
            "ticker": r.ticker,
            "decision_date": r.decision_date,
            "realized_return": r.realized_return,
        }
        for r in buy_negative_sorted[:5]
    ]

    return {
        "buy_total": len(buy_records),
        "buy_negative_count": len(buy_negative),
        "buy_negative_pct": buy_negative_pct,
        "sell_total": len(sell_records),
        "sell_positive_count": len(sell_positive),
        "sell_positive_pct": sell_positive_pct,
        "hold_total": hold_total,
        "sufficient_samples": sufficient_samples,
        "min_samples": min_samples,
        "top_false_positives": top_false_positives,
    }
