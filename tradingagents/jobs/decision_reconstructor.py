"""Decision reconstruction by joining three per-layer JSONL audit logs.

This job reconstructs the complete Research -> Auditor -> Risk decision trail
for a given ticker and trade-date by joining the three existing separate decision
logs (research_thesis_log, auditor_log, risk_log). No new LLM invocation or
prompt capture is added — this is a read-only join over already-persisted data
from Phases 2-4 (D-01/D-02).

Design:
- Given ticker + trade_date, scans all three logs for matching records
- Gracefully degrades on missing/malformed files (returns None for that layer)
- Mirrors reflection_reader's most-recent-match-wins semantics when duplicates exist
- Never raises; always returns a dict with nullable layer keys
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)


def _find_latest_match(
    log_path: Path, ticker: str, trade_date: str
) -> dict[str, Any] | None:
    """Find the most recent record matching ticker+trade_date in a JSONL file.

    Scans the entire file and returns the LAST matching record found (file order).
    Mirrors reflection_reader.read_reflection_for_ticker's semantics.

    Gracefully degradation:
    - Missing file → return None
    - Malformed JSON line → log warning, skip, continue
    - No matching record → return None
    - I/O error → log error, return None

    Args:
        log_path: Path to the JSONL log file
        ticker: Stock ticker to match
        trade_date: Trade date (YYYY-MM-DD) to match

    Returns:
        The parsed JSON dict of the matching record, or None if not found
    """
    if not log_path.exists():
        return None

    most_recent: dict[str, Any] | None = None

    try:
        with open(log_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError(f"expected a JSON object, got {type(record).__name__}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse line {line_num} in {log_path.name}: {e}"
                    )
                    continue

                # Check if this record matches
                if (
                    record.get("ticker") == ticker
                    and record.get("trade_date") == trade_date
                ):
                    most_recent = record  # Keep updating, last match wins

        return most_recent

    except OSError as e:
        logger.error(f"Error reading {log_path}: {e}")
        return None


def reconstruct_decision(ticker: str, trade_date: str) -> dict[str, Any]:
    """Reconstruct the complete decision trail by joining three per-layer logs.

    Given a ticker and trade-date, joins the research, auditor, and risk decision
    logs into a single readable dict. Returns None for any layer's key if that
    layer's log is missing/malformed or has no matching record.

    Args:
        ticker: Stock ticker (e.g. "AAPL")
        trade_date: Trade date in YYYY-MM-DD format (e.g. "2026-08-01")

    Returns:
        Dict with structure:
        {
            "ticker": ticker,
            "trade_date": trade_date,
            "research": {...full_record...} or None,
            "auditor": {...full_record...} or None,
            "risk": {...full_record...} or None,
        }
    """
    config = get_config()

    # Resolve the three log paths from config
    research_log_path = Path(config["research_thesis_log_path"])
    auditor_log_path = Path(config["auditor_log_path"])
    risk_log_path = Path(config["risk_log_path"])

    # Find matching records in each layer
    research_record = _find_latest_match(research_log_path, ticker, trade_date)
    auditor_record = _find_latest_match(auditor_log_path, ticker, trade_date)
    risk_record = _find_latest_match(risk_log_path, ticker, trade_date)

    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "research": research_record,
        "auditor": auditor_record,
        "risk": risk_record,
    }
