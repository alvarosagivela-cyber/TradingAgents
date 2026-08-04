"""Batch job for processing realized outcomes and generating reflections.

This job closes the write side of the reflection loop: for each of the three
already-separated per-layer decision audit logs (research/auditor/risk), it finds
decisions old enough to have a realized outcome, computes that outcome via the
Plan 05-01 calculator, and appends an idempotent reflection record to that
layer's dedicated store.

Design:
- Each layer is processed independently with its own decision log and reflection store
- No cross-layer writes (research decisions go to research reflections only, etc.)
- Idempotent: re-running against an unchanged decision log never duplicates reflections
- Graceful degradation: decisions still inside their holding window (NoMarketDataError)
  are logged and skipped, to be retried on a later job run
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.agents.reflectors.reflection_schema import (
    ReflectionRecord,
    build_lesson_text,
    classify_realized_return,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.realized_return_calculator import (
    REALIZED_RETURN_WINDOW_DAYS,
    compute_realized_return,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError

logger = logging.getLogger(__name__)

# Explicit mappings from layer name to config key (not derived from string interpolation)
_DECISION_LOG_CONFIG_KEYS = {
    "research": "research_thesis_log_path",
    "auditor": "auditor_log_path",
    "risk": "risk_log_path",
}

_REFLECTION_LOG_CONFIG_KEYS = {
    "research": "research_reflections_log_path",
    "auditor": "auditor_reflections_log_path",
    "risk": "risk_reflections_log_path",
}


def _extract_research_verdict(record: dict) -> str | None:
    """Extract Research layer verdict from a decision record.

    Args:
        record: A JSON dict from research_thesis_log

    Returns:
        The verdict string (e.g. "Buy"), or None if not present
    """
    return record.get("verdict") or None


def _extract_auditor_verdict(record: dict) -> str | None:
    """Extract Auditor layer verdict from a decision record.

    Auditor records with auditor_verdict == "INCONCLUSIVE" return None
    (these are skipped, as the Auditor's compute phase failed).

    Args:
        record: A JSON dict from auditor_log

    Returns:
        The verdict string (e.g. "Sell"), or None if INCONCLUSIVE or not present
    """
    verdict = record.get("auditor_verdict")
    if verdict in (None, "", "INCONCLUSIVE"):
        return None
    return verdict


def _extract_risk_verdict(record: dict) -> str | None:
    """Extract Risk layer verdict from a decision record.

    Risk verdicts are derived from the boolean final_veto field:
    - final_veto: True → "VETO"
    - final_veto: False → "APPROVE"

    This always returns a value (never None), as Risk cycles always
    resolve to one of the two verdicts via the fail-safe aggregation (RISK-01).

    Args:
        record: A JSON dict from risk_log

    Returns:
        "VETO" or "APPROVE"
    """
    final_veto = record.get("final_veto", False)
    return "VETO" if final_veto else "APPROVE"


def _read_jsonl(path: Path) -> list[dict]:
    """Read and parse JSONL file, gracefully handling errors.

    Each line is expected to be valid JSON. Lines that fail to parse
    are logged with a warning and skipped (never crash the entire read).

    Args:
        path: Path to the JSONL file

    Returns:
        List of parsed JSON dicts; empty list if file does not exist
    """
    if not path.exists():
        return []

    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse line {line_num} in {path.name}: {e}"
                    )
                    continue
    except IOError as e:
        logger.error(f"Error reading {path}: {e}")
        return []

    return records


def _existing_reflection_keys(path: Path) -> set[tuple[str, str]]:
    """Extract existing (ticker, decision_date) pairs from a reflections store.

    Used for idempotency checking: if a (ticker, decision_date) pair already
    exists in the store, that decision is skipped on the next job run.

    Args:
        path: Path to the reflections JSONL file

    Returns:
        Set of (ticker, decision_date) tuples; empty set if file does not exist
    """
    if not path.exists():
        return set()

    keys = set()
    for record in _read_jsonl(path):
        ticker = record.get("ticker")
        decision_date = record.get("decision_date")
        if ticker and decision_date:
            keys.add((ticker, decision_date))

    return keys


def _append_reflection(reflection_path: Path, reflection: ReflectionRecord) -> None:
    """Append a reflection record to a JSONL file.

    Creates parent directories if needed. Appends one JSON line.

    Args:
        reflection_path: Path to the reflections JSONL file
        reflection: The ReflectionRecord to append
    """
    reflection_path.parent.mkdir(parents=True, exist_ok=True)

    with open(reflection_path, "a", encoding="utf-8") as f:
        line = json.dumps(reflection.to_json_dict()) + "\n"
        f.write(line)


def _process_layer(
    layer: str,
    verdict_extractor,
    dry_run: bool = False,
    since_date: str | None = None,
) -> int:
    """Process a single layer's decision log and generate reflections.

    For each decision in the layer's log:
    1. Skip if verdict is None (e.g., Auditor INCONCLUSIVE)
    2. Skip if (ticker, decision_date) already in the reflection store (idempotency)
    3. Call compute_realized_return
    4. On NoMarketDataError, log and skip (retry later)
    5. On success, build reflection and append to layer's store (unless dry_run)
    6. Return count of processed decisions

    Args:
        layer: One of "research", "auditor", "risk"
        verdict_extractor: Function to extract verdict from a record
        dry_run: If True, compute but do not write
        since_date: Optional YYYY-MM-DD filter (only process records >= this date)

    Returns:
        Number of reflections processed (written or would-have-been-written)
    """
    config = get_config()

    decision_log_path = Path(config[_DECISION_LOG_CONFIG_KEYS[layer]])
    reflection_log_path = Path(config[_REFLECTION_LOG_CONFIG_KEYS[layer]])

    # Read decision log and existing reflection keys
    decisions = _read_jsonl(decision_log_path)
    existing_keys = _existing_reflection_keys(reflection_log_path)

    processed_count = 0

    for record in decisions:
        ticker = record.get("ticker")
        decision_date = record.get("trade_date")
        verdict = verdict_extractor(record)

        # Skip if verdict is None (e.g., Auditor INCONCLUSIVE)
        if verdict is None:
            continue

        # Skip if already processed (idempotency)
        if (ticker, decision_date) in existing_keys:
            continue

        # Try to compute realized return
        try:
            realized_return = compute_realized_return(
                ticker, decision_date, window_days=REALIZED_RETURN_WINDOW_DAYS
            )
        except NoMarketDataError as e:
            logger.debug(
                f"Skipping {layer}/{ticker}/{decision_date}: {e} (will retry later)"
            )
            continue

        # Filter by since_date if provided
        if since_date and decision_date < since_date:
            continue

        # Build reflection record
        classification = classify_realized_return(realized_return)
        lesson_text = build_lesson_text(
            verdict, realized_return, classification, REALIZED_RETURN_WINDOW_DAYS
        )
        created_at = datetime.now(timezone.utc).isoformat()

        reflection = ReflectionRecord(
            ticker=ticker,
            decision_date=decision_date,
            decision_verdict=verdict,
            realized_return=realized_return,
            classification=classification,
            lesson_text=lesson_text,
            created_at=created_at,
        )

        # Append to reflection store (unless dry_run)
        if not dry_run:
            _append_reflection(reflection_log_path, reflection)

        processed_count += 1

    return processed_count


def process_pending_outcomes(
    dry_run: bool = False, since_date: str | None = None
) -> dict[str, int]:
    """Process all pending outcomes across all three layers.

    Main entry point: reads each layer's decision log, computes realized
    returns, and appends idempotent reflections to each layer's dedicated
    reflection store. Never cross-writes between layers.

    Args:
        dry_run: If True, compute and log but do not write reflections
        since_date: Optional YYYY-MM-DD filter (only process decisions >= this date)

    Returns:
        Dict with per-layer counts: {"research": N, "auditor": N, "risk": N}
    """
    logger.info(f"process_pending_outcomes: dry_run={dry_run}, since_date={since_date}")

    results = {}

    # Process research layer
    results["research"] = _process_layer(
        "research", _extract_research_verdict, dry_run=dry_run, since_date=since_date
    )

    # Process auditor layer
    results["auditor"] = _process_layer(
        "auditor", _extract_auditor_verdict, dry_run=dry_run, since_date=since_date
    )

    # Process risk layer
    results["risk"] = _process_layer(
        "risk", _extract_risk_verdict, dry_run=dry_run, since_date=since_date
    )

    logger.info(f"process_pending_outcomes complete: {results}")

    return results
