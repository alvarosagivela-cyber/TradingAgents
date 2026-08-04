"""Reflection reader for Phase 5 memory/feedback loop.

This module provides the read-side interface for per-layer reflection stores.
The reader gracefully degrades on missing or malformed files, never raising
exceptions for I/O issues (only for programmer errors like invalid layer names).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tradingagents.dataflows.config import get_config

from .reflection_schema import ReflectionRecord

logger = logging.getLogger(__name__)

# Explicit allow-list of valid reflection layers (programmer error to use anything else)
_VALID_LAYERS = ("research", "auditor", "risk")


def read_reflection_for_ticker(layer: str, ticker: str) -> ReflectionRecord | None:
    """Read the most recent reflection for a ticker from a layer's store.

    Queries the per-layer JSONL reflection store and returns the most recently
    recorded ReflectionRecord matching the given ticker, or None if:
    - The layer name is invalid (raises ValueError)
    - The store file does not exist
    - No matching ticker is found in the store
    - The store cannot be read (I/O error)
    - All lines in the store are malformed JSON

    Malformed JSON lines (e.g. from a crashed job's partial write) are logged
    and skipped; they never cause this function to raise or return None unless
    they're the ONLY lines in the file.

    Args:
        layer: One of "research", "auditor", or "risk"
        ticker: Stock ticker to look up (e.g. "AAPL")

    Returns:
        ReflectionRecord if a match is found, None otherwise

    Raises:
        ValueError: If layer is not one of the valid layers
    """
    # Programmer error: invalid layer name — fail loud
    if layer not in _VALID_LAYERS:
        raise ValueError(f"Unknown reflection layer: {layer!r}")

    # Derive the config key for this layer
    config_key = f"{layer}_reflections_log_path"
    config = get_config()
    store_path = Path(config[config_key])

    # I/O error or missing file: graceful degradation
    if not store_path.exists():
        return None

    try:
        most_recent: ReflectionRecord | None = None

        with open(store_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        f"Malformed JSON in {store_path}:{line_num} — skipping: {exc}"
                    )
                    continue

                # Reconstruct the record
                try:
                    record = ReflectionRecord.from_json_dict(data)
                except (KeyError, TypeError) as exc:
                    logger.warning(
                        f"Invalid ReflectionRecord in {store_path}:{line_num} — skipping: {exc}"
                    )
                    continue

                # Check if this record matches the ticker
                if record.ticker == ticker:
                    most_recent = record

        return most_recent

    except Exception as exc:
        # Catch-all for any other I/O or processing error
        logger.warning(f"Error reading reflections from {store_path}: {exc}")
        return None
