"""Cost tracking schema and persistence (Phase 6, D-04/D-05).

This module implements:
1. CostRecord: frozen dataclass matching ReflectionRecord pattern
2. persist_cost_record: append-only JSONL writer (never raises, mirrors thesis_validator pattern)
3. read_cost_records: graceful JSONL reader (skips malformed lines, mirrors realized_outcome_processor pattern)
4. record_cost_from_response: extract usage_metadata from LLM responses and record cost

The cost tracking layer must never block the pipeline — all errors are logged
and swallowed (D-05: fail-closed guarantee).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cost_pricing import calculate_cost
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostRecord:
    """Immutable record of an LLM invocation cost.

    Attributes:
        timestamp: ISO 8601 UTC timestamp of the LLM invocation
        layer: Layer name (e.g. "research", "auditor", "conservative", "balanced", "aggressive")
        model: Model ID (e.g. "claude-haiku-4-5", "claude-sonnet-5")
        ticker: Stock ticker symbol (may be empty if call predates ticker selection)
        trade_date: Trade date in YYYY-MM-DD format (may be empty if call predates date selection)
        input_tokens: Exact input token count from usage_metadata
        output_tokens: Exact output token count from usage_metadata
        cache_read_input_tokens: Cache hit tokens (always 0 today, no cache_control used)
        cache_creation_input_tokens: Cache write tokens (always 0 today, no cache_control used)
        cost_usd: Calculated cost via calculate_cost()
    """

    timestamp: str
    layer: str
    model: str
    ticker: str
    trade_date: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: float

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict for JSONL persistence.

        Returns:
            Dict with all ten fields suitable for json.dumps()
        """
        return {
            "timestamp": self.timestamp,
            "layer": self.layer,
            "model": self.model,
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> CostRecord:
        """Deserialize from a JSON dict.

        Args:
            data: Dict with ten keys matching CostRecord fields

        Returns:
            Reconstructed CostRecord instance
        """
        return cls(
            timestamp=data["timestamp"],
            layer=data["layer"],
            model=data["model"],
            ticker=data["ticker"],
            trade_date=data["trade_date"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            cache_read_input_tokens=data["cache_read_input_tokens"],
            cache_creation_input_tokens=data["cache_creation_input_tokens"],
            cost_usd=data["cost_usd"],
        )


def persist_cost_record(record: CostRecord, log_path: Path) -> None:
    """Append a cost record to the JSONL audit trail (D-04).

    Creates parent directories if needed. Never raises — failures are logged
    and swallowed (D-05: fail-closed guarantee on cost tracking).

    Args:
        record: CostRecord to persist
        log_path: Path to append-only JSONL file
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_json_dict()) + "\n")
    except Exception as exc:
        logger.exception(f"Failed to persist cost record to {log_path}: {exc}")


def read_cost_records(log_path: Path) -> list[CostRecord]:
    """Read cost records from JSONL file, gracefully handling errors.

    Returns empty list if file doesn't exist. Skips malformed JSON lines
    (logged with a warning) but returns well-formed records around them
    (established pattern from realized_outcome_processor._read_jsonl).

    Args:
        log_path: Path to the JSONL file

    Returns:
        List of parsed CostRecords; empty list if file does not exist
    """
    if not log_path.exists():
        return []

    records = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = CostRecord.from_json_dict(data)
                    records.append(record)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(
                        f"Failed to parse line {line_num} in {log_path.name}: {e}"
                    )
                    continue
    except IOError as e:
        logger.error(f"Error reading {log_path}: {e}")
        return []

    return records


def record_cost_from_response(
    response: Any,
    *,
    layer: str,
    model: str,
    ticker: str = "",
    trade_date: str = "",
) -> CostRecord | None:
    """Extract usage_metadata from an LLM response and persist cost record (D-03/D-04).

    Wraps entire body in try/except to guarantee D-05: a cost-tracking failure
    never raises out of invoke() and never blocks the pipeline.

    Args:
        response: Response object from ChatAnthropic.invoke() carrying usage_metadata
        layer: Layer name (e.g. "auditor", "research", "conservative")
        model: Model ID (e.g. "claude-sonnet-5")
        ticker: Stock ticker (optional, defaults to empty string)
        trade_date: Trade date (optional, defaults to empty string)

    Returns:
        CostRecord if successfully extracted and persisted, None on any error
    """
    try:
        # Extract usage_metadata from response
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            logger.warning(
                f"No usage_metadata found on response for {layer}/{model} — "
                f"cost tracking skipped"
            )
            return None

        # Extract token counts
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        details = usage.get("input_token_details") or {}
        cache_read = details.get("cache_read", 0)
        cache_creation = details.get("cache_creation", 0)

        # Calculate cost via pricing table
        cost_usd = calculate_cost(model, input_tokens, output_tokens)

        # Build CostRecord
        record = CostRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            layer=layer,
            model=model,
            ticker=ticker,
            trade_date=trade_date,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            cost_usd=cost_usd,
        )

        # Persist to JSONL
        config = get_config()
        log_path = Path(config["cost_log_path"])
        persist_cost_record(record, log_path)

        return record

    except Exception as exc:
        logger.exception(
            f"Unexpected error during cost tracking for {layer}/{model}: {exc} — "
            f"pipeline continues (D-05)"
        )
        return None
