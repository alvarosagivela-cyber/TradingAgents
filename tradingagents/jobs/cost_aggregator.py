"""Cost aggregation, projection, and budget alert (Phase 6, D-05/D-06).

This module implements:
1. summarize_costs(): Aggregate CostRecord JSONL into running total, layer/model breakdown, and projected-annual-spend
2. Budget alert when projected spend reaches 80% of $630 target (D-05/D-06)

The alert is non-blocking (logged + printed in CLI, D-05) and never stops the pipeline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.dataflows.config import get_config
from tradingagents.llm_clients.cost_tracker import read_cost_records

logger = logging.getLogger(__name__)

# Below this elapsed window, a linear extrapolation to a full year is too noisy to
# alert on: a burst of calls within a single short session (e.g. a few tickers run
# in one sitting) hits the 1-hour floor below and gets multiplied by up to 365*24x,
# which can cross the budget threshold on literally the first session ever run.
# The raw projection is still returned for transparency; only the alert is gated.
MIN_DAYS_ELAPSED_FOR_ALERT = 1.0


def summarize_costs() -> dict[str, Any]:
    """Aggregate cost records into running total, layer/model breakdown, and projected-annual spend.

    Returns a dict with keys:
    - total_cost_usd: Sum of all cost_usd values
    - call_count: Number of cost records
    - by_layer: Dict[str, float] keyed by layer, summed cost per layer
    - by_model: Dict[str, float] keyed by model, summed cost per model
    - projected_annual_usd: Linear extrapolation from elapsed time to a full year
    - annual_budget_target_usd: Resolved from config (default 630.0)
    - budget_alert_threshold_pct: Resolved from config (default 0.80)
    - over_threshold: True if projected_annual_usd >= target * threshold_pct AND at
      least MIN_DAYS_ELAPSED_FOR_ALERT has elapsed since the earliest record (a burst
      of calls within a single short session should not trigger a false alarm)

    When over_threshold is True, logs a warning containing "Budget alert".
    When records is empty, returns zeroed dict with no errors.

    Never raises for any input shape — all I/O errors are swallowed by read_cost_records.
    """
    config = get_config()
    log_path = Path(config["cost_log_path"])
    records = read_cost_records(log_path)

    # Short-circuit: empty log
    if not records:
        target = config.get("annual_budget_target_usd", 630.0)
        threshold_pct = config.get("budget_alert_threshold_pct", 0.80)
        return {
            "total_cost_usd": 0.0,
            "call_count": 0,
            "by_layer": {},
            "by_model": {},
            "projected_annual_usd": 0.0,
            "annual_budget_target_usd": target,
            "budget_alert_threshold_pct": threshold_pct,
            "over_threshold": False,
        }

    # Aggregate totals and breakdowns
    total_cost_usd = sum(r.cost_usd for r in records)
    call_count = len(records)

    by_layer: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    for record in records:
        by_layer[record.layer] += record.cost_usd
        by_model[record.model] += record.cost_usd

    # Find earliest timestamp for extrapolation
    earliest = min(datetime.fromisoformat(r.timestamp) for r in records)
    now = datetime.now(timezone.utc)
    days_elapsed = max((now - earliest).total_seconds() / 86400.0, 1.0 / 24.0)

    # Linear extrapolation to annual spend
    projected_annual_usd = total_cost_usd * (365.0 / days_elapsed)

    # Resolve budget config
    target = config.get("annual_budget_target_usd", 630.0)
    threshold_pct = config.get("budget_alert_threshold_pct", 0.80)
    threshold_usd = target * threshold_pct

    # Check if over threshold — gated on a minimum elapsed window (see
    # MIN_DAYS_ELAPSED_FOR_ALERT) so a short burst of calls doesn't get
    # multiplied into a false alarm before there's enough data to trust the trend.
    over_threshold = (
        days_elapsed >= MIN_DAYS_ELAPSED_FOR_ALERT
        and projected_annual_usd >= threshold_usd
    )

    if over_threshold:
        logger.warning(
            f"Budget alert: projected annual LLM spend ${projected_annual_usd:.2f} "
            f"has reached {threshold_pct:.0%} of the ${target:.2f}/year target (D-05/D-06) — "
            f"non-blocking, review spend before Phase 7's extended paper-trading window"
        )

    return {
        "total_cost_usd": total_cost_usd,
        "call_count": call_count,
        "by_layer": dict(by_layer),  # Convert defaultdict to plain dict
        "by_model": dict(by_model),
        "projected_annual_usd": projected_annual_usd,
        "annual_budget_target_usd": target,
        "budget_alert_threshold_pct": threshold_pct,
        "over_threshold": over_threshold,
    }
