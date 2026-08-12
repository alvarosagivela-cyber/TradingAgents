"""Phase 7 daily production runner — Windows Task Scheduler invocable wrapper.

D-02: A thin, Windows-Task-Scheduler-invoked wrapper around the checkpointed
TradingAgentsGraph.propagate() path (D-01). No new long-running process, no new
infrastructure — only a daily invocation of the already-tested checkpoint-capable
propagate() method per ticker in the D-03 fixed basket.

D-08: Implements retry-once-then-skip: if a single ticker's propagate() call
raises an exception, we retry exactly once after a short delay. If the retry
also fails, we log the skip and continue to the next ticker. The day's run
always continues to completion and always exits 0 — a bad day never halts the
4-6 week window.

D-11: Every invocation prints the current budget status (from Phase 6's
cost_aggregator.summarize_costs()) to stdout, visible via Windows Task Scheduler's
run history — passive, always-fresh visibility with zero new command the user
must remember to run.

Persistent logging: Task Scheduler does not capture a launched process's stdout/
stderr, and this script always exits 0 (D-08), so console output and exit code are
not durable records for an unattended 4-6 week window. Every run also appends to
a log file under DEFAULT_CONFIG["results_dir"] so per-ticker outcomes and budget
status remain reviewable after the fact regardless of how the process was invoked.

Explicit scope boundary: This script is invoked once per day by Task Scheduler.
It does NOT itself register the scheduled task — see setup_phase7_scheduler.ps1
for the one-time, human-run, documented setup artifact.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.jobs.cost_aggregator import summarize_costs
from tradingagents.trading.validation_universe import PHASE7_TICKER_BASKET

# Persistent log file: Task Scheduler doesn't capture stdout/stderr and this
# script always exits 0, so this file is the only durable record of daily
# outcomes across the unattended validation window.
_LOG_DIR = DEFAULT_CONFIG["results_dir"]
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(_LOG_DIR, "phase7_daily_runner.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# D-08: Short delay before retry, long enough to ride out a transient API blip,
# short enough not to meaningfully slow the run.
RETRY_DELAY_SECONDS = 30


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


def run_ticker_with_retry(
    ticker: str,
    trade_date: str,
    config: dict,
    selected_analysts: list[str],
) -> bool:
    """Run a single ticker through the trading agents graph with retry logic.

    D-08: Retry logic is bounded: exactly 2 attempts (1 retry). If the first
    attempt fails, sleep for RETRY_DELAY_SECONDS and retry. If the second
    attempt also fails, log the skip and return False. If either attempt
    succeeds, return True immediately.

    D-01 dependency: checkpoint_enabled must be True in config. A single
    TradingAgentsGraph instance is constructed once it first succeeds (not
    reconstructed per attempt) — a fresh checkpointed propagate() retry
    resumes from wherever the first attempt's exception left the checkpoint,
    reusing D-01's resumability. Construction itself happens inside the
    retry loop too: a transient failure while building the graph (LLM client
    init, memory log setup, etc.) is retried exactly like a propagate()
    failure rather than escaping and aborting the rest of the day's tickers.

    Args:
        ticker: The ticker symbol (e.g., "AAPL")
        trade_date: The trade date in YYYY-MM-DD format
        config: Configuration dict with checkpoint_enabled=True
        selected_analysts: List of analyst types to include

    Returns:
        True if either attempt succeeded, False if both attempts failed
    """
    graph = None

    for attempt in (1, 2):
        try:
            if graph is None:
                graph = TradingAgentsGraph(
                    selected_analysts, config=config, debug=False
                )
            final_state, signal = graph.propagate(
                ticker, trade_date, asset_type="stock"
            )
            logger.info(
                "OK ticker=%s trade_date=%s signal=%s",
                ticker,
                trade_date,
                signal,
            )
            return True
        except Exception as exc:
            logger.error(
                "FAILED attempt=%d ticker=%s trade_date=%s error=%s",
                attempt,
                ticker,
                trade_date,
                exc,
            )
            if attempt == 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                # attempt == 2: exhausted retry
                logger.error(
                    "SKIPPED ticker=%s trade_date=%s after retry exhausted (D-08)",
                    ticker,
                    trade_date,
                )
                return False

    # Unreachable (loop bound is [1, 2]), but satisfy type checker
    return False


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the daily runner.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:] when None)

    Returns:
        Exit code (always 0, per D-08: a bad day never halts the window)
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trade-date",
        type=_validate_trade_date,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Trade date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated override of the D-03 basket (default: use PHASE7_TICKER_BASKET)",
    )
    parser.add_argument(
        "--analysts",
        default="market",
        help="Comma-separated analyst types (default: 'market' to avoid unused spend on Sentiment/News/Fundamentals)",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Resolve tickers: use override if provided, else use D-03 basket
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = PHASE7_TICKER_BASKET

    # Resolve selected analysts
    selected_analysts = [a.strip() for a in args.analysts.split(",")]

    # Build config: force checkpoint_enabled=True (D-01 dependency, defensive)
    config = dict(DEFAULT_CONFIG)
    config["checkpoint_enabled"] = True

    logger.info(
        "Phase 7 daily runner started: trade_date=%s, tickers=%d, analysts=%s",
        args.trade_date,
        len(tickers),
        ", ".join(selected_analysts),
    )

    # Run each ticker with retry-then-skip
    results: dict[str, bool] = {}
    for ticker in tickers:
        results[ticker] = run_ticker_with_retry(
            ticker,
            args.trade_date,
            config,
            selected_analysts,
        )

    # Print summary
    succeeded = [t for t, ok in results.items() if ok]
    skipped = [t for t, ok in results.items() if not ok]
    print()
    print(
        f"Day summary: {len(succeeded)}/{len(tickers)} tickers succeeded, "
        f"{len(skipped)} skipped"
    )
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")

    # Print budget status (D-11)
    print()
    budget = summarize_costs()
    print(f"Budget status at end of run:")
    print(
        f"  Total cost: ${budget['total_cost_usd']:.2f} | "
        f"Projected annual: ${budget['projected_annual_usd']:.2f} / "
        f"${budget['annual_budget_target_usd']:.2f} target | "
        f"Status: {'OVER THRESHOLD' if budget['over_threshold'] else 'within budget'}"
    )
    print()

    # D-08: Always return 0 — per-ticker skips are expected, recoverable behavior
    return 0


if __name__ == "__main__":
    sys.exit(main())
