"""Phase 7 daily production runner — Windows Task Scheduler invocable wrapper.

D-02: A thin, Windows-Task-Scheduler-invoked wrapper around the checkpointed
TradingAgentsGraph.propagate() path (D-01). No new long-running process, no new
infrastructure — only a daily invocation of the already-tested checkpoint-capable
propagate() method per ticker in the D-03 fixed basket.

D-08: Implements retry-once-then-skip: if a single ticker's propagate() call
raises an exception, we retry exactly once after a short delay. If the retry
also fails, we log the skip and continue to the next ticker. The day's run
always continues to completion regardless of individual ticker failures — a
bad ticker never halts the rest of the day.

Exit code: 0 unless a *majority* of the tickers actually attempted this run
failed, in which case exit 1. This is deliberately narrower than "any skip" --
D-08's per-ticker tolerance above is preserved, one flaky ticker still exits 0
-- but a systemic failure (everything or almost everything failing) must be
loud. Confirmed necessary in production: an Anthropic org spend-cap outage,
then separately an invalid-API-key outage, each ran for a full week completely
undetected because this always returned 0 and GitHub Actions only alerts on a
non-zero exit -- every day still showed green in Actions despite processing
zero real decisions, both times only caught by a manual "how's it going" check.

D-11: Every invocation prints the current budget status (from Phase 6's
cost_aggregator.summarize_costs()) to stdout, visible via Windows Task Scheduler's
run history — passive, always-fresh visibility with zero new command the user
must remember to run.

Persistent logging: Task Scheduler does not capture a launched process's stdout/
stderr, and a run with only isolated ticker skips still exits 0, so console
output alone is not a durable record for an unattended 4-6 week window. Every
run also appends to a log file under DEFAULT_CONFIG["results_dir"] so per-ticker
outcomes and budget status remain reviewable after the fact regardless of how
the process was invoked or what its exit code was.

Explicit scope boundary: This script is invoked once per day by Task Scheduler.
It does NOT itself register the scheduled task — see setup_phase7_scheduler.ps1
for the one-time, human-run, documented setup artifact.

Idempotency guard: before processing a ticker, checks whether an Auditor cycle
is already persisted for (ticker, trade_date) and skips it if so. This exists
because a scheduled trigger and a manual workflow_dispatch test run can land on
the same calendar day (confirmed in production: a manual re-run to verify an
API-limit fix, followed a couple hours later by that day's normal cron trigger,
double-processed all 13 tickers -- doubling real LLM spend and, worse, doubling
every persisted decision record, which would silently inflate VALID-01's
Auditor-refutation count on any day with a real mismatch verdict). The guard
also means a day that partially failed (some tickers skipped) can be safely
re-run later to backfill only the missing tickers, without redoing the ones
that already succeeded. Pass --force to bypass and reprocess unconditionally.
"""

from __future__ import annotations

import argparse
import json
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


def get_already_processed_tickers(trade_date: str, auditor_log_path: str) -> set[str]:
    """Return tickers that already have a persisted Auditor cycle for trade_date.

    The Auditor is the last step in the Research->Auditor chain to persist
    (unconditionally, regardless of comparison_result -- AUDIT-04), so its
    presence for a given (ticker, trade_date) is a reliable "this cycle already
    completed" signal. Fails open (returns an empty set, i.e. process everything)
    on any read error, so a corrupt or unreadable log never blocks a real run --
    consistent with this module's D-08 fail-open philosophy.

    Args:
        trade_date: The trade date in YYYY-MM-DD format
        auditor_log_path: Path to the auditor JSONL log

    Returns:
        Set of ticker symbols already processed for trade_date
    """
    processed: set[str] = set()
    if not os.path.exists(auditor_log_path):
        return processed

    try:
        with open(auditor_log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # A line can be valid JSON but not an object (e.g. a bare
                    # number or array) -- .get() on that raises AttributeError,
                    # not JSONDecodeError. Same bug class Phase 6 hardened
                    # read_cost_records()/decision_reconstructor against; skip
                    # only this one malformed line, don't lose the guard for
                    # every other valid record in the file over it.
                    if record.get("trade_date") == trade_date:
                        ticker = record.get("ticker")
                        if ticker:
                            processed.add(ticker)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    continue
    except Exception:
        logger.exception(
            "Failed reading auditor log for idempotency check at %s; "
            "proceeding without the skip-already-done guard",
            auditor_log_path,
        )
        return set()

    return processed


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
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the idempotency guard and reprocess every ticker even if "
            "already persisted for this trade_date. Use only for a deliberate "
            "re-run; the default (off) is what prevents a manual test run and "
            "a same-day scheduled trigger from double-processing everything."
        ),
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

    # Idempotency guard: skip tickers already persisted for this trade_date,
    # unless --force. Prevents a manual test run + the day's normal scheduled
    # trigger from double-processing (and double-spending/double-recording).
    already_processed: set[str] = set()
    if not args.force:
        already_processed = get_already_processed_tickers(
            args.trade_date, config["auditor_log_path"]
        )
        for ticker in already_processed:
            if ticker in tickers:
                logger.info(
                    "ALREADY PROCESSED ticker=%s trade_date=%s "
                    "(idempotency guard -- pass --force to reprocess)",
                    ticker,
                    args.trade_date,
                )

    # Run each ticker with retry-then-skip
    results: dict[str, bool] = {}
    for ticker in tickers:
        if ticker in already_processed:
            continue
        results[ticker] = run_ticker_with_retry(
            ticker,
            args.trade_date,
            config,
            selected_analysts,
        )

    # Print summary
    succeeded = [t for t, ok in results.items() if ok]
    skipped = [t for t, ok in results.items() if not ok]
    reused = [t for t in tickers if t in already_processed]
    print()
    print(
        f"Day summary: {len(succeeded)}/{len(tickers)} tickers succeeded, "
        f"{len(reused)} already processed (skipped duplicate), "
        f"{len(skipped)} skipped"
    )
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    if reused:
        print(f"  Already processed: {', '.join(reused)}")

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

    # D-08 preserved for individual ticker hiccups (transient data/API blips):
    # a single skipped ticker never fails the run. But a *majority* of the
    # tickers actually attempted this run failing is a systemic problem, not
    # a normal day -- confirmed twice in production (an Anthropic spend-cap
    # outage, then an invalid-API-key outage), both invisible for a full week
    # each because this always returned 0 and GitHub Actions only alerts on a
    # non-zero exit, so every day still showed green despite processing
    # nothing. `attempted` excludes idempotency-guard skips (already-done
    # tickers aren't failures) so re-running an already-complete day never
    # trips this.
    attempted = len(tickers) - len(reused)
    if attempted > 0 and len(skipped) > attempted / 2:
        print(
            f"WARNING: majority of attempted tickers failed "
            f"({len(skipped)}/{attempted}) -- treating this as a systemic "
            f"failure (exit 1) so the run shows red and any configured "
            f"GitHub Actions failure alert actually fires."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
