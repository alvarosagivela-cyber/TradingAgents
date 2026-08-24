# Code Review: Phase 7 idempotency guard (commit 5dab7f7 + follow-up fixes)

**Reviewed**: 2026-08-25
**Mode**: Local review (no uncommitted diff at review time — reviewed the most recent real code commit, `5dab7f7`, plus fixes applied during this review)
**Decision**: APPROVE (with comments — 2 findings fixed during review, 1 MEDIUM hardening recommendation left open)

## Summary

The idempotency guard (`get_already_processed_tickers` + its use in `main()`) correctly solves the confirmed production bug (same-day manual + scheduled runs double-processing all 13 tickers). Design is sound: fail-open on read errors, keyed on the Auditor's unconditional persistence point, `--force` escape hatch. Two real issues found during this review were fixed inline (a ruff regression, and a Phase-6-style "valid JSON, non-dict" crash-the-whole-read bug). One MEDIUM-severity design note is left as a recommendation, not fixed, since it's a defense-in-depth improvement rather than a confirmed live bug.

## Findings

### CRITICAL
None.

### HIGH
None.

### MEDIUM

**1. Idempotency check keys on `auditor_log_path`, an earlier pipeline step than true completion.**

`get_already_processed_tickers` treats "Auditor persisted for (ticker, trade_date)" as "this ticker is done." But the pipeline continues past the Auditor into Risk Squad, Portfolio Manager, and Paper Execution. If a real exception occurred *after* Auditor persistence but *before* the cycle truly finished, and that exception propagated up to `run_ticker_with_retry` (causing a skip for that run), a later re-run/backfill would see the Auditor record and treat the ticker as done — permanently skipping it even though no real decision/execution ever completed for that date.

**Verified this is currently low-probability, not a live bug**: every node downstream of the Auditor that makes a real external call already catches its own exceptions internally and never propagates them —
- `conservative_perspective.py:174` (and balanced/aggressive, same pattern): catches LLM invocation failures
- `portfolio_snapshot.py:94`: catches the real Alpaca `get_account()` call
- `paper_execution.py`: catches all Alpaca `submit_order()` exceptions (verified earlier this session)

So today, an exception reaching `run_ticker_with_retry` almost always means the failure happened *before* Auditor persistence (Research/Momentum/Auditor's own LLM calls aren't wrapped this way — confirmed by the observed billing-outage behavior: all 13 tickers failed cleanly with no partial Auditor writes). The gap is real but narrow: it would need a bug in one of those three defensive wrappers, or a new node added later without matching defensive handling.

**Recommendation** (not applied — a design choice, not a bug fix): key the guard on `risk_log_path` (`decisions.jsonl`) instead of `auditor_log_path`. It's written later in the pipeline (closer to true completion) and, if it's also written unconditionally like the Auditor's, would be a strictly stronger "really done" signal at no extra cost. Worth doing before this matters — i.e., before any future node is added between Risk Squad and Paper Execution without the same fail-safe wrapping.

**Status**: Left open, documented here for a future pass. Not blocking — no live incident traces to this gap.

### Findings fixed during this review

**2. [Fixed] Valid-JSON-but-non-dict line would silently disable the guard for the entire file, not just that line.**

Original code: the inner `try/except json.JSONDecodeError` only caught malformed *syntax*. A line that's valid JSON but not an object (e.g. a bare `42`) would pass `json.loads()` fine, then `record.get("trade_date")` would raise `AttributeError` — caught only by the *outer* `except Exception`, which aborts the whole read and returns `set()`. Net effect: one corrupted line anywhere in the log would silently re-enable double-processing for every ticker that day, reintroducing the exact class of bug this guard exists to prevent.

This is the same bug class Phase 6 already hardened `read_cost_records()`/`decision_reconstructor` against elsewhere in this codebase (per this session's earlier work) — this new function reintroduced a milder version of it.

**Fix applied**: moved the type-unsafe `.get()` calls inside the per-line try/except and widened it to `(json.JSONDecodeError, AttributeError, TypeError)`, so one bad line is skipped without losing the guard for the rest of the file. Regression test added: `test_get_already_processed_tickers_skips_only_one_bad_line_not_whole_file`.

**3. [Fixed] Minor ruff regression in new code.**

`open(auditor_log_path, "r", encoding="utf-8")` — the explicit `"r"` mode is the default and redundant (ruff UP015). Fixed to `open(auditor_log_path, encoding="utf-8")`. (The other 3 ruff findings in this file/test — `SIM108`, `F541`, `F841` — are pre-existing, in code this change didn't touch, already covered by the separately-tracked ruff cleanup task.)

### LOW

**4. Minor duplicated computation.** `main()` computes which tickers are already-processed twice: once in the logging loop (`for ticker in already_processed: if ticker in tickers: ...`) and again for the `reused` list (`[t for t in tickers if t in already_processed]`). Not a bug — both are `O(n)` over a 13-element basket — just a small readability nit. Not fixed (cosmetic only, judged not worth the diff noise on top of everything else in this commit).

## Validation Results

| Check | Result |
|---|---|
| Tests (`pytest -q`, full suite) | Pass — 936 passed, 2 skipped, 1 deselected |
| Tests (`test_phase7_daily_runner.py` specifically) | Pass — 13/13, including 2 new tests added during this review |
| Lint (`ruff check` on changed files) | Pass, after fixing UP015 in new code; 3 pre-existing findings in untouched code left to the tracked ruff-cleanup task |
| Type check | N/A (no configured type checker for this project) |
| Build | N/A (Python script, no build step) |

## Files Reviewed

- `scripts/phase7_daily_runner.py` — Modified (idempotency guard added, 2 fixes applied during review)
- `tests/test_phase7_daily_runner.py` — Modified (5 new tests from the original commit + 1 new test added during this review)
- `phase7-state/auditor/audits.jsonl`, `phase7-state/research/theses.jsonl`, `phase7-state/risk/decisions.jsonl` — Modified (duplicate 2026-08-24 entries removed; data files, not reviewed as code)
