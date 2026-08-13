"""Unit tests for the Phase 7 cloud daily runner GitHub Actions workflow.

Mirrors the pattern used for scripts/setup_phase7_scheduler.ps1: reads and
parses the workflow file without actually running it (GitHub Actions can't be
exercised locally), and asserts on structure/content. This workflow exists so
the 4-6 week validation window can keep running while the user travels without
their PC -- if state persistence or the required secrets silently regress,
the window quietly loses days of data with no local signal.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

WORKFLOW_PATH = (
    pathlib.Path(__file__).parent.parent
    / ".github"
    / "workflows"
    / "phase7-cloud-daily.yml"
)


@pytest.fixture
def workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_workflow_file_exists():
    assert WORKFLOW_PATH.exists(), f"Workflow not found at {WORKFLOW_PATH}"


@pytest.mark.unit
def test_workflow_is_valid_yaml(workflow):
    assert isinstance(workflow, dict)


@pytest.mark.unit
def test_workflow_has_schedule_and_manual_trigger(workflow):
    """Runs on a cron schedule, plus workflow_dispatch for manual testing.

    PyYAML parses the bare `on:` key as boolean True (YAML 1.1 quirk) --
    GitHub's own workflow parser handles it correctly regardless; this is a
    known, harmless artifact also present when parsing the existing ci.yml
    the same way, not a bug in this file.
    """
    triggers = workflow[True]
    assert "schedule" in triggers
    assert triggers["schedule"][0]["cron"] == "30 20 * * *"
    assert "workflow_dispatch" in triggers


@pytest.mark.unit
def test_workflow_has_write_permission_for_state_commit(workflow):
    """The final step commits phase7-state/ back to the repo -- needs contents: write."""
    assert workflow["permissions"]["contents"] == "write"


@pytest.mark.unit
def test_workflow_never_cancels_a_running_trading_cycle(workflow):
    """A cancelled mid-flight run could leave a partial/incomplete Alpaca order."""
    assert workflow["concurrency"]["cancel-in-progress"] is False


@pytest.mark.unit
def test_workflow_uses_required_secrets(workflow):
    """Credentials must come from repo secrets, never hardcoded in the workflow."""
    steps = workflow["jobs"]["daily-run"]["steps"]
    run_step = next(s for s in steps if s.get("name") == "Run Phase 7 daily basket")
    env = run_step["env"]

    assert env["ANTHROPIC_API_KEY"] == "${{ secrets.ANTHROPIC_API_KEY }}"
    assert env["ALPACA_PAPER_API_KEY"] == "${{ secrets.ALPACA_PAPER_API_KEY }}"
    assert env["ALPACA_PAPER_SECRET_KEY"] == "${{ secrets.ALPACA_PAPER_SECRET_KEY }}"


@pytest.mark.unit
def test_workflow_forces_anthropic_provider(workflow):
    """DEFAULT_CONFIG defaults to OpenAI (llm_provider='openai') -- the workflow
    must override to Anthropic explicitly, or a real run would try to call
    OpenAI with no OPENAI_API_KEY secret configured."""
    steps = workflow["jobs"]["daily-run"]["steps"]
    run_step = next(s for s in steps if s.get("name") == "Run Phase 7 daily basket")
    env = run_step["env"]

    assert env["TRADINGAGENTS_LLM_PROVIDER"] == "anthropic"
    assert "claude" in env["TRADINGAGENTS_DEEP_THINK_LLM"]
    assert "claude" in env["TRADINGAGENTS_QUICK_THINK_LLM"]


@pytest.mark.unit
def test_workflow_redirects_all_state_paths_into_the_repo(workflow):
    """All 5 persistence paths must be redirected into phase7-state/ inside the
    checkout -- GitHub's ephemeral runners discard anything outside the repo
    (e.g. the ~/.tradingagents/ default) between runs. Missing even one of
    these means that category of audit data silently stops accumulating."""
    steps = workflow["jobs"]["daily-run"]["steps"]
    run_step = next(s for s in steps if s.get("name") == "Run Phase 7 daily basket")
    env = run_step["env"]

    required_path_vars = [
        "TRADINGAGENTS_RESULTS_DIR",
        "TRADINGAGENTS_CACHE_DIR",
        "TRADINGAGENTS_RESEARCH_LOG_PATH",
        "TRADINGAGENTS_AUDITOR_LOG_PATH",
        "TRADINGAGENTS_RISK_LOG_PATH",
        "TRADINGAGENTS_COST_LOG_PATH",
    ]
    for var in required_path_vars:
        assert var in env, f"Missing state redirection for {var}"
        assert "phase7-state" in env[var], (
            f"{var} does not point into phase7-state/: {env[var]}"
        )


@pytest.mark.unit
def test_workflow_runs_the_full_default_basket(workflow):
    """No --tickers override -- must run the full PHASE7_TICKER_BASKET (13
    instruments), not a partial/test subset, on the real schedule."""
    steps = workflow["jobs"]["daily-run"]["steps"]
    run_step = next(s for s in steps if s.get("name") == "Run Phase 7 daily basket")

    assert run_step["run"].strip() == "python scripts/phase7_daily_runner.py"


@pytest.mark.unit
def test_workflow_commits_and_pushes_state_after_run(workflow):
    """Without this step, every run's audit trail is discarded when the
    ephemeral runner is torn down -- the whole point of this workflow."""
    steps = workflow["jobs"]["daily-run"]["steps"]
    commit_step = next(
        s for s in steps if s.get("name") == "Commit updated audit trail"
    )
    run_script = commit_step["run"]

    assert "git add phase7-state/" in run_script
    assert "git commit" in run_script
    assert "git push" in run_script
