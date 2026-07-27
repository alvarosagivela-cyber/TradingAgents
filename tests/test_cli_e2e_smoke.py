"""End-to-end smoke test: CLI analyze with real Anthropic key and yfinance data.

This test validates SETUP-01 and SETUP-02:
- SETUP-01: Fork runs locally end-to-end without errors, using Claude instead of OpenAI
- SETUP-02: yfinance + Alpha Vantage fallback vendor chain works

SQLite checkpointing (D-10) is NOT exercised here: it is wired into
TradingAgentsGraph.propagate(), which the interactive `analyze` CLI command
does not call (it streams the compiled graph directly). Checkpointing itself
is verified by tests/test_checkpoint_resume.py against propagate()/get_checkpointer()
directly — see that file, and the note inline below for details.

The test requires a real ANTHROPIC_API_KEY (lives-API test, marked @pytest.mark.smoke).
When run without the key, the test is skipped cleanly.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest


def _has_real_anthropic_key():
    """Check if a real Anthropic API key is set (not placeholder)."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return bool(key) and key != "placeholder"


@pytest.mark.smoke
@pytest.mark.skipif(
    not _has_real_anthropic_key(),
    reason="ANTHROPIC_API_KEY not set (or placeholder); skipping live Anthropic + yfinance e2e smoke test",
)
class TestCliE2ESmoke(unittest.TestCase):
    """End-to-end CLI test: analyze flow with real LLM + data sources."""

    def test_analyze_end_to_end_produces_report(self):
        """Full CLI analyze flow: ticker → analysts → debate → report on disk."""
        import cli.main as m
        from cli.models import AnalystType

        # Separate temp roots for results and cache
        results_root = tempfile.mkdtemp()
        cache_root = tempfile.mkdtemp()

        try:
            # Build fake config with temps and Claude settings
            fake_cfg = dict(m.DEFAULT_CONFIG)
            fake_cfg.update({
                "results_dir": results_root,
                "data_cache_dir": cache_root,
                "llm_provider": "anthropic",
                "deep_think_llm": "claude-sonnet-5",
                "quick_think_llm": "claude-haiku-4-5",
                "anthropic_effort": "medium",
                "checkpoint_enabled": True,
                "max_debate_rounds": 1,
                "max_risk_discuss_rounds": 1,
                "output_language": "English",
                # Preserve other vendor categories but set core_stock_apis fallback
                "data_vendors": dict(
                    m.DEFAULT_CONFIG["data_vendors"],
                    core_stock_apis="yfinance,alpha_vantage"
                ),
            })

            ticker = "AAPL"
            date = "2026-06-02"  # Past date relative to current system date

            # Combine all mocks: env vars, config, CLI prompts
            with mock.patch.dict(os.environ, {
                "TRADINGAGENTS_LLM_PROVIDER": "anthropic",
                "TRADINGAGENTS_DEEP_THINK_LLM": "claude-sonnet-5",
                "TRADINGAGENTS_QUICK_THINK_LLM": "claude-haiku-4-5",
                "TRADINGAGENTS_ANTHROPIC_EFFORT": "medium",
                "TRADINGAGENTS_CHECKPOINT_ENABLED": "true",
                "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "1",
                "TRADINGAGENTS_MAX_RISK_ROUNDS": "1",
            }, clear=False), \
            mock.patch.object(m, "DEFAULT_CONFIG", fake_cfg), \
            mock.patch.object(m, "fetch_announcements", return_value=None), \
            mock.patch.object(m, "display_announcements"), \
            mock.patch.object(m, "get_ticker", return_value=ticker), \
            mock.patch.object(m, "get_analysis_date", return_value=date), \
            mock.patch.object(m, "select_analysts", return_value=[AnalystType.MARKET]), \
            mock.patch.object(m, "ask_output_language", return_value="English"), \
            mock.patch.object(m, "ensure_api_key"), \
            mock.patch.object(m.typer, "prompt", return_value="N"):
                # Call run_analysis with checkpoint=None so env value is honored
                m.run_analysis(checkpoint=None)

            # Assert: report directory exists and contains *.md files
            report_path = Path(results_root) / ticker / date / "reports"
            self.assertTrue(
                report_path.exists(),
                f"Report directory {report_path} not created",
            )

            md_files = list(report_path.glob("*.md"))
            self.assertTrue(
                len(md_files) > 0,
                f"No *.md report files found in {report_path}",
            )

            # Assert: at least one report file has non-empty content
            has_content = False
            for md_file in md_files:
                content = md_file.read_text(encoding="utf-8").strip()
                if content and content != "NO_DATA_AVAILABLE":
                    has_content = True
                    break
            self.assertTrue(
                has_content,
                f"All report files are empty or contain only NO_DATA_AVAILABLE sentinel",
            )

            # Assert: message_tool.log exists
            log_file = Path(results_root) / ticker / date / "message_tool.log"
            self.assertTrue(
                log_file.exists(),
                f"Log file {log_file} not created",
            )

            # NOTE on D-10 (SQLite checkpointing): checkpointing is wired into
            # TradingAgentsGraph.propagate() (recompiles the workflow with a
            # SqliteSaver when config["checkpoint_enabled"] is set — see
            # tradingagents/graph/trading_graph.py:362-402). The interactive CLI's
            # run_analysis() does NOT call propagate() — it streams the compiled
            # graph directly (graph.graph.stream(...), no checkpointer attached),
            # so no checkpoint DB is ever created via this `tradingagents analyze`
            # code path. This is confirmed upstream fork behavior, not a bug
            # introduced here: the fork's own tests/test_checkpoint_resume.py
            # exercises checkpointing exclusively through propagate()/get_checkpointer(),
            # never through cli.main.run_analysis(). That suite already passed in
            # Task 1 (regression) and is the authoritative proof the mechanism works.
            # Asserting a checkpoint file here would test a code path that
            # structurally doesn't have one, and would require another full
            # (costly) LLM run through propagate() to exercise correctly — better
            # left to a later phase if the production runner needs propagate()-based
            # resumability (relevant for Phase 7's continuous unattended runs).

        finally:
            # Clean up temp directories
            import shutil
            shutil.rmtree(results_root, ignore_errors=True)
            shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
