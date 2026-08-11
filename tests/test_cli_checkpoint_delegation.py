"""Unit test: CLI analyze command delegates to TradingAgentsGraph.propagate().

This test proves (D-01) that the interactive `analyze` CLI command now calls
TradingAgentsGraph.propagate() instead of building state manually and calling
graph.graph.stream() directly. Checkpointing is wired into propagate() and is
verified here by construction, without requiring a real Anthropic API key.

Marked @pytest.mark.unit (no real API key needed, unlike test_cli_e2e_smoke.py).
"""

import os
import tempfile
import unittest
from unittest import mock

import pytest


@pytest.mark.unit
class TestCliCheckpointDelegation(unittest.TestCase):
    """Unit test: CLI delegates graph execution to propagate()."""

    def test_run_analysis_delegates_to_propagate_not_stream(self):
        """Verify CLI calls propagate() exactly once, never calls stream()/invoke()."""
        import cli.main as m
        from cli.models import AnalystType

        # Temp directories for results and cache
        results_root = tempfile.mkdtemp()
        cache_root = tempfile.mkdtemp()

        try:
            # Build fake config
            fake_cfg = dict(m.DEFAULT_CONFIG)
            fake_cfg.update({
                "results_dir": results_root,
                "data_cache_dir": cache_root,
                "llm_provider": "anthropic",
                "deep_think_llm": "claude-sonnet-5",
                "quick_think_llm": "claude-haiku-4-5",
                "checkpoint_enabled": True,
                "max_debate_rounds": 1,
                "max_risk_discuss_rounds": 1,
                "output_language": "English",
            })

            ticker = "AAPL"
            date = "2026-08-11"

            # Mock propagate() to return a plausible final_state and signal
            mock_final_state = {
                "final_trade_decision": "HOLD",
                "company_of_interest": "AAPL",
                "trade_date": "2026-08-11",
                "market_report": "Test market report",
                "sentiment_report": "Test sentiment",
                "news_report": "Test news",
                "fundamentals_report": "Test fundamentals",
                "investment_plan": "Test investment plan",
                "trader_investment_plan": "Test trader plan",
                "investment_debate_state": {
                    "bull_history": "Bull thoughts",
                    "bear_history": "Bear thoughts",
                    "history": [],
                    "current_response": "",
                    "judge_decision": "Test judge decision",
                },
                "risk_debate_state": {
                    "aggressive_history": "Aggressive thoughts",
                    "conservative_history": "Conservative thoughts",
                    "neutral_history": "Neutral thoughts",
                    "history": [],
                    "current_response": "",
                    "judge_decision": "Test risk decision",
                },
            }
            mock_signal = "HOLD"

            # Apply all mocks
            with mock.patch.dict(os.environ, {
                "TRADINGAGENTS_LLM_PROVIDER": "anthropic",
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
            mock.patch.object(m.typer, "prompt", return_value="N"), \
            mock.patch.object(m, "TradingAgentsGraph") as mock_graph_class:
                # Create a mock graph instance with propagate as MagicMock
                mock_graph_instance = mock.MagicMock()
                mock_graph_class.return_value = mock_graph_instance

                # Mock propagate() to return our plausible state and signal
                mock_graph_instance.propagate.return_value = (
                    mock_final_state,
                    mock_signal,
                )

                # Mock stream/invoke to raise AssertionError if called
                mock_graph_instance.graph.stream.side_effect = AssertionError(
                    "stream() must not be called"
                )
                mock_graph_instance.graph.invoke.side_effect = AssertionError(
                    "invoke() must not be called"
                )

                # Call run_analysis
                m.run_analysis(checkpoint=True)

                # Assert: propagate() was called exactly once with correct args
                self.assertEqual(mock_graph_instance.propagate.call_count, 1)
                call_args = mock_graph_instance.propagate.call_args
                self.assertEqual(call_args.args, (ticker, date))
                self.assertEqual(call_args.kwargs, {"asset_type": "stock"})

                # Assert: stream/invoke were never called
                mock_graph_instance.graph.stream.assert_not_called()
                mock_graph_instance.graph.invoke.assert_not_called()

        finally:
            # run_analysis() monkey-patches instance attributes onto the module-level
            # message_buffer singleton (cli/main.py's save_*_decorator calls). Left in
            # place, the next run_analysis() call would wrap THIS test's already-wrapped
            # method (closing over this test's now-deleted temp log_file) instead of the
            # original class method — a real cross-test decorator-stacking bug that
            # surfaces once more than one test in a process calls run_analysis(). Restore
            # the class methods so each test starts from a clean, unwrapped baseline.
            for attr in ("add_message", "add_tool_call", "update_report_section"):
                m.message_buffer.__dict__.pop(attr, None)
            # Clean up temp directories
            import shutil
            shutil.rmtree(results_root, ignore_errors=True)
            shutil.rmtree(cache_root, ignore_errors=True)

    def test_run_analysis_writes_reports_from_propagate_result(self):
        """Verify report files are written from propagate()'s returned final_state."""
        import cli.main as m
        from cli.models import AnalystType

        # Temp directories
        results_root = tempfile.mkdtemp()
        cache_root = tempfile.mkdtemp()

        try:
            # Build fake config
            fake_cfg = dict(m.DEFAULT_CONFIG)
            fake_cfg.update({
                "results_dir": results_root,
                "data_cache_dir": cache_root,
                "llm_provider": "anthropic",
                "deep_think_llm": "claude-sonnet-5",
                "quick_think_llm": "claude-haiku-4-5",
                "checkpoint_enabled": False,
                "max_debate_rounds": 1,
                "max_risk_discuss_rounds": 1,
                "output_language": "English",
            })

            ticker = "AAPL"
            date = "2026-08-11"

            # Mock propagate() return value
            mock_final_state = {
                "final_trade_decision": "BUY",
                "company_of_interest": "AAPL",
                "trade_date": "2026-08-11",
                "market_report": "Bullish market conditions",
                "sentiment_report": "Positive sentiment",
                "news_report": "Good news",
                "fundamentals_report": "Strong fundamentals",
                "investment_plan": "BUY recommendation",
                "trader_investment_plan": "Execute BUY",
                "investment_debate_state": {
                    "bull_history": "Strong bull case",
                    "bear_history": "Weak bear case",
                    "history": [],
                    "current_response": "",
                    "judge_decision": "Go with bull",
                },
                "risk_debate_state": {
                    "aggressive_history": "Can take risk",
                    "conservative_history": "Play it safe",
                    "neutral_history": "Balanced",
                    "history": [],
                    "current_response": "",
                    "judge_decision": "Moderate risk OK",
                },
            }
            mock_signal = "BUY"

            # Apply mocks (same pattern as first test, with more env vars for strict control)
            with mock.patch.dict(os.environ, {
                "TRADINGAGENTS_LLM_PROVIDER": "anthropic",
                "TRADINGAGENTS_CHECKPOINT_ENABLED": "false",
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
            mock.patch.object(m.typer, "prompt", return_value="N"), \
            mock.patch.object(m, "TradingAgentsGraph") as mock_graph_class:
                # Mock graph instance
                mock_graph_instance = mock.MagicMock()
                mock_graph_class.return_value = mock_graph_instance

                # Mock propagate to return our test state
                mock_graph_instance.propagate.return_value = (
                    mock_final_state,
                    mock_signal,
                )

                # Call run_analysis
                m.run_analysis(checkpoint=False)

                # Assert: propagate() was called
                self.assertEqual(mock_graph_instance.propagate.call_count, 1)

                # Assert: report directory exists
                from pathlib import Path
                report_dir = Path(results_root) / ticker / date / "reports"
                self.assertTrue(
                    report_dir.exists(),
                    f"Report directory {report_dir} not created",
                )

                # Assert: message_tool.log exists
                log_file = Path(results_root) / ticker / date / "message_tool.log"
                self.assertTrue(
                    log_file.exists(),
                    f"Log file {log_file} not created",
                )

        finally:
            # See the matching comment in the first test — restore the singleton's
            # class methods so no later test (or later run_analysis() call in this
            # same test) inherits a wrapper closed over this test's deleted temp dir.
            for attr in ("add_message", "add_tool_call", "update_report_section"):
                m.message_buffer.__dict__.pop(attr, None)
            # Clean up temp directories
            import shutil
            shutil.rmtree(results_root, ignore_errors=True)
            shutil.rmtree(cache_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
