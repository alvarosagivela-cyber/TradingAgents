"""Integration tests for the Research Thesis generation pipeline (compute -> LLM -> validate -> persist).

Tests the full chain: momentum compute node, LLM node, validation node, and disk persistence.
All tests use mocked yfinance/load_ohlcv and LLM, so no network or API calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.schemas import ResearchThesis, TraderAction

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_momentum_state(
    company: str = "AAPL",
    date: str = "2026-07-29",
    retorno_12_1: float = 0.1834,
    momentum_z_score: float = 2.3,
    momentum_valid: bool = True,
    confidence_level: str = "high",
    data_points: dict | None = None,
) -> dict:
    """Build a minimal AgentState dict pre-populated with momentum fields for testing.

    By default, all required fields are populated correctly. Override as needed
    for specific test cases (e.g., mismatch tests, insufficient-data tests).
    """
    if data_points is None:
        data_points = {"retorno_12_1": retorno_12_1, "z_score": momentum_z_score}

    return {
        "company_of_interest": company,
        "trade_date": date,
        "retorno_12_1": retorno_12_1,
        "momentum_z_score": momentum_z_score,
        "momentum_valid": momentum_valid,
        "confidence_level": confidence_level,
        "market_report": "The broader market shows signs of momentum continuation.",
        "sentiment_report": "Bullish sentiment from social media.",
        "news_report": "Positive earnings surprise.",
        "fundamentals_report": "Strong fundamentals.",
        "thesis": "",
        "thesis_verdict": "",
        "refutation_criterion": "",
        "data_points": data_points,
        "verified": False,
    }


def _structured_thesis_llm(captured: dict, thesis: ResearchThesis | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real ResearchThesis.
    """
    if thesis is None:
        thesis = ResearchThesis(
            verdict=TraderAction.BUY,
            reasoning="The 12-month momentum of 18.34% with a z-score of 2.3 suggests strong persistence. "
                      "Historical data shows that at this z-score level, momentum continues 65% of the time. "
                      "Current market breadth supports continuation into the next quarter.",
            refutation_criterion="Thesis fails if price closes below the 50-day MA on 2x average volume within 10 trading days.",
            data_points={"retorno_12_1": 0.1834, "z_score": 2.3},
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or thesis
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMomentumThesisPipeline:

    def test_happy_path_compute_llm_validate_persist(self, tmp_path):
        """Test 1 (happy path): valid momentum state -> valid thesis -> persisted JSONL."""
        # Import the node factories here so they're only needed if test runs
        from tradingagents.agents.researchers.momentum_researcher import (
            create_momentum_llm_node,
        )
        from tradingagents.agents.utils.thesis_validator import (
            create_momentum_validate_node,
        )

        # Setup
        state = _make_momentum_state()
        captured = {}
        mock_llm = _structured_thesis_llm(captured)

        # Patch get_config to use tmp_path for thesis output
        with patch(
            "tradingagents.agents.utils.thesis_validator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "research_thesis_log_path": str(tmp_path / "theses.jsonl")
            }

            # Run the pipeline (state is pre-populated, we skip compute node)
            llm_node = create_momentum_llm_node(mock_llm)
            llm_update = llm_node(state)
            state.update(llm_update)

            validate_node = create_momentum_validate_node()
            validate_update = validate_node(state)
            state.update(validate_update)

            # Assertions
            assert state.get("verified") is True, "Thesis should be verified"
            assert state.get("thesis_verdict") == "Buy"
            assert (tmp_path / "theses.jsonl").exists(), "JSONL file should be created"

            # Verify JSONL content
            lines = (tmp_path / "theses.jsonl").read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1, "Should have exactly one record"

    def test_numeric_mismatch_rejects_thesis(self, tmp_path):
        """Test 2 (numeric mismatch -> reject, D-11): LLM-cited retorno_12_1 differs from computed."""
        from tradingagents.agents.researchers.momentum_researcher import (
            create_momentum_llm_node,
        )
        from tradingagents.agents.utils.thesis_validator import (
            create_momentum_validate_node,
        )

        # Setup: state has computed retorno_12_1=0.1834, but LLM returns 0.20 (differs by >0.0001)
        state = _make_momentum_state(retorno_12_1=0.1834)
        captured = {}

        # Create a mock LLM that returns mismatched data_points
        mismatch_thesis = ResearchThesis(
            verdict=TraderAction.BUY,
            reasoning="Momentum signal is strong based on recent price action.",
            refutation_criterion="Thesis fails if price closes below the 50-day MA.",
            data_points={"retorno_12_1": 0.20, "z_score": 2.3},  # Mismatch!
        )
        mock_llm = _structured_thesis_llm(captured, mismatch_thesis)

        with patch(
            "tradingagents.agents.utils.thesis_validator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "research_thesis_log_path": str(tmp_path / "theses.jsonl")
            }

            # Run LLM and validation
            llm_node = create_momentum_llm_node(mock_llm)
            llm_update = llm_node(state)
            state.update(llm_update)

            validate_node = create_momentum_validate_node()
            validate_update = validate_node(state)
            state.update(validate_update)

            # Assertions
            assert state.get("verified") is False, "Thesis should be rejected due to numeric mismatch"
            assert not (
                tmp_path / "theses.jsonl"
            ).exists(), "JSONL file should NOT be created for rejected thesis"

    def test_insufficient_ohlcv_fails_open(self, tmp_path):
        """Test 3 (insufficient OHLCV -> fail-open, D-10): momentum_valid=False short-circuits LLM."""
        from tradingagents.agents.researchers.momentum_researcher import (
            create_momentum_llm_node,
        )
        from tradingagents.agents.utils.thesis_validator import (
            create_momentum_validate_node,
        )

        # Setup: momentum_valid=False signals insufficient history
        state = _make_momentum_state(momentum_valid=False)
        captured = {}
        mock_llm = _structured_thesis_llm(captured)

        with patch(
            "tradingagents.agents.utils.thesis_validator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "research_thesis_log_path": str(tmp_path / "theses.jsonl")
            }

            # Run LLM node
            llm_node = create_momentum_llm_node(mock_llm)
            llm_update = llm_node(state)

            # Assertions: LLM should NOT have been called
            mock_llm.with_structured_output.assert_not_called()
            assert llm_update.get("verified") is False
            assert llm_update.get("data_points") == {}

            # Update state and run validation
            state.update(llm_update)
            validate_node = create_momentum_validate_node()
            validate_update = validate_node(state)
            state.update(validate_update)

            # JSONL should never be created
            assert not (
                tmp_path / "theses.jsonl"
            ).exists(), "JSONL file should NOT be created for insufficient data"
