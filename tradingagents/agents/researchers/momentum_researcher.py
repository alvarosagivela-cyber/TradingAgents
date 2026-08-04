"""LLM and compute nodes for the momentum research layer.

This module provides the core pipeline:
1. Compute node: deterministic Python momentum calculation
2. LLM node: reasoning on computed momentum metrics
3. (Validate node: in thesis_validator.py)

The LLM node rejects on structured-output failure (no free-text fallback) to ensure
every thesis goes through validation (D-11).
"""

from __future__ import annotations

import logging

from tradingagents.agents.reflectors.reflection_reader import read_reflection_for_ticker
from tradingagents.agents.schemas import ResearchThesis
from tradingagents.agents.utils.momentum_calculator import compute_momentum
from tradingagents.agents.utils.structured import bind_structured

logger = logging.getLogger(__name__)


def create_momentum_compute_node():
    """Factory for the momentum compute node.

    Returns a node function that:
    1. Reads state["company_of_interest"] and state["trade_date"]
    2. Calls compute_momentum(...) to deterministically calculate metrics
    3. Returns an update dict with the computed values

    The compute node fails open (valid=False) on any data fetch error; the LLM node
    later short-circuits when it sees valid=False (D-10).

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
    """

    def compute_node(state: dict) -> dict:
        """Compute momentum metrics for the current ticker.

        Args:
            state: AgentState dict with company_of_interest, trade_date

        Returns:
            Dict with {retorno_12_1, momentum_z_score, confidence_level, momentum_valid}
        """
        ticker = state.get("company_of_interest")
        trade_date = state.get("trade_date")

        logger.info(f"Computing momentum for {ticker} as of {trade_date}")

        metrics = compute_momentum(ticker, trade_date)

        return {
            "retorno_12_1": metrics.retorno_12_1,
            "momentum_z_score": metrics.z_score,
            "confidence_level": metrics.confidence_level,
            "momentum_valid": metrics.valid,
        }

    return compute_node


def create_momentum_llm_node(llm):
    """Factory for the momentum LLM node.

    The LLM node:
    1. Short-circuits if momentum_valid=False (fail-open, D-10)
    2. Otherwise, builds a prompt with the computed momentum metrics as input evidence (D-16)
    3. Invokes the LLM with ResearchThesis structured output
    4. Rejects (no free-text fallback) on structured-output failure (D-11)

    Args:
        llm: LLM client to use for inference

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
    """

    def llm_node(state: dict) -> dict:
        """Generate research thesis based on momentum metrics.

        Args:
            state: AgentState dict with computed momentum fields, market_report, etc.

        Returns:
            Dict with {thesis, thesis_verdict, refutation_criterion, data_points}
            or rejection dict {verified=False, ...} if valid=False or inference fails
        """
        ticker = state.get("company_of_interest")
        momentum_valid = state.get("momentum_valid", False)

        # Short-circuit if insufficient OHLCV history (D-10)
        if not momentum_valid:
            logger.info(
                f"Momentum for {ticker} has insufficient history; skipping LLM (fail-open)"
            )
            return {
                "verified": False,
                "thesis": "REJECTED: insufficient OHLCV history to compute retorno_12_1 (D-10 precondition not met)",
                "thesis_verdict": "",
                "refutation_criterion": "",
                "data_points": {},
            }

        # Extract momentum metrics for the prompt
        retorno_12_1 = state.get("retorno_12_1", 0.0)
        z_score = state.get("momentum_z_score", 0.0)
        market_report = state.get("market_report", "")

        # Read prior reflection from the research layer (before prompt construction)
        reflection = read_reflection_for_ticker("research", ticker)

        # Build reflection context if available
        reflection_context = ""
        if reflection is not None:
            reflection_context = f"""
## Prior Analysis: {ticker}
- **Decision Date**: {reflection.decision_date}
- **Decision**: {reflection.decision_verdict}
- **Realized Return**: {reflection.realized_return:+.2%}
- **Classification**: {reflection.classification.upper()}
- **Lesson**: {reflection.lesson_text}

Please weigh this prior outcome in your current analysis.

"""

        # Build the prompt (D-16: state the z-score as input evidence, not the conclusion)
        prompt = f"""You are a research analyst evaluating momentum-based investment theses.

**Ticker**: {ticker}

**Momentum Metrics** (pre-computed by Python, input evidence for your reasoning):
- 12-Month Momentum (skip-month formula): {retorno_12_1:.4f} ({retorno_12_1*100:.2f}%)
- Z-Score (vs 5-year distribution): {z_score:.2f}

**Market Context**:
{market_report}

{reflection_context}

Your task:
1. Treat the momentum metrics above as input data to weigh alongside other context
2. Reason causally about why momentum should persist or reverse for this ticker
3. Do NOT simply restate the z-score or momentum number as your conclusion
4. Provide a concrete, falsifiable refutation criterion (what observation would disprove this thesis?)
5. Cite the exact momentum and z-score values in your response so they can be validated

Output a structured thesis with:
- verdict: Buy, Hold, or Sell
- reasoning: 150-300 word causal explanation
- refutation_criterion: A concrete, falsifiable observation
- data_points: {{retorno_12_1: (the exact value above), z_score: (the exact value above)}}
"""

        # Bind LLM to structured output
        structured_llm = bind_structured(llm, ResearchThesis, "MomentumResearcher")

        # Invoke structured LLM (no free-text fallback — reject on failure per D-11)
        if structured_llm is None:
            logger.error("MomentumResearcher: structured output not supported by provider")
            return {
                "verified": False,
                "thesis": "REJECTED: provider does not support structured output",
                "thesis_verdict": "",
                "refutation_criterion": "",
                "data_points": {},
            }

        try:
            thesis = structured_llm.invoke(prompt)
            if thesis is None:
                logger.error("MomentumResearcher: structured output returned None")
                return {
                    "verified": False,
                    "thesis": "REJECTED: structured output returned no result",
                    "thesis_verdict": "",
                    "refutation_criterion": "",
                    "data_points": {},
                }

            # Success: return the thesis fields for validation
            return {
                "thesis": thesis.reasoning,
                "thesis_verdict": thesis.verdict.value,
                "refutation_criterion": thesis.refutation_criterion,
                "data_points": thesis.data_points,
            }

        except Exception as exc:
            logger.exception(f"MomentumResearcher: LLM invocation failed: {exc}")
            return {
                "verified": False,
                "thesis": f"REJECTED: LLM invocation failed: {exc}",
                "thesis_verdict": "",
                "refutation_criterion": "",
                "data_points": {},
            }

    return llm_node
