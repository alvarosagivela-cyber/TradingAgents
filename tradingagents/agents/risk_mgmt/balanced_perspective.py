"""Balanced/neutral perspective node for the Risk Squad (Phase 4).

This module provides a balanced perspective that evaluates trades with
a neutral weighting of upside and downside considerations.

The balanced perspective:
1. Uses a fresh Haiku LLM instance (temperature=0.3) per invocation
2. Reads only pre-computed portfolio context (never shared risk_debate_state)
3. Produces a structured RiskDecision verdict (VETO/APPROVE)
4. Writes only to its own balanced_verdict field in AgentState
"""

from __future__ import annotations

import logging

from tradingagents.agents.reflectors.reflection_reader import read_reflection_for_ticker
from tradingagents.agents.schemas import RiskDecision
from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.agents.utils.structured import bind_structured
from tradingagents.dataflows.config import get_config
from tradingagents.llm_clients.factory import create_llm_client

logger = logging.getLogger(__name__)

# Balanced uses Haiku with moderate temperature for balanced assessment
RISK_MODEL = "claude-haiku-4-5"


def create_balanced_perspective(model: str = RISK_MODEL):
    """Factory for the balanced Risk Squad perspective.

    The balanced perspective evaluates each trade with neutral weighting of upside
    and downside considerations. It uses a fresh Haiku instance with temperature=0.3
    (balanced, moderate randomness).

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
                               with only the balanced_verdict field populated
    """

    def balanced_node(state: dict) -> dict:
        """Evaluate trade risk from a balanced perspective.

        Reads pre-computed portfolio context and produces a binary VETO/APPROVE
        verdict balancing both opportunities and risks.

        Args:
            state: AgentState dict with portfolio_*_value, risk_concentration_pct, etc.

        Returns:
            Dict with balanced_verdict (RiskDecision.model_dump()) or REJECTED dict on error
        """
        ticker = state.get("company_of_interest", "UNKNOWN")
        proposed_side = state.get("proposed_side", "Hold")
        portfolio_total_value = state.get("portfolio_total_value", 0.0)
        existing_position_value = state.get("existing_position_value", 0.0)
        proposed_notional_usd = state.get("proposed_notional_usd", 0.0)
        risk_concentration_pct = state.get("risk_concentration_pct", 1.0)
        risk_max_concentration_pct = get_config().get("risk_max_concentration_pct", 0.05)

        logger.info(
            f"Balanced perspective: Evaluating {ticker} {proposed_side} "
            f"(concentration={risk_concentration_pct:.4f}, threshold={risk_max_concentration_pct:.4f})"
        )

        # Read prior reflection from the risk layer (before prompt construction)
        reflection = read_reflection_for_ticker("risk", ticker)

        # Build reflection context if available
        reflection_context = ""
        if reflection is not None:
            reflection_context = f"""
## Prior Risk Assessment: {ticker}
- **Decision Date**: {reflection.decision_date}
- **Decision**: {reflection.decision_verdict}
- **Realized Return**: {reflection.realized_return:+.2%}
- **Classification**: {reflection.classification.upper()}
- **Lesson**: {reflection.lesson_text}

Weigh this prior outcome alongside both risk and opportunity.

"""

        # Build balanced prompt with neutral weighting of risk and opportunity
        prompt = f"""You are a balanced portfolio risk manager tasked with a binary VETO or APPROVE decision
on a proposed trade. Your role is to weigh both upside opportunity and downside risk fairly.

**Ticker**: {ticker}
**Proposed Action**: {proposed_side}

**Portfolio Context** (pre-computed, do NOT estimate):
- Total Portfolio Equity: ${portfolio_total_value:,.2f}
- Existing Position Value: ${existing_position_value:,.2f}
- Proposed Position Size: ${proposed_notional_usd:,.2f}
- Total Concentration After Trade: {risk_concentration_pct:.4f} ({risk_concentration_pct*100:.2f}%)
- Concentration Threshold: {risk_max_concentration_pct:.4f} ({risk_max_concentration_pct*100:.2f}%)

{reflection_context}

**Your Task**:
1. Balance risk and opportunity: what is the risk/reward ratio?
2. Assess concentration pragmatically: is the sizing reasonable for this portfolio?
3. Consider diversification: does this position fit within overall portfolio strategy?
4. Make a binary decision: VETO (unbalanced risk) or APPROVE (acceptable risk/reward)

**Output Requirements**:
- verdict: Either "VETO" or "APPROVE"
- confidence: Your confidence in this decision (0.0–1.0)
- reasoning: 100-300 word explanation grounded in the exact pre-computed values above, balancing both sides
- risk_factors: Specific concerns if any
- cited_concentration_pct: Repeat the Total Concentration value verbatim from above
""" + get_language_instruction()

        # Create a fresh LLM instance (D-02: NOT shared, NOT reused, D-03: temperature=0.3)
        logger.info(f"Balanced: Creating fresh Haiku instance for {ticker} (D-02, D-03)")
        try:
            balanced_llm = create_llm_client(
                "anthropic",
                model,
                temperature=0.3,
                max_tokens=1000,
            ).get_llm()
        except Exception as exc:
            logger.exception(f"Balanced: Failed to create LLM client: {exc}")
            return {
                "balanced_verdict": {
                    "verdict": "VETO",
                    "confidence": 0.0,
                    "reasoning": f"REJECTED: Failed to create LLM client: {exc}",
                    "risk_factors": ["llm_error"],
                    "cited_concentration_pct": 0.0,
                }
            }

        # Bind LLM to structured output
        structured_llm = bind_structured(balanced_llm, RiskDecision, "BalancedPerspective")

        # Invoke structured LLM (no free-text fallback — reject on failure per D-11)
        if structured_llm is None:
            logger.error("Balanced: structured output not supported by provider")
            return {
                "balanced_verdict": {
                    "verdict": "VETO",
                    "confidence": 0.0,
                    "reasoning": "REJECTED: provider does not support structured output",
                    "risk_factors": ["llm_error"],
                    "cited_concentration_pct": 0.0,
                }
            }

        try:
            verdict = structured_llm.invoke(prompt)
            if verdict is None:
                logger.error("Balanced: structured output returned None")
                return {
                    "balanced_verdict": {
                        "verdict": "VETO",
                        "confidence": 0.0,
                        "reasoning": "REJECTED: structured output returned no result",
                        "risk_factors": ["llm_error"],
                        "cited_concentration_pct": 0.0,
                    }
                }

            # Success: return the verdict (D-02: ONLY balanced_verdict, never touch others)
            logger.info(f"Balanced: Produced {verdict.verdict} verdict for {ticker}")
            return {"balanced_verdict": verdict.model_dump()}

        except Exception as exc:
            logger.exception(f"Balanced: LLM invocation failed: {exc}")
            return {
                "balanced_verdict": {
                    "verdict": "VETO",
                    "confidence": 0.0,
                    "reasoning": f"REJECTED: LLM invocation failed: {exc}",
                    "risk_factors": ["llm_error"],
                    "cited_concentration_pct": 0.0,
                }
            }

    return balanced_node
