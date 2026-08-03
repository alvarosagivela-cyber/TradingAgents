"""Aggressive/opportunistic perspective node for the Risk Squad (Phase 4).

This module provides an aggressive perspective that evaluates trades with
a focus on upside opportunity, while still respecting hard concentration limits.

The aggressive perspective:
1. Uses a fresh Haiku LLM instance (temperature=0.4) per invocation
2. Reads only pre-computed portfolio context (never shared risk_debate_state)
3. Produces a structured RiskDecision verdict (VETO/APPROVE)
4. Writes only to its own aggressive_verdict field in AgentState

Note: The aggressive perspective is risk-tolerant, NOT risk-blind. It still recommends
VETO when concentration clearly breaches thresholds, even while weighing upside more heavily.
"""

from __future__ import annotations

import logging

from tradingagents.agents.schemas import RiskDecision
from tradingagents.agents.utils.structured import bind_structured
from tradingagents.dataflows.config import get_config
from tradingagents.llm_clients.factory import create_llm_client

logger = logging.getLogger(__name__)

# Aggressive uses Haiku with higher temperature for opportunity-focused assessment
RISK_MODEL = "claude-haiku-4-5"


def create_aggressive_perspective(model: str = RISK_MODEL):
    """Factory for the aggressive Risk Squad perspective.

    The aggressive perspective evaluates each trade with a focus on upside opportunity,
    weighing growth potential more heavily than downside risk. It uses a fresh Haiku
    instance with temperature=0.4 (risk-tolerant, higher randomness).

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
                               with only the aggressive_verdict field populated
    """

    def aggressive_node(state: dict) -> dict:
        """Evaluate trade risk from an aggressive perspective.

        Reads pre-computed portfolio context and produces a binary VETO/APPROVE
        verdict emphasizing upside opportunity while maintaining hard limits.

        Args:
            state: AgentState dict with portfolio_*_value, risk_concentration_pct, etc.

        Returns:
            Dict with aggressive_verdict (RiskDecision.model_dump()) or REJECTED dict on error
        """
        ticker = state.get("company_of_interest", "UNKNOWN")
        proposed_side = state.get("proposed_side", "Hold")
        portfolio_total_value = state.get("portfolio_total_value", 0.0)
        existing_position_value = state.get("existing_position_value", 0.0)
        proposed_notional_usd = state.get("proposed_notional_usd", 0.0)
        risk_concentration_pct = state.get("risk_concentration_pct", 1.0)
        risk_max_concentration_pct = get_config().get("risk_max_concentration_pct", 0.05)

        logger.info(
            f"Aggressive perspective: Evaluating {ticker} {proposed_side} "
            f"(concentration={risk_concentration_pct:.4f}, threshold={risk_max_concentration_pct:.4f})"
        )

        # Build aggressive prompt with focus on opportunity but hard boundaries
        prompt = f"""You are an aggressive portfolio manager tasked with a binary VETO or APPROVE decision
on a proposed trade. Your role is to maximize upside opportunity while respecting absolute risk boundaries.

**Ticker**: {ticker}
**Proposed Action**: {proposed_side}

**Portfolio Context** (pre-computed, do NOT estimate):
- Total Portfolio Equity: ${portfolio_total_value:,.2f}
- Existing Position Value: ${existing_position_value:,.2f}
- Proposed Position Size: ${proposed_notional_usd:,.2f}
- Total Concentration After Trade: {risk_concentration_pct:.4f} ({risk_concentration_pct*100:.2f}%)
- Concentration Threshold: {risk_max_concentration_pct:.4f} ({risk_max_concentration_pct*100:.2f}%)

**Your Task**:
1. Emphasize upside: what is the growth potential of this trade?
2. Assess concentration strictly: VETO if concentration clearly breaches the threshold (no exceptions)
3. Balance within acceptable bounds: can this position fit within the portfolio strategy?
4. Make a binary decision: VETO (violates hard limits) or APPROVE (opportunity outweighs risk within bounds)

**Critical Constraint**:
Even though you favor upside, you MUST recommend VETO if concentration >= threshold.
This is a hard boundary, not a soft guideline.

**Output Requirements**:
- verdict: Either "VETO" or "APPROVE"
- confidence: Your confidence in this decision (0.0–1.0)
- reasoning: 100-300 word explanation grounded in the exact pre-computed values above, emphasizing opportunity
- risk_factors: Specific concerns if concentration is near/at threshold
- cited_concentration_pct: Repeat the Total Concentration value verbatim from above
"""

        # Create a fresh LLM instance (D-02: NOT shared, NOT reused, D-03: temperature=0.4)
        logger.info(f"Aggressive: Creating fresh Haiku instance for {ticker} (D-02, D-03)")
        try:
            aggressive_llm = create_llm_client(
                "anthropic",
                model,
                temperature=0.4,
                max_tokens=500,
            ).get_llm()
        except Exception as exc:
            logger.exception(f"Aggressive: Failed to create LLM client: {exc}")
            return {
                "aggressive_verdict": {
                    "verdict": "VETO",
                    "confidence": 0.0,
                    "reasoning": f"REJECTED: Failed to create LLM client: {exc}",
                    "risk_factors": ["llm_error"],
                    "cited_concentration_pct": 0.0,
                }
            }

        # Bind LLM to structured output
        structured_llm = bind_structured(aggressive_llm, RiskDecision, "AggressivePerspective")

        # Invoke structured LLM (no free-text fallback — reject on failure per D-11)
        if structured_llm is None:
            logger.error("Aggressive: structured output not supported by provider")
            return {
                "aggressive_verdict": {
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
                logger.error("Aggressive: structured output returned None")
                return {
                    "aggressive_verdict": {
                        "verdict": "VETO",
                        "confidence": 0.0,
                        "reasoning": "REJECTED: structured output returned no result",
                        "risk_factors": ["llm_error"],
                        "cited_concentration_pct": 0.0,
                    }
                }

            # Success: return the verdict (D-02: ONLY aggressive_verdict, never touch others)
            logger.info(f"Aggressive: Produced {verdict.verdict} verdict for {ticker}")
            return {"aggressive_verdict": verdict.model_dump()}

        except Exception as exc:
            logger.exception(f"Aggressive: LLM invocation failed: {exc}")
            return {
                "aggressive_verdict": {
                    "verdict": "VETO",
                    "confidence": 0.0,
                    "reasoning": f"REJECTED: LLM invocation failed: {exc}",
                    "risk_factors": ["llm_error"],
                    "cited_concentration_pct": 0.0,
                }
            }

    return aggressive_node
