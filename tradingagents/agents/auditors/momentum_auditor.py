"""LLM and compute nodes for the independent Auditor layer (Phase 3).

This module provides the core Auditor pipeline:
1. Compute node: deterministic Python momentum calculation (independent fetch, AUDIT-01)
2. LLM node: reasoning under an asymmetric refutation mandate with a fresh Sonnet instance (AUDIT-02, D-03/D-04)

The Auditor is designed as a genuinely independent check on Research: it reads only the Research
verdict (for comparison) and its own computed momentum metrics, never Research's cached reports
or shared LLM state. The LLM prompt is framed asymmetrically — explicitly tasking the model with
finding reasons the Research thesis could be WRONG, not neutral ticker analysis.
"""

from __future__ import annotations

import logging

from tradingagents.agents.reflectors.reflection_reader import read_reflection_for_ticker
from tradingagents.agents.schemas import AuditorVerdict
from tradingagents.agents.utils.momentum_calculator import (
    compute_momentum,
    derive_confidence_level,
)
from tradingagents.agents.utils.structured import bind_structured
from tradingagents.llm_clients.factory import create_llm_client

logger = logging.getLogger(__name__)

# Auditor uses Sonnet for deeper reasoning on refutation (distinct from Research's Haiku, D-03)
AUDITOR_MODEL = "claude-sonnet-5"


def create_auditor_compute_node():
    """Factory for the Auditor's momentum compute node.

    Returns a node function that:
    1. Reads state["company_of_interest"] and state["trade_date"]
    2. Calls compute_momentum(...) to deterministically calculate metrics (AUDIT-01: independent invocation)
    3. Returns an update dict with the computed values

    The compute node fails open (phase1_status=fail) on any data fetch error; the LLM node
    later short-circuits when it sees this status (D-02).

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
    """

    def compute_node(state: dict) -> dict:
        """Compute momentum metrics for the current ticker (Auditor's own independent fetch).

        Args:
            state: AgentState dict with company_of_interest, trade_date

        Returns:
            Dict with auditor_phase1_status, auditor_phase1_failure_reason, auditor_retorno_12_1,
            auditor_z_score, auditor_confidence_level
        """
        ticker = state.get("company_of_interest")
        trade_date = state.get("trade_date")

        logger.info(f"Auditor: Computing momentum for {ticker} as of {trade_date} (independent fetch, AUDIT-01)")

        metrics = compute_momentum(ticker, trade_date)

        if not metrics.valid:
            logger.warning(f"Auditor: Momentum computation failed for {ticker} (insufficient OHLCV history)")
            return {
                "auditor_phase1_status": "fail",
                "auditor_phase1_failure_reason": "insufficient_ohlcv_history",
                "auditor_retorno_12_1": 0.0,
                "auditor_z_score": 0.0,
                "auditor_confidence_level": "low",
            }

        return {
            "auditor_phase1_status": "pass",
            "auditor_phase1_failure_reason": "",
            "auditor_retorno_12_1": metrics.retorno_12_1,
            "auditor_z_score": metrics.z_score,
            "auditor_confidence_level": derive_confidence_level(metrics.z_score),
        }

    return compute_node


def create_auditor_llm_node(model: str = AUDITOR_MODEL):
    """Factory for the Auditor's LLM node.

    The Auditor LLM node:
    1. Short-circuits if auditor_phase1_status != "pass" (D-02: fail-open)
    2. Otherwise, builds a prompt with an asymmetric refutation mandate (D-04)
    3. Creates a FRESH LLM instance via create_llm_client(...) on every call (AUDIT-02, D-03)
    4. Invokes the LLM with AuditorVerdict structured output (AUDIT-02)
    5. Rejects (returns INCONCLUSIVE) on structured-output failure (D-02)

    The prompt explicitly instructs the model to find concrete reasons the Research thesis
    could be WRONG, not to perform neutral analysis. This asymmetry is critical to the
    independent verification value (D-04).

    Args:
        model: Model name to use for Auditor reasoning (default: claude-sonnet-5 for deeper analysis, D-03)

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
    """

    def llm_node(state: dict) -> dict:
        """Generate Auditor refutation verdict based on independent momentum metrics.

        Args:
            state: AgentState dict with auditor_phase1_status, thesis_verdict, auditor computed fields

        Returns:
            Dict with auditor_verdict, auditor_reasoning, auditor_refutation_criterion, auditor_data_points
            or INCONCLUSIVE dict if phase1_status != "pass" or LLM fails
        """
        ticker = state.get("company_of_interest")
        phase1_status = state.get("auditor_phase1_status", "fail")

        # Short-circuit if compute phase failed (D-02: fail-open, never invoke LLM)
        if phase1_status != "pass":
            logger.info(
                f"Auditor: Phase 1 failed for {ticker} ({phase1_status}); skipping LLM (fail-open, D-02)"
            )
            return {
                "auditor_verdict": "INCONCLUSIVE",
                "auditor_reasoning": f"REJECTED: Auditor's own momentum calculation failed ({state.get('auditor_phase1_failure_reason')}), cannot produce refutation",
                "auditor_refutation_criterion": "",
                "auditor_data_points": {},
            }

        # Extract Auditor's own computed metrics (AUDIT-01: independent)
        auditor_retorno_12_1 = state.get("auditor_retorno_12_1", 0.0)
        auditor_z_score = state.get("auditor_z_score", 0.0)
        research_verdict = state.get("thesis_verdict", "")

        # Read prior reflection from the auditor layer (before prompt construction)
        reflection = read_reflection_for_ticker("auditor", ticker)

        # Build reflection context if available
        reflection_context = ""
        if reflection is not None:
            reflection_context = f"""
## Prior Auditor Analysis: {ticker}
- **Decision Date**: {reflection.decision_date}
- **Decision**: {reflection.decision_verdict}
- **Realized Return**: {reflection.realized_return:+.2%}
- **Classification**: {reflection.classification.upper()}
- **Lesson**: {reflection.lesson_text}

Incorporate this prior analysis into your refutation reasoning.

"""

        # Build prompt with asymmetric refutation mandate (D-04)
        # The key difference from Research's neutral analysis: explicit instruction to find reasons
        # the Research thesis could be WRONG.
        prompt = f"""You are an independent Auditor tasked with refuting investment theses.

**Your Role**: NOT neutral analysis. Your explicit job is to find concrete, ruthless reasons
the Research thesis COULD BE WRONG. Treat the Research verdict as a claim to scrutinize,
not as evidence to confirm.

**Ticker**: {ticker}

**Auditor's Own Momentum Metrics** (computed independently of Research):
- 12-Month Momentum (skip-month formula): {auditor_retorno_12_1:.4f} ({auditor_retorno_12_1*100:.2f}%)
- Z-Score (vs 5-year distribution): {auditor_z_score:.2f}

**Research's Verdict** (what you are tasked to refute):
{research_verdict}

{reflection_context}

Your task:
1. Treat the Research verdict as a hypothesis to test, not as established fact
2. Use YOUR OWN computed momentum metrics (above) as evidence, not Research's interpretation
3. Find 2-3 concrete, specific reasons why the Research thesis could fail
4. Ground your refutation in measurable, falsifiable observations (not vague doubts)
5. State explicitly: what market condition would prove this thesis is wrong?
6. Cite the exact values (auditor_retorno_12_1 and auditor_z_score above) in your refutation
7. Produce a structured verdict: Buy, Hold, or Sell based on YOUR ruthless analysis

Output a structured verdict with:
- verdict: Buy, Hold, or Sell (your independent call, possibly opposite Research's)
- reasoning: 250-400 word explanation of why Research thesis could be wrong
- refutation_criterion: Concrete, falsifiable market observation that would disprove Research's thesis
- data_points: {{retorno_12_1: (the exact value from Auditor's metrics above), z_score: (the exact value from Auditor's metrics above)}}
"""

        # Create a fresh LLM instance (AUDIT-02, D-03: NOT shared quick_thinking_llm)
        logger.info(f"Auditor: Creating fresh Sonnet instance for {ticker} (AUDIT-02, D-03)")
        try:
            auditor_llm = create_llm_client(
                "anthropic",
                model,
                max_tokens=1000,
            ).get_llm()
        except Exception as exc:
            logger.exception(f"Auditor: Failed to create LLM client: {exc}")
            return {
                "auditor_verdict": "INCONCLUSIVE",
                "auditor_reasoning": f"REJECTED: Failed to create LLM client: {exc}",
                "auditor_refutation_criterion": "",
                "auditor_data_points": {},
            }

        # Bind LLM to structured output
        structured_llm = bind_structured(auditor_llm, AuditorVerdict, "Auditor")

        # Invoke structured LLM (no free-text fallback — reject on failure per D-11)
        if structured_llm is None:
            logger.error("Auditor: structured output not supported by provider")
            return {
                "auditor_verdict": "INCONCLUSIVE",
                "auditor_reasoning": "REJECTED: provider does not support structured output",
                "auditor_refutation_criterion": "",
                "auditor_data_points": {},
            }

        try:
            verdict = structured_llm.invoke(prompt)
            if verdict is None:
                logger.error("Auditor: structured output returned None")
                return {
                    "auditor_verdict": "INCONCLUSIVE",
                    "auditor_reasoning": "REJECTED: structured output returned no result",
                    "auditor_refutation_criterion": "",
                    "auditor_data_points": {},
                }

            # Success: return the verdict fields for comparison
            logger.info(f"Auditor: Produced {verdict.verdict.value} verdict for {ticker}")
            return {
                "auditor_verdict": verdict.verdict.value,
                "auditor_reasoning": verdict.reasoning,
                "auditor_refutation_criterion": verdict.refutation_criterion,
                "auditor_data_points": verdict.data_points,
            }

        except Exception as exc:
            logger.exception(f"Auditor: LLM invocation failed: {exc}")
            return {
                "auditor_verdict": "INCONCLUSIVE",
                "auditor_reasoning": f"REJECTED: LLM invocation failed: {exc}",
                "auditor_refutation_criterion": "",
                "auditor_data_points": {},
            }

    return llm_node
