"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)

        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        # Read Risk Squad verdicts via .get() for safe defaults
        # (These fields are populated by the Risk Squad nodes; if not present,
        # use empty dicts as defaults so the node doesn't crash on old test fixtures)
        conservative = state.get("conservative_verdict", {})
        balanced = state.get("balanced_verdict", {})
        aggressive = state.get("aggressive_verdict", {})
        final_veto = state.get("final_veto", False)

        # Build risk summary showing each perspective's verdict + reasoning
        risk_summary = (
            f"Conservative: {conservative.get('verdict', 'N/A')} — "
            f"{conservative.get('reasoning', '')}\n"
            f"Balanced: {balanced.get('verdict', 'N/A')} — "
            f"{balanced.get('reasoning', '')}\n"
            f"Aggressive: {aggressive.get('verdict', 'N/A')} — "
            f"{aggressive.get('reasoning', '')}\n"
            f"Risk Squad Decision: {'VETOED' if final_veto else 'APPROVED'}"
        )

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Squad Verdicts:**
{risk_summary}

---

Be decisive and ground every conclusion in specific evidence from the analysts.

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        return {
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
