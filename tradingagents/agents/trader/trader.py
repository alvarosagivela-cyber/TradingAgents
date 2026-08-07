"""Trader: deterministic mapping from Auditor's comparison_result to transaction proposal.

Phase 05.1 D-04: The Trader is no longer an LLM node. It is a pure function
that implements D-03's 3-case mapping:
  - comparison_result == "match" -> thesis_verdict passes through
  - comparison_result == "mismatch" -> veto to Hold (refutation is never averaged)
  - comparison_result == "not_reached" -> fail-closed to Hold (incomplete audit is never implicit approval)

This removes one LLM call per decision cycle and ensures the Auditor's verdict
gates execution, not just informs it.
"""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderAction, TraderProposal, render_trader_proposal


def _map_comparison_to_action(comparison_result: str, thesis_verdict: str) -> TraderAction:
    """Pure function implementing D-03's comparison_result -> action mapping.

    Args:
        comparison_result: One of "match" | "mismatch" | "not_reached"
        thesis_verdict: The research verdict ("Buy" | "Hold" | "Sell"), or empty string

    Returns:
        TraderAction: The deterministic action mapped from the comparison result
    """
    if comparison_result == "match":
        # Research verdict confirmed by Auditor, passes through
        try:
            return TraderAction(thesis_verdict)
        except ValueError:
            # thesis_verdict is empty or malformed; fail-closed
            return TraderAction.HOLD
    else:
        # comparison_result == "mismatch" or "not_reached"
        # Refutation is a veto (never averaged). Incomplete audit is never implicit approval.
        return TraderAction.HOLD


def create_trader():
    """Factory for the deterministic Trader node (D-04).

    The Trader reads the Auditor's comparison_result and the research thesis_verdict,
    then applies D-03's 3-case mapping to produce a transaction proposal.
    No LLM calls, no arguments to the factory.

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
    """

    def trader_node(state, name):
        comparison_result = state.get("comparison_result", "not_reached")
        thesis_verdict = state.get("thesis_verdict", "")

        # Apply the deterministic mapping
        action = _map_comparison_to_action(comparison_result, thesis_verdict)

        # Build reasoning that names the mapping and the inputs it consumed
        reasoning = (
            f"Deterministic mapping from Auditor's comparison_result='{comparison_result}' "
            f"and thesis_verdict='{thesis_verdict}': action={action.value} "
            f"(D-03: match→pass-through, mismatch→hold, not_reached→hold)"
        )

        # Construct the proposal
        proposal = TraderProposal(
            action=action,
            reasoning=reasoning,
            entry_price=None,
            stop_loss=None,
            position_sizing=None,
        )

        # Render to markdown (same format as the old LLM node, so downstream code doesn't change)
        trader_plan = render_trader_proposal(proposal)

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
