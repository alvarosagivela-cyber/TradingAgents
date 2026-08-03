"""Test D-04 topological guarantee: vetoed trades are structurally unreachable at Paper Execution.

This test verifies that the veto mechanism is implemented as a topological guarantee
(no unconditional edge to Paper Execution exists), not as a runtime if-statement.
A vetoed trade can NEVER reach the Paper Execution node, by design, because the only
path to Paper Execution is through the Portfolio Manager's conditional router, which
routes to "__end__" when final_veto=True.
"""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.agents.utils.agent_states import AgentState


@pytest.mark.unit
def test_portfolio_manager_routes_to_paper_execution_when_veto_false():
    """Test that should_execute_paper_trade returns 'Paper Execution' when final_veto=False."""
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    # State with final_veto=False (trade approved)
    state = {
        "final_veto": False,
        "messages": [],
        "company_of_interest": "TEST",
        "asset_type": "stock",
        "instrument_context": "",
        "trade_date": "2026-08-04",
        "past_context": "",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
    }

    result = conditional_logic.should_execute_paper_trade(state)
    assert result == "Paper Execution", (
        f"Expected 'Paper Execution' when final_veto=False, got '{result}'"
    )


@pytest.mark.unit
def test_portfolio_manager_routes_to_end_when_veto_true():
    """Test that should_execute_paper_trade returns '__end__' when final_veto=True."""
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    # State with final_veto=True (trade vetoed)
    state = {
        "final_veto": True,
        "messages": [],
        "company_of_interest": "TEST",
        "asset_type": "stock",
        "instrument_context": "",
        "trade_date": "2026-08-04",
        "past_context": "",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
    }

    result = conditional_logic.should_execute_paper_trade(state)
    assert result == "__end__", (
        f"Expected '__end__' when final_veto=True, got '{result}'"
    )


@pytest.mark.unit
def test_no_unconditional_edge_to_paper_execution():
    """Test that there is NO unconditional edge from Portfolio Manager to Paper Execution.

    This is the critical structural guarantee (D-04). The ONLY way to reach Paper Execution
    is through the conditional router. An unconditional edge would allow bypassing the veto.
    """
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Assert there is NO unconditional edge from Portfolio Manager to Paper Execution
    assert ("Portfolio Manager", "Paper Execution") not in workflow.edges, (
        f"Found unconditional edge from 'Portfolio Manager' to 'Paper Execution'. "
        f"This violates D-04: vetoed trades can bypass the veto by using the unconditional edge."
    )

    # The only edge from Portfolio Manager must be a conditional edge
    # (verified by the presence of conditional routing logic)
    portfolio_manager_edges = [
        (src, tgt) for src, tgt in workflow.edges if src == "Portfolio Manager"
    ]
    assert len(portfolio_manager_edges) == 0, (
        f"Portfolio Manager has unconditional edges (not conditional): {portfolio_manager_edges}"
    )


@pytest.mark.unit
def test_paper_execution_only_reachable_via_approved_veto():
    """Test that Paper Execution is only reachable when final_veto=False.

    The conditional edge returns "__end__" when final_veto=True, making Paper Execution
    topologically unreachable. When final_veto=False, it returns "Paper Execution", but
    the veto must be set correctly by the Risk Aggregator before Portfolio Manager runs.
    """
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Paper Execution must exist as a node
    assert "Paper Execution" in set(workflow.nodes), (
        f"Paper Execution node does not exist. "
        f"Available nodes: {set(workflow.nodes)}"
    )

    # Paper Execution must unconditionally edge to END (terminal state)
    assert ("Paper Execution", "__end__") in workflow.edges, (
        f"Paper Execution does not edge to '__end__'. "
        f"The graph structure is incomplete."
    )

    # Paper Execution can ONLY be reached from Portfolio Manager via conditional router
    # This is verified by:
    # 1. No other node edges to Paper Execution (structural check)
    # 2. Portfolio Manager uses conditional routing (already tested)
    incoming_edges_to_paper_execution = [
        (src, tgt) for src, tgt in workflow.edges if tgt == "Paper Execution"
    ]
    assert len(incoming_edges_to_paper_execution) == 0, (
        f"Paper Execution has incoming edges that don't go through conditional routing: "
        f"{incoming_edges_to_paper_execution}. "
        f"This violates the structural veto guarantee."
    )


@pytest.mark.unit
def test_routing_always_returns_valid_path_map_key():
    """Test that should_execute_paper_trade always returns a key present in PORTFOLIO_MANAGER_PATH_MAP.

    This ensures no routing typos can crash the graph with an invalid node reference.
    """
    from tradingagents.graph.setup import PORTFOLIO_MANAGER_PATH_MAP

    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    # Test with veto=False
    state_approved = {"final_veto": False, "messages": []}
    result_approved = conditional_logic.should_execute_paper_trade(state_approved)
    assert result_approved in PORTFOLIO_MANAGER_PATH_MAP, (
        f"Router returned '{result_approved}', "
        f"which is not in PORTFOLIO_MANAGER_PATH_MAP: {PORTFOLIO_MANAGER_PATH_MAP}"
    )

    # Test with veto=True
    state_vetoed = {"final_veto": True, "messages": []}
    result_vetoed = conditional_logic.should_execute_paper_trade(state_vetoed)
    assert result_vetoed in PORTFOLIO_MANAGER_PATH_MAP, (
        f"Router returned '{result_vetoed}', "
        f"which is not in PORTFOLIO_MANAGER_PATH_MAP: {PORTFOLIO_MANAGER_PATH_MAP}"
    )

    # Test with missing final_veto (defaults to False)
    state_missing = {"messages": []}
    result_missing = conditional_logic.should_execute_paper_trade(state_missing)
    assert result_missing in PORTFOLIO_MANAGER_PATH_MAP, (
        f"Router returned '{result_missing}' with missing final_veto, "
        f"which is not in PORTFOLIO_MANAGER_PATH_MAP: {PORTFOLIO_MANAGER_PATH_MAP}"
    )
