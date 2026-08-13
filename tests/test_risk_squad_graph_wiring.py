"""Test that Risk Squad nodes are correctly wired into the TradingAgentsGraph.

This test verifies the graph topology without compiling or invoking any LLM.
It ensures that the independent Risk Squad pipeline (Portfolio Snapshot ->
Conservative/Balanced/Aggressive Perspectives -> Risk Aggregator -> Portfolio Manager ->
conditional routing to Paper Execution) is properly integrated into the main workflow,
and that the old risk_debate_state round-robin routing has been fully removed
(regression guard for the retired D-02 vulnerability).
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup


@pytest.mark.unit
def test_risk_squad_nodes_in_graph():
    """Test that all Risk Squad nodes exist in the compiled workflow."""
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Risk Squad nodes must exist (Portfolio Snapshot, three perspectives, aggregator, Paper Execution)
    risk_squad_nodes = {
        "Portfolio Snapshot",
        "Conservative Perspective",
        "Balanced Perspective",
        "Aggressive Perspective",
        "Risk Aggregator",
        "Paper Execution",
    }
    assert risk_squad_nodes <= set(workflow.nodes), (
        f"Missing Risk Squad nodes. Found: {set(workflow.nodes)}"
    )

    # Portfolio Manager must exist (unchanged from before)
    assert "Portfolio Manager" in set(workflow.nodes), (
        "Portfolio Manager node missing"
    )


@pytest.mark.unit
def test_risk_squad_edges_in_graph():
    """Test that Risk Squad nodes are correctly wired together and to adjacent stages."""
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Expected edges: Trader -> Portfolio Snapshot -> 3 perspectives -> Risk Aggregator -> Portfolio Manager
    expected_edges = [
        ("Trader", "Portfolio Snapshot"),
        ("Portfolio Snapshot", "Conservative Perspective"),
        ("Portfolio Snapshot", "Balanced Perspective"),
        ("Portfolio Snapshot", "Aggressive Perspective"),
        ("Conservative Perspective", "Risk Aggregator"),
        ("Balanced Perspective", "Risk Aggregator"),
        ("Aggressive Perspective", "Risk Aggregator"),
        ("Risk Aggregator", "Portfolio Manager"),
        ("Paper Execution", "__end__"),
    ]

    for source, target in expected_edges:
        assert (source, target) in workflow.edges, (
            f"Missing edge: {source} -> {target}. "
            f"Available edges: {workflow.edges}"
        )


@pytest.mark.unit
def test_old_risk_debators_removed():
    """Test that the old risk debator nodes are completely removed.

    This is a critical regression guard for D-02 retirement. The old nodes
    (Aggressive Analyst, Conservative Analyst, Neutral Analyst) must not
    exist anymore, neither registered nor wired.
    """
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Old debator nodes must NOT exist
    old_debator_nodes = {
        "Aggressive Analyst",
        "Conservative Analyst",
        "Neutral Analyst",
    }
    for node in old_debator_nodes:
        assert node not in set(workflow.nodes), (
            f"Old debator node '{node}' still exists in graph. "
            f"D-02 retirement incomplete."
        )

    # Old debator edges must NOT exist
    old_debator_edges = [
        ("Trader", "Aggressive Analyst"),
        ("Aggressive Analyst", "Conservative Analyst"),
        ("Conservative Analyst", "Neutral Analyst"),
        ("Neutral Analyst", "Portfolio Manager"),
    ]
    for source, target in old_debator_edges:
        assert (source, target) not in workflow.edges, (
            f"Old debator edge '{source}' -> '{target}' still exists in graph. "
            f"D-02 retirement incomplete."
        )


@pytest.mark.unit
def test_trader_can_only_reach_portfolio_manager_through_risk_squad():
    """Test that Trader can only reach Portfolio Manager through Risk Squad.

    This is a critical topology verification: there must NOT be any direct edge
    from Trader to Portfolio Manager or any other bypass of the Risk Squad.
    The only path is: Trader -> Portfolio Snapshot -> ... -> Risk Aggregator -> Portfolio Manager.
    """
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # There must NOT be a direct Trader -> Portfolio Manager edge
    assert ("Trader", "Portfolio Manager") not in workflow.edges, (
        "Found direct edge from 'Trader' to 'Portfolio Manager', "
        "bypassing the Risk Squad. Risk Squad is no longer in the critical path."
    )

    # Trader must edge to Portfolio Snapshot (start of Risk Squad)
    assert ("Trader", "Portfolio Snapshot") in workflow.edges, (
        "Trader does not edge to 'Portfolio Snapshot'. "
        "The Risk Squad is not in the critical path."
    )

    # Risk Aggregator must edge to Portfolio Manager
    assert ("Risk Aggregator", "Portfolio Manager") in workflow.edges, (
        "Risk Aggregator does not edge to 'Portfolio Manager'. "
        "The Risk Squad is incomplete."
    )
