"""Test that Auditor nodes are correctly wired into the TradingAgentsGraph.

This test verifies the graph topology without compiling or invoking any LLM.
It ensures that the independent Auditor pipeline (Compute -> LLM -> Compare) is
properly integrated into the main workflow, wired between Research Manager and Trader,
and that the direct bypass edge has been removed (regression guard for AI-SPEC Critical
Failure Mode #1/#3).
"""

import pytest
from unittest.mock import MagicMock

from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.conditional_logic import ConditionalLogic


@pytest.mark.unit
def test_auditor_nodes_in_graph():
    """Test that all three auditor nodes exist in the compiled workflow."""
    # Setup mocks
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    # Build the graph (without compiling)
    workflow = graph_setup.setup_graph()

    # Assert auditor nodes exist
    auditor_nodes = {"Auditor Compute", "Auditor LLM", "Auditor Compare"}
    assert auditor_nodes <= set(workflow.nodes), (
        f"Missing auditor nodes. Found: {set(workflow.nodes)}"
    )


@pytest.mark.unit
def test_auditor_edges_in_graph():
    """Test that auditor nodes are correctly wired together and to adjacent stages."""
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Assert all 4 edges in the full chain:
    # Research Manager -> Auditor Compute -> Auditor LLM -> Auditor Compare -> Trader
    expected_edges = [
        ("Research Manager", "Auditor Compute"),
        ("Auditor Compute", "Auditor LLM"),
        ("Auditor LLM", "Auditor Compare"),
        ("Auditor Compare", "Trader"),
    ]

    for source, target in expected_edges:
        assert (source, target) in workflow.edges, (
            f"Missing edge: {source} -> {target}. "
            f"Available edges: {workflow.edges}"
        )


@pytest.mark.unit
def test_research_manager_no_longer_edges_directly_to_trader():
    """Test that the Research Manager no longer bypasses the Auditor to reach Trader.

    This is a critical regression guard: if this test fails, the Auditor is no longer
    in the mandatory execution path, and Trader can bypass the independent verification.
    This directly tests AI-SPEC Critical Failure Modes #1 and #3.
    """
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Assert the old direct bypass edge is GONE
    assert ("Research Manager", "Trader") not in workflow.edges, (
        f"Found direct edge from 'Research Manager' to 'Trader', "
        f"which bypasses the Auditor. The Auditor is no longer in the critical path."
    )

    # Assert Trader is now topologically reachable from Research Manager ONLY through
    # the full Auditor chain (Compute -> LLM -> Compare)
    assert ("Research Manager", "Auditor Compute") in workflow.edges, (
        f"Research Manager does not edge to 'Auditor Compute'. "
        f"The Auditor pipeline is not in the critical path."
    )

    assert ("Auditor Compare", "Trader") in workflow.edges, (
        f"Auditor Compare does not edge to 'Trader'. "
        f"The Auditor pipeline is incomplete."
    )
