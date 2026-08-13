"""Test that Momentum nodes are correctly wired into the TradingAgentsGraph.

This test verifies the graph topology without compiling or invoking any LLM.
It ensures that the momentum research pipeline (Compute -> LLM -> Validate) is
properly integrated into the main workflow, connecting from the last analyst
to the Auditor chain (Phase 05.1 post-wiring).
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup


@pytest.mark.unit
def test_momentum_nodes_in_graph():
    """Test that all three momentum nodes exist in the compiled workflow."""
    # Setup mocks
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    # Build the graph (without compiling)
    workflow = graph_setup.setup_graph()

    # Assert momentum nodes exist
    momentum_nodes = {"Momentum Compute", "Momentum LLM", "Momentum Validate"}
    assert momentum_nodes <= set(workflow.nodes), (
        f"Missing momentum nodes. Found: {set(workflow.nodes)}"
    )


@pytest.mark.unit
def test_momentum_edges_in_graph():
    """Test that momentum nodes are correctly wired together and to Auditor Compute."""
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Assert edges between momentum nodes
    expected_edges = [
        ("Momentum Compute", "Momentum LLM"),
        ("Momentum LLM", "Momentum Validate"),
        ("Momentum Validate", "Auditor Compute"),
    ]

    for source, target in expected_edges:
        assert (source, target) in workflow.edges, (
            f"Missing edge: {source} -> {target}. "
            f"Available edges: {workflow.edges}"
        )


@pytest.mark.unit
def test_last_analyst_edges_to_momentum():
    """Test that the last analyst's clear node edges into Momentum Compute, not Bull Researcher.

    This is a regression guard: if this test fails, the momentum subgraph is no longer
    in the critical path and theses are bypassing validation.
    """
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Get the last analyst's clear node
    plan = build_analyst_execution_plan(("market", "social", "news", "fundamentals"))
    last_analyst_clear_node = plan.specs[-1].clear_node

    # Assert it edges to Momentum Compute, not directly to Bull Researcher
    assert (last_analyst_clear_node, "Momentum Compute") in workflow.edges, (
        f"Last analyst clear node '{last_analyst_clear_node}' does not edge to "
        f"'Momentum Compute'. This means the momentum pipeline is not in the "
        f"critical path."
    )

    # Ensure there is no edge bypassing momentum
    assert (last_analyst_clear_node, "Bull Researcher") not in workflow.edges, (
        f"Found direct edge from '{last_analyst_clear_node}' to 'Bull Researcher', "
        f"bypassing momentum validation."
    )


@pytest.mark.unit
def test_momentum_validate_reaches_auditor_compute():
    """Test that momentum validation output reaches Auditor Compute directly (Phase 05.1 wiring).

    After Phase 05.1, momentum validation flows directly to the Auditor chain,
    replacing the retired Bull Researcher/Bear Researcher/Research Manager debate layer.
    """
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Walk from Momentum Validate to ensure it reaches Auditor Compute
    assert ("Momentum Validate", "Auditor Compute") in workflow.edges
