"""Test that Auditor nodes are correctly wired into the TradingAgentsGraph.

This test verifies the graph topology without compiling or invoking any LLM.
It ensures that the independent Auditor pipeline (Compute -> LLM -> Compare) is
properly integrated into the main workflow, wired as the sole gate into Trader
(Phase 05.1 post-wiring: Momentum Validate flows directly to Auditor Compute),
and includes a data-dependency mutation test proving the Auditor's verdict
actually changes the Trader's output (not just edge presence, but causal data dependency).
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

    # Assert all 4 edges in the full chain (Phase 05.1 post-wiring):
    # Momentum Validate -> Auditor Compute -> Auditor LLM -> Auditor Compare -> Trader
    expected_edges = [
        ("Momentum Validate", "Auditor Compute"),
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
def test_auditor_is_the_sole_gate_into_trader():
    """Test that Auditor is the sole gate into Trader (Phase 05.1 topology).

    After Phase 05.1, Research Manager is not in the graph at all (not merely bypassed).
    Momentum Validate flows directly to Auditor Compute, which is the only path
    feeding into Trader. This is a stronger guarantee than mere edge absence:
    it asserts the node itself is disconnected.

    This regression guard ensures the Auditor is topologically mandatory for execution.
    """
    tool_nodes = {
        k: MagicMock() for k in ("market", "social", "news", "fundamentals")
    }
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    graph_setup = GraphSetup(MagicMock(), MagicMock(), tool_nodes, conditional_logic)

    workflow = graph_setup.setup_graph()

    # Assert Research Manager node is completely absent (stronger guarantee)
    assert "Research Manager" not in workflow.nodes, (
        f"Found 'Research Manager' in workflow.nodes. "
        f"Phase 05.1 should have disconnected this node entirely."
    )

    # Assert Momentum Validate flows directly to Auditor Compute
    assert ("Momentum Validate", "Auditor Compute") in workflow.edges, (
        f"Momentum Validate does not edge to 'Auditor Compute'. "
        f"The Auditor pipeline is not in the critical path."
    )

    # Assert Auditor Compare feeds directly to Trader
    assert ("Auditor Compare", "Trader") in workflow.edges, (
        f"Auditor Compare does not edge to 'Trader'. "
        f"The Auditor pipeline is incomplete."
    )


@pytest.mark.unit
def test_auditor_verdict_causally_gates_trader_proposed_side():
    """Test that the Auditor's comparison_result causally changes the Trader's output.

    This is the data-dependency mutation test (CONTEXT.md finding #2).
    It proves the Auditor's verdict doesn't just exist as a graph edge, but actually
    changes what the Trader proposes. The old test_research_manager_no_longer_edges_directly_to_trader
    only checked edge topology (false confidence that let the bug live undetected).
    This test verifies real causal dependency: holding thesis_verdict fixed and changing
    comparison_result from "match" to "mismatch" must change the proposed action from
    "Buy" to "Hold".

    Rationale: topology alone is not enough. A well-intentioned future refactoring could
    re-introduce a Trader bypass (reading investment_plan directly) while keeping the edge
    present. This test catches that class of regression: it asserts the Trader's output
    actually depends on the Auditor's verdict.
    """
    from tradingagents.agents.trader.trader import create_trader

    # Create the deterministic trader node (zero arguments, per Phase 05.1)
    trader = create_trader()

    # Base state with thesis_verdict and auditor_verdict held constant
    base_state = {
        "company_of_interest": "NVDA",
        "thesis_verdict": "Buy",
        "auditor_verdict": "Buy",
    }

    # Invoke trader with comparison_result = "match"
    state_match = {**base_state, "comparison_result": "match"}
    result_match = trader(state_match, name="Trader")
    plan_match = result_match["trader_investment_plan"]

    # Invoke trader with comparison_result = "mismatch" (auditor refutes)
    state_mismatch = {**base_state, "comparison_result": "mismatch"}
    result_mismatch = trader(state_mismatch, name="Trader")
    plan_mismatch = result_mismatch["trader_investment_plan"]

    # Assert the match case renders a BUY trailer
    assert "FINAL TRANSACTION PROPOSAL: **BUY**" in plan_match, (
        f"Expected BUY trailer in match case, got: {plan_match}"
    )

    # Assert the mismatch case renders a HOLD trailer
    assert "FINAL TRANSACTION PROPOSAL: **HOLD**" in plan_mismatch, (
        f"Expected HOLD trailer in mismatch case, got: {plan_mismatch}"
    )

    # Assert the two plans are materially different (not just whitespace)
    assert plan_match != plan_mismatch, (
        f"Expected different plans for match vs mismatch, but got identical output. "
        f"The Auditor's verdict is not causally gating the Trader's output."
    )
