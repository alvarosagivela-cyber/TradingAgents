"""Source-code isolation test suite for Auditor agents (AUDIT-01/02, D-09 memory-log guard).

This test suite verifies, via source inspection, that the Auditor pipeline
never shares state or data sources with Research. It ensures the Auditor code
adheres to the independence requirements from the TradeTrap security model.

These tests complement the graph wiring tests (test_auditor_graph_wiring.py) by
confirming the actual implementation never references forbidden shared resources.
"""

import inspect
import pytest

from tradingagents.agents.auditors.momentum_auditor import (
    create_auditor_compute_node,
    create_auditor_llm_node,
)
from tradingagents.agents.auditors.thesis_comparator import create_auditor_compare_node


@pytest.mark.unit
def test_compute_node_never_reads_shared_reports():
    """Verify Auditor compute node never reads shared analyst reports (AUDIT-01).

    Forbidden patterns (shared state from Research layer):
    - state["market_report"]
    - state["sentiment_report"]
    - state["news_report"]
    - state["fundamentals_report"]

    The compute node should read only:
    - state["company_of_interest"]
    - state["trade_date"]
    """
    compute_node_factory = create_auditor_compute_node()
    source = inspect.getsource(compute_node_factory)

    forbidden_reports = [
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
    ]

    for report_field in forbidden_reports:
        # Ensure the field is not accessed via state["field"] or state.get("field")
        assert f'["{report_field}"]' not in source, (
            f"Compute node must not read state['{report_field}'] "
            f"(shared analyst report). This violates AUDIT-01 isolation."
        )
        assert f"['{report_field}']" not in source, (
            f"Compute node must not read state['{report_field}'] "
            f"(shared analyst report). This violates AUDIT-01 isolation."
        )
        assert f'.get("{report_field}")' not in source, (
            f"Compute node must not read state.get('{report_field}') "
            f"(shared analyst report). This violates AUDIT-01 isolation."
        )
        assert f".get('{report_field}')" not in source, (
            f"Compute node must not read state.get('{report_field}') "
            f"(shared analyst report). This violates AUDIT-01 isolation."
        )


@pytest.mark.unit
def test_compute_node_never_reads_shared_thesis_verdict():
    """Verify Auditor compute node never reads Research's thesis verdict prose (AUDIT-01).

    The compute node should NOT read:
    - state["thesis_verdict"]  (Research's prose verdict)

    (The LLM node reads thesis_verdict for comparison purposes, but compute does not.)
    """
    compute_node_factory = create_auditor_compute_node()
    source = inspect.getsource(compute_node_factory)

    # The compute node's inner function should not reference thesis_verdict
    # (it only reads company_of_interest and trade_date)
    assert '["thesis_verdict"]' not in source, (
        "Compute node must not read state['thesis_verdict'] "
        "(Research's verdict is for comparison only, not for compute)"
    )
    assert "['thesis_verdict']" not in source, (
        "Compute node must not read state['thesis_verdict'] "
        "(Research's verdict is for comparison only, not for compute)"
    )


@pytest.mark.unit
def test_llm_node_creates_isolated_llm_instance():
    """Verify Auditor LLM node creates a fresh LLM instance per call (AUDIT-02, D-03).

    Forbidden patterns (shared LLM state):
    - self.llm
    - self.quick_thinking_llm  (as a variable reference)
    - ChatAnthropic( without create_llm_client factory wrapper)

    Required pattern:
    - create_llm_client(...)
    """
    llm_node_factory = create_auditor_llm_node()
    source = inspect.getsource(llm_node_factory)

    # Ensure the code calls create_llm_client
    assert "create_llm_client" in source, (
        "LLM node must use create_llm_client factory to create a fresh instance per call (AUDIT-02, D-03)"
    )

    # Ensure it doesn't use pre-shared LLM instances
    # Check for assignment or usage patterns (not just mentions in comments)
    assert "auditor_llm = " in source or "auditor_llm=" in source, (
        "LLM node must create a local auditor_llm instance (AUDIT-02)"
    )

    assert "self.llm" not in source, (
        "LLM node must not use self.llm (would require shared closure). "
        "Use create_llm_client() instead (AUDIT-02)."
    )


@pytest.mark.unit
def test_llm_node_never_reads_shared_reports():
    """Verify Auditor LLM node never reads shared analyst reports (AUDIT-01).

    Forbidden patterns:
    - state["market_report"]
    - state["sentiment_report"]
    - state["news_report"]
    - state["fundamentals_report"]

    The LLM node should read only:
    - state["thesis_verdict"] (for comparison)
    - state["auditor_retorno_12_1"] (Auditor's own computed metric)
    - state["auditor_z_score"] (Auditor's own computed metric)
    - state["auditor_phase1_status"] (Auditor's own phase tracking)
    """
    llm_node_factory = create_auditor_llm_node()
    source = inspect.getsource(llm_node_factory)

    forbidden_reports = [
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
    ]

    for report_field in forbidden_reports:
        assert f'["{report_field}"]' not in source, (
            f"LLM node must not read state['{report_field}'] "
            f"(shared analyst report). This violates AUDIT-01 isolation."
        )
        assert f"['{report_field}']" not in source, (
            f"LLM node must not read state['{report_field}'] "
            f"(shared analyst report). This violates AUDIT-01 isolation."
        )
        assert f'.get("{report_field}")' not in source, (
            f"LLM node must not read state.get('{report_field}') "
            f"(shared analyst report). This violates AUDIT-01 isolation."
        )
        assert f".get('{report_field}')" not in source, (
            f"LLM node must not read state.get('{report_field}') "
            f"(shared analyst report). This violates AUDIT-01 isolation."
        )


@pytest.mark.unit
def test_compare_node_reads_only_verdict_fields():
    """Verify Auditor compare node reads only verdicts (AUDIT-03).

    Forbidden patterns (shared analyst reports):
    - state["market_report"]
    - state["sentiment_report"]
    - state["news_report"]
    - state["fundamentals_report"]

    Allowed patterns (verdicts for comparison):
    - state["thesis_verdict"] (Research's verdict)
    - state["auditor_verdict"] (Auditor's verdict)
    - state["auditor_data_points"] (Auditor's own data)
    """
    compare_node_factory = create_auditor_compare_node()
    source = inspect.getsource(compare_node_factory)

    forbidden_reports = [
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
    ]

    for report_field in forbidden_reports:
        assert f'["{report_field}"]' not in source, (
            f"Compare node must not read state['{report_field}'] "
            f"(shared analyst report). This violates AUDIT-03 isolation."
        )
        assert f"['{report_field}']" not in source, (
            f"Compare node must not read state['{report_field}'] "
            f"(shared analyst report). This violates AUDIT-03 isolation."
        )
        assert f'.get("{report_field}")' not in source, (
            f"Compare node must not read state.get('{report_field}') "
            f"(shared analyst report). This violates AUDIT-03 isolation."
        )
        assert f".get('{report_field}')" not in source, (
            f"Compare node must not read state.get('{report_field}') "
            f"(shared analyst report). This violates AUDIT-03 isolation."
        )


@pytest.mark.unit
def test_auditor_never_touches_memory_log():
    """Verify Auditor pipeline never reads/writes the shared memory log (D-09).

    Forbidden patterns:
    - memory_log
    - TradingMemoryLog
    - get_past_context

    The Auditor should never inject past reflections into its reasoning.
    This prevents persistent state corruption if the memory log is compromised.
    """
    compute_factory = create_auditor_compute_node()
    llm_factory = create_auditor_llm_node()
    compare_factory = create_auditor_compare_node()

    sources = [
        inspect.getsource(compute_factory),
        inspect.getsource(llm_factory),
        inspect.getsource(compare_factory),
    ]

    forbidden_patterns = [
        "memory_log",
        "TradingMemoryLog",
        "get_past_context",
    ]

    for source in sources:
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"Auditor factories must not reference '{pattern}'. "
                f"The Auditor must not touch the shared memory log (D-09). "
                f"This prevents persistent state injection if memory is compromised."
            )
