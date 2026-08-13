"""Source-code isolation test suite for reflection readers across all 5 node factories (D-01, D-09).

This test suite verifies, via source inspection, that the Phase 5 reflection mechanism
maintains strict per-layer isolation across all five reflection-reading node factories:
- Research layer: create_momentum_llm_node
- Auditor layer: create_auditor_llm_node
- Risk layer (3 perspectives):
  - create_conservative_perspective
  - create_balanced_perspective
  - create_aggressive_perspective

D-01 (per-layer isolation) is the highest-stakes boundary: the Auditor must never read
Research's reflection store (that would silently reintroduce the exact TradeTrap coupling
AUDIT-02 exists to prevent). Similarly, Risk perspectives must never read Research's or
Auditor's stores. This suite proves, by source inspection, that all five node factories
respect their designated layer boundaries.

D-09 (memory-log guard) is extended from Auditor-only (test_auditor_isolation.py::test_auditor_never_touches_memory_log)
to cover all five factories: no factory may reference the pre-existing shared TradingMemoryLog machinery.
"""

import inspect

import pytest

from tradingagents.agents.auditors.momentum_auditor import create_auditor_llm_node
from tradingagents.agents.researchers.momentum_researcher import create_momentum_llm_node
from tradingagents.agents.risk_mgmt.aggressive_perspective import (
    create_aggressive_perspective,
)
from tradingagents.agents.risk_mgmt.balanced_perspective import (
    create_balanced_perspective,
)
from tradingagents.agents.risk_mgmt.conservative_perspective import (
    create_conservative_perspective,
)

# ---------------------------------------------------------------------------
# Test 1: Research reads only research layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_research_reads_only_research_reflection():
    """Verify Research LLM node reads 'research' layer only, not 'auditor' or 'risk'."""
    source = inspect.getsource(create_momentum_llm_node)

    # Should contain "research" as the layer argument
    assert '"research"' in source, "Research should read from 'research' layer"

    # Should NOT contain references to auditor or risk layers
    assert "auditor_reflections" not in source, "Research should not reference auditor_reflections"
    assert "risk_reflections" not in source, "Research should not reference risk_reflections"


# ---------------------------------------------------------------------------
# Test 2: Auditor reads only auditor layer (D-01 highest-stakes boundary)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_auditor_reads_only_auditor_reflection():
    """Verify Auditor LLM node reads 'auditor' layer only, not 'research' or 'risk' (D-01).

    This is the single highest-stakes isolation boundary: a cross-read here would
    silently reintroduce the exact TradeTrap coupling AUDIT-02 was designed to prevent.
    """
    source = inspect.getsource(create_auditor_llm_node)

    # Should contain "auditor" as the layer argument
    assert '"auditor"' in source, "Auditor should read from 'auditor' layer"

    # Should NOT contain references to research or risk layers — this is not optional
    assert (
        "research_reflections" not in source
    ), "Auditor must not reference research_reflections (violates AUDIT-02, D-01)"
    assert (
        "risk_reflections" not in source
    ), "Auditor must not reference risk_reflections (violates D-01)"


# ---------------------------------------------------------------------------
# Test 3: Conservative reads only risk layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_conservative_reads_only_risk_reflection():
    """Verify conservative perspective reads 'risk' layer only, not 'research' or 'auditor'."""
    source = inspect.getsource(create_conservative_perspective)

    # Should contain "risk" as the layer argument
    assert '"risk"' in source, "Conservative should read from 'risk' layer"

    # Should NOT contain references to research or auditor layers
    assert (
        "research_reflections" not in source
    ), "Conservative should not reference research_reflections"
    assert (
        "auditor_reflections" not in source
    ), "Conservative should not reference auditor_reflections"


# ---------------------------------------------------------------------------
# Test 4: Balanced reads only risk layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_balanced_reads_only_risk_reflection():
    """Verify balanced perspective reads 'risk' layer only, not 'research' or 'auditor'."""
    source = inspect.getsource(create_balanced_perspective)

    # Should contain "risk" as the layer argument
    assert '"risk"' in source, "Balanced should read from 'risk' layer"

    # Should NOT contain references to research or auditor layers
    assert (
        "research_reflections" not in source
    ), "Balanced should not reference research_reflections"
    assert (
        "auditor_reflections" not in source
    ), "Balanced should not reference auditor_reflections"


# ---------------------------------------------------------------------------
# Test 5: Aggressive reads only risk layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aggressive_reads_only_risk_reflection():
    """Verify aggressive perspective reads 'risk' layer only, not 'research' or 'auditor'."""
    source = inspect.getsource(create_aggressive_perspective)

    # Should contain "risk" as the layer argument
    assert '"risk"' in source, "Aggressive should read from 'risk' layer"

    # Should NOT contain references to research or auditor layers
    assert (
        "research_reflections" not in source
    ), "Aggressive should not reference research_reflections"
    assert (
        "auditor_reflections" not in source
    ), "Aggressive should not reference auditor_reflections"


# ---------------------------------------------------------------------------
# Test 6: No reflection reader touches shared memory log (D-09 extended guard)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_no_reflection_reader_touches_shared_memory_log():
    """Verify all 5 reflection-reading node factories never reference TradingMemoryLog machinery (D-09).

    Extends the existing Auditor-only guard (test_auditor_isolation.py::test_auditor_never_touches_memory_log)
    to cover all five new/modified node factories wired for reflection in Phase 5.
    The pre-existing shared memory system must remain untouched by every part of Phase 5.
    """
    factories = [
        ("Research", create_momentum_llm_node),
        ("Auditor", create_auditor_llm_node),
        ("Conservative", create_conservative_perspective),
        ("Balanced", create_balanced_perspective),
        ("Aggressive", create_aggressive_perspective),
    ]

    for name, factory in factories:
        source = inspect.getsource(factory)

        # Should NOT reference the pre-existing memory machinery
        assert (
            "memory_log" not in source
        ), f"{name} must not reference memory_log (D-09)"
        assert (
            "TradingMemoryLog" not in source
        ), f"{name} must not reference TradingMemoryLog (D-09)"
        assert (
            "get_past_context" not in source
        ), f"{name} must not reference get_past_context (D-09)"
