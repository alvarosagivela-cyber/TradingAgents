"""Source-code isolation test suite for Risk Squad perspective agents (RISK-02, D-02, D-03).

This test suite verifies, via source inspection, that each of the three Risk Squad
perspectives (conservative/balanced/aggressive) maintains strict isolation guarantees:
- No shared risk_debate_state between perspectives
- Each creates fresh LLM instances (not reused or shared)
- Each perspective writes only to its own verdict field
- Temperature distinctiveness is mechanically enforced (0.2/0.3/0.4)
- All perspectives read pre-computed context (risk_concentration_pct) for grounding

These tests complement the integration test (test_risk_squad_pipeline.py) by
confirming each perspective node is individually, structurally isolated — not just
narratively independent. This prevents regressions like reintroducing shared state
before it reaches the graph wiring layer.
"""

import inspect
import pytest

from tradingagents.agents.risk_mgmt.conservative_perspective import (
    create_conservative_perspective,
)
from tradingagents.agents.risk_mgmt.balanced_perspective import (
    create_balanced_perspective,
)
from tradingagents.agents.risk_mgmt.aggressive_perspective import (
    create_aggressive_perspective,
)


class TestConservativePerspectiveIsolation:
    """Source-inspection tests for conservative perspective isolation."""

    @pytest.mark.unit
    def test_conservative_never_reads_risk_debate_state(self):
        """Verify conservative perspective never reads shared risk_debate_state (RISK-02, D-02).

        Forbidden pattern (shared mutable state):
        - state["risk_debate_state"]
        - state.get("risk_debate_state")

        This prevents accidental coupling to shared debate context.
        """
        factory = create_conservative_perspective()
        source = inspect.getsource(factory)

        assert '["risk_debate_state"]' not in source, (
            "Conservative perspective must not read state['risk_debate_state'] "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert "['risk_debate_state']" not in source, (
            "Conservative perspective must not read state['risk_debate_state'] "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert '.get("risk_debate_state")' not in source, (
            "Conservative perspective must not read state.get('risk_debate_state') "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert ".get('risk_debate_state')" not in source, (
            "Conservative perspective must not read state.get('risk_debate_state') "
            "(shared mutable state). This violates RISK-02 isolation."
        )

    @pytest.mark.unit
    def test_conservative_creates_isolated_llm_instance(self):
        """Verify conservative perspective creates fresh LLM per call (D-02, D-03).

        Forbidden patterns (shared LLM state):
        - self.llm
        - self.quick_thinking_llm
        - ChatAnthropic( without create_llm_client wrapper)

        Required pattern:
        - create_llm_client(...)
        """
        factory = create_conservative_perspective()
        source = inspect.getsource(factory)

        # Ensure create_llm_client is called
        assert "create_llm_client" in source, (
            "Conservative perspective must use create_llm_client factory to create "
            "a fresh instance per call (D-02, D-03)"
        )

        # Ensure local variable is assigned
        assert "conservative_llm" in source, (
            "Conservative perspective must create a local conservative_llm instance (D-02)"
        )

        # Ensure no self.llm or pre-shared LLM references
        assert "self.llm" not in source, (
            "Conservative perspective must not use self.llm (would require shared closure). "
            "Use create_llm_client() instead (D-02)."
        )
        assert "self.quick_thinking_llm" not in source, (
            "Conservative perspective must not use self.quick_thinking_llm. "
            "Use create_llm_client() instead (D-02)."
        )

    @pytest.mark.unit
    def test_conservative_never_writes_other_verdict_fields(self):
        """Verify conservative perspective writes only to conservative_verdict (D-02).

        Forbidden patterns (writing to other perspectives' fields):
        - balanced_verdict
        - aggressive_verdict

        Required pattern:
        - conservative_verdict
        """
        factory = create_conservative_perspective()
        source = inspect.getsource(factory)

        # Check return statements only write conservative_verdict
        assert '"conservative_verdict"' in source or "'conservative_verdict'" in source, (
            "Conservative perspective must return conservative_verdict "
            "(D-02 field isolation)."
        )

        # Ensure it never reads or writes other verdicts
        assert 'balanced_verdict' not in source, (
            "Conservative perspective must not reference balanced_verdict "
            "(D-02: each perspective owns only its own verdict)."
        )
        assert 'aggressive_verdict' not in source, (
            "Conservative perspective must not reference aggressive_verdict "
            "(D-02: each perspective owns only its own verdict)."
        )

    @pytest.mark.unit
    def test_conservative_uses_temperature_0_2(self):
        """Verify conservative perspective uses temperature=0.2 literal (RISK-02, D-03).

        Temperature distinctiveness is mechanically enforced in source code,
        not just stated in documentation. This ensures each perspective has
        measurably different decision-making stochasticity.
        """
        factory = create_conservative_perspective()
        source = inspect.getsource(factory)

        assert "0.2" in source, (
            "Conservative perspective must contain the literal temperature=0.2 "
            "in its source code (RISK-02: provable distinctiveness by construction)."
        )

    @pytest.mark.unit
    def test_conservative_reads_pre_computed_context(self):
        """Verify conservative perspective reads risk_concentration_pct (grounded inputs).

        Forbidden pattern (hallucination risk):
        - Free-text-only prompt with no numeric input validation

        Required pattern:
        - risk_concentration_pct (pre-computed, not hallucinated)
        """
        factory = create_conservative_perspective()
        source = inspect.getsource(factory)

        assert "risk_concentration_pct" in source, (
            "Conservative perspective must read pre-computed risk_concentration_pct "
            "to ground its reasoning in real portfolio data (prevents hallucination)."
        )


class TestBalancedPerspectiveIsolation:
    """Source-inspection tests for balanced perspective isolation."""

    @pytest.mark.unit
    def test_balanced_never_reads_risk_debate_state(self):
        """Verify balanced perspective never reads shared risk_debate_state (RISK-02, D-02)."""
        factory = create_balanced_perspective()
        source = inspect.getsource(factory)

        assert '["risk_debate_state"]' not in source, (
            "Balanced perspective must not read state['risk_debate_state'] "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert "['risk_debate_state']" not in source, (
            "Balanced perspective must not read state['risk_debate_state'] "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert '.get("risk_debate_state")' not in source, (
            "Balanced perspective must not read state.get('risk_debate_state') "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert ".get('risk_debate_state')" not in source, (
            "Balanced perspective must not read state.get('risk_debate_state') "
            "(shared mutable state). This violates RISK-02 isolation."
        )

    @pytest.mark.unit
    def test_balanced_creates_isolated_llm_instance(self):
        """Verify balanced perspective creates fresh LLM per call (D-02, D-03)."""
        factory = create_balanced_perspective()
        source = inspect.getsource(factory)

        assert "create_llm_client" in source, (
            "Balanced perspective must use create_llm_client factory to create "
            "a fresh instance per call (D-02, D-03)"
        )

        assert "balanced_llm" in source, (
            "Balanced perspective must create a local balanced_llm instance (D-02)"
        )

        assert "self.llm" not in source, (
            "Balanced perspective must not use self.llm. "
            "Use create_llm_client() instead (D-02)."
        )
        assert "self.quick_thinking_llm" not in source, (
            "Balanced perspective must not use self.quick_thinking_llm. "
            "Use create_llm_client() instead (D-02)."
        )

    @pytest.mark.unit
    def test_balanced_never_writes_other_verdict_fields(self):
        """Verify balanced perspective writes only to balanced_verdict (D-02)."""
        factory = create_balanced_perspective()
        source = inspect.getsource(factory)

        assert '"balanced_verdict"' in source or "'balanced_verdict'" in source, (
            "Balanced perspective must return balanced_verdict "
            "(D-02 field isolation)."
        )

        assert 'conservative_verdict' not in source, (
            "Balanced perspective must not reference conservative_verdict "
            "(D-02: each perspective owns only its own verdict)."
        )
        assert 'aggressive_verdict' not in source, (
            "Balanced perspective must not reference aggressive_verdict "
            "(D-02: each perspective owns only its own verdict)."
        )

    @pytest.mark.unit
    def test_balanced_uses_temperature_0_3(self):
        """Verify balanced perspective uses temperature=0.3 literal (RISK-02, D-03)."""
        factory = create_balanced_perspective()
        source = inspect.getsource(factory)

        assert "0.3" in source, (
            "Balanced perspective must contain the literal temperature=0.3 "
            "in its source code (RISK-02: provable distinctiveness by construction)."
        )

    @pytest.mark.unit
    def test_balanced_reads_pre_computed_context(self):
        """Verify balanced perspective reads risk_concentration_pct (grounded inputs)."""
        factory = create_balanced_perspective()
        source = inspect.getsource(factory)

        assert "risk_concentration_pct" in source, (
            "Balanced perspective must read pre-computed risk_concentration_pct "
            "to ground its reasoning in real portfolio data (prevents hallucination)."
        )


class TestAggressivePerspectiveIsolation:
    """Source-inspection tests for aggressive perspective isolation."""

    @pytest.mark.unit
    def test_aggressive_never_reads_risk_debate_state(self):
        """Verify aggressive perspective never reads shared risk_debate_state (RISK-02, D-02)."""
        factory = create_aggressive_perspective()
        source = inspect.getsource(factory)

        assert '["risk_debate_state"]' not in source, (
            "Aggressive perspective must not read state['risk_debate_state'] "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert "['risk_debate_state']" not in source, (
            "Aggressive perspective must not read state['risk_debate_state'] "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert '.get("risk_debate_state")' not in source, (
            "Aggressive perspective must not read state.get('risk_debate_state') "
            "(shared mutable state). This violates RISK-02 isolation."
        )
        assert ".get('risk_debate_state')" not in source, (
            "Aggressive perspective must not read state.get('risk_debate_state') "
            "(shared mutable state). This violates RISK-02 isolation."
        )

    @pytest.mark.unit
    def test_aggressive_creates_isolated_llm_instance(self):
        """Verify aggressive perspective creates fresh LLM per call (D-02, D-03)."""
        factory = create_aggressive_perspective()
        source = inspect.getsource(factory)

        assert "create_llm_client" in source, (
            "Aggressive perspective must use create_llm_client factory to create "
            "a fresh instance per call (D-02, D-03)"
        )

        assert "aggressive_llm" in source, (
            "Aggressive perspective must create a local aggressive_llm instance (D-02)"
        )

        assert "self.llm" not in source, (
            "Aggressive perspective must not use self.llm. "
            "Use create_llm_client() instead (D-02)."
        )
        assert "self.quick_thinking_llm" not in source, (
            "Aggressive perspective must not use self.quick_thinking_llm. "
            "Use create_llm_client() instead (D-02)."
        )

    @pytest.mark.unit
    def test_aggressive_never_writes_other_verdict_fields(self):
        """Verify aggressive perspective writes only to aggressive_verdict (D-02)."""
        factory = create_aggressive_perspective()
        source = inspect.getsource(factory)

        assert '"aggressive_verdict"' in source or "'aggressive_verdict'" in source, (
            "Aggressive perspective must return aggressive_verdict "
            "(D-02 field isolation)."
        )

        assert 'conservative_verdict' not in source, (
            "Aggressive perspective must not reference conservative_verdict "
            "(D-02: each perspective owns only its own verdict)."
        )
        assert 'balanced_verdict' not in source, (
            "Aggressive perspective must not reference balanced_verdict "
            "(D-02: each perspective owns only its own verdict)."
        )

    @pytest.mark.unit
    def test_aggressive_uses_temperature_0_4(self):
        """Verify aggressive perspective uses temperature=0.4 literal (RISK-02, D-03)."""
        factory = create_aggressive_perspective()
        source = inspect.getsource(factory)

        assert "0.4" in source, (
            "Aggressive perspective must contain the literal temperature=0.4 "
            "in its source code (RISK-02: provable distinctiveness by construction)."
        )

    @pytest.mark.unit
    def test_aggressive_reads_pre_computed_context(self):
        """Verify aggressive perspective reads risk_concentration_pct (grounded inputs)."""
        factory = create_aggressive_perspective()
        source = inspect.getsource(factory)

        assert "risk_concentration_pct" in source, (
            "Aggressive perspective must read pre-computed risk_concentration_pct "
            "to ground its reasoning in real portfolio data (prevents hallucination)."
        )


class TestTemperatureDistinctiveness:
    """Cross-perspective tests for temperature distinctiveness (RISK-02)."""

    @pytest.mark.unit
    def test_all_temperatures_distinct(self):
        """Verify all three perspectives use distinct temperatures (RISK-02).

        This test confirms that temperature distinctiveness is mechanically
        proven by the source code, not just documented in comments.
        """
        conservative_factory = create_conservative_perspective()
        balanced_factory = create_balanced_perspective()
        aggressive_factory = create_aggressive_perspective()

        conservative_source = inspect.getsource(conservative_factory)
        balanced_source = inspect.getsource(balanced_factory)
        aggressive_source = inspect.getsource(aggressive_factory)

        # Verify each has its unique temperature literal
        assert "0.2" in conservative_source, (
            "Conservative perspective must use temperature=0.2"
        )
        assert "0.3" in balanced_source, (
            "Balanced perspective must use temperature=0.3"
        )
        assert "0.4" in aggressive_source, (
            "Aggressive perspective must use temperature=0.4"
        )

        # Verify they don't incorrectly use each other's temperatures
        assert "temperature=0.3" not in conservative_source, (
            "Conservative perspective must not use balanced temperature (0.3)"
        )
        assert "temperature=0.4" not in conservative_source, (
            "Conservative perspective must not use aggressive temperature (0.4)"
        )

        assert "temperature=0.2" not in balanced_source, (
            "Balanced perspective must not use conservative temperature (0.2)"
        )
        assert "temperature=0.4" not in balanced_source, (
            "Balanced perspective must not use aggressive temperature (0.4)"
        )

        assert "temperature=0.2" not in aggressive_source, (
            "Aggressive perspective must not use conservative temperature (0.2)"
        )
        assert "temperature=0.3" not in aggressive_source, (
            "Aggressive perspective must not use balanced temperature (0.3)"
        )
