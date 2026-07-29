"""Unit tests for thesis_validator — tolerance boundaries, persistence, quality checks."""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.utils.thesis_validator import (
    create_momentum_validate_node,
    validate_thesis,
    _check_refutation_quality,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_thesis_state(
    retorno_12_1: float = 0.1834,
    z_score: float = 2.3,
    llm_retorno: float = 0.1834,
    llm_z_score: float = 2.3,
    verdict: str = "Buy",
    refutation: str = "Thesis fails if price closes below the 50-day MA.",
) -> dict:
    """Build a minimal state dict for thesis validation."""
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2026-07-29",
        "retorno_12_1": retorno_12_1,
        "momentum_z_score": z_score,
        "thesis_verdict": verdict,
        "refutation_criterion": refutation,
        "data_points": {"retorno_12_1": llm_retorno, "z_score": llm_z_score},
        "thesis": "Sample reasoning.",
        "confidence_level": "high",
    }


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToleranceBoundaries:
    """Test the D-11 tolerance boundaries for numeric validation."""

    def test_validate_thesis_tolerance_boundary_retorno(self):
        """Retorno diff of 0.00009 passes; 0.00011 fails (D-11 exact boundary)."""
        # Tolerance is ±0.0001
        base_retorno = 0.1834

        # Just within tolerance (0.00009 < 0.0001)
        state_pass = _make_thesis_state(
            retorno_12_1=base_retorno,
            llm_retorno=base_retorno + 0.00009,
        )
        result_pass = validate_thesis(state_pass)
        assert result_pass.get("verified") is True, "Diff of 0.00009 should pass"

        # Just outside tolerance (0.00011 > 0.0001)
        state_fail = _make_thesis_state(
            retorno_12_1=base_retorno,
            llm_retorno=base_retorno + 0.00011,
        )
        result_fail = validate_thesis(state_fail)
        assert result_fail.get("verified") is False, "Diff of 0.00011 should fail"

    def test_validate_thesis_tolerance_boundary_z_score(self):
        """Z-score diff of 0.009 passes; 0.011 fails (D-11 exact boundary)."""
        # Tolerance is ±0.01
        base_z = 2.3

        # Just within tolerance (0.009 < 0.01)
        state_pass = _make_thesis_state(
            z_score=base_z,
            llm_z_score=base_z + 0.009,
        )
        result_pass = validate_thesis(state_pass)
        assert result_pass.get("verified") is True, "Diff of 0.009 should pass"

        # Just outside tolerance (0.011 > 0.01)
        state_fail = _make_thesis_state(
            z_score=base_z,
            llm_z_score=base_z + 0.011,
        )
        result_fail = validate_thesis(state_fail)
        assert result_fail.get("verified") is False, "Diff of 0.011 should fail"


@pytest.mark.unit
class TestRequiredFields:
    """Test validation of required thesis fields."""

    def test_validate_thesis_missing_data_points(self):
        """Empty data_points dict results in verified=False."""
        state = _make_thesis_state()
        state["data_points"] = {}
        result = validate_thesis(state)
        assert result.get("verified") is False

    def test_validate_thesis_missing_retorno_in_data_points(self):
        """Missing retorno_12_1 key in data_points results in verified=False."""
        state = _make_thesis_state()
        state["data_points"] = {"z_score": 2.3}  # Missing retorno_12_1
        result = validate_thesis(state)
        assert result.get("verified") is False

    def test_validate_thesis_missing_z_score_in_data_points(self):
        """Missing z_score key in data_points results in verified=False."""
        state = _make_thesis_state()
        state["data_points"] = {"retorno_12_1": 0.1834}  # Missing z_score
        result = validate_thesis(state)
        assert result.get("verified") is False

    def test_validate_thesis_missing_verdict(self):
        """Empty thesis_verdict results in verified=False."""
        state = _make_thesis_state()
        state["thesis_verdict"] = ""
        result = validate_thesis(state)
        assert result.get("verified") is False

    def test_validate_thesis_missing_refutation_criterion(self):
        """Empty refutation_criterion results in verified=False."""
        state = _make_thesis_state()
        state["refutation_criterion"] = ""
        result = validate_thesis(state)
        assert result.get("verified") is False


@pytest.mark.unit
class TestPersistence:
    """Test JSONL audit trail persistence."""

    def test_persist_writes_verified_fields_scope(self, tmp_path):
        """Valid state → JSONL record with verified_fields: ['retorno_12_1', 'z_score'] (D-12)."""
        state = _make_thesis_state()
        state["refutation_quality"] = "ok"

        with patch(
            "tradingagents.agents.utils.thesis_validator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "research_thesis_log_path": str(tmp_path / "theses.jsonl")
            }

            validate_node = create_momentum_validate_node()
            validate_node(state)

            # Read and parse JSONL
            assert (tmp_path / "theses.jsonl").exists()
            import json
            with open(tmp_path / "theses.jsonl") as f:
                record = json.loads(f.readline())

            # Verify scope disclosure (D-12)
            assert record["verified_fields"] == ["retorno_12_1", "z_score"]
            assert record["ticker"] == "AAPL"
            assert record["verdict"] == "Buy"

    def test_rejected_thesis_never_persisted(self, tmp_path):
        """Numeric mismatch → thesis NOT persisted (D-11, D-13)."""
        state = _make_thesis_state(
            retorno_12_1=0.1834,
            llm_retorno=0.20,  # Mismatch!
        )

        with patch(
            "tradingagents.agents.utils.thesis_validator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "research_thesis_log_path": str(tmp_path / "theses.jsonl")
            }

            validate_node = create_momentum_validate_node()
            validate_node(state)

            # JSONL file should not be created for rejected thesis
            assert not (tmp_path / "theses.jsonl").exists()


@pytest.mark.unit
class TestRefutationQuality:
    """Test refutation criterion quality checks (D-17)."""

    def test_refutation_quality_flags_vague_criterion(self):
        """Vague phrases flag criterion as needs_review, but don't block verification."""
        state = _make_thesis_state(
            refutation="I'm not 100% certain about this thesis, but the momentum looks good."
        )
        result = validate_thesis(state)

        # Thesis is verified despite vague language (flag, not block)
        assert result.get("verified") is True
        assert result.get("refutation_quality") == "needs_review"

    def test_refutation_quality_ok_for_concrete_criterion(self):
        """Concrete, substantive criterion is marked as ok."""
        state = _make_thesis_state(
            refutation="Thesis fails if the stock closes below $150 for 3 consecutive trading days."
        )
        result = validate_thesis(state)

        assert result.get("verified") is True
        assert result.get("refutation_quality") == "ok"

    @pytest.mark.parametrize("vague_phrase", [
        "not 100% certain",
        "there could be",
        "markets can be irrational",
        "hard to say",
        "no way to know",
    ])
    def test_refutation_quality_detects_banned_phrases(self, vague_phrase):
        """All banned phrases trigger needs_review flag."""
        state = _make_thesis_state(
            refutation=f"The thesis might fail. {vague_phrase} when volatility spikes."
        )
        result = validate_thesis(state)

        assert result.get("verified") is True, "Vague phrase should not block verification"
        assert result.get("refutation_quality") == "needs_review"

    def test_check_refutation_quality_case_insensitive(self):
        """Banned phrase detection is case-insensitive."""
        result = _check_refutation_quality("NOT 100% CERTAIN that this will work.")
        assert result == "needs_review"

        result = _check_refutation_quality("THERE COULD BE issues with this approach.")
        assert result == "needs_review"


@pytest.mark.unit
class TestPersistenceDetails:
    """Test details of JSONL persistence format."""

    def test_persist_includes_timestamp(self, tmp_path):
        """Persisted record includes ISO timestamp (D-13)."""
        state = _make_thesis_state()
        state["refutation_quality"] = "ok"

        with patch(
            "tradingagents.agents.utils.thesis_validator.get_config"
        ) as mock_config:
            mock_config.return_value = {
                "research_thesis_log_path": str(tmp_path / "theses.jsonl")
            }

            validate_node = create_momentum_validate_node()
            validate_node(state)

            import json
            with open(tmp_path / "theses.jsonl") as f:
                record = json.loads(f.readline())

            # Verify timestamp is ISO format
            assert "timestamp" in record
            assert record["timestamp"].endswith("Z")  # ISO UTC indicator

    def test_persist_appends_not_overwrites(self, tmp_path):
        """Multiple calls append to JSONL (not overwrite)."""
        log_path = tmp_path / "theses.jsonl"

        with patch(
            "tradingagents.agents.utils.thesis_validator.get_config"
        ) as mock_config:
            mock_config.return_value = {"research_thesis_log_path": str(log_path)}

            validate_node = create_momentum_validate_node()

            state1 = _make_thesis_state()
            state1["refutation_quality"] = "ok"
            validate_node(state1)

            state2 = _make_thesis_state()
            state2["company_of_interest"] = "MSFT"
            state2["refutation_quality"] = "ok"
            validate_node(state2)

            # Should have 2 lines
            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2

            import json
            records = [json.loads(line) for line in lines]
            assert records[0]["ticker"] == "AAPL"
            assert records[1]["ticker"] == "MSFT"
