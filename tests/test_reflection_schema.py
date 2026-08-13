"""Unit tests for reflection schema — ReflectionRecord, serialization, classification."""


import pytest

from tradingagents.agents.reflectors.reflection_schema import (
    ReflectionRecord,
    build_lesson_text,
    classify_realized_return,
)

# ---------------------------------------------------------------------------
# Test Classification Thresholds
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClassifyRealizedReturn:
    """Test the classify_realized_return function thresholds."""

    def test_classify_negative_return(self):
        """Returns < -0.01 classify as 'negative'."""
        assert classify_realized_return(-0.02) == "negative"
        assert classify_realized_return(-0.05) == "negative"
        assert classify_realized_return(-1.0) == "negative"

    def test_classify_positive_return(self):
        """Returns > 0.01 classify as 'positive'."""
        assert classify_realized_return(0.02) == "positive"
        assert classify_realized_return(0.05) == "positive"
        assert classify_realized_return(1.0) == "positive"

    def test_classify_neutral_return(self):
        """Returns between -0.01 and 0.01 (inclusive at boundary) classify as 'neutral'."""
        assert classify_realized_return(0.0) == "neutral"
        assert classify_realized_return(-0.01) == "neutral"  # Exact boundary
        assert classify_realized_return(0.01) == "neutral"   # Exact boundary
        assert classify_realized_return(0.005) == "neutral"
        assert classify_realized_return(-0.005) == "neutral"

    def test_classify_boundary_values(self):
        """Verify strict < and > comparisons (not <= or >=)."""
        # Just below negative threshold
        assert classify_realized_return(-0.0101) == "negative"
        # Just above positive threshold
        assert classify_realized_return(0.0101) == "positive"


# ---------------------------------------------------------------------------
# Test Lesson Text Generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildLessonText:
    """Test build_lesson_text generates deterministic prose."""

    def test_build_lesson_text_no_llm_call(self):
        """build_lesson_text is pure Python, no LLM call."""
        # This test just verifies the function returns a string without attempting
        # any network/LLM call (no mock needed, just invocation)
        text = build_lesson_text("Sell", -0.03, "negative", window_days=10)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_build_lesson_text_contains_verdict(self):
        """Lesson text includes the decision verdict."""
        text = build_lesson_text("Buy", 0.05, "positive", window_days=10)
        assert "Buy" in text

    def test_build_lesson_text_contains_percentage(self):
        """Lesson text includes the formatted realized return as percentage."""
        text = build_lesson_text("Sell", -0.03, "negative", window_days=10)
        assert "-3.00%" in text

    def test_build_lesson_text_contains_classification(self):
        """Lesson text includes the classification."""
        text = build_lesson_text("Hold", 0.0, "neutral", window_days=10)
        assert "NEUTRAL" in text

    def test_build_lesson_text_contains_window_days(self):
        """Lesson text includes the window_days."""
        text = build_lesson_text("Buy", 0.02, "positive", window_days=10)
        assert "10" in text


# ---------------------------------------------------------------------------
# Test Schema Serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReflectionRecordRoundTrip:
    """Test ReflectionRecord serialization and deserialization."""

    def test_record_round_trip(self):
        """ReflectionRecord to_json_dict and from_json_dict preserve all fields."""
        original = ReflectionRecord(
            ticker="AAPL",
            decision_date="2026-07-20",
            decision_verdict="Sell",
            realized_return=-0.03,
            classification="negative",
            lesson_text="Prior analysis showed weakness; thesis invalidated.",
            created_at="2026-07-30T00:00:00Z",
        )

        # Serialize
        json_dict = original.to_json_dict()
        assert isinstance(json_dict, dict)

        # Deserialize
        restored = ReflectionRecord.from_json_dict(json_dict)

        # Verify all fields match
        assert restored.ticker == original.ticker
        assert restored.decision_date == original.decision_date
        assert restored.decision_verdict == original.decision_verdict
        assert restored.realized_return == original.realized_return
        assert restored.classification == original.classification
        assert restored.lesson_text == original.lesson_text
        assert restored.created_at == original.created_at

    def test_record_is_frozen(self):
        """ReflectionRecord is frozen (immutable)."""
        record = ReflectionRecord(
            ticker="AAPL",
            decision_date="2026-07-20",
            decision_verdict="Sell",
            realized_return=-0.03,
            classification="negative",
            lesson_text="...",
            created_at="2026-07-30T00:00:00Z",
        )
        with pytest.raises((AttributeError, TypeError)):
            record.ticker = "MSFT"

    def test_record_json_dict_keys(self):
        """to_json_dict includes exactly the 7 expected fields."""
        record = ReflectionRecord(
            ticker="AAPL",
            decision_date="2026-07-20",
            decision_verdict="Sell",
            realized_return=-0.03,
            classification="negative",
            lesson_text="...",
            created_at="2026-07-30T00:00:00Z",
        )
        json_dict = record.to_json_dict()
        expected_keys = {
            "ticker",
            "decision_date",
            "decision_verdict",
            "realized_return",
            "classification",
            "lesson_text",
            "created_at",
        }
        assert set(json_dict.keys()) == expected_keys
