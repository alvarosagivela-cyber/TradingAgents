"""Unit tests for reflection reader — layer validation, graceful degradation, most-recent-wins."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.reflectors.reflection_schema import ReflectionRecord
from tradingagents.agents.reflectors.reflection_reader import (
    read_reflection_for_ticker,
    _VALID_LAYERS,
)


# ---------------------------------------------------------------------------
# Test Layer Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLayerValidation:
    """Test that read_reflection_for_ticker validates the layer parameter."""

    def test_valid_layers(self):
        """Valid layers are 'research', 'auditor', 'risk'."""
        assert _VALID_LAYERS == ("research", "auditor", "risk")

    def test_invalid_layer_raises_value_error(self):
        """Unknown layer raises ValueError immediately."""
        with patch("tradingagents.agents.reflectors.reflection_reader.get_config") as mock_config:
            mock_config.return_value = {"portfolio_reflections_log_path": "/tmp/test.jsonl"}
            with pytest.raises(ValueError, match="Unknown reflection layer"):
                read_reflection_for_ticker("portfolio", "AAPL")

    def test_invalid_layer_fails_loud(self):
        """ValueError is raised, not silently degraded to None."""
        with patch("tradingagents.agents.reflectors.reflection_reader.get_config"):
            with pytest.raises(ValueError):
                read_reflection_for_ticker("auditors", "AAPL")  # Typo: 'auditors' not 'auditor'


# ---------------------------------------------------------------------------
# Test Missing Store Behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMissingStore:
    """Test that missing reflection store returns None gracefully."""

    def test_missing_store_returns_none(self, tmp_path):
        """When reflection log file does not exist, return None."""
        nonexistent_path = tmp_path / "nonexistent" / "reflections.jsonl"

        with patch("tradingagents.agents.reflectors.reflection_reader.get_config") as mock_config:
            mock_config.return_value = {
                "research_reflections_log_path": str(nonexistent_path)
            }
            result = read_reflection_for_ticker("research", "AAPL")
            assert result is None


# ---------------------------------------------------------------------------
# Test Most-Recent-Wins Behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMostRecentWins:
    """Test that the most-recent matching record is returned."""

    def test_most_recent_aapl_returned(self, tmp_path):
        """With two AAPL records (old, new) and one MSFT, return the newest AAPL."""
        store_path = tmp_path / "reflections.jsonl"

        # Write two AAPL records and one MSFT
        record_old = ReflectionRecord(
            ticker="AAPL",
            decision_date="2026-07-15",
            decision_verdict="Buy",
            realized_return=0.05,
            classification="positive",
            lesson_text="Old lesson",
            created_at="2026-07-20T00:00:00Z",
        )
        record_new = ReflectionRecord(
            ticker="AAPL",
            decision_date="2026-07-20",
            decision_verdict="Sell",
            realized_return=-0.02,
            classification="negative",
            lesson_text="Newer lesson",
            created_at="2026-07-25T00:00:00Z",
        )
        record_msft = ReflectionRecord(
            ticker="MSFT",
            decision_date="2026-07-18",
            decision_verdict="Hold",
            realized_return=0.0,
            classification="neutral",
            lesson_text="MSFT lesson",
            created_at="2026-07-22T00:00:00Z",
        )

        with open(store_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record_old.to_json_dict()) + "\n")
            f.write(json.dumps(record_msft.to_json_dict()) + "\n")
            f.write(json.dumps(record_new.to_json_dict()) + "\n")

        with patch("tradingagents.agents.reflectors.reflection_reader.get_config") as mock_config:
            mock_config.return_value = {"research_reflections_log_path": str(store_path)}
            result = read_reflection_for_ticker("research", "AAPL")

            assert result is not None
            assert result.ticker == "AAPL"
            assert result.decision_date == "2026-07-20"
            assert result.lesson_text == "Newer lesson"


# ---------------------------------------------------------------------------
# Test Malformed Line Handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMalformedLineHandling:
    """Test that malformed JSON lines do not crash the reader."""

    def test_malformed_line_skipped(self, tmp_path):
        """A malformed JSON line is skipped; valid lines after it are still read."""
        store_path = tmp_path / "reflections.jsonl"

        valid_record = ReflectionRecord(
            ticker="AAPL",
            decision_date="2026-07-20",
            decision_verdict="Sell",
            realized_return=-0.03,
            classification="negative",
            lesson_text="Valid lesson",
            created_at="2026-07-25T00:00:00Z",
        )

        with open(store_path, "w", encoding="utf-8") as f:
            f.write('{"broken": json without closing brace\n')
            f.write(json.dumps(valid_record.to_json_dict()) + "\n")

        with patch("tradingagents.agents.reflectors.reflection_reader.get_config") as mock_config:
            mock_config.return_value = {"research_reflections_log_path": str(store_path)}
            result = read_reflection_for_ticker("research", "AAPL")

            assert result is not None
            assert result.ticker == "AAPL"
            assert result.lesson_text == "Valid lesson"

    def test_multiple_malformed_lines(self, tmp_path):
        """Multiple malformed lines are all skipped gracefully."""
        store_path = tmp_path / "reflections.jsonl"

        valid_record = ReflectionRecord(
            ticker="AAPL",
            decision_date="2026-07-20",
            decision_verdict="Sell",
            realized_return=-0.03,
            classification="negative",
            lesson_text="Valid",
            created_at="2026-07-25T00:00:00Z",
        )

        with open(store_path, "w", encoding="utf-8") as f:
            f.write("not valid json\n")
            f.write('{"also": "broken\n')
            f.write(json.dumps(valid_record.to_json_dict()) + "\n")
            f.write("more garbage\n")

        with patch("tradingagents.agents.reflectors.reflection_reader.get_config") as mock_config:
            mock_config.return_value = {"research_reflections_log_path": str(store_path)}
            result = read_reflection_for_ticker("research", "AAPL")

            assert result is not None
            assert result.ticker == "AAPL"


# ---------------------------------------------------------------------------
# Test All Layers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAllLayers:
    """Test that all three layers work correctly."""

    @pytest.mark.parametrize("layer", ["research", "auditor", "risk"])
    def test_read_from_each_layer(self, layer, tmp_path):
        """Each layer can be read independently."""
        store_path = tmp_path / f"{layer}.jsonl"

        record = ReflectionRecord(
            ticker="AAPL",
            decision_date="2026-07-20",
            decision_verdict="Buy",
            realized_return=0.05,
            classification="positive",
            lesson_text=f"Lesson from {layer}",
            created_at="2026-07-25T00:00:00Z",
        )

        with open(store_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record.to_json_dict()) + "\n")

        config_key = f"{layer}_reflections_log_path"
        with patch("tradingagents.agents.reflectors.reflection_reader.get_config") as mock_config:
            mock_config.return_value = {config_key: str(store_path)}
            result = read_reflection_for_ticker(layer, "AAPL")

            assert result is not None
            assert result.ticker == "AAPL"
            assert result.lesson_text == f"Lesson from {layer}"
