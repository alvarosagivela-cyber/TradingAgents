"""Tests for cost tracking persistence (Phase 6, D-04/D-05).

Tests verify that:
1. CostRecord schema matches ReflectionRecord pattern (frozen dataclass)
2. JSONL persistence handles I/O errors gracefully (never raise)
3. Reading JSONL skips malformed lines but returns well-formed records
4. record_cost_from_response extracts usage_metadata and calculates cost
5. record_cost_from_response handles missing usage_metadata gracefully
6. record_cost_from_response handles unknown models gracefully
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from tradingagents.llm_clients.cost_tracker import (
    CostRecord,
    persist_cost_record,
    read_cost_records,
    record_cost_from_response,
)


@pytest.mark.unit
class TestCostRecord:
    """Verify CostRecord schema and serialization."""

    def test_cost_record_is_frozen(self):
        """CostRecord must be a frozen dataclass (immutable)."""
        record = CostRecord(
            timestamp="2026-08-05T10:00:00+00:00",
            layer="auditor",
            model="claude-sonnet-5",
            ticker="AAPL",
            trade_date="2026-08-05",
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cost_usd=0.0003,
        )
        # Attempt to modify should raise
        with pytest.raises((AttributeError, TypeError)):
            record.timestamp = "changed"

    def test_cost_record_to_json_dict(self):
        """to_json_dict() must serialize all fields."""
        record = CostRecord(
            timestamp="2026-08-05T10:00:00+00:00",
            layer="research",
            model="claude-haiku-4-5",
            ticker="MSFT",
            trade_date="2026-08-05",
            input_tokens=1000,
            output_tokens=200,
            cache_read_input_tokens=10,
            cache_creation_input_tokens=5,
            cost_usd=0.002,
        )
        json_dict = record.to_json_dict()
        assert json_dict["timestamp"] == "2026-08-05T10:00:00+00:00"
        assert json_dict["layer"] == "research"
        assert json_dict["model"] == "claude-haiku-4-5"
        assert json_dict["ticker"] == "MSFT"
        assert json_dict["trade_date"] == "2026-08-05"
        assert json_dict["input_tokens"] == 1000
        assert json_dict["output_tokens"] == 200
        assert json_dict["cache_read_input_tokens"] == 10
        assert json_dict["cache_creation_input_tokens"] == 5
        assert json_dict["cost_usd"] == 0.002

    def test_cost_record_roundtrip(self):
        """to_json_dict() → from_json_dict() must produce identical instance."""
        original = CostRecord(
            timestamp="2026-08-05T10:00:00+00:00",
            layer="balanced",
            model="claude-haiku-4-5",
            ticker="TSLA",
            trade_date="2026-08-05",
            input_tokens=500,
            output_tokens=100,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            cost_usd=0.001,
        )
        json_dict = original.to_json_dict()
        restored = CostRecord.from_json_dict(json_dict)
        assert restored == original


@pytest.mark.unit
class TestPersistCostRecord:
    """Verify JSONL persistence (D-04)."""

    def test_persist_creates_parent_directories(self):
        """persist_cost_record must create parent dirs if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "deeply" / "nested" / "cost.jsonl"
            record = CostRecord(
                timestamp="2026-08-05T10:00:00+00:00",
                layer="auditor",
                model="claude-sonnet-5",
                ticker="AAPL",
                trade_date="2026-08-05",
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0003,
            )
            persist_cost_record(record, log_path)
            assert log_path.exists()

    def test_persist_appends_to_jsonl(self):
        """persist_cost_record must append JSON line to existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "cost.jsonl"
            record1 = CostRecord(
                timestamp="2026-08-05T10:00:00+00:00",
                layer="research",
                model="claude-haiku-4-5",
                ticker="AAPL",
                trade_date="2026-08-05",
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0003,
            )
            record2 = CostRecord(
                timestamp="2026-08-05T10:01:00+00:00",
                layer="auditor",
                model="claude-sonnet-5",
                ticker="MSFT",
                trade_date="2026-08-05",
                input_tokens=200,
                output_tokens=100,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0006,
            )
            persist_cost_record(record1, log_path)
            persist_cost_record(record2, log_path)

            # Read and verify both records
            with open(log_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["layer"] == "research"
            assert json.loads(lines[1])["layer"] == "auditor"

    def test_persist_graceful_error(self, monkeypatch):
        """persist_cost_record must not raise when write fails (D-05)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "cost.jsonl"
            record = CostRecord(
                timestamp="2026-08-05T10:00:00+00:00",
                layer="auditor",
                model="claude-sonnet-5",
                ticker="AAPL",
                trade_date="2026-08-05",
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0003,
            )

            # Mock open to raise OSError
            def mock_open_raises(*args, **kwargs):
                raise OSError("Disk write failed (simulated)")

            monkeypatch.setattr("builtins.open", mock_open_raises)

            # Should not raise, just log
            persist_cost_record(record, log_path)


@pytest.mark.unit
class TestReadCostRecords:
    """Verify JSONL reading with error handling."""

    def test_read_nonexistent_file_returns_empty(self):
        """read_cost_records must return [] if file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nonexistent.jsonl"
            records = read_cost_records(log_path)
            assert records == []

    def test_read_jsonl_skips_malformed_lines(self):
        """read_cost_records must skip malformed JSON and return well-formed records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "cost.jsonl"

            # Write a mix of valid and invalid lines
            with open(log_path, "w") as f:
                # Line 1: valid
                record1 = CostRecord(
                    timestamp="2026-08-05T10:00:00+00:00",
                    layer="research",
                    model="claude-haiku-4-5",
                    ticker="AAPL",
                    trade_date="2026-08-05",
                    input_tokens=100,
                    output_tokens=50,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.0003,
                )
                f.write(json.dumps(record1.to_json_dict()) + "\n")

                # Line 2: invalid JSON
                f.write("{ invalid json\n")

                # Line 3: valid
                record2 = CostRecord(
                    timestamp="2026-08-05T10:01:00+00:00",
                    layer="auditor",
                    model="claude-sonnet-5",
                    ticker="MSFT",
                    trade_date="2026-08-05",
                    input_tokens=200,
                    output_tokens=100,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cost_usd=0.0006,
                )
                f.write(json.dumps(record2.to_json_dict()) + "\n")

            # Read and verify: should get 2 records (skipped malformed line 2)
            records = read_cost_records(log_path)
            assert len(records) == 2
            assert records[0].layer == "research"
            assert records[1].layer == "auditor"

    def test_read_jsonl_skips_valid_json_non_dict_lines(self):
        """A line that is valid JSON but not an object (e.g. a list or int) must be
        skipped gracefully, not raise TypeError out of read_cost_records (regression
        for the gap found in Phase 6 code review: from_json_dict()'s dict subscripts
        raise TypeError on a non-dict input, which the original except tuple didn't catch).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "cost.jsonl"

            record = CostRecord(
                timestamp="2026-08-05T10:00:00+00:00",
                layer="research",
                model="claude-haiku-4-5",
                ticker="AAPL",
                trade_date="2026-08-05",
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                cost_usd=0.0003,
            )
            with open(log_path, "w") as f:
                f.write(json.dumps(record.to_json_dict()) + "\n")
                f.write(json.dumps([1, 2, 3]) + "\n")  # valid JSON, not a dict
                f.write(json.dumps(42) + "\n")  # valid JSON, not a dict

            records = read_cost_records(log_path)
            assert len(records) == 1
            assert records[0].layer == "research"


@pytest.mark.unit
class TestRecordCostFromResponse:
    """Verify extraction of usage_metadata and cost calculation."""

    def test_extract_usage_metadata_and_record(self):
        """record_cost_from_response must extract usage_metadata and calculate cost."""
        # Mock response with usage_metadata
        response = mock.MagicMock()
        response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {
                "cache_read": 0,
                "cache_creation": 0,
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "cost.jsonl"

            # Mock get_config to return our temp path
            with mock.patch("tradingagents.llm_clients.cost_tracker.get_config") as mock_config:
                mock_config.return_value = {"cost_log_path": str(log_path)}

                record = record_cost_from_response(
                    response,
                    layer="auditor",
                    model="claude-sonnet-5",
                    ticker="AAPL",
                    trade_date="2026-08-05",
                )

                assert record is not None
                assert record.layer == "auditor"
                assert record.model == "claude-sonnet-5"
                assert record.ticker == "AAPL"
                assert record.trade_date == "2026-08-05"
                assert record.input_tokens == 100
                assert record.output_tokens == 50
                assert record.cache_read_input_tokens == 0
                assert record.cache_creation_input_tokens == 0
                assert record.cost_usd > 0  # Sonnet 5: 100*$2/1M + 50*$10/1M

                # Verify persisted to JSONL
                assert log_path.exists()
                with open(log_path) as f:
                    persisted = json.loads(f.read().strip())
                assert persisted["layer"] == "auditor"

    def test_missing_usage_metadata_returns_none(self):
        """record_cost_from_response must return None if usage_metadata is missing."""
        # Mock response without usage_metadata
        response = mock.MagicMock()
        response.usage_metadata = None

        record = record_cost_from_response(
            response,
            layer="auditor",
            model="claude-sonnet-5",
        )
        assert record is None

    def test_unknown_model_returns_none(self):
        """record_cost_from_response must return None if model is unknown (not raise)."""
        response = mock.MagicMock()
        response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
        }

        record = record_cost_from_response(
            response,
            layer="auditor",
            model="claude-unknown-model",
        )
        assert record is None

    def test_no_bare_raise_in_function(self):
        """record_cost_from_response must have try/except wrapping entire body."""
        # This is more of a code review assertion, but we verify robustness by
        # having record_cost_from_response continue even on unexpected exceptions
        response = mock.MagicMock()
        response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
        }

        # Mock calculate_cost to raise an unexpected exception
        with mock.patch(
            "tradingagents.llm_clients.cost_tracker.calculate_cost"
        ) as mock_calc:
            mock_calc.side_effect = RuntimeError("Unexpected error")

            # Should not raise, just log and return None
            record = record_cost_from_response(
                response,
                layer="auditor",
                model="claude-sonnet-5",
            )
            assert record is None

    def test_defaults_for_optional_params(self):
        """record_cost_from_response must use defaults for ticker/trade_date."""
        response = mock.MagicMock()
        response.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 50,
            "input_token_details": {"cache_read": 0, "cache_creation": 0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "cost.jsonl"
            with mock.patch("tradingagents.llm_clients.cost_tracker.get_config") as mock_config:
                mock_config.return_value = {"cost_log_path": str(log_path)}

                # Call without ticker/trade_date
                record = record_cost_from_response(
                    response,
                    layer="research",
                    model="claude-haiku-4-5",
                )

                assert record is not None
                assert record.ticker == ""
                assert record.trade_date == ""
