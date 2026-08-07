"""Tests for cost pricing table and calculation (Phase 6, D-03).

Tests verify that cost calculations use exact token counts from usage_metadata
and a versioned pricing table, never estimates.
"""

import pytest

from tradingagents.llm_clients.cost_pricing import PRICING_TABLE, calculate_cost


@pytest.mark.unit
class TestPricingTable:
    """Verify pricing table structure and content."""

    def test_pricing_table_has_expected_models(self):
        """PRICING_TABLE must contain exactly the two models in use."""
        assert set(PRICING_TABLE.keys()) == {"claude-haiku-4-5", "claude-sonnet-5"}

    def test_pricing_table_has_required_fields(self):
        """Each model entry must have input and output rates."""
        for model, rates in PRICING_TABLE.items():
            assert "input_per_1m_usd" in rates, f"{model} missing input_per_1m_usd"
            assert "output_per_1m_usd" in rates, f"{model} missing output_per_1m_usd"
            assert isinstance(rates["input_per_1m_usd"], (int, float))
            assert isinstance(rates["output_per_1m_usd"], (int, float))


@pytest.mark.unit
class TestCalculateCost:
    """Verify cost calculation against real pricing rates."""

    def test_haiku_input_only_cost(self):
        """Haiku input cost: $1.00 per 1M tokens."""
        # 1M input tokens = $1.00
        cost = calculate_cost("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)
        assert cost == 1.00

    def test_haiku_output_only_cost(self):
        """Haiku output cost: $5.00 per 1M tokens."""
        # 1M output tokens = $5.00
        cost = calculate_cost("claude-haiku-4-5", input_tokens=0, output_tokens=1_000_000)
        assert cost == 5.00

    def test_sonnet5_input_only_cost(self):
        """Sonnet 5 introductory input cost: $2.00 per 1M tokens (through 2026-08-31)."""
        # 1M input tokens = $2.00
        cost = calculate_cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
        assert cost == 2.00

    def test_sonnet5_output_only_cost(self):
        """Sonnet 5 introductory output cost: $10.00 per 1M tokens (through 2026-08-31)."""
        # 1M output tokens = $10.00
        cost = calculate_cost("claude-sonnet-5", input_tokens=0, output_tokens=1_000_000)
        assert cost == 10.00

    def test_zero_tokens_zero_cost(self):
        """Zero input and output tokens produce zero cost."""
        cost = calculate_cost("claude-haiku-4-5", input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_combined_input_output_cost(self):
        """Verify combined input+output calculation."""
        # Haiku: 1000 input ($1 per 1M) + 2000 output ($5 per 1M)
        cost = calculate_cost("claude-haiku-4-5", input_tokens=1_000, output_tokens=2_000)
        expected = (1_000 * 1.00 / 1_000_000) + (2_000 * 5.00 / 1_000_000)
        assert abs(cost - expected) < 1e-9

    def test_unknown_model_raises_valueerror(self):
        """Unknown model must raise ValueError, not silently return 0."""
        with pytest.raises(ValueError) as exc_info:
            calculate_cost("claude-opus-9-unknown", input_tokens=100, output_tokens=100)
        assert "Unknown model" in str(exc_info.value)
        assert "claude-opus-9-unknown" in str(exc_info.value)

    def test_partial_usage_calculation(self):
        """Verify partial token usage calculation (1000 tokens)."""
        # Haiku: 1000 input = 1000 * $1 / 1M = $0.001
        cost_input = calculate_cost("claude-haiku-4-5", input_tokens=1_000, output_tokens=0)
        assert abs(cost_input - 0.001) < 1e-9

        # Haiku: 1000 output = 1000 * $5 / 1M = $0.005
        cost_output = calculate_cost("claude-haiku-4-5", input_tokens=0, output_tokens=1_000)
        assert abs(cost_output - 0.005) < 1e-9
