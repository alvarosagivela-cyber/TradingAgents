"""Unit tests for the trader action parser (Buy/Hold/Sell extraction).

Tests the parse_trader_action function in isolation without any LLM or
dependency mocking (pure Python parsing logic).
"""

import pytest

from tradingagents.agents.utils.trader_action import parse_trader_action


@pytest.mark.unit
class TestTraderActionParser:
    """Test cases for deterministic 3-tier trader action extraction."""

    def test_exact_trailer_match_buy(self):
        """Parse exact trailer: 'FINAL TRANSACTION PROPOSAL: **BUY**' -> 'Buy'."""
        text = """Some reasoning here.

FINAL TRANSACTION PROPOSAL: **BUY**
"""
        assert parse_trader_action(text) == "Buy"

    def test_exact_trailer_match_hold(self):
        """Parse exact trailer: 'FINAL TRANSACTION PROPOSAL: **HOLD**' -> 'Hold'."""
        text = """Some reasoning here.

FINAL TRANSACTION PROPOSAL: **HOLD**
"""
        assert parse_trader_action(text) == "Hold"

    def test_exact_trailer_match_sell(self):
        """Parse exact trailer: 'FINAL TRANSACTION PROPOSAL: **SELL**' -> 'Sell'."""
        text = """Some reasoning here.

FINAL TRANSACTION PROPOSAL: **SELL**
"""
        assert parse_trader_action(text) == "Sell"

    def test_trailer_without_bold_markers(self):
        """Parse trailer without markdown bold: 'FINAL TRANSACTION PROPOSAL: BUY' -> 'Buy'."""
        text = "FINAL TRANSACTION PROPOSAL: BUY"
        assert parse_trader_action(text) == "Buy"

    def test_trailer_case_insensitive(self):
        """Parse trailer case-insensitively: 'final transaction proposal: buy' -> 'Buy'."""
        text = "final transaction proposal: buy"
        assert parse_trader_action(text) == "Buy"

    def test_trailer_with_extra_spaces(self):
        """Parse trailer with extra spaces: 'FINAL  TRANSACTION  PROPOSAL:  **BUY**' -> 'Buy'."""
        text = "FINAL  TRANSACTION  PROPOSAL:  **BUY**"
        assert parse_trader_action(text) == "Buy"

    def test_missing_trailer_returns_default(self):
        """Missing trailer returns default 'Hold'."""
        text = "Some analysis without a transaction proposal trailer."
        assert parse_trader_action(text) == "Hold"

    def test_missing_trailer_custom_default(self):
        """Missing trailer with custom default returns 'Sell'."""
        text = "Some analysis without a transaction proposal trailer."
        assert parse_trader_action(text, default="Sell") == "Sell"

    def test_invalid_action_in_trailer_returns_default(self):
        """Invalid action word in trailer ('INVALID') returns default."""
        text = "FINAL TRANSACTION PROPOSAL: **INVALID**"
        assert parse_trader_action(text) == "Hold"

    def test_parse_in_longer_document(self):
        """Parse trailer from a longer markdown document."""
        text = """
**Reasoning**: The momentum is strong. Buy signals are aligned.

**Entry Price**: $150.00

**Stop Loss**: $140.00

FINAL TRANSACTION PROPOSAL: **BUY**

---
End of analysis.
"""
        assert parse_trader_action(text) == "Buy"
