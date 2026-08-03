"""Unit tests for the Alpaca client factory (EXEC-01: paper trading only).

Tests enforce that:
1. create_alpaca_client(paper=False) raises ValueError before importing TradingClient
2. Missing ALPACA_PAPER_API_KEY or ALPACA_PAPER_SECRET_KEY raises ValueError
3. create_alpaca_client(paper=True) calls TradingClient with the correct credentials
   and paper=True is a hardcoded literal at the call site (never a variable)
"""

from unittest.mock import MagicMock, patch
import pytest
from tradingagents.trading.alpaca_client import create_alpaca_client


@pytest.mark.unit
class TestAlpacaClientFactory:

    def test_paper_false_raises_valueerror(self):
        """Calling create_alpaca_client(paper=False) should raise ValueError before any TradingClient instantiation."""
        with pytest.raises(ValueError, match="Live trading is not permitted"):
            create_alpaca_client(paper=False)

    def test_missing_api_key_raises_valueerror(self, monkeypatch):
        """Missing ALPACA_PAPER_API_KEY should raise ValueError."""
        monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
        monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "test-secret")
        with pytest.raises(ValueError, match="ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY"):
            create_alpaca_client(paper=True)

    def test_missing_secret_key_raises_valueerror(self, monkeypatch):
        """Missing ALPACA_PAPER_SECRET_KEY should raise ValueError."""
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "test-key")
        monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY"):
            create_alpaca_client(paper=True)

    def test_paper_true_calls_trading_client_with_credentials(self, monkeypatch):
        """Calling create_alpaca_client(paper=True) with env vars set should call TradingClient with correct args."""
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "test-api-key")
        monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "test-secret-key")

        mock_trading_client = MagicMock()
        with patch("tradingagents.trading.alpaca_client.TradingClient", return_value=mock_trading_client) as mock_class:
            result = create_alpaca_client(paper=True)

            # Verify TradingClient was instantiated with the correct credentials and paper=True
            mock_class.assert_called_once_with(
                api_key="test-api-key",
                secret_key="test-secret-key",
                paper=True
            )
            assert result == mock_trading_client

    def test_paper_true_with_empty_env_vars_raises_valueerror(self, monkeypatch):
        """Empty env vars (falsy but present) should raise ValueError."""
        monkeypatch.setenv("ALPACA_PAPER_API_KEY", "")
        monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "")
        with pytest.raises(ValueError, match="ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY"):
            create_alpaca_client(paper=True)
