"""Unit tests for position_cleanup module (D-09: close all positions at window end).

Tests verify:
- D-09: close_all_open_positions() wraps close_all_positions(cancel_orders=True)
- EXEC-01: Never constructs TradingClient, accepts one as parameter
- Fail-safe pattern: never raises, logs errors and returns status dict
"""

from unittest.mock import Mock, MagicMock, patch
import pytest

from tradingagents.trading.position_cleanup import close_all_open_positions


class TestCloseAllOpenPositions:
    """Test close_all_open_positions() wrapper and error handling."""

    def test_close_all_open_positions_success_list_response(self):
        """D-09: Successful close returns list of responses."""
        # Mock Alpaca SDK's close_all_positions returning a list of 3 response objects
        mock_position_1 = Mock()
        mock_position_2 = Mock()
        mock_position_3 = Mock()

        mock_client = Mock()
        mock_client.close_all_positions.return_value = [
            mock_position_1,
            mock_position_2,
            mock_position_3,
        ]

        result = close_all_open_positions(mock_client)

        # Verify structure
        assert isinstance(result, dict)
        assert result["status"] == "closed"
        assert result["positions_closed"] == 3
        assert result["error"] is None

        # Verify close_all_positions was called with cancel_orders=True
        mock_client.close_all_positions.assert_called_once_with(cancel_orders=True)

    def test_close_all_open_positions_empty_list(self):
        """D-09: No open positions (empty list response)."""
        mock_client = Mock()
        mock_client.close_all_positions.return_value = []

        result = close_all_open_positions(mock_client)

        assert result["status"] == "closed"
        assert result["positions_closed"] == 0
        assert result["error"] is None

    def test_close_all_open_positions_dict_response(self):
        """D-09: Handle dict response (Union[list, dict] from Alpaca SDK)."""
        # Alpaca SDK returns either list or dict; count only extractable from list
        mock_client = Mock()
        mock_client.close_all_positions.return_value = {"status": "success"}

        result = close_all_open_positions(mock_client)

        assert result["status"] == "closed"
        # positions_closed = 0 when response is dict (count not extractable)
        assert result["positions_closed"] == 0
        assert result["error"] is None

    def test_close_all_open_positions_failure(self):
        """D-09: Exception is caught, logged, never propagates. Returns status=failed."""
        mock_client = Mock()
        exception_message = "API connection timeout"
        mock_client.close_all_positions.side_effect = Exception(exception_message)

        result = close_all_open_positions(mock_client)

        # Should return error status without raising
        assert result["status"] == "failed"
        assert result["positions_closed"] == 0
        assert result["error"] == exception_message
        assert "failed to close all positions" in result["error"] or exception_message in result["error"]

    def test_close_all_open_positions_never_constructs_client(self):
        """EXEC-01: Function never constructs TradingClient; accepts one as parameter."""
        import inspect
        import tradingagents.trading.position_cleanup as module

        source = inspect.getsource(module)

        # Verify "TradingClient(" does not appear in source (except in docstrings/comments)
        # Strip comments to be safe
        lines = source.split('\n')
        non_comment_lines = [
            line for line in lines
            if not line.strip().startswith('#')
        ]
        source_without_comments = '\n'.join(non_comment_lines)

        assert "TradingClient(" not in source_without_comments, \
            "EXEC-01 violation: position_cleanup.py constructs TradingClient directly"

    @patch('tradingagents.trading.position_cleanup.logger')
    def test_close_all_open_positions_logs_success(self, mock_logger):
        """D-09: Successful close logs info message."""
        mock_client = Mock()
        mock_client.close_all_positions.return_value = [Mock(), Mock()]

        result = close_all_open_positions(mock_client)

        # Verify info log was called
        assert mock_logger.info.called
        call_args = mock_logger.info.call_args[0][0]
        assert "closed" in call_args.lower()

    @patch('tradingagents.trading.position_cleanup.logger')
    def test_close_all_open_positions_logs_failure(self, mock_logger):
        """D-09: Failed close logs exception."""
        mock_client = Mock()
        mock_client.close_all_positions.side_effect = Exception("Network error")

        result = close_all_open_positions(mock_client)

        # Verify exception log was called
        assert mock_logger.exception.called
