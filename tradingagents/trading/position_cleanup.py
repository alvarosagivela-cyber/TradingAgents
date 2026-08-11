"""Position cleanup wrapper for Phase 7 window end (D-09).

D-09: All open positions are closed at the end of the validation window,
producing a clean, bounded Sharpe calculation rather than leaving positions
open indefinitely. Reuses Phase 4 Alpaca primitives, never leaving positions
open per CONCERNS.md's "Missing Critical Features #1: Position Exit / Stop-Loss Mechanism".

The close_all_open_positions() function wraps the empirically-verified
TradingClient.close_all_positions(cancel_orders=True) atomic method — no
per-position loop, no custom REST calls.

EXEC-01: This module accepts a TradingClient parameter and never constructs
one itself (factory create_alpaca_client(paper=True) is the sole call site).

Error handling follows the fail-safe pattern: any exception is caught,
logged, and returned as a status dict (never propagates to caller).
"""

import logging
from typing import Any

from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)


def close_all_open_positions(client: TradingClient) -> dict[str, Any]:
    """Close all open positions and cancel pending orders at window end (D-09).

    Wraps Alpaca's TradingClient.close_all_positions(cancel_orders=True) atomic
    method. Never raises exceptions — all errors are logged and returned in the
    status dict with status='failed'.

    Args:
        client: Alpaca TradingClient (must be paper=True, enforced by caller)

    Returns:
        Dict with keys:
        - status: 'closed' (success) or 'failed' (exception)
        - positions_closed: Number of positions closed (int); 0 if response is dict (count not extractable)
        - error: Exception message (str) if status='failed', else None
    """
    try:
        # D-09: Call empirically-verified close_all_positions(cancel_orders=True)
        responses = client.close_all_positions(cancel_orders=True)

        # Alpaca SDK returns Union[list, dict] — extract count only from list
        if isinstance(responses, list):
            positions_closed = len(responses)
        else:
            # Dict response — count not extractable (documented, not a bug)
            positions_closed = 0

        logger.info("D-09: closed %d position(s), cancelled pending orders", positions_closed)

        return {
            "status": "closed",
            "positions_closed": positions_closed,
            "error": None,
        }

    except Exception as exc:
        error_message = str(exc)
        logger.exception("D-09: failed to close all positions: %s", error_message)

        return {
            "status": "failed",
            "positions_closed": 0,
            "error": error_message,
        }
