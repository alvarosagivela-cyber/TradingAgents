"""Paper execution node: submit approved trades to Alpaca paper trading (D-04, EXEC-01).

This node is reached only when the Risk Squad aggregator has decided final_veto=False
(topology enforced in 04-04-PLAN.md). It submits a notional (dollar-based) MarketOrderRequest
to the Alpaca paper trading account, using paper=True enforced at the TradingClient level.

The node uses notional sizing (not share quantity) to avoid introducing a live-price-fetch
dependency in this phase — the Portfolio Manager and Risk Squad have already pre-computed
the dollar sizing in proposed_notional_usd (D-03).

On any Alpaca submission error (network, API, auth), the node catches the exception and
returns paper_execution_status='failed' without propagating, to prevent a single Alpaca
error from crashing the entire graph.
"""

import logging

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from tradingagents.trading.alpaca_client import create_alpaca_client

logger = logging.getLogger(__name__)


def create_paper_execution_node():
    """Create and return a paper execution node.

    Returns:
        A callable node that takes a state dict and returns an updated state dict
        with paper_execution_status and paper_order_id fields.
    """

    def paper_execution_node(state: dict) -> dict:
        """Submit an approved trade to Alpaca paper trading if proposed_side != 'Hold'.

        Reads:
        - company_of_interest: ticker symbol
        - proposed_side: Buy|Hold|Sell (deterministically parsed by portfolio_snapshot_node)
        - proposed_notional_usd: dollar amount to trade (deterministically sized by portfolio_snapshot_node)
        - execution_log: audit trail list to append to

        If proposed_side == 'Hold':
        - Returns immediately with paper_execution_status='skipped_hold'
        - Does NOT create an Alpaca client (critical for mocking/testing)
        - No order is submitted

        If proposed_side in ('Buy', 'Sell'):
        - Creates an Alpaca client (paper=True enforced at factory level)
        - Submits a MarketOrderRequest with:
          - symbol=ticker
          - notional=proposed_notional_usd (dollar-based, not share quantity)
          - side=OrderSide.BUY or OrderSide.SELL
          - time_in_force=TimeInForce.DAY
        - On success: paper_execution_status='submitted', paper_order_id=order.id
        - On Alpaca exception: paper_execution_status='failed', paper_order_id='', exception logged

        Args:
            state: LangGraph state dict with company_of_interest, proposed_side, proposed_notional_usd

        Returns:
            State dict with paper_execution_status and paper_order_id fields added/updated
        """
        ticker = state.get("company_of_interest", "")
        proposed_side = state.get("proposed_side", "Hold")
        notional = state.get("proposed_notional_usd", 0.0)
        execution_log = state.get("execution_log", [])

        # If no action proposed, skip submission
        if proposed_side == "Hold":
            return {
                **state,
                "paper_execution_status": "skipped_hold",
                "paper_order_id": "",
                "execution_log": execution_log + [
                    "Paper Execution: skipped (Trader proposed Hold, nothing to execute)"
                ],
            }

        # Convert proposed_side to Alpaca OrderSide
        order_side = OrderSide.BUY if proposed_side == "Buy" else OrderSide.SELL

        try:
            # Create Alpaca client (paper=True hardcoded at factory level)
            client = create_alpaca_client(paper=True)

            # Build notional order (dollar-based, no live price fetch needed)
            order_request = MarketOrderRequest(
                symbol=ticker,
                notional=notional,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            # Submit order to Alpaca paper trading
            order = client.submit_order(order_request)
            order_id = str(order.id)

            return {
                **state,
                "paper_execution_status": "submitted",
                "paper_order_id": order_id,
                "execution_log": execution_log
                + [
                    f"Paper Execution: order {order_id} submitted "
                    f"({proposed_side} ${notional:.2f} notional {ticker}) to Alpaca paper trading"
                ],
            }

        except Exception as exc:
            # Catch all Alpaca exceptions (network, API, auth, etc.)
            # Log and return failed status without propagating
            logger.exception(
                "Paper Execution: order submission failed for %s (%s $%.2f): %s",
                ticker,
                proposed_side,
                notional,
                exc,
            )
            return {
                **state,
                "paper_execution_status": "failed",
                "paper_order_id": "",
                "execution_log": execution_log
                + [f"Paper Execution: order submission failed: {exc}"],
            }

    return paper_execution_node
