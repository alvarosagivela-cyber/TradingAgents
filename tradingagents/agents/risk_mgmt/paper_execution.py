"""Paper execution node: submit approved trades to Alpaca paper trading (D-04, EXEC-01).

This node is reached only when the Risk Squad aggregator has decided final_veto=False
(topology enforced in 04-04-PLAN.md). It submits a notional (dollar-based) MarketOrderRequest
to the Alpaca paper trading account, using paper=True enforced at the TradingClient level.

The node uses notional sizing (not share quantity) to avoid introducing a live-price-fetch
dependency in this phase — the Portfolio Manager and Risk Squad have already pre-computed
the dollar sizing in proposed_notional_usd (D-03).

Crypto tickers (e.g. BTC-USD, the yfinance/data-pipeline convention) are translated to
Alpaca's slash symbol format (BTC/USD) at this execution boundary only -- every other
part of the pipeline (data fetch, momentum calc, reports) keeps using the yfinance form.
Alpaca's crypto trading API only accepts `gtc`/`ioc` time_in_force (confirmed against
Alpaca's docs: "For Crypto Trading, Alpaca only supports gtc and ioc. OPG, fok, day, and
CLS are not supported") -- submitting `day` for a crypto order is rejected outright, so
this node picks the TIF deterministically from the ticker string, never from an LLM.

On any Alpaca submission error (network, API, auth), the node catches the exception and
returns paper_execution_status='failed' without propagating, to prevent a single Alpaca
error from crashing the entire graph.
"""

import logging

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from tradingagents.dataflows.symbol_utils import crypto_base
from tradingagents.trading.alpaca_client import create_alpaca_client

logger = logging.getLogger(__name__)


def _to_alpaca_symbol(ticker: str) -> str:
    """Translate a yfinance-style crypto ticker (BASE-USD) to Alpaca's BASE/USD.

    Non-crypto tickers (equities, ETFs) pass through unchanged.
    """
    base = crypto_base(ticker)
    return f"{base}/USD" if base else ticker


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
          - symbol=ticker translated to Alpaca's crypto format (BASE/USD) when the
            ticker is a recognized crypto base; equities/ETFs pass through unchanged
          - notional=proposed_notional_usd (dollar-based, not share quantity)
          - side=OrderSide.BUY or OrderSide.SELL
          - time_in_force=TimeInForce.GTC for crypto (Alpaca rejects DAY for crypto),
            TimeInForce.DAY for equities/ETFs
        - On success: paper_execution_status='submitted', paper_order_id=order.id
        - On Alpaca exception: paper_execution_status='failed', paper_order_id='', exception logged

        Args:
            state: LangGraph state dict with company_of_interest, proposed_side, proposed_notional_usd

        Returns:
            State dict with paper_execution_status and paper_order_id fields added/updated
        """
        ticker = state.get("company_of_interest", "")
        proposed_side = state.get("proposed_side", "Hold")
        # Alpaca rejects notional values with more than 2 decimal places ("notional
        # value must be limited to 2 decimal places", code 42210000) -- confirmed
        # live in production against a real BTC/USD Sell order, whose unrounded
        # notional (risk_position_size_pct * portfolio_total_value, a real Alpaca
        # equity float) had more precision than that. round() must happen before
        # the order is built, not just at display/log time.
        notional = round(state.get("proposed_notional_usd", 0.0), 2)
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

        # Crypto trades 24/7 and Alpaca's crypto API only accepts gtc/ioc (day is
        # rejected outright); equities close daily and use day, matching prior behavior.
        alpaca_symbol = _to_alpaca_symbol(ticker)
        is_crypto = crypto_base(ticker) is not None
        time_in_force = TimeInForce.GTC if is_crypto else TimeInForce.DAY

        try:
            # Create Alpaca client (paper=True hardcoded at factory level)
            client = create_alpaca_client(paper=True)

            # Build notional order (dollar-based, no live price fetch needed)
            order_request = MarketOrderRequest(
                symbol=alpaca_symbol,
                notional=notional,
                side=order_side,
                time_in_force=time_in_force,
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
                    f"({proposed_side} ${notional:.2f} notional {alpaca_symbol}) to Alpaca paper trading"
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
