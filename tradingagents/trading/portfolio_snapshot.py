"""Portfolio snapshot node: deterministic pre-computation of portfolio data from Alpaca paper account (RISK-01).

This node reads real portfolio state from the Alpaca paper trading account and pre-computes
the inputs that the Risk Squad nodes use to evaluate position size and concentration risk.
All values are computed by Python from the live account (never LLM-estimated), and failures
are handled with a fail-safe sentinel (risk_concentration_pct=1.0) that forces a deterministic
veto in the Risk Squad aggregator (RISK-01 safety net).

The node is deterministic and side-effect-free: it reads from Alpaca and the provided state,
and returns an updated state dict without modifying the input state.
"""

import logging

from tradingagents.agents.utils.trader_action import parse_trader_action
from tradingagents.dataflows.config import get_config
from tradingagents.trading.alpaca_client import create_alpaca_client

logger = logging.getLogger(__name__)


def create_portfolio_snapshot_node():
    """Create and return a portfolio snapshot node.

    Returns:
        A callable node that takes a state dict and returns an updated state dict
        with portfolio_snapshot_status, portfolio_total_value, existing_position_value,
        proposed_notional_usd, risk_concentration_pct, and proposed_side fields.
    """

    def portfolio_snapshot_node(state: dict) -> dict:
        """Snapshot the Alpaca paper account and pre-compute portfolio risk metrics.

        Reads the Alpaca paper trading account to obtain:
        - portfolio_total_value: account equity (real, not estimated)
        - existing_position_value: current position value in company_of_interest (0.0 if none)
        - proposed_notional_usd: dollar sizing = risk_position_size_pct * portfolio_total_value
        - risk_concentration_pct: (existing + proposed) / total
        - proposed_side: Buy|Hold|Sell extracted from trader_investment_plan

        On any Alpaca fetch failure (network error, API error, timeout):
        - portfolio_snapshot_status='fail'
        - portfolio_total_value=0.0, existing_position_value=0.0, proposed_notional_usd=0.0
        - risk_concentration_pct=1.0 (fail-safe sentinel forcing deterministic veto in aggregator)
        - proposed_side is still correctly parsed (doesn't depend on Alpaca)

        Args:
            state: LangGraph state dict with company_of_interest and trader_investment_plan

        Returns:
            State dict with portfolio_* and risk_* fields added/updated
        """
        ticker = state.get("company_of_interest", "")
        trader_plan = state.get("trader_investment_plan", "")

        # Parse proposed_side deterministically (doesn't depend on Alpaca, always succeeds)
        proposed_side = parse_trader_action(trader_plan)

        # Get risk position sizing config
        risk_position_size_pct = get_config().get("risk_position_size_pct", 0.02)

        try:
            # Fetch live Alpaca paper account state
            client = create_alpaca_client(paper=True)
            account = client.get_account()
            portfolio_total_value = float(account.equity)

            # Fetch existing position for the ticker (may not exist)
            try:
                position = client.get_open_position(ticker)
                existing_position_value = float(position.market_value)
            except Exception:
                # No position exists for this ticker (expected case) — not an error
                existing_position_value = 0.0

            # Compute deterministic sizing and concentration
            proposed_notional_usd = risk_position_size_pct * portfolio_total_value
            if portfolio_total_value > 0:
                risk_concentration_pct = (existing_position_value + proposed_notional_usd) / portfolio_total_value
            else:
                # Edge case: zero equity (shouldn't happen on a live account, but handle safely)
                risk_concentration_pct = 1.0

            return {
                **state,
                "proposed_side": proposed_side,
                "portfolio_snapshot_status": "pass",
                "portfolio_total_value": portfolio_total_value,
                "existing_position_value": existing_position_value,
                "proposed_notional_usd": proposed_notional_usd,
                "risk_concentration_pct": risk_concentration_pct,
            }

        except Exception as exc:
            # Alpaca account fetch failed (network, API error, etc.)
            # Return fail-safe: concentration=1.0 forces aggregator to veto (RISK-01 safety net)
            logger.exception("Portfolio snapshot failed: %s", exc)
            return {
                **state,
                "proposed_side": proposed_side,
                "portfolio_snapshot_status": "fail",
                "portfolio_total_value": 0.0,
                "existing_position_value": 0.0,
                "proposed_notional_usd": 0.0,
                "risk_concentration_pct": 1.0,
            }

    return portfolio_snapshot_node
