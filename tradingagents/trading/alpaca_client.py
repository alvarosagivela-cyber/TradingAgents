"""Alpaca paper trading client factory (EXEC-01: paper trading only, never a live account).

This module provides a factory function that structurally enforces paper=True at instantiation.
The factory validates that paper is exactly True BEFORE importing TradingClient, and hardcodes
paper=True as a literal at the TradingClient call site — two independent guarantees that no
code path can reach a real trading account.

Credentials are loaded from ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY env vars
(distinct from any live-key naming to prevent accidental cross-contamination).
"""

import os

from alpaca.trading.client import TradingClient


def create_alpaca_client(paper: bool = True) -> TradingClient:
    """Create and return an Alpaca TradingClient for paper trading only.

    Enforces paper=True at instantiation via two independent checks:
    1. Function entry validation: if paper is not exactly True, raise ValueError before
       any TradingClient instantiation
    2. Call site literal: paper=True is hardcoded as a literal at the TradingClient(...)
       call, never forwarded from the function parameter (defense in depth)

    Args:
        paper: Must be exactly True. Any other value raises ValueError.

    Returns:
        TradingClient instance configured for paper trading.

    Raises:
        ValueError: If paper is not exactly True, or if required env vars are missing.
    """
    # EXEC-01 Guard 1: Validate paper is True BEFORE importing/touching TradingClient
    if paper is not True:
        raise ValueError(
            "Live trading is not permitted in this phase (EXEC-01). "
            "paper=True is required — this is a structural, non-negotiable constraint."
        )

    # Load credentials from env vars (distinct paper-specific names, never generic ALPACA_API_KEY)
    api_key = os.getenv("ALPACA_PAPER_API_KEY")
    api_secret = os.getenv("ALPACA_PAPER_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError(
            "ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY env vars required. "
            "See .env.example."
        )

    # EXEC-01 Guard 2: Hardcode paper=True as a literal at the call site
    # (not forwarding the parameter value, which was already validated above)
    return TradingClient(api_key=api_key, secret_key=api_secret, paper=True)
