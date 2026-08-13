"""Fixed instrument basket for Phase 7 paper trading validation (D-03/D-04).

D-03: A fixed, sector-varied basket for the full Phase 7 window, expanding the
original 4 (AAPL/XOM/JPM/ETSY) to generate more accumulated decisions for the
Phase 6 false-positive report, plus gold/silver (via ETF proxy) and BTC to widen
asset-class coverage beyond US equities.

D-04: This module contains zero automatic instrument discovery or LLM-driven
ticker-selection logic. The basket is a hardcoded Python list, and no function
in this module calls out to any external discovery mechanism.

10 tickers represent 10 distinct GICS sectors (one ticker per sector):
1. Technology (AAPL)
2. Energy (XOM)
3. Financials (JPM)
4. Consumer Discretionary (ETSY)
5. Health Care (JNJ)
6. Consumer Staples (PG)
7. Utilities (NEE)
8. Industrials (CAT)
9. Materials (FCX)
10. Real Estate (AMT)

Plus 3 non-equity instruments for asset-class diversity:
11. Gold, via the SPDR Gold Shares ETF (GLD) -- Alpaca has no direct commodities/
    futures market, so the ETF proxy is what's actually tradable through this
    pipeline's broker integration. Ordinary equity ticker as far as data-fetch,
    momentum calc, and order execution are concerned -- no special-casing needed.
12. Silver, via the iShares Silver Trust ETF (SLV) -- same rationale as gold.
13. Bitcoin (BTC-USD, the yfinance/data-pipeline ticker convention). Trades 24/7,
    so momentum_calculator.py uses calendar-day windows instead of trading-day
    windows for this ticker, and paper_execution.py translates the symbol to
    Alpaca's BTC/USD format and uses time_in_force=GTC (Alpaca rejects DAY for
    crypto orders) -- see the crypto-handling notes in both modules.
"""

# D-03: Fixed, expanded instrument basket
# D-04: No discovery logic (this is a static list only)
PHASE7_TICKER_BASKET: list[str] = [
    "AAPL",     # Technology
    "XOM",      # Energy
    "JPM",      # Financials
    "ETSY",     # Consumer Discretionary
    "JNJ",      # Health Care
    "PG",       # Consumer Staples
    "NEE",      # Utilities
    "CAT",      # Industrials
    "FCX",      # Materials
    "AMT",      # Real Estate
    "GLD",      # Gold (ETF proxy)
    "SLV",      # Silver (ETF proxy)
    "BTC-USD",  # Bitcoin (24/7 crypto)
]
