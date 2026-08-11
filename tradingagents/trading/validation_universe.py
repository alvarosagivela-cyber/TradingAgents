"""Fixed ticker basket for Phase 7 paper trading validation (D-03/D-04).

D-03: A fixed, sector-varied, 10-ticker basket for the full Phase 7 window,
expanding the original 4 (AAPL/XOM/JPM/ETSY) to generate more accumulated decisions
for the Phase 6 false-positive report.

D-04: This module contains zero automatic instrument discovery or LLM-driven
ticker-selection logic. The basket is a hardcoded Python list, and no function
in this module calls out to any external discovery mechanism.

The 10 tickers represent 10 distinct GICS sectors (one ticker per sector):
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
"""

# D-03: Fixed, expanded 10-ticker basket
# D-04: No discovery logic (this is a static list only)
PHASE7_TICKER_BASKET: list[str] = [
    "AAPL",   # Technology
    "XOM",    # Energy
    "JPM",    # Financials
    "ETSY",   # Consumer Discretionary
    "JNJ",    # Health Care
    "PG",     # Consumer Staples
    "NEE",    # Utilities
    "CAT",    # Industrials
    "FCX",    # Materials
    "AMT",    # Real Estate
]
