"""Unit tests for validation_universe module (D-03/D-04: fixed ticker basket, no discovery).

Tests verify:
- D-03: Fixed, sector-varied 10-ticker basket for Phase 7 window
- D-04: Zero market-scanning, news-scanning, or LLM-driven ticker-selection logic
"""

import re
from tradingagents.trading.validation_universe import PHASE7_TICKER_BASKET


def test_basket_has_ten_sector_varied_tickers():
    """D-03: PHASE7_TICKER_BASKET has exactly 10 unique uppercase string tickers.

    Includes all 4 originals (AAPL, XOM, JPM, ETSY) and 6 additional sector-varied tickers.
    """
    # Exactly 10 tickers
    assert len(PHASE7_TICKER_BASKET) == 10, \
        f"Expected 10 tickers, got {len(PHASE7_TICKER_BASKET)}"

    # All unique (no duplicates)
    assert len(set(PHASE7_TICKER_BASKET)) == 10, \
        f"Duplicate tickers found: {PHASE7_TICKER_BASKET}"

    # All uppercase strings
    for ticker in PHASE7_TICKER_BASKET:
        assert isinstance(ticker, str), f"Ticker {ticker} is not a string"
        assert ticker.isupper(), f"Ticker {ticker} is not uppercase"

    # Includes the 4 originals
    required_originals = {"AAPL", "XOM", "JPM", "ETSY"}
    assert required_originals.issubset(set(PHASE7_TICKER_BASKET)), \
        f"Missing original tickers. Required: {required_originals}, Got: {set(PHASE7_TICKER_BASKET)}"


def test_basket_module_has_no_discovery_logic():
    """D-04: Negative guarantee - module contains zero discovery/scanning/news logic.

    Checks actual CODE tokens (not comments, not string/docstring literals) for
    none of: "requests.get", "scan", "discover", "news" (case-insensitive). Using
    tokenize rather than a line-based comment strip means the module's own D-04
    docstring can freely explain "no discovery logic" in prose without tripping
    the guard on its own words -- only executable identifiers/calls count.
    """
    import inspect
    import io
    import tokenize

    import tradingagents.trading.validation_universe as module

    source = inspect.getsource(module)

    excluded_types = {
        tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
        tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
    }
    code_tokens = [
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type not in excluded_types
    ]
    code_only_lower = " ".join(code_tokens).lower()

    # Search for forbidden substrings (case-insensitive) in code only
    forbidden = ["requests.get", "scan", "discover", "news"]

    for pattern in forbidden:
        assert pattern not in code_only_lower, \
            f"D-04 violation: found forbidden pattern '{pattern}' in module code"
