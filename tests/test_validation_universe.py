"""Unit tests for validation_universe module (D-03/D-04: fixed ticker basket, no discovery).

Tests verify:
- D-03: Fixed, sector-varied 13-instrument basket for Phase 7 window (10 equities
  across 10 GICS sectors + gold/silver ETF proxies + BTC-USD)
- D-04: Zero market-scanning, news-scanning, or LLM-driven ticker-selection logic
"""

from tradingagents.trading.validation_universe import PHASE7_TICKER_BASKET


def test_basket_has_thirteen_instruments():
    """D-03: PHASE7_TICKER_BASKET has exactly 13 unique uppercase string tickers.

    Includes all 4 originals (AAPL, XOM, JPM, ETSY), 6 additional sector-varied
    tickers, and 3 non-equity instruments (GLD, SLV, BTC-USD).
    """
    # Exactly 13 tickers
    assert len(PHASE7_TICKER_BASKET) == 13, \
        f"Expected 13 tickers, got {len(PHASE7_TICKER_BASKET)}"

    # All unique (no duplicates)
    assert len(set(PHASE7_TICKER_BASKET)) == 13, \
        f"Duplicate tickers found: {PHASE7_TICKER_BASKET}"

    # All uppercase strings
    for ticker in PHASE7_TICKER_BASKET:
        assert isinstance(ticker, str), f"Ticker {ticker} is not a string"
        assert ticker.isupper(), f"Ticker {ticker} is not uppercase"

    # Includes the 4 original equities
    required_originals = {"AAPL", "XOM", "JPM", "ETSY"}
    assert required_originals.issubset(set(PHASE7_TICKER_BASKET)), \
        f"Missing original tickers. Required: {required_originals}, Got: {set(PHASE7_TICKER_BASKET)}"

    # Includes the 3 non-equity instruments
    required_non_equity = {"GLD", "SLV", "BTC-USD"}
    assert required_non_equity.issubset(set(PHASE7_TICKER_BASKET)), \
        f"Missing non-equity instruments. Required: {required_non_equity}, Got: {set(PHASE7_TICKER_BASKET)}"


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
