"""Versioned pricing table for Anthropic Claude models (Phase 6, D-03).

This module contains the canonical source for Claude model pricing rates.
When Anthropic pricing changes, update PRICING_TABLE here once — all cost
calculations automatically use the new rates (no scattered hardcoding).

Pricing as of 2026-08-05 from https://platform.claude.com/docs/en/about-claude/pricing
Standard pricing (not batch, not cache discounts):
  - Claude Haiku 4.5: $1.00 / 1M input, $5.00 / 1M output
  - Claude Sonnet 5: $2.00 / 1M input, $10.00 / 1M output (introductory rates through 2026-08-31)
    After 2026-08-31, Sonnet 5 reverts to $3.00 / 1M input, $15.00 / 1M output

Cache hits cost 10% of the standard input price (Anthropic's prompt caching feature).
Currently no call site sets cache_control, so cache tokens are always 0 (YAGNI: cache-aware
pricing deferred until cache behavior is observed in production).

Updated: 2026-08-05
Valid until: 2026-08-31 (Sonnet 5 introductory rates expire; verify pricing before that date)
"""

# Per-million-token rates (input, output) in USD
PRICING_TABLE = {
    "claude-haiku-4-5": {
        "input_per_1m_usd": 1.00,
        "output_per_1m_usd": 5.00,
    },
    "claude-sonnet-5": {
        # NOTE: These are introductory rates (through 2026-08-31).
        # After 2026-08-31, update to input=3.00, output=15.00
        "input_per_1m_usd": 2.00,
        "output_per_1m_usd": 10.00,
    },
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate exact cost in USD given token counts and model.

    D-03 discipline: precise Python calculation from real usage_metadata,
    never estimation or averages.

    Args:
        model: Model identifier (must exist in PRICING_TABLE)
        input_tokens: Number of input tokens (from usage_metadata)
        output_tokens: Number of output tokens (from usage_metadata)

    Returns:
        Cost in USD (float)

    Raises:
        ValueError: If model is not in PRICING_TABLE (programmer error, fail loud)
    """
    rates = PRICING_TABLE.get(model)
    if not rates:
        raise ValueError(
            f"Unknown model for cost calculation: {model!r} — "
            f"add it to PRICING_TABLE in cost_pricing.py"
        )

    input_cost = input_tokens * rates["input_per_1m_usd"] / 1_000_000
    output_cost = output_tokens * rates["output_per_1m_usd"] / 1_000_000
    return input_cost + output_cost
