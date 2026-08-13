"""Reflection record schema and utility functions for Phase 5 memory/feedback loop.

This module defines the data contract for per-layer reflections: the
ReflectionRecord frozen dataclass plus helper functions for classification
and lesson text generation.

Reflections are pure data (never LLM-generated content re-injected into later
LLM calls), enabling safe deterministic injection into future prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Thresholds for classifying realized returns (Claude's Discretion per 05-CONTEXT.md)
NEGATIVE_RETURN_THRESHOLD = -0.01
POSITIVE_RETURN_THRESHOLD = 0.01


@dataclass(frozen=True)
class ReflectionRecord:
    """A frozen record of a past decision and its realized outcome.

    Attributes:
        ticker: Stock ticker symbol (e.g. "AAPL")
        decision_date: Date the decision was made (YYYY-MM-DD format)
        decision_verdict: The decision made (e.g. "Buy", "Sell", "Hold", "VETO", "APPROVE")
        realized_return: Actual return over the observation window (float, e.g. -0.03 for -3%)
        classification: Outcome classification ("positive", "negative", or "neutral")
        lesson_text: Deterministic prose describing the outcome
        created_at: ISO 8601 UTC timestamp of when this record was created
    """

    ticker: str
    decision_date: str
    decision_verdict: str
    realized_return: float
    classification: str
    lesson_text: str
    created_at: str

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dict.

        Returns:
            Dict with all seven fields suitable for json.dumps()
        """
        return {
            "ticker": self.ticker,
            "decision_date": self.decision_date,
            "decision_verdict": self.decision_verdict,
            "realized_return": self.realized_return,
            "classification": self.classification,
            "lesson_text": self.lesson_text,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> ReflectionRecord:
        """Deserialize from a JSON dict.

        Args:
            data: Dict with seven keys matching the ReflectionRecord fields

        Returns:
            Reconstructed ReflectionRecord instance
        """
        return cls(
            ticker=data["ticker"],
            decision_date=data["decision_date"],
            decision_verdict=data["decision_verdict"],
            realized_return=data["realized_return"],
            classification=data["classification"],
            lesson_text=data["lesson_text"],
            created_at=data["created_at"],
        )


def classify_realized_return(realized_return: float) -> str:
    """Classify a realized return as positive, negative, or neutral.

    Uses strict < and > comparisons:
    - realized_return < NEGATIVE_RETURN_THRESHOLD (-0.01) → "negative"
    - realized_return > POSITIVE_RETURN_THRESHOLD (0.01) → "positive"
    - otherwise → "neutral" (includes exactly 0)

    Args:
        realized_return: The realized return (e.g. -0.03 for -3%)

    Returns:
        One of "negative", "positive", or "neutral"
    """
    if realized_return < NEGATIVE_RETURN_THRESHOLD:
        return "negative"
    if realized_return > POSITIVE_RETURN_THRESHOLD:
        return "positive"
    return "neutral"


def build_lesson_text(
    decision_verdict: str, realized_return: float, classification: str, window_days: int
) -> str:
    """Generate deterministic lesson prose from a decision outcome.

    This function is pure Python with no LLM call in its path. The "lesson"
    can be safely injected into future prompts as pre-computed data.

    Args:
        decision_verdict: The decision that was made (e.g. "Buy", "Sell", "Hold")
        realized_return: The actual return observed (e.g. -0.03)
        classification: The classification of that return ("positive", "negative", "neutral")
        window_days: The number of trading days the window spanned (e.g. 10)

    Returns:
        A single deterministic sentence citing all parameters
    """
    formatted_return = f"{realized_return:+.2%}"
    return (
        f"{decision_verdict} decision resulted in a {formatted_return} return "
        f"over the {window_days}-trading-day window, classified as {classification.upper()}."
    )
