import contextlib
import re
from typing import Any

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .cost_tracker import record_cost_from_response
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens", "temperature",
    "callbacks", "http_client", "http_async_client", "effort",
)

# Anthropic's extended-thinking ``effort`` parameter is accepted by Opus 4.5+,
# Sonnet 4.6+, and the Claude 5 family (Sonnet 5, Fable 5). Sonnet 4.5 and any
# Haiku version 400 with ``"This model does not support the effort parameter"``
# (#831). Versions may be dotted (``opus-4-8``) or single-number (``sonnet-5``,
# ``fable-5``); the per-family minimum below is forward-compatible.
_EFFORT_EXACT = {
    "claude-mythos-preview",  # non-standard preview name; effort-capable
    "claude-mythos-5",        # Fable 5 twin (Project Glasswing); effort-capable
}
_EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet|fable)-(\d+)(?:-(\d+))?$")
_EFFORT_MIN_VERSION = {"opus": (4, 5), "sonnet": (4, 6), "fable": (5, 0)}


def _supports_effort(model: str) -> bool:
    """Whether Anthropic accepts the ``effort`` parameter for this model."""
    model_lc = model.lower()
    if model_lc in _EFFORT_EXACT:
        return True
    match = _EFFORT_MODEL.match(model_lc)
    if not match:
        return False
    family = match.group(1)
    major = int(match.group(2))
    minor = int(match.group(3)) if match.group(3) else 0
    return (major, minor) >= _EFFORT_MIN_VERSION[family]


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output and cost tracking.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.

    Cost tracking (Phase 6, D-03/D-04):
    - Captures usage_metadata from every invoke() call
    - Records cost to JSONL without blocking pipeline (D-05: fail-closed)
    - Labels records with layer/model/ticker/trade_date for audit trail

    New fields (optional):
        cost_layer: Layer name for cost record (default: "unclassified")
        cost_ticker: Stock ticker for cost record (default: "")
        cost_trade_date: Trade date for cost record (default: "")
    """

    # Pydantic-declared fields with defaults (compatible with installed langchain-anthropic)
    cost_layer: str = "unclassified"
    cost_ticker: str = ""
    cost_trade_date: str = ""

    def invoke(self, input, config=None, **kwargs):
        response = super().invoke(input, config, **kwargs)

        # Capture cost from usage_metadata (defense in depth: even if
        # record_cost_from_response fails, we still return the response).
        # D-05: cost tracking never blocks the pipeline.
        # record_cost_from_response already logs all exceptions internally,
        # so we just swallow here for defense in depth.
        with contextlib.suppress(Exception):
            record_cost_from_response(
                response,
                layer=self.cost_layer,
                model=self.model,
                ticker=self.cost_ticker,
                trade_date=self.cost_trade_date,
            )

        return normalize_content(response)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "effort" and not _supports_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # Extract cost tracking labels from kwargs (never added to ChatAnthropic itself,
        # only passed to our NormalizedChatAnthropic cost_* fields)
        cost_layer = self.kwargs.get("layer", "unclassified")
        cost_ticker = self.kwargs.get("ticker", "")
        cost_trade_date = self.kwargs.get("trade_date", "")

        return NormalizedChatAnthropic(
            **llm_kwargs,
            cost_layer=cost_layer,
            cost_ticker=cost_ticker,
            cost_trade_date=cost_trade_date,
        )

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
