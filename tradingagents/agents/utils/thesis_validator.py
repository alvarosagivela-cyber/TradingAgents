"""Post-LLM validation and audit-trail persistence for research theses.

This module validates that LLM-reported numeric data-points match the Python-computed
ground truth within tight tolerances (D-11), and persists valid theses to an append-only
JSONL audit log with explicit scope disclosure (D-12, D-13).

Rejected theses (numeric mismatches) are never persisted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

# Validation tolerances (exact values from D-11, `02-CONTEXT.md`)
TOLERANCE_RETORNO = 0.0001
TOLERANCE_Z_SCORE = 0.01

# Banned phrases for refutation criterion quality check (D-17)
VAGUE_REFUTATION_PHRASES = (
    "not 100% certain",
    "there could be",
    "markets can be irrational",
    "hard to say",
    "no way to know",
)


def create_momentum_validate_node():
    """Factory for the validation node in the momentum research pipeline.

    Returns a node function that:
    1. Calls validate_thesis(state) to check numeric accuracy
    2. On verified=True, calls _persist_thesis_to_disk(state) for audit trail

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
    """

    def validate_node(state: dict) -> dict:
        """Validate thesis and optionally persist to disk.

        Args:
            state: AgentState dict with computed and LLM-reported values

        Returns:
            Dict with at minimum {verified: bool}; includes refutation_quality if verified=True
        """
        result = validate_thesis(state)
        if result.get("verified"):
            _persist_thesis_to_disk(state)
        return result

    return validate_node


def validate_thesis(state: dict) -> dict:
    """Validate that LLM-reported data-points match Python-computed values (D-11).

    Tolerance thresholds:
        retorno_12_1: ±0.0001 (4 decimal places)
        z_score: ±0.01 (2 decimal places)

    Also requires:
        - data_points dict is non-empty and contains both keys
        - thesis_verdict is non-empty
        - refutation_criterion is non-empty

    Args:
        state: AgentState dict

    Returns:
        {verified: True/False, refutation_quality: "ok"|"needs_review"} if verified=True
        {verified: False} if validation fails
    """
    # Extract values
    computed_retorno = state.get("retorno_12_1")
    computed_z_score = state.get("momentum_z_score")
    data_points = state.get("data_points", {})
    thesis_verdict = state.get("thesis_verdict", "")
    refutation_criterion = state.get("refutation_criterion", "")

    # Require non-empty data_points
    if not data_points:
        logger.warning("Thesis validation failed: empty data_points dict")
        return {"verified": False}

    # Extract LLM-reported values
    llm_retorno = data_points.get("retorno_12_1")
    llm_z_score = data_points.get("z_score")

    # Check for missing keys or None values
    if llm_retorno is None or llm_z_score is None:
        logger.warning("Thesis validation failed: missing data_points keys (retorno_12_1 or z_score)")
        return {"verified": False}

    # Check tolerance for retorno_12_1
    if abs(llm_retorno - computed_retorno) > TOLERANCE_RETORNO:
        logger.warning(
            f"Thesis validation failed: retorno_12_1 mismatch "
            f"(computed={computed_retorno}, llm={llm_retorno}, tolerance={TOLERANCE_RETORNO})"
        )
        return {"verified": False}

    # Check tolerance for z_score
    if abs(llm_z_score - computed_z_score) > TOLERANCE_Z_SCORE:
        logger.warning(
            f"Thesis validation failed: z_score mismatch "
            f"(computed={computed_z_score}, llm={llm_z_score}, tolerance={TOLERANCE_Z_SCORE})"
        )
        return {"verified": False}

    # Require non-empty verdict and refutation criterion
    if not thesis_verdict:
        logger.warning("Thesis validation failed: empty thesis_verdict")
        return {"verified": False}

    if not refutation_criterion:
        logger.warning("Thesis validation failed: empty refutation_criterion")
        return {"verified": False}

    # Check refutation quality (flag only, doesn't block verification — D-17)
    refutation_quality = _check_refutation_quality(refutation_criterion)

    return {
        "verified": True,
        "refutation_quality": refutation_quality,
    }


def _check_refutation_quality(criterion: str) -> str:
    """Check if refutation criterion contains vague language (D-17).

    Flags criteria containing banned phrases but does not reject the thesis.
    This is a guardrail for humans to review, not an automated gate.

    Args:
        criterion: The refutation criterion text

    Returns:
        "ok" if criterion looks substantive, "needs_review" if it contains vague language
    """
    criterion_lower = criterion.lower()
    for phrase in VAGUE_REFUTATION_PHRASES:
        if phrase in criterion_lower:
            return "needs_review"
    return "ok"


def _persist_thesis_to_disk(state: dict) -> None:
    """Append thesis to JSONL audit trail (D-13).

    Creates parent directories if needed. Each line is a JSON record containing:
    - timestamp (UTC ISO format)
    - ticker, trade_date, verdict
    - retorno_12_1, z_score, confidence_level (the verified metrics)
    - verified_fields: ["retorno_12_1", "z_score"] — scope disclosure (D-12)
    - reasoning, refutation_criterion
    - refutation_quality (from validate_thesis)

    Rejected theses (verified=False) should never reach this function.

    Args:
        state: AgentState dict with populated thesis fields
    """
    config = get_config()
    log_path = Path(config.get("research_thesis_log_path"))

    # Create parent directory if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the audit record
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ticker": state.get("company_of_interest"),
        "trade_date": state.get("trade_date"),
        "verdict": state.get("thesis_verdict"),
        "retorno_12_1": state.get("retorno_12_1"),
        "z_score": state.get("momentum_z_score"),
        "confidence_level": state.get("confidence_level"),
        "verified_fields": ["retorno_12_1", "z_score"],  # D-12: scope disclosure
        "reasoning": state.get("thesis"),
        "refutation_criterion": state.get("refutation_criterion"),
        "refutation_quality": state.get("refutation_quality", "ok"),
    }

    # Append to JSONL
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Persisted thesis to {log_path}")
    except Exception as exc:
        logger.exception(f"Failed to persist thesis to {log_path}: {exc}")
