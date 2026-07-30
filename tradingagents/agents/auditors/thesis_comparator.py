"""Comparison and persistence for Auditor verdicts.

This module compares the Auditor's verdict against Research's verdict and persists
all cycles (including refuted and inconclusive) to an append-only JSONL audit log.

Key difference from Research's validation (thesis_validator.py):
- Research persistence is verified-gated: only valid theses are persisted
- Auditor persistence is UNCONDITIONAL: ALL comparison results (match/mismatch/not_reached)
  are persisted, even refuted cycles and numeric hallucinations (D-06)

This deliberate asymmetry ensures the audit trail is complete and unbiased, enabling
Phase 7 to accurately measure the Auditor's value-add independent of verification status.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from tradingagents.agents.utils.thesis_validator import (
    TOLERANCE_RETORNO,
    TOLERANCE_Z_SCORE,
)
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)


def create_auditor_compare_node():
    """Factory for the Auditor's comparison and persistence node.

    Returns a node function that:
    1. Validates the Auditor's cited data_points against its own computed values (AUDIT-03)
    2. Compares the Auditor's verdict against Research's verdict (D-06)
    3. Calls _persist_audit_cycle(...) to unconditionally append the cycle to JSONL (D-06)

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
    """

    def compare_node(state: dict) -> dict:
        """Compare Auditor and Research verdicts; validate and persist the cycle.

        Args:
            state: AgentState dict with auditor and research verdict fields

        Returns:
            Dict with auditor_verified and comparison_result
        """
        ticker = state.get("company_of_interest")

        # Validate Auditor's data_points against its own computed values (AUDIT-03, D-11)
        auditor_verified = _validate_auditor_data_points(state)
        logger.info(f"Auditor {ticker}: verified={auditor_verified}")

        # Compute comparison result (match/mismatch/not_reached)
        auditor_verdict = state.get("auditor_verdict", "")
        research_verdict = state.get("thesis_verdict", "")

        if auditor_verdict == "INCONCLUSIVE":
            comparison_result = "not_reached"
        elif auditor_verdict == research_verdict:
            comparison_result = "match"
        else:
            comparison_result = "mismatch"

        logger.info(f"Auditor {ticker}: comparison_result={comparison_result} (research={research_verdict}, auditor={auditor_verdict})")

        # Persist the cycle UNCONDITIONALLY (D-06: including refuted, including not_reached)
        _persist_audit_cycle(state, {
            "auditor_verified": auditor_verified,
            "comparison_result": comparison_result,
        })

        return {
            "auditor_verified": auditor_verified,
            "comparison_result": comparison_result,
        }

    return compare_node


def _validate_auditor_data_points(state: dict) -> bool:
    """Validate that Auditor-cited data_points match its own computed values (AUDIT-03, D-11).

    Tolerance thresholds (reused from Research's thesis_validator):
        retorno_12_1: ±0.0001 (4 decimal places)
        z_score: ±0.01 (2 decimal places)

    Unlike Research's validation which gates persistence, this validation only sets
    the auditor_verified flag — persistence is unconditional (D-06).

    Args:
        state: AgentState dict

    Returns:
        True if data_points match computed values within tolerance, False otherwise
    """
    # If Auditor never reached LLM (INCONCLUSIVE), skip validation
    if state.get("auditor_verdict") == "INCONCLUSIVE":
        return False

    # Extract Auditor's own computed values
    computed_retorno = state.get("auditor_retorno_12_1")
    computed_z_score = state.get("auditor_z_score")
    auditor_data_points = state.get("auditor_data_points", {})

    # Require non-empty data_points
    if not auditor_data_points:
        logger.warning("Auditor validation: empty auditor_data_points")
        return False

    # Extract Auditor-reported values from the LLM
    llm_retorno = auditor_data_points.get("retorno_12_1")
    llm_z_score = auditor_data_points.get("z_score")

    # Check for missing keys or None values
    if llm_retorno is None or llm_z_score is None:
        logger.warning("Auditor validation: missing auditor_data_points keys (retorno_12_1 or z_score)")
        return False

    # Check tolerance for retorno_12_1
    if abs(llm_retorno - computed_retorno) > TOLERANCE_RETORNO:
        logger.warning(
            f"Auditor validation: retorno_12_1 mismatch "
            f"(computed={computed_retorno}, llm={llm_retorno}, tolerance={TOLERANCE_RETORNO})"
        )
        return False

    # Check tolerance for z_score
    if abs(llm_z_score - computed_z_score) > TOLERANCE_Z_SCORE:
        logger.warning(
            f"Auditor validation: z_score mismatch "
            f"(computed={computed_z_score}, llm={llm_z_score}, tolerance={TOLERANCE_Z_SCORE})"
        )
        return False

    return True


def _persist_audit_cycle(state: dict, result: dict) -> None:
    """Append Auditor cycle to JSONL audit trail (D-06, D-13).

    This persistence is UNCONDITIONAL across all comparison_result values:
    - "match": Auditor and Research agree
    - "mismatch": Auditor refutes Research (this is the critical refutation test case)
    - "not_reached": Auditor's compute phase failed, LLM never invoked

    Creates parent directories if needed. Each line is a JSON record containing:
    - timestamp (UTC ISO format)
    - ticker, trade_date, phase1_status, phase1_failure_reason
    - auditor_retorno_12_1, auditor_z_score (the Auditor's own computed values)
    - auditor_verdict, auditor_reasoning, auditor_refutation_criterion
    - research_verdict (what the Auditor is tasked with refuting)
    - auditor_data_points (LLM-cited values for validation)
    - comparison_result (match/mismatch/not_reached)
    - verified (auditor_verified: whether LLM-cited data matched computed values)

    Args:
        state: AgentState dict with populated auditor fields
        result: Dict with auditor_verified and comparison_result
    """
    config = get_config()
    log_path = Path(config.get("auditor_log_path"))

    # Create parent directory if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the audit record
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ticker": state.get("company_of_interest"),
        "trade_date": state.get("trade_date"),
        "phase1_status": state.get("auditor_phase1_status"),
        "phase1_failure_reason": state.get("auditor_phase1_failure_reason"),
        "auditor_retorno_12_1": state.get("auditor_retorno_12_1"),
        "auditor_z_score": state.get("auditor_z_score"),
        "auditor_verdict": state.get("auditor_verdict"),
        "auditor_reasoning": state.get("auditor_reasoning"),
        "auditor_refutation_criterion": state.get("auditor_refutation_criterion"),
        "research_verdict": state.get("thesis_verdict"),
        "auditor_data_points": state.get("auditor_data_points"),
        "comparison_result": result.get("comparison_result"),
        "verified": result.get("auditor_verified"),
    }

    # Append to JSONL (UNCONDITIONAL, D-06)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(f"Persisted Auditor cycle to {log_path} (comparison_result={result.get('comparison_result')})")
    except Exception as exc:
        logger.exception(f"Failed to persist Auditor cycle to {log_path}: {exc}")
