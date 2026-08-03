"""Deterministic aggregation and persistence for Risk Squad verdicts.

This module aggregates the three perspective verdicts (conservative/balanced/aggressive)
into a final binary VETO/APPROVE decision, with a deterministic fail-safe override
and unconditional audit trail persistence (RISK-01, D-06 discipline).

Key responsibilities:
1. Compute final_veto from any-perspective-VETO OR portfolio-snapshot-fail (RISK-01)
2. Validate concentration values cited by perspectives against computed values (D-11)
3. Persist ALL cycles (vetoed, approved, hallucinated) to JSONL unconditionally (D-06)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

# Concentration tolerance: perspectives must cite within ±0.5pp (0.005)
RISK_CONCENTRATION_TOLERANCE_PCT = 0.005


def create_risk_aggregator():
    """Factory for the Risk Squad's deterministic aggregator node.

    Returns a node function that:
    1. Reads the three perspective verdicts (conservative/balanced/aggressive)
    2. Computes any_llm_veto and applies the fail-safe override rule (RISK-01)
    3. Validates perspective concentration values (D-11)
    4. Calls _persist_risk_cycle(...) to unconditionally append to JSONL (D-06)

    Returns:
        Callable[[dict], dict]: Node function that takes state dict, returns update dict
                               with final_veto, concentration_verified, execution_log
    """

    def aggregator_node(state: dict) -> dict:
        """Aggregate Risk Squad verdicts into a final veto decision.

        Args:
            state: AgentState dict with conservative/balanced/aggressive_verdict fields

        Returns:
            Dict with final_veto, concentration_verified, execution_log (updated)
        """
        ticker = state.get("company_of_interest", "UNKNOWN")
        logger.info(f"Risk Aggregator: Processing {ticker}")

        # Read the three perspective verdicts (D-02: isolated field reads)
        conservative = state.get("conservative_verdict", {})
        balanced = state.get("balanced_verdict", {})
        aggressive = state.get("aggressive_verdict", {})

        # Compute any_llm_veto: True if ANY perspective returned VETO
        any_llm_veto = any(
            v.get("verdict") == "VETO" for v in (conservative, balanced, aggressive)
        )
        logger.info(f"Risk Aggregator {ticker}: any_llm_veto={any_llm_veto}")

        # Compute final_veto: the deterministic RISK-01 rule (fail-safe override, never bypassed)
        portfolio_snapshot_status = state.get("portfolio_snapshot_status", "fail")
        final_veto = any_llm_veto or portfolio_snapshot_status == "fail"
        logger.info(
            f"Risk Aggregator {ticker}: final_veto={final_veto} "
            f"(any_llm_veto={any_llm_veto}, snapshot_status={portfolio_snapshot_status})"
        )

        # Validate concentration values cited by perspectives (D-11 discipline applied 3x)
        concentration_verified = _validate_risk_concentration(state)
        logger.info(f"Risk Aggregator {ticker}: concentration_verified={concentration_verified}")

        # Build audit trail entry
        execution_log = state.get("execution_log", [])
        audit_entry = (
            f"Risk cycle: {conservative.get('verdict', 'N/A')} | "
            f"{balanced.get('verdict', 'N/A')} | "
            f"{aggressive.get('verdict', 'N/A')} → "
            f"final_veto={final_veto}, verified={concentration_verified}"
        )
        execution_log.append(audit_entry)

        # Persist the cycle UNCONDITIONALLY (D-06: including vetoed, including not_reached)
        _persist_risk_cycle(state, final_veto, concentration_verified)

        return {
            "final_veto": final_veto,
            "concentration_verified": concentration_verified,
            "execution_log": execution_log,
        }

    return aggregator_node


def _validate_risk_concentration(state: dict) -> bool:
    """Validate that all perspectives cited concentration within tolerance (D-11, RISK-02).

    Tolerance threshold (0.5pp = 0.005 absolute):
    - Perspectives must cite within ±0.005 of state's computed risk_concentration_pct

    This validation is orthogonal to final_veto: even if all perspectives approve,
    if they hallucinate concentration values, concentration_verified remains False.

    Args:
        state: AgentState dict with all three verdict dicts and risk_concentration_pct

    Returns:
        True if all three perspectives cite concentration within tolerance, False otherwise
    """
    computed_concentration = state.get("risk_concentration_pct", 1.0)
    conservative = state.get("conservative_verdict", {})
    balanced = state.get("balanced_verdict", {})
    aggressive = state.get("aggressive_verdict", {})

    for perspective_name, verdict in [
        ("conservative", conservative),
        ("balanced", balanced),
        ("aggressive", aggressive),
    ]:
        cited_concentration = verdict.get("cited_concentration_pct")

        if cited_concentration is None:
            logger.warning(
                f"Risk concentration validation: {perspective_name} missing cited_concentration_pct"
            )
            return False

        if abs(cited_concentration - computed_concentration) > RISK_CONCENTRATION_TOLERANCE_PCT:
            logger.warning(
                f"Risk concentration validation: {perspective_name} hallucination detected "
                f"(cited={cited_concentration:.4f}, computed={computed_concentration:.4f}, "
                f"tolerance={RISK_CONCENTRATION_TOLERANCE_PCT:.4f})"
            )
            return False

    return True


def _persist_risk_cycle(state: dict, final_veto: bool, concentration_verified: bool) -> None:
    """Append Risk Squad cycle to JSONL audit trail (D-06, RISK-01).

    This persistence is UNCONDITIONAL across all final_veto values:
    - True: Vetoed cycle (critical for measuring risk-layer effectiveness)
    - False: Approved cycle (routine execution case)

    Creates parent directories if needed. Each line is a JSON record containing:
    - timestamp (UTC ISO format)
    - ticker, trade_date, proposed_side
    - portfolio snapshot status and values
    - risk concentration metrics
    - all three verdict dicts (conservative, balanced, aggressive)
    - final_veto, concentration_verified (aggregation results)

    Args:
        state: AgentState dict with all risk/verdict/portfolio fields
        final_veto: Boolean aggregation result
        concentration_verified: Boolean validation result
    """
    config = get_config()
    risk_log_path = config.get("risk_log_path") if config else None
    if not risk_log_path:
        logger.error("risk_log_path not configured; cannot persist Risk cycle")
        return

    log_path = Path(risk_log_path)

    # Create parent directory if needed
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the audit record
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ticker": state.get("company_of_interest"),
        "trade_date": state.get("trade_date"),
        "proposed_side": state.get("proposed_side"),
        "portfolio_snapshot_status": state.get("portfolio_snapshot_status"),
        "portfolio_total_value": state.get("portfolio_total_value"),
        "existing_position_value": state.get("existing_position_value"),
        "proposed_notional_usd": state.get("proposed_notional_usd"),
        "risk_concentration_pct": state.get("risk_concentration_pct"),
        "conservative_verdict": state.get("conservative_verdict"),
        "balanced_verdict": state.get("balanced_verdict"),
        "aggressive_verdict": state.get("aggressive_verdict"),
        "final_veto": final_veto,
        "concentration_verified": concentration_verified,
    }

    # Append to JSONL (UNCONDITIONAL, D-06, RISK-01)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            f"Persisted Risk cycle to {log_path} (final_veto={final_veto}, verified={concentration_verified})"
        )
    except Exception as exc:
        logger.exception(f"Failed to persist Risk cycle to {log_path}: {exc}")
