#!/bin/bash

# audit_auditor_isolation.sh
#
# Isolation verification script for the Auditor pipeline (AUDIT-03).
# This is a standalone, manually-run verification tool per D-11's explicit discretion.
# It is NOT wired into pytest or any CI gate.
#
# Exits with 0 if all checks pass (all lines show PASS).
# Exits with non-zero if any check fails.

echo "=== Auditor Isolation Verification (AUDIT-03) ==="
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Check 1: Auditor compute node does not read state["market_report"], state["sentiment_report"], etc.
echo "Check 1: Compute node does not read shared analyst reports..."
if grep -E 'market_report|sentiment_report|news_report|fundamentals_report' tradingagents/agents/auditors/momentum_auditor.py > /dev/null 2>&1; then
    echo "  FAIL: Found forbidden shared report fields in compute node"
    ((FAIL_COUNT++))
else
    echo "  PASS: Compute node never reads shared reports"
    ((PASS_COUNT++))
fi

# Check 2: Auditor LLM node does not read state["market_report"], state["sentiment_report"], etc.
echo "Check 2: LLM node does not read shared analyst reports..."
if grep -E 'state\["(market_report|sentiment_report|news_report|fundamentals_report)"\]|state\.get\("(market_report|sentiment_report|news_report|fundamentals_report)"\)' tradingagents/agents/auditors/momentum_auditor.py > /dev/null 2>&1; then
    echo "  FAIL: Found forbidden shared report fields accessed via state"
    ((FAIL_COUNT++))
else
    echo "  PASS: LLM node never reads shared reports via state"
    ((PASS_COUNT++))
fi

# Check 3: Auditor LLM node creates isolated instance via create_llm_client
echo "Check 3: LLM node creates isolated instance..."
if grep -q "create_llm_client" tradingagents/agents/auditors/momentum_auditor.py; then
    echo "  PASS: LLM node uses create_llm_client (isolated instance)"
    ((PASS_COUNT++))
else
    echo "  FAIL: LLM node does not use create_llm_client factory"
    ((FAIL_COUNT++))
fi

# Check 4: Auditor code does not use self.llm or raw ChatAnthropic factory calls
echo "Check 4: Auditor does not use shared LLM instances..."
if grep -E 'self\.llm|ChatAnthropic\(' tradingagents/agents/auditors/*.py > /dev/null 2>&1; then
    echo "  FAIL: Found shared LLM instance references"
    ((FAIL_COUNT++))
else
    echo "  PASS: Auditor never uses shared LLM instances"
    ((PASS_COUNT++))
fi

# Check 5: graph/setup.py does not contain direct bypass edge "Research Manager" -> "Trader"
echo "Check 5: No direct bypass edge from Research Manager to Trader..."
if grep -q 'add_edge("Research Manager", "Trader")' tradingagents/graph/setup.py; then
    echo "  FAIL: Found direct bypass edge Research Manager -> Trader"
    ((FAIL_COUNT++))
else
    echo "  PASS: Direct bypass edge has been replaced with Auditor chain"
    ((PASS_COUNT++))
fi

# Check 6: agent_states.py contains Auditor state fields (auditor_verdict, auditor_data_points, etc.)
echo "Check 6: AgentState contains Auditor fields..."
AUDITOR_FIELD_COUNT=$(grep -c "auditor_" tradingagents/agents/utils/agent_states.py || echo 0)
if [ "$AUDITOR_FIELD_COUNT" -ge 10 ]; then
    echo "  PASS: Found $AUDITOR_FIELD_COUNT auditor_* field declarations"
    ((PASS_COUNT++))
else
    echo "  FAIL: Expected >=10 auditor_* fields, found $AUDITOR_FIELD_COUNT"
    ((FAIL_COUNT++))
fi

echo ""
echo "=== Results ==="
echo "PASS: $PASS_COUNT"
echo "FAIL: $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "✓ All isolation checks passed"
    exit 0
else
    echo "✗ Some isolation checks failed"
    exit 1
fi
