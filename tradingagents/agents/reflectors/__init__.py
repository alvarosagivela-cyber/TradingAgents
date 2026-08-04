"""Per-layer reflection storage and queries for Phase 5 memory/feedback loop.

This package implements the foundational reflection infrastructure for the
research/auditor/risk layers (MEM-01). Each layer maintains its own isolated
reflection store, queryable by ticker, to remember past decisions and their
realized outcomes for injection into future reasoning.

This is a new, separate system from the pre-existing TradingMemoryLog/Reflector
machinery (agents/utils/memory.py, graph/reflection.py), which is reserved for
the Bull/Bear/Trader/Portfolio-Manager pipeline.
"""
