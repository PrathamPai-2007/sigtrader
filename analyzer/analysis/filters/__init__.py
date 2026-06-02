"""Entry filters — gate which setups are tradable before the scorer emits them.

  long_entry_filters.py  — momentum, structure, trend, and trap checks for LONG entries
  long_pipeline_log.py   — debug instrumentation for the LONG filter funnel
"""
from analyzer.analysis.filters.long_entry_filters import (
    MomentumExpansionResult,
    DelayedEntryResult,
    StructureConfirmationResult,
    TrendDominanceResult,
    PullbackTrapResult,
    LongEntryFilterResult,
    momentum_expansion_check,
    delayed_entry_confirmation,
    structure_confirmation_check,
    check_trend_dominance,
    detect_pullback_trap,
    apply_long_entry_filters,
    adjust_long_confidence_threshold,
)
from analyzer.analysis.filters.long_pipeline_log import LongPipelineLog
