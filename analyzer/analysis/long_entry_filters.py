# Backward-compatibility shim — canonical location is analyzer.analysis.filters.long_entry_filters
from analyzer.analysis.filters.long_entry_filters import *  # noqa: F401, F403
from analyzer.analysis.filters.long_entry_filters import MomentumExpansionResult, DelayedEntryResult, StructureConfirmationResult, TrendDominanceResult, PullbackTrapResult, LongEntryFilterResult, momentum_expansion_check, delayed_entry_confirmation, structure_confirmation_check, check_trend_dominance, detect_pullback_trap, apply_long_entry_filters, adjust_long_confidence_threshold
