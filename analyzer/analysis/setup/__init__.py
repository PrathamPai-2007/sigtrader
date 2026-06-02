"""Trade setup construction — entry/stop/target placement and quality scoring.

  geometry.py          — entry/stop/target placement, swing points, quality score
  leverage.py          — leverage suggestion and drawdown adjustment
  enhanced_metrics.py  — supplementary market microstructure metrics
  reversal_confluence.py — reversal signal and confluence detection (pure functions)
"""
from analyzer.analysis.setup.geometry import (
    find_swing_points,
    select_best_stop,
    select_best_target,
    place_entry_stop_target,
    geometry_quality_score,
    _compute_entry,
    _build_stop_candidates,
    _build_target_candidates,
)
from analyzer.analysis.setup.leverage import (
    _leverage_suggestion,
    DrawdownAdjuster,
)
from analyzer.analysis.setup.enhanced_metrics import (
    _calculate_enhanced_metrics,
)
from analyzer.analysis.setup.reversal_confluence import (
    _detect_early_reversal_signals,
    _round_number_proximity,
    _score_confluence,
)
