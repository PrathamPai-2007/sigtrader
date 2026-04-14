# Scorer.py Decomposition Guide

This document describes how to safely decompose `scorer.py` (~3000 lines) into modular components.

## Current Structure

The file contains these logical sections:

### 1. Data Models (Lines 1-170)
- `IndicatorBundle` - all indicator data
- `NormalizedSignals` - normalized [0,1] signals
- `EvidenceVector` - weighted evidence results
- `SwingPoints` - swing high/low data
- `EntryGeometry` - entry/stop/target geometry
- `_Contribution` - signal contribution details
- `_EvidenceSnapshot` - evidence state snapshot
- `_SideMetrics` - complete side metrics data

### 2. Geometry Functions (Lines 166-450)
- `find_swing_points()` - find swing pivots
- `select_best_stop()` - select optimal stop
- `select_best_target()` - select optimal target
- `place_entry_stop_target()` - complete geometry placement
- `geometry_quality_score()` - score geometry quality

### 3. Signal Processing (Lines 525-640)
- `normalize_signals()` - convert raw indicators to [0,1]
- `_oi_funding_biases()` - funding bias calculation

### 4. Evidence Computation (Lines 640-740)
- `compute_graded_evidence()` - weighted evidence scoring
- `_regime_weight_profile()` - regime-specific weights
- `_regime_alignment()` - regime alignment
- `_regime_penalty()` - regime penalties

### 5. Confidence Mapping (Lines 742-790)
- `logistic_confidence()` - logistic sigmoid mapping
- `logistic_confidence_from_config()` - config wrapper

### 6. Indicator Computation (Lines 790-1060)
- `_compute_atr()` - ATR calculation
- `_ema_value()` - EMA calculation
- `compute_all_indicators()` - master indicator computation

### 7. Utility Functions (Lines 1060-1130)
- `build_timeframe_plan()` - timeframe configuration
- Various small helpers

### 8. Helper Functions (Lines 1132-1455)
- `_clamp()` - value clamping
- `_atr()` - ATR (duplicate?)
- `_structure()` - support/resistance
- `_momentum()` - momentum calculation
- `_trend_strength()` - trend measurement
- `_volume_surge_ratio()` - volume surge detection
- `_buy_sell_pressure()` - pressure calculation
- `_range_span()` - range measurement
- `_volume_divergence_penalties()` - divergence detection
- `_confirmation_penalty()` - confirmation scoring
- `_funding_momentum()` - funding rate momentum
- `_classify_regime()` - regime classification
- `_quantize()` - price quantization
- `_quality_label()` - quality labeling
- `_quality_score_cap_from_confidence()` - quality cap
- `_leverage_suggestion()` - leverage recommendation
- `_calculate_enhanced_metrics()` - enhanced metrics

### 9. Scoring Pipeline Helpers (Lines 1516-1790)
- `_contributor_catalog()` - signal contributor metadata
- `_to_contributor_details()` - contributor formatting
- `_timeframe_alignment_score()` - cross-timeframe scoring
- `_detect_early_reversal_signals()` - reversal detection
- `_apply_reversal_penalty()` - reversal penalties
- `_round_number_proximity()` - round number detection
- `_score_confluence()` - confluence scoring
- `_apply_confluence_boost()` - confluence boosts

### 10. SetupAnalyzer Class (Lines 1820-3001)
- `__init__()` - initialization
- `_mode_params()` - mode-specific parameters
- `_trade_filter_params()` - filter configuration
- `_trade_filter_reasons()` - filter failure reasons
- `analyze()` - main analysis orchestrator
- `_build_setup()` - setup construction
- `_apply_deliberation()` - deliberation logic
- `_apply_enhanced_metrics_boost()` - metric boosts

## Safe Decomposition Strategy

### Phase 1: Extract Pure Functions (No Dependencies)
These functions have no dependencies on other scorer.py functions:
1. `_clamp()` - can move to a `utils.py` module
2. `_quantize()` - can move to `utils.py`
3. `_quality_label()` - can move to `utils.py`

### Phase 2: Extract Geometry Module
Create `geometry.py` with:
- All geometry-related functions
- Import `_clamp` and `_quantize` from utils

### Phase 3: Extract Indicator Module  
Create `indicators_scorer.py` with:
- Indicator computation functions
- Import what they need from geometry/utils

### Phase 4: Extract Normalization Module
Create `normalization.py`:
- Signal normalization logic
- Depends on indicators and config

### Phase 5: Extract Evidence Module
Create `evidence.py`:
- Evidence computation
- Depends on normalization

### Phase 6: Extract SetupAnalyzer
Create `setup_analyzer.py`:
- Import from all above modules
- Contains orchestration logic only

### Phase 7: Create Re-exports
Update `scorer.py` to re-export everything for backward compatibility

## Testing Strategy

After EACH phase:
1. Run `pytest -q` to ensure no regressions
2. Test a few manual cases if possible
3. Verify imports work correctly

## Risks to Avoid

1. **Circular imports**: Carefully manage import dependencies
2. **Breaking changes**: Maintain exact API compatibility via re-exports
3. **Logic changes**: Don't modify any logic during extraction
4. **Missing functions**: Ensure all functions are accounted for

## Recommended Order

Based on dependencies:
```
utils.py (pure helpers)
  ↑
geometry.py (uses utils)
  ↑
indicators_scorer.py (uses utils, geometry)
  ↑
normalization.py (uses indicators_scorer)
  ↑
evidence.py (uses normalization)
  ↑
confidence.py (uses evidence)
  ↑
quality.py (uses geometry, evidence)
  ↑
filters.py (uses all above)
  ↑
setup_analyzer.py (orchestrates everything)
  ↑
scorer.py (re-exports for backward compat)
```

## File Size Targets

- `utils.py`: ~100 lines
- `geometry.py`: ~350 lines
- `indicators_scorer.py`: ~250 lines
- `normalization.py`: ~200 lines
- `evidence.py`: ~180 lines
- `confidence.py`: ~100 lines
- `quality.py`: ~150 lines
- `filters.py`: ~400 lines
- `setup_analyzer.py`: ~800 lines
- `scorer.py`: ~50 lines (re-exports only)

Total: ~2580 lines (vs current 3001, some cleanup expected)
