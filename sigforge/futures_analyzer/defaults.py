"""
Single source of truth for all safe fallback constants.

These are used ONLY when a config key is missing — not as strategy values.
All actual strategy parameters must live in futures_analyzer.config.json.
"""
from __future__ import annotations

# ── Scoring pipeline ──────────────────────────────────────────────────────────
SCORE_CONFIDENCE_DIVISOR: float = 18.0
QUALITY_BASE_SCORE: float = 32.0
QUALITY_MIN_SCORE: float = 10.0
QUALITY_MAX_SCORE: float = 95.0
QUALITY_POSITIVE_CAP: float = 10.0
QUALITY_POSITIVE_WEIGHT: float = 4.8
QUALITY_RR_WEIGHT: float = 6.0
QUALITY_RR_CAP: float = 3.0
QUALITY_INVALIDATION_WEIGHT: float = 10.0
QUALITY_NEGATIVE_CAP: float = 10.0
QUALITY_NEGATIVE_WEIGHT: float = 5.2

# ── Signal transforms (sigmoid/tanh scaling) ─────────────────────────────────
TREND_SIGMOID_SCALE: float = 10.0
MOMENTUM_SIGMOID_SCALE: float = 15.0
PRESSURE_SIGMOID_SCALE: float = 5.0
FUNDING_MOMENTUM_SIGMOID_SCALE: float = 5.0
MACD_HISTOGRAM_SIGMOID_SCALE: float = 100.0
VWAP_DEV_SIGMOID_SCALE: float = 20.0
VOLUME_SURGE_CAP: float = 2.0
OI_FUNDING_NORM: float = 0.5
POC_PROXIMITY_PCT_CAP: float = 2.0
TREND_TANH_SCALE: float = 100.0
MOMENTUM_TANH_SCALE: float = 50.0

# ── Indicator computation ─────────────────────────────────────────────────────
EMA_FAST_PERIOD: int = 20
EMA_SLOW_PERIOD: int = 50
ROC_PERIOD: int = 14
VOLUME_WINDOW: int = 20
ATR_PERIOD: int = 14
PIVOT_N: int = 2

# ── Regime classifier ─────────────────────────────────────────────────────────
ADX_TREND_THRESHOLD: float = 25.0
ADX_WEAK_THRESHOLD: float = 20.0
ATR_VOLATILE_PERCENTILE: float = 85.0
ATR_TREND_MAX_PERCENTILE: float = 75.0
ADX_SLOPE_EXHAUSTION: float = -2.0
ADX_SLOPE_BREAKOUT: float = 3.0
CONSENSUS_WEIGHT_HIGHER: float = 0.50
CONSENSUS_WEIGHT_CONTEXT: float = 0.35
CONSENSUS_WEIGHT_TRIGGER: float = 0.15
ADX_CONF_BASE: float = 0.5
ADX_CONF_RANGE: float = 0.45
ADX_CONF_ADX_BASE: float = 20.0
ADX_CONF_ADX_RANGE: float = 60.0
DI_BIAS_MARGIN: float = 1.1

# ── Regime gate thresholds ────────────────────────────────────────────────────
REGIME_GATE_TREND: float = 0.4
REGIME_GATE_BREAKOUT: float = 0.3
REGIME_GATE_EXHAUSTION: float = 0.3
REGIME_GATE_RANGE: float = 0.3
REGIME_GATE_VOLATILE_CHOP: float = 0.5
REGIME_GATE_TRANSITION: float = 0.35

# ── Logistic confidence ───────────────────────────────────────────────────────
LOGISTIC_DEFAULT_STEEPNESS: float = 6.0
LOGISTIC_DEFAULT_MIDPOINT: float = 0.5

# ── Geometry / risk ───────────────────────────────────────────────────────────
ATR_BUFFER_FACTOR: float = 0.5
MIN_RISK_PX_FACTOR: float = 0.001
TARGET_CAP_ATR_MULT: float = 3.0
MIN_RR_RATIO: float = 1.5
STOP_SWING_PROXIMITY_ATR: float = 2.5
STOP_MIN_OFFSET_ATR: float = 0.1
TARGET_MIN_OFFSET_ATR: float = 0.3
INVALIDATION_STRENGTH_ATR_DIVISOR: float = 3.0

# ── Geometry quality scoring ──────────────────────────────────────────────────
GQ_RR_WEIGHT: float = 30.0
GQ_RR_CAP: float = 3.0
GQ_STOP_ATR_IDEAL_MIN: float = 0.8
GQ_STOP_ATR_IDEAL_MAX: float = 2.0
GQ_STOP_ATR_BONUS: float = 10.0
GQ_STOP_ATR_PENALTY: float = 8.0
GQ_ANCHOR_SWING_BONUS: float = 12.0
GQ_ANCHOR_VWAP_BONUS: float = 8.0
GQ_ANCHOR_VP_BONUS: float = 6.0
GQ_ANCHOR_ATR_PENALTY: float = 5.0
GQ_ENTRY_CONFLUENCE_BONUS: float = 3.0
GQ_TARGET_CONFLUENCE_BONUS: float = 2.0
GQ_BASE_SCORE: float = 20.0
GQ_MIN_SCORE: float = 10.0
GQ_MAX_SCORE: float = 95.0

# ── Leverage suggestion ───────────────────────────────────────────────────────
LEVERAGE_STOP_WIDE: float = 3.0
LEVERAGE_STOP_MID: float = 2.0
LEVERAGE_STOP_NARROW: float = 1.2
LEVERAGE_STOP_TIGHT: float = 0.8
LEVERAGE_CONFIDENCE_HIGH: float = 0.8
LEVERAGE_CONFIDENCE_LOW: float = 0.6
LEVERAGE_RR_GOOD: float = 2.0
LEVERAGE_RR_POOR: float = 1.5
LEVERAGE_VOLATILE_CHOP_CAP: int = 4
LEVERAGE_MIN_CONFIDENCE: float = 0.45

# ── OI / funding ──────────────────────────────────────────────────────────────
FUNDING_RATE_SCALE: float = 10_000.0
FUNDING_RATE_DIVISOR: float = 50.0
OI_FUNDING_PARTIAL_WEIGHT: float = 0.6
FUNDING_MOMENTUM_WEIGHT: float = 0.5
FUNDING_MOMENTUM_SLOPE_SCALE: float = 20_000.0

# ── Confirmation / penalty ────────────────────────────────────────────────────
CONFIRMATION_LOW_CUTOFF: int = 2
CONFIRMATION_MID_CUTOFF: int = 4
CONFIRMATION_LOW_PENALTY: float = 1.3
CONFIRMATION_MID_PENALTY: float = 0.55
DUAL_TREND_PENALTY: float = 0.35
DUAL_MOMENTUM_PENALTY: float = 0.35
DUAL_PRESSURE_PENALTY: float = 0.2
CONFIRMATION_PENALTY_WEIGHT: float = 3.0
TARGET_AMBITION_PENALTY_WEIGHT: float = 10.0
REGIME_PENALTY_WEIGHT: float = 4.0

# ── Reversal penalties ────────────────────────────────────────────────────────
REVERSAL_1_CONFIDENCE_MULT: float = 0.90
REVERSAL_2_CONFIDENCE_MULT: float = 0.75
REVERSAL_3_CONFIDENCE_MULT: float = 0.60
REVERSAL_2_QUALITY_DEDUCTION: float = 8.0
REVERSAL_3_QUALITY_DEDUCTION: float = 15.0

# ── Confluence boosts ─────────────────────────────────────────────────────────
CONFLUENCE_ENTRY_3_BOOST: float = 10.0
CONFLUENCE_ENTRY_2_BOOST: float = 6.0
CONFLUENCE_ENTRY_1_BOOST: float = 2.0
CONFLUENCE_TARGET_2_BOOST: float = 5.0
CONFLUENCE_TARGET_1_BOOST: float = 2.0

# ── Enhanced metrics ──────────────────────────────────────────────────────────
RSI_NEUTRAL_CONFIDENCE_MULT: float = 1.05
RSI_EXTREME_CONFIDENCE_MULT: float = 0.85
RSI_MILD_EXTREME_CONFIDENCE_MULT: float = 0.90
MACD_CONFIRM_CONFIDENCE_DELTA: float = 0.03
BB_MID_QUALITY_BOOST: float = 5.0
BB_EXTREME_QUALITY_PENALTY: float = 3.0
BB_MID_LOW: float = 0.35
BB_MID_HIGH: float = 0.65
BB_EXTREME_LOW: float = 0.2
BB_EXTREME_HIGH: float = 0.8
OB_IMBALANCE_THRESHOLD: float = 0.3
OB_IMBALANCE_CONFIDENCE_DELTA: float = 0.02
VOL_RANK_HIGH_THRESHOLD: float = 75.0
VOL_RANK_LOW_THRESHOLD: float = 25.0
VOL_RANK_HIGH_STOP_MULT: float = 1.15
VOL_RANK_LOW_STOP_MULT: float = 0.85

# ── Timeframe alignment ───────────────────────────────────────────────────────
ALIGNMENT_STRONG_THRESHOLD: float = 0.6
ALIGNMENT_NEUTRAL_HIGH: float = 0.2
ALIGNMENT_NEUTRAL_LOW: float = -0.2
ALIGNMENT_STRONG_CONFIDENCE_MULT: float = 1.10
ALIGNMENT_WEAK_CONFIDENCE_MULT: float = 0.90
ALIGNMENT_OPPOSED_CONFIDENCE_MULT: float = 0.75
ALIGNMENT_NORM_SCALE: float = 20.0

# ── Invalidation strength ─────────────────────────────────────────────────────
INVALIDATION_STRUCTURE_GAP_WEIGHT: float = 0.35
INVALIDATION_ATR_BUFFER_WEIGHT: float = 0.45
INVALIDATION_RANGE_CONTEXT_WEIGHT: float = 0.2
INVALIDATION_RANGE_CONTEXT_CAP: float = 3.0
INVALIDATION_FORMULA_PENALTY: float = 0.2
INVALIDATION_PROXIMITY_THRESHOLD_PCT: float = 0.0015

# ── Volume profile ────────────────────────────────────────────────────────────
VOLUME_POC_NEAR_SCORE: float = 0.8
VOLUME_POC_SIDE_SCORE: float = 0.4
VOLUME_PROFILE_ATR_PROXIMITY: float = 1.5

# ── Volatile chop ─────────────────────────────────────────────────────────────
VOLATILE_CHOP_BOOST_PER_SIGNAL: float = 0.04
VOLATILE_CHOP_CONFIDENCE_CEILING_CAP: float = 0.68
VOLATILE_CHOP_STRONG_SIGNAL_THRESHOLD: float = 0.15
VOLATILE_CHOP_QUALITY_CAP: float = 50.0
RANGE_QUALITY_CAP: float = 52.0

# ── Confidence quality caps ───────────────────────────────────────────────────
CONFIDENCE_QUALITY_CAP_BELOW_045: float = 54.9
CONFIDENCE_QUALITY_CAP_BELOW_070: float = 74.9
CONFIDENCE_QUALITY_CAP_DEFAULT: float = 95.0

# ── Default signal weights (normalized, sum = 1.0) ───────────────────────────
DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "higher_trend": 0.15,
    "context_trend": 0.12,
    "trigger_momentum": 0.10,
    "entry_momentum": 0.08,
    "volume_surge": 0.06,
    "buy_pressure": 0.08,
    "oi_funding_bias": 0.06,
    "funding_momentum": 0.05,
    "structure_position": 0.06,
    "rsi_alignment": 0.05,
    "macd_alignment": 0.05,
    "bb_alignment": 0.04,
    "vwap_alignment": 0.06,
    "market_structure_align": 0.08,
    "cumulative_delta_align": 0.06,
    "volume_poc_proximity": 0.06,
}

# ── RSI alignment centers ─────────────────────────────────────────────────────
RSI_LONG_CENTER: float = 45.0
RSI_SHORT_CENTER: float = 55.0
RSI_ALIGNMENT_WIDTH: float = 20.0

# ── Pressure threshold ────────────────────────────────────────────────────────
PRESSURE_THRESHOLD: float = 0.05
VOLUME_SURGE_THRESHOLD: float = 1.05
