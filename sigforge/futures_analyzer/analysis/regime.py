"""Market regime classifier — all thresholds come from config."""
from __future__ import annotations

from dataclasses import dataclass

from futures_analyzer.analysis.indicators import adx, compute_adx_slope
from futures_analyzer.analysis.models import Candle, MarketRegime
from futures_analyzer.config import load_app_config, RegimeClassifierConfig


@dataclass
class PerTFRegime:
    timeframe: str
    regime: MarketRegime
    adx: float
    plus_di: float
    minus_di: float
    atr_percentile: float
    adx_slope: float


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float
    per_tf: list[PerTFRegime]
    higher_bias: str


_ATR_WINDOW = 100


def _atr_percentile(candles: list[Candle], window: int = _ATR_WINDOW) -> float:
    """ATR percentile of the most recent bar within a rolling window. Returns 50.0 on insufficient data."""
    if len(candles) < 2:
        return 50.0
    atr_values: list[float] = []
    for i in range(1, len(candles)):
        prev = candles[i - 1]
        curr = candles[i]
        atr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close),
        )
        atr_values.append(atr)
    window_atrs = atr_values[-window:]
    current_atr = window_atrs[-1]
    historical = window_atrs[:-1]
    if not historical:
        return 50.0
    count_lte = sum(1 for v in historical if v <= current_atr)
    return float(max(0.0, min(100.0, count_lte / len(historical) * 100.0)))


def _load_regime_cfg() -> RegimeClassifierConfig:
    """Load regime classifier config. Raises error if config is unavailable."""
    return load_app_config().strategy.regime_classifier


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _classify_single_tf(
    candles: list[Candle],
    tf_name: str,
    cfg: RegimeClassifierConfig,
) -> PerTFRegime:
    """Classify a single timeframe. All thresholds come from config."""
    # Resolve thresholds from config
    adx_trend   = cfg.adx_trend_threshold
    adx_weak    = cfg.adx_weak_threshold
    atr_volatile = cfg.atr_volatile_percentile
    atr_trend_max = cfg.atr_trend_max_percentile
    slope_exhaustion = cfg.adx_slope_exhaustion
    slope_breakout   = cfg.adx_slope_breakout

    adx_result = adx(candles)
    adx_val  = adx_result["adx"]
    plus_di  = adx_result["plus_di"]
    minus_di = adx_result["minus_di"]
    atr_pct  = _atr_percentile(candles)
    adx_slope = compute_adx_slope(candles, lookback=5)

    if adx_val >= adx_trend and atr_pct <= atr_trend_max:
        if adx_slope < slope_exhaustion:
            regime = MarketRegime.EXHAUSTION
        else:
            regime = MarketRegime.BULLISH_TREND if plus_di > minus_di else MarketRegime.BEARISH_TREND
    elif adx_val < adx_weak and adx_slope > slope_breakout:
        regime = MarketRegime.BREAKOUT
    elif atr_pct >= atr_volatile:
        regime = MarketRegime.VOLATILE_CHOP
    elif adx_weak <= adx_val < adx_trend:
        regime = MarketRegime.TRANSITION
    else:
        regime = MarketRegime.RANGE

    return PerTFRegime(
        timeframe=tf_name,
        regime=regime,
        adx=adx_val,
        plus_di=plus_di,
        minus_di=minus_di,
        atr_percentile=atr_pct,
        adx_slope=adx_slope,
    )


# Tie-breaking conservatism order
_CONSERVATISM_ORDER: list[MarketRegime] = [
    MarketRegime.RANGE,
    MarketRegime.VOLATILE_CHOP,
    MarketRegime.TRANSITION,
]


def classify_regime(
    context_candles: list[Candle],
    higher_candles: list[Candle],
    trigger_atr: float,
    px: float,
) -> tuple[MarketRegime, float]:
    """Backward-compatible wrapper around classify_regime_consensus."""
    result = classify_regime_consensus(
        context_candles=context_candles,
        higher_candles=higher_candles,
        trigger_candles=context_candles,
        trigger_atr=trigger_atr,
        px=px,
    )
    return (result.regime, result.confidence)


def classify_regime_consensus(
    context_candles: list[Candle],
    higher_candles: list[Candle],
    trigger_candles: list[Candle],
    trigger_atr: float,
    px: float,
) -> RegimeResult:
    """
    Multi-timeframe weighted consensus regime classifier.
    All weights and thresholds come from config.strategy.regime_classifier.
    """
    cfg = _load_regime_cfg()

    w_higher  = cfg.consensus_weight_higher
    w_context = cfg.consensus_weight_context
    w_trigger = cfg.consensus_weight_trigger

    tf_inputs = [
        (higher_candles,  "higher",  w_higher),
        (context_candles, "context", w_context),
        (trigger_candles, "trigger", w_trigger),
    ]

    per_tf: list[PerTFRegime] = [
        _classify_single_tf(candles, tf_name, cfg)
        for candles, tf_name, _ in tf_inputs
    ]

    # Weighted vote
    vote_counts: dict[MarketRegime, float] = {}
    for tf_result, (_, _, weight) in zip(per_tf, tf_inputs):
        vote_counts[tf_result.regime] = vote_counts.get(tf_result.regime, 0.0) + weight

    max_weight = max(vote_counts.values())
    tied = [r for r, w in vote_counts.items() if w == max_weight]

    if len(tied) == 1:
        consensus_regime = tied[0]
    else:
        higher_tf_regime = per_tf[0].regime
        if higher_tf_regime in tied:
            consensus_regime = higher_tf_regime
        else:
            for conservative in _CONSERVATISM_ORDER:
                if conservative in tied:
                    consensus_regime = conservative
                    break
            else:
                consensus_regime = tied[0]

    # Confidence: consensus_score × adx_conf_factor (all params from config)
    adx_conf_base     = cfg.adx_conf_base
    adx_conf_range    = cfg.adx_conf_range
    adx_conf_adx_base = cfg.adx_conf_adx_base
    adx_conf_adx_range = cfg.adx_conf_adx_range
    di_bias_margin    = cfg.di_bias_margin

    higher_tf = per_tf[0]
    adx_conf_factor = _clamp(
        adx_conf_base + (higher_tf.adx - adx_conf_adx_base) / adx_conf_adx_range * adx_conf_range,
        adx_conf_base,
        adx_conf_base + adx_conf_range,
    )
    confidence = _clamp(vote_counts[consensus_regime] * adx_conf_factor, 0.0, 1.0)

    # Higher TF directional bias
    if higher_tf.plus_di > higher_tf.minus_di * di_bias_margin:
        higher_bias = "bullish"
    elif higher_tf.minus_di > higher_tf.plus_di * di_bias_margin:
        higher_bias = "bearish"
    else:
        higher_bias = "neutral"

    return RegimeResult(
        regime=consensus_regime,
        confidence=float(confidence),
        per_tf=per_tf,
        higher_bias=higher_bias,
    )
