from __future__ import annotations

from futures_analyzer.analysis.models import (
    IndicatorBundle,
    MarketMeta,
    MarketRegime,
    NormalizedSignals,
)
from futures_analyzer.analysis.indicators import sigmoid, gaussian_peak
from futures_analyzer.analysis.scoring.utils import _clamp
from futures_analyzer.config import AppConfig, load_app_config


def normalize_distension(price: float, mean: float, atr: float) -> float:
    """Return how overextended price is from the mean, normalised to [0.0, 1.0].

    Maps the absolute distance in ATR multiples onto [0, 1], capped at 2.5 ATRs.
    A value of 1.0 means price is ≥ 2.5 ATRs from the mean (maximum distension).
    """
    if atr <= 0:
        return 0.0
    distance_atr = abs(price - mean) / atr
    return _clamp(distance_atr / 2.5, 0.0, 1.0)


def normalize_signals(
    bundle: IndicatorBundle,
    market_meta: MarketMeta,
    side: str,
    regime: MarketRegime,
    config: AppConfig | None = None,
    signal_transforms=None,
) -> NormalizedSignals:
    """Convert raw indicator values to normalized [0, 1] signal strengths per side.

    All scaling factors come from config.strategy.signal_transforms — no magic numbers.
    All outputs are clamped to [0.0, 1.0]. No NaN or infinite values are produced.
    """
    cfg = config or load_app_config()
    t = signal_transforms if signal_transforms is not None else cfg.strategy.signal_transforms

    direction = 1.0 if side == "long" else -1.0
    px = market_meta.mark_price

    # Trend signals: sigmoid of raw trend value × direction × config scale
    higher_trend = sigmoid(bundle.higher_trend * direction * t.higher_trend_scale)
    context_trend = sigmoid(bundle.context_trend * direction * t.context_trend_scale)
    trigger_momentum = sigmoid(bundle.trigger_momentum * direction * t.trigger_momentum_scale)
    entry_momentum = sigmoid(bundle.entry_momentum * direction * t.entry_momentum_scale)

    # Volume: surge ratio → [0, 1] via min-max with cap from config
    volume_surge = _clamp((bundle.trigger_volume_surge - 1.0) / t.volume_surge_cap, 0.0, 1.0)

    # Pressure: cumulative_delta × direction → sigmoid with config scale
    buy_pressure = sigmoid(bundle.cumulative_delta * direction * t.buy_pressure_scale)

    # OI/funding: normalized to [0, 1] using config normalization factor
    raw_oi_funding_long, raw_oi_funding_short = _oi_funding_biases(
        bundle.funding_rate, bundle.oi_change_pct
    )
    if side == "long":
        oi_funding_bias = _clamp(raw_oi_funding_long / t.oi_funding_norm, 0.0, 1.0)
    else:
        oi_funding_bias = _clamp(raw_oi_funding_short / t.oi_funding_norm, 0.0, 1.0)

    # Funding momentum with config scale
    funding_momentum = sigmoid(bundle.funding_momentum * direction * t.funding_momentum_scale)

    # Structure position: proximity to support (long) or resistance (short)
    range_width = max(bundle.vah - bundle.val, 1e-9)
    if side == "long":
        dist_to_support = max(bundle.val - px, 0.0)
        structure_position = _clamp(1.0 - dist_to_support / range_width, 0.0, 1.0)
    else:
        dist_to_resistance = max(px - bundle.vah, 0.0)
        structure_position = _clamp(1.0 - dist_to_resistance / range_width, 0.0, 1.0)

    # RSI alignment: gaussian_peak centered on favorable zone
    if side == "long":
        rsi_alignment = gaussian_peak(bundle.rsi_14, center=45.0, width=20.0)
    else:
        rsi_alignment = gaussian_peak(bundle.rsi_14, center=55.0, width=20.0)

    # MACD alignment: histogram direction matches side, scaled by config
    if side == "long":
        macd_alignment = sigmoid(bundle.macd_histogram * t.macd_histogram_scale)
    else:
        macd_alignment = sigmoid(-bundle.macd_histogram * t.macd_histogram_scale)

    # BB alignment: price position favorable for entry
    bb_pos = bundle.bb_position
    if side == "long":
        bb_alignment = _clamp(1.0 - bb_pos, 0.0, 1.0)
    else:
        bb_alignment = _clamp(bb_pos, 0.0, 1.0)

    # VWAP alignment: price below VWAP favors longs, above favors shorts
    vwap_dev = (px - bundle.vwap) / max(bundle.vwap, 1e-9)
    if side == "long":
        vwap_alignment = sigmoid(-vwap_dev * t.vwap_dev_scale)
    else:
        vwap_alignment = sigmoid(vwap_dev * t.vwap_dev_scale)

    # Market structure alignment
    if side == "long" and bundle.market_structure == "HH_HL":
        market_structure_align = 1.0
    elif side == "short" and bundle.market_structure == "LH_LL":
        market_structure_align = 1.0
    elif bundle.market_structure == "mixed":
        market_structure_align = 0.5
    else:
        market_structure_align = 0.0

    # Cumulative delta alignment with config scale
    cumulative_delta_align = sigmoid(bundle.cumulative_delta * direction * t.buy_pressure_scale)

    # Volume POC proximity: config-driven percentage cap
    poc_dist_pct = abs(px - bundle.poc) / max(bundle.poc, 1e-9) * 100.0
    volume_poc_proximity = _clamp(1.0 - poc_dist_pct / t.poc_proximity_pct_cap, 0.0, 1.0)

    return NormalizedSignals(
        higher_trend=higher_trend,
        context_trend=context_trend,
        trigger_momentum=trigger_momentum,
        entry_momentum=entry_momentum,
        volume_surge=volume_surge,
        buy_pressure=buy_pressure,
        oi_funding_bias=oi_funding_bias,
        funding_momentum=funding_momentum,
        structure_position=structure_position,
        rsi_alignment=rsi_alignment,
        macd_alignment=macd_alignment,
        bb_alignment=bb_alignment,
        vwap_alignment=vwap_alignment,
        market_structure_align=market_structure_align,
        cumulative_delta_align=cumulative_delta_align,
        volume_poc_proximity=volume_poc_proximity,
    )


def _funding_momentum(history: list[float], window: int = 4) -> float:
    """Compute the slope of the last `window` funding rate values.

    Returns a normalised value in roughly [-1, 1]:
    positive = funding rising (crowded longs), negative = falling (crowded shorts).
    """
    slope_scale = load_app_config().strategy.funding_momentum_slope_scale
    tail = history[-window:] if len(history) >= window else history
    n = len(tail)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(tail) / n
    num = sum((xs[i] - x_mean) * (tail[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    return _clamp(slope * slope_scale, -1.0, 1.0)


def _oi_funding_biases(
    funding_rate: float | None,
    oi_change_pct: float | None,
    funding_rate_history: list[float] | None = None,
) -> tuple[float, float]:
    s = load_app_config().strategy
    oi_delta = (oi_change_pct or 0.0) / 100.0
    if oi_delta < 0:
        oi_delta = 0.0
    funding = funding_rate or 0.0
    long_funding = max(-funding * s.funding_rate_scale, 0.0) / s.funding_rate_divisor
    short_funding = max(funding * s.funding_rate_scale, 0.0) / s.funding_rate_divisor
    long_oi = oi_delta * (1.0 if funding <= 0 else s.oi_funding_partial_weight)
    short_oi = oi_delta * (1.0 if funding >= 0 else s.oi_funding_partial_weight)

    momentum = _funding_momentum(funding_rate_history or [])
    long_momentum = max(-momentum, 0.0) * s.funding_momentum_weight
    short_momentum = max(momentum, 0.0) * s.funding_momentum_weight

    return (
        long_funding + long_oi + long_momentum,
        short_funding + short_oi + short_momentum,
    )
