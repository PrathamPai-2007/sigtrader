from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from statistics import mean

from futures_analyzer.analysis.models import (
    AnalysisResult,
    Candle,
    ContributorDetail,
    ContributorDirection,
    EnhancedMetrics,
    MarketMode,
    MarketMeta,
    MarketRegime,
    QualityLabel,
    StrategyStyle,
    TimeframePlan,
    TradeSetup,
)
from futures_analyzer.analysis.indicators import (
    rsi,
    macd,
    stochastic,
    bollinger_bands,
    rsi_divergence,
    volume_profile_strength,
)


def build_timeframe_plan(
    *,
    style: StrategyStyle = StrategyStyle.CONSERVATIVE,
    market_mode: MarketMode = MarketMode.INTRADAY,
) -> TimeframePlan:
    from futures_analyzer.config import load_app_config
    config = load_app_config()
    timeframe = config.timeframe_for(market_mode)
    return TimeframePlan(
        profile_name=timeframe.profile_name,
        style=style,
        market_mode=market_mode,
        entry_timeframe=timeframe.entry_timeframe,
        context_timeframe=timeframe.context_timeframe,
        trigger_timeframe=timeframe.trigger_timeframe,
        higher_timeframe=timeframe.higher_timeframe,
        lookback_bars=timeframe.lookback_bars,
    )


@dataclass
class _Contribution:
    key: str
    label: str
    value: float
    impact: float
    direction: ContributorDirection
    summary: str


@dataclass
class _EvidenceSnapshot:
    agreement: int
    total: int
    summary: str
    raw_checks: dict[str, bool]


@dataclass
class _SideMetrics:
    side: str
    score: float
    confidence: float
    quality_label: QualityLabel
    quality_score: float
    leverage_suggestion: str
    entry: float
    stop: float
    target: float
    rationale: str
    top_positive_contributors: list[ContributorDetail]
    top_negative_contributors: list[ContributorDetail]
    components: dict[str, float]
    structure_points: dict[str, float]
    risk_reward_ratio: float
    stop_distance_pct: float
    target_distance_pct: float
    atr_multiple_to_stop: float
    atr_multiple_to_target: float
    invalidation_strength: float
    is_tradable: bool
    tradable_reasons: list[str]
    evidence_agreement: int
    evidence_total: int
    deliberation_summary: str


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    trs: list[float] = []
    prev_close = candles[0].close
    for candle in candles[1:]:
        tr = max(candle.high - candle.low, abs(candle.high - prev_close), abs(candle.low - prev_close))
        trs.append(tr)
        prev_close = candle.close
    if not trs:
        return 0.0
    tail = trs[-period:] if len(trs) >= period else trs
    return mean(tail)


def _structure(candles: list[Candle]) -> tuple[float, float]:
    if not candles:
        raise ValueError("No candles available for structure detection")
    recent = candles[-30:] if len(candles) >= 30 else candles
    resistance = max(candle.high for candle in recent)
    support = min(candle.low for candle in recent)
    return support, resistance


def _structure_biases(px: float, support: float, resistance: float) -> tuple[float, float]:
    range_width = max(resistance - support, 1e-9)
    position = _clamp((px - support) / range_width, 0.0, 1.0)
    # Longs should be favored closer to support; shorts closer to resistance.
    return 1.0 - position, position


def _momentum(candles: list[Candle], window: int = 10) -> float:
    if len(candles) < 2:
        return 0.0
    width = min(window, len(candles) - 1)
    first = candles[-(width + 1)].close
    last = candles[-1].close
    if first <= 0:
        return 0.0
    return (last / first) - 1.0


def _trend_strength(candles: list[Candle], *, fast_window: int = 5, slow_window: int = 20) -> float:
    if len(candles) < slow_window:
        return 0.0
    closes = [candle.close for candle in candles[-slow_window:]]
    ma_fast = mean(closes[-fast_window:])
    ma_slow = mean(closes)
    if ma_slow == 0:
        return 0.0
    return (ma_fast / ma_slow) - 1.0


def _volume_surge_ratio(candles: list[Candle], window: int = 20) -> float:
    if len(candles) < 2:
        return 1.0
    if len(candles) > window:
        recent = candles[-(window + 1) : -1]
    else:
        recent = candles[:-1]
    if not recent:
        return 1.0
    base = mean(candle.volume for candle in recent)
    if base <= 0:
        return 1.0
    return candles[-1].volume / base


def _buy_sell_pressure(candles: list[Candle], window: int = 20) -> float:
    if not candles:
        return 0.0
    tail = candles[-window:] if len(candles) >= window else candles
    weighted = 0.0
    volume_sum = 0.0
    for candle in tail:
        rng = max(candle.high - candle.low, 1e-9)
        body_position = (candle.close - candle.open) / rng
        weighted += body_position * candle.volume
        volume_sum += candle.volume
    if volume_sum <= 0:
        return 0.0
    return _clamp(weighted / volume_sum, -1.0, 1.0)


def _range_span(candles: list[Candle], window: int = 20) -> float:
    if not candles:
        return 0.0
    tail = candles[-window:] if len(candles) >= window else candles
    high = max(candle.high for candle in tail)
    low = min(candle.low for candle in tail)
    close_ref = max(mean(candle.close for candle in tail), 1e-9)
    return (high - low) / close_ref


def _volume_divergence_penalties(momentum: float, pressure: float, volume_surge: float) -> tuple[float, float]:
    long_penalty = 0.0
    short_penalty = 0.0
    if momentum > 0 and (pressure < 0 or volume_surge < 1.0):
        long_penalty = abs(momentum) + max(-pressure, 0.0) + max(1.0 - volume_surge, 0.0)
    if momentum < 0 and (pressure > 0 or volume_surge < 1.0):
        short_penalty = abs(momentum) + max(pressure, 0.0) + max(1.0 - volume_surge, 0.0)
    return long_penalty, short_penalty


def _confirmation_penalty(
    side: str,
    *,
    higher_trend: float,
    context_trend: float,
    trigger_momentum: float,
    entry_momentum: float,
    trigger_pressure: float,
    entry_pressure: float,
    trigger_volume_surge: float,
    entry_volume_surge: float,
) -> float:
    from futures_analyzer.config import load_app_config
    strategy = load_app_config().strategy
    direction = 1.0 if side == "long" else -1.0
    checks = [
        higher_trend * direction > 0,
        context_trend * direction > 0,
        trigger_momentum * direction > 0,
        entry_momentum * direction > 0,
        trigger_pressure * direction > strategy.pressure_threshold,
        entry_pressure * direction > strategy.pressure_threshold,
        trigger_volume_surge >= strategy.volume_surge_threshold,
        entry_volume_surge >= strategy.volume_surge_threshold,
    ]
    confirmations = sum(1 for ok in checks if ok)
    penalty = 0.0
    if confirmations <= strategy.confirmation_low_cutoff:
        penalty += strategy.confirmation_low_penalty
    elif confirmations <= strategy.confirmation_mid_cutoff:
        penalty += strategy.confirmation_mid_penalty
    if (higher_trend * direction) <= 0 and (context_trend * direction) <= 0:
        penalty += strategy.dual_trend_penalty
    if (trigger_momentum * direction) <= 0 and (entry_momentum * direction) <= 0:
        penalty += strategy.dual_momentum_penalty
    if (trigger_pressure * direction) <= 0 and (entry_pressure * direction) <= 0:
        penalty += strategy.dual_pressure_penalty
    return penalty


def _oi_funding_biases(funding_rate: float | None, oi_change_pct: float | None) -> tuple[float, float]:
    oi_delta = (oi_change_pct or 0.0) / 100.0
    if oi_delta < 0:
        oi_delta = 0.0
    funding = funding_rate or 0.0
    long_funding = max(-funding * 10_000.0, 0.0) / 50.0
    short_funding = max(funding * 10_000.0, 0.0) / 50.0
    long_oi = oi_delta * (1.0 if funding <= 0 else 0.6)
    short_oi = oi_delta * (1.0 if funding >= 0 else 0.6)
    return long_funding + long_oi, short_funding + short_oi


def _classify_regime(
    context_candles: list[Candle],
    higher_candles: list[Candle],
    trigger_atr: float,
    px: float,
) -> tuple[MarketRegime, float]:
    context_trend = _trend_strength(context_candles)
    higher_trend = _trend_strength(higher_candles, fast_window=4, slow_window=16)
    trend = (context_trend * 0.65) + (higher_trend * 0.35)
    trend_agreement = 1.0 if (context_trend == 0 or higher_trend == 0 or context_trend * higher_trend > 0) else 0.75
    span = _range_span(context_candles)
    atr_ratio = trigger_atr / max(px, 1e-9)
    if trend >= 0.01:
        confidence = min(0.55 + (trend * 18.0 * trend_agreement), 0.95)
        return MarketRegime.BULLISH_TREND, confidence
    if trend <= -0.01:
        confidence = min(0.55 + (abs(trend) * 18.0 * trend_agreement), 0.95)
        return MarketRegime.BEARISH_TREND, confidence
    if atr_ratio >= 0.012 or span >= 0.04:
        volatility_signal = max(atr_ratio / 0.012, span / 0.04)
        confidence = min(0.45 + (volatility_signal * 0.15), 0.9)
        return MarketRegime.VOLATILE_CHOP, confidence
    confidence = min(0.45 + max(span, atr_ratio) * 10.0, 0.85)
    return MarketRegime.RANGE, confidence


def _quantize(price: float, tick: float | None) -> float:
    if tick is None or tick <= 0:
        return round(price, 6)
    try:
        tick_decimal = Decimal(str(tick)).normalize()
        price_decimal = Decimal(str(price))
    except InvalidOperation:
        return round(price, 10)
    if tick_decimal <= 0:
        return round(price, 10)
    units = (price_decimal / tick_decimal).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    quantized = units * tick_decimal
    precision = max(0, -tick_decimal.as_tuple().exponent)
    return float(format(quantized, f".{min(precision, 16)}f"))


def _quality_label(quality_score: float) -> QualityLabel:
    if quality_score >= 75:
        return QualityLabel.HIGH
    if quality_score >= 55:
        return QualityLabel.MEDIUM
    return QualityLabel.LOW


def _quality_score_cap_from_confidence(confidence: float) -> float:
    from futures_analyzer.config import load_app_config
    caps = load_app_config().strategy.confidence_quality_caps
    if confidence < 0.45:
        return caps["below_0_45"]
    if confidence < 0.7:
        return caps["below_0_70"]
    return caps["default"]


def _leverage_suggestion(
    *,
    stop_distance_pct: float,
    quality_label: QualityLabel,
    confidence: float,
    regime: MarketRegime,
    risk_reward_ratio: float = 2.0,
    is_tradable: bool = True,
) -> str:
    if not is_tradable or confidence < 0.45:
        return "1x"

    from futures_analyzer.config import load_app_config
    strategy = load_app_config().strategy
    cap_map = {
        QualityLabel.LOW: strategy.leverage_caps["low"],
        QualityLabel.MEDIUM: strategy.leverage_caps["medium"],
        QualityLabel.HIGH: strategy.leverage_caps["high"],
    }
    floor_map = {
        QualityLabel.LOW: strategy.leverage_floors["low"],
        QualityLabel.MEDIUM: strategy.leverage_floors["medium"],
        QualityLabel.HIGH: strategy.leverage_floors["high"],
    }

    cap = cap_map[quality_label]
    floor = floor_map[quality_label]

    if stop_distance_pct >= 3.0:
        base = floor
    elif stop_distance_pct >= 2.0:
        base = floor + 1
    elif stop_distance_pct >= 1.2:
        base = floor + 2
    elif stop_distance_pct >= 0.8:
        base = floor + 3
    else:
        base = cap

    if confidence >= 0.8:
        base += 1
    elif confidence < 0.6:
        base -= 1

    if risk_reward_ratio >= 2.0:
        base += 1
    elif risk_reward_ratio < 1.5:
        base -= 1

    if regime == MarketRegime.VOLATILE_CHOP:
        cap = min(cap, 4)
        floor = min(floor, cap)
        base -= 2
    elif regime == MarketRegime.RANGE:
        base -= 1

    base = int(_clamp(base, floor, cap))
    return f"{base}x"


def _regime_weight_profile(regime: MarketRegime, side: str) -> tuple[dict[str, float], float, float]:
    from futures_analyzer.config import load_app_config
    strategy = load_app_config().strategy
    regime_key = regime.value
    side_key = f"{regime_key}_{side}"
    weights = dict(strategy.regime_weights.get("default", {}))
    if regime in {MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND}:
        weights.update(strategy.regime_weights.get(side_key, {}))
    else:
        weights.update(strategy.regime_weights.get(regime_key, {}))
    penalty_multiplier = strategy.regime_penalty_multiplier.get(regime_key, strategy.regime_penalty_multiplier["default"])
    confidence_ceiling = strategy.regime_confidence_ceiling.get(side_key, strategy.regime_confidence_ceiling.get(regime_key, strategy.regime_confidence_ceiling["default"]))
    return weights, penalty_multiplier, confidence_ceiling


def _regime_alignment(regime: MarketRegime, side: str) -> float:
    from futures_analyzer.config import load_app_config
    strategy = load_app_config().strategy
    return strategy.regime_alignment.get(f"{regime.value}_{side}", strategy.regime_alignment.get(regime.value, 0.0))


def _regime_penalty(regime: MarketRegime, side: str) -> float:
    from futures_analyzer.config import load_app_config
    strategy = load_app_config().strategy
    return strategy.regime_penalty.get(f"{regime.value}_{side}", strategy.regime_penalty.get(regime.value, 0.0))


def _calculate_enhanced_metrics(
    entry_candles: list[Candle],
    trigger_candles: list[Candle],
    context_candles: list[Candle],
    market_meta: MarketMeta,
    order_book_data: dict[str, float] | None = None,
    volatility_data: dict[str, float] | None = None,
    vwap_data: dict[str, float] | None = None,
) -> EnhancedMetrics:
    """Calculate enhanced technical indicators and market microstructure metrics."""
    
    # Advanced technical indicators on entry timeframe
    rsi_14 = rsi(entry_candles, period=14)
    macd_result = macd(entry_candles, fast=12, slow=26, signal=9)
    stoch_result = stochastic(entry_candles, period=14, smooth_k=3, smooth_d=3)
    bb_result = bollinger_bands(entry_candles, period=20, std_dev=2.0)
    div_detected, div_type = rsi_divergence(entry_candles, period=14, lookback=20)
    
    # Order book metrics
    bid_ask_spread = order_book_data.get("spread_pct", 0.0) if order_book_data else 0.0
    bid_ask_ratio = order_book_data.get("bid_ask_ratio", 1.0) if order_book_data else 1.0
    ob_imbalance = order_book_data.get("imbalance", 0.0) if order_book_data else 0.0
    
    # Volatility metrics
    vol_rank = volatility_data.get("volatility_rank", 50.0) if volatility_data else 50.0
    vol_regime = volatility_data.get("volatility_regime", "normal") if volatility_data else "normal"
    
    # VWAP metrics
    vwap = vwap_data.get("vwap", entry_candles[-1].close if entry_candles else 0.0) if vwap_data else (entry_candles[-1].close if entry_candles else 0.0)
    vwap_dev = vwap_data.get("vwap_deviation_pct", 0.0) if vwap_data else 0.0
    
    # Liquidity score (placeholder - would come from enhanced data provider)
    liquidity_score = 50.0
    slippage_est = bid_ask_spread * 1.5
    
    return EnhancedMetrics(
        rsi_14=rsi_14,
        macd_value=macd_result.value,
        macd_signal=macd_result.signal or 0.0,
        macd_histogram=macd_result.histogram or 0.0,
        stochastic_k=stoch_result.value,
        stochastic_d=stoch_result.signal or 0.0,
        bollinger_upper=bb_result["upper"],
        bollinger_middle=bb_result["middle"],
        bollinger_lower=bb_result["lower"],
        bollinger_bandwidth_pct=bb_result["bandwidth"],
        bollinger_position=bb_result["position"],
        rsi_divergence=div_detected,
        rsi_divergence_type=div_type,
        bid_ask_spread_pct=bid_ask_spread,
        bid_ask_ratio=bid_ask_ratio,
        order_book_imbalance=ob_imbalance,
        volatility_rank=vol_rank,
        volatility_regime=vol_regime,
        vwap=vwap,
        vwap_deviation_pct=vwap_dev,
        liquidity_score=liquidity_score,
        slippage_estimate_pct=slippage_est,
    )


def _contributor_catalog(side: str, regime: MarketRegime) -> dict[str, tuple[str, str]]:
    side_label = "bullish" if side == "long" else "bearish"
    regime_side = "trend" if regime in {MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND} else regime.value.replace("_", " ")
    return {
        "higher_trend": ("Macro Trend", f"The higher timeframe trend is supporting the {side_label} case."),
        "momentum": ("Trigger Momentum", f"The trigger timeframe momentum is supporting the {side_label} case."),
        "trend": ("Context Trend", f"The context timeframe trend is helping this {side_label} setup."),
        "entry_confirmation": ("Entry Timing", "The lower timeframe timing is aligned with the setup."),
        "structure": ("Structure", "Nearby market structure gives the setup room to work."),
        "volume_surge": ("Volume Surge", "Participation is expanding in the setup direction."),
        "buy_sell_pressure": ("Order Flow Pressure", "Candle pressure is aligned with the selected side."),
        "oi_funding_bias": ("Funding / OI", "Funding and open-interest context are leaning in this direction."),
        "regime_alignment": ("Regime Fit", f"The current {regime_side} regime supports this setup."),
        "volume_divergence_penalty": (
            "Volume Divergence",
            "Momentum is not fully confirmed by pressure or participation.",
        ),
        "confirmation_penalty": (
            "Confirmation Gap",
            "The timeframes are not agreeing cleanly enough yet.",
        ),
        "target_ambition_penalty": (
            "Target Stretch",
            "The target is ambitious relative to current ATR.",
        ),
        "regime_penalty": (
            "Regime Friction",
            "The current regime reduces setup quality for this side.",
        ),
        "volume_poc_proximity": (
            "Volume POC",
            "Price is near a high-volume node — strong support/resistance.",
        ),
        "timeframe_alignment": (
            "Timeframe Alignment",
            "All timeframes agree on direction — high-conviction setup.",
        ),
        "reversal_signal_count": (
            "Reversal Warning",
            "Multiple signals suggest the move may be exhausting against this side.",
        ),
        "entry_confluence": (
            "Entry Confluence",
            "Multiple factors align at the entry level — stronger support/resistance.",
        ),
        "target_confluence": (
            "Target Confluence",
            "Multiple factors align at the target level — more likely to be reached.",
        ),
    }


def _to_contributor_details(items: list[_Contribution]) -> list[ContributorDetail]:
    return [
        ContributorDetail(
            key=item.key,
            label=item.label,
            value=round(item.value, 6),
            impact=round(item.impact, 6),
            direction=item.direction,
            summary=item.summary,
        )
        for item in items
    ]


def _timeframe_alignment_score(
    side: str,
    *,
    higher_trend: float,
    context_trend: float,
    trigger_momentum: float,
    entry_momentum: float,
) -> float:
    """Return an alignment score in [-1, 1].

    Positive = all timeframes agree with `side`.
    Negative = timeframes conflict with `side`.
    Each of the four signals is normalised to [-1, 1] before averaging.
    """
    direction = 1.0 if side == "long" else -1.0
    signals = [
        higher_trend * direction,
        context_trend * direction,
        trigger_momentum * direction,
        entry_momentum * direction,
    ]
    normalised = [_clamp(s * 20.0, -1.0, 1.0) for s in signals]
    return sum(normalised) / len(normalised)


def _detect_early_reversal_signals(
    side: str,
    *,
    trigger_candles: list[Candle],
    entry_candles: list[Candle],
    support: float,
    resistance: float,
    rsi_divergence_type: str,
    macd_histogram: float,
) -> dict[str, bool]:
    """Return a dict of fired reversal signals for the given side.

    True = that signal is warning against entering this side right now.
    """
    signals: dict[str, bool] = {}

    # 1. RSI divergence against the side
    if side == "long":
        signals["rsi_divergence"] = rsi_divergence_type == "bearish"
    else:
        signals["rsi_divergence"] = rsi_divergence_type == "bullish"

    # 2. MACD histogram flip against the side (histogram crossed zero)
    if len(trigger_candles) >= 2:
        prev_hist = macd(trigger_candles[:-1]).histogram or 0.0
        if side == "long":
            signals["macd_flip"] = prev_hist > 0 and macd_histogram < 0
        else:
            signals["macd_flip"] = prev_hist < 0 and macd_histogram > 0
    else:
        signals["macd_flip"] = False

    # 3. Momentum divergence: trigger TF still going one way but entry TF reversed
    trigger_mom = _momentum(trigger_candles)
    entry_mom = _momentum(entry_candles, window=12)
    if side == "long":
        signals["momentum_divergence"] = trigger_mom > 0 and entry_mom < -0.002
    else:
        signals["momentum_divergence"] = trigger_mom < 0 and entry_mom > 0.002

    # 4. Structure break: price already through the key level (0.2% buffer)
    px = trigger_candles[-1].close
    if side == "long":
        signals["structure_break"] = px < support * 0.998
    else:
        signals["structure_break"] = px > resistance * 1.002

    return signals


def _apply_reversal_penalty(
    setup: _SideMetrics,
    signals: dict[str, bool],
    regime: MarketRegime,
) -> None:
    """Apply confidence/quality penalty when reversal signals fire.

    1 signal  → mild confidence reduction, no block.
    2 signals → meaningful reduction.
    3+        → strong reduction + trade blocked.
    Volatile chop: even 1 signal blocks.
    """
    fired = [k for k, v in signals.items() if v]
    count = len(fired)

    if count == 0:
        setup.components["reversal_signal_count"] = 0.0
        return

    if count == 1:
        setup.confidence = _clamp(setup.confidence * 0.90, 0.0, 1.0)
    elif count == 2:
        setup.confidence = _clamp(setup.confidence * 0.75, 0.0, 1.0)
        setup.quality_score = _clamp(setup.quality_score - 8.0, 10.0, 95.0)
    else:
        setup.confidence = _clamp(setup.confidence * 0.60, 0.0, 1.0)
        setup.quality_score = _clamp(setup.quality_score - 15.0, 10.0, 95.0)
        setup.tradable_reasons.append(
            f"early reversal: {count} signals fired ({', '.join(fired)})"
        )

    if regime == MarketRegime.VOLATILE_CHOP and count >= 1:
        if not any("early reversal" in r for r in setup.tradable_reasons):
            setup.tradable_reasons.append(
                f"early reversal in volatile chop: {', '.join(fired)}"
            )

    setup.components["reversal_signal_count"] = float(count)
    setup.is_tradable = not setup.tradable_reasons


def _round_number_proximity(price: float, threshold_pct: float = 0.5) -> bool:
    """True if price is within threshold_pct% of a psychologically significant round number.

    Checks multiples of the largest power of 10 that fits in the price,
    and half that magnitude (e.g. for a 5-digit price: 10000s and 5000s).
    """
    if price <= 0:
        return False
    digits = len(str(int(price)))
    magnitude = 10 ** max(digits - 1, 0)
    # Check multiples of magnitude and magnitude/2 (catches e.g. 65000, 2500)
    for step in (magnitude, magnitude // 2 if magnitude >= 2 else magnitude):
        if step <= 0:
            continue
        nearest = round(price / step) * step
        if nearest > 0 and abs(price - nearest) / nearest * 100 < threshold_pct:
            return True
    return False


def _score_confluence(
    side: str,
    *,
    entry: float,
    target: float,
    support: float,
    resistance: float,
    atr: float,
    volume_profile: dict | None = None,
    near_threshold_pct: float = 1.0,
) -> dict[str, float]:
    """Count independent factors aligning at entry and target price levels.

    Returns entry_confluence and target_confluence counts (0–4 each).
    volume_profile is optional — skipped when None (task 1 not yet done).
    """
    def near(price: float, level: float) -> bool:
        return level > 0 and abs(price - level) / level * 100 < near_threshold_pct

    entry_factors = 0
    target_factors = 0

    if side == "long":
        if near(entry, support):
            entry_factors += 1
        if volume_profile and near(entry, volume_profile.get("val", 0.0)):
            entry_factors += 1
        if volume_profile and near(entry, volume_profile.get("poc", 0.0)):
            entry_factors += 1
        if _round_number_proximity(entry):
            entry_factors += 1

        if near(target, resistance):
            target_factors += 1
        if volume_profile and near(target, volume_profile.get("vah", 0.0)):
            target_factors += 1
        if volume_profile and near(target, volume_profile.get("poc", 0.0)):
            target_factors += 1
        if _round_number_proximity(target):
            target_factors += 1
    else:
        if near(entry, resistance):
            entry_factors += 1
        if volume_profile and near(entry, volume_profile.get("vah", 0.0)):
            entry_factors += 1
        if volume_profile and near(entry, volume_profile.get("poc", 0.0)):
            entry_factors += 1
        if _round_number_proximity(entry):
            entry_factors += 1

        if near(target, support):
            target_factors += 1
        if volume_profile and near(target, volume_profile.get("val", 0.0)):
            target_factors += 1
        if volume_profile and near(target, volume_profile.get("poc", 0.0)):
            target_factors += 1
        if _round_number_proximity(target):
            target_factors += 1

    return {
        "entry_confluence": float(entry_factors),
        "target_confluence": float(target_factors),
    }


def _apply_confluence_boost(
    setup: _SideMetrics,
    confluence: dict[str, float],
) -> None:
    """Boost quality_score when multiple factors align at entry/target levels.

    Entry confluence (safer stop): up to +10 pts.
    Target confluence (more reachable): up to +5 pts.
    """
    entry_c = confluence["entry_confluence"]
    target_c = confluence["target_confluence"]

    if entry_c >= 3:
        setup.quality_score = _clamp(setup.quality_score + 10.0, 10.0, 95.0)
    elif entry_c == 2:
        setup.quality_score = _clamp(setup.quality_score + 6.0, 10.0, 95.0)
    elif entry_c == 1:
        setup.quality_score = _clamp(setup.quality_score + 2.0, 10.0, 95.0)

    if target_c >= 2:
        setup.quality_score = _clamp(setup.quality_score + 5.0, 10.0, 95.0)
    elif target_c == 1:
        setup.quality_score = _clamp(setup.quality_score + 2.0, 10.0, 95.0)

    setup.components["entry_confluence"] = entry_c
    setup.components["target_confluence"] = target_c
    setup.quality_label = _quality_label(setup.quality_score)


class SetupAnalyzer:
    def __init__(
        self,
        *,
        risk_reward: float = 2.0,
        style: StrategyStyle = StrategyStyle.CONSERVATIVE,
        market_mode: MarketMode = MarketMode.INTRADAY,
        filter_overrides: dict[str, float] | None = None,
    ) -> None:
        self.risk_reward = max(risk_reward, 0.5)
        self.style = style
        self.market_mode = market_mode
        self._filter_overrides: dict[str, float] = filter_overrides or {}

    def _mode_params(self) -> dict[str, float]:
        from futures_analyzer.config import load_app_config
        config = load_app_config()
        tuning = config.style_tuning(self.style)
        mode_settings = config.market_mode_settings(self.market_mode)
        return {
            "fallback_risk_reward": tuning.fallback_risk_reward,
            "target_cap_atr_mult": (
                mode_settings.target_cap_atr_mult
                if mode_settings.target_cap_atr_mult is not None
                else tuning.target_cap_atr_mult
            ),
            "ambition_penalty_start_atr": tuning.ambition_penalty_start_atr,
            "ambition_penalty_slope": tuning.ambition_penalty_slope,
        }

    def _trade_filter_params(self) -> dict[str, float]:
        from futures_analyzer.config import load_app_config
        config = load_app_config()
        tuning = config.style_tuning(self.style)
        mode_settings = config.market_mode_settings(self.market_mode)
        params = {
            "min_confidence": (
                mode_settings.min_confidence
                if mode_settings.min_confidence is not None
                else tuning.min_confidence
            ),
            "min_quality": tuning.min_quality,
            "min_rr_ratio": tuning.min_rr_ratio,
            "max_stop_distance_pct": (
                mode_settings.max_stop_distance_pct
                if mode_settings.max_stop_distance_pct is not None
                else tuning.max_stop_distance_pct
            ),
            "min_evidence_agreement": (
                mode_settings.min_evidence_agreement
                if mode_settings.min_evidence_agreement is not None
                else tuning.min_evidence_agreement
            ),
            "min_evidence_edge": (
                mode_settings.min_evidence_edge
                if mode_settings.min_evidence_edge is not None
                else tuning.min_evidence_edge
            ),
        }
        params.update(self._filter_overrides)
        return params

    def _trade_filter_reasons(
        self,
        *,
        confidence: float,
        quality_score: float,
        risk_reward_ratio: float,
        stop_distance_pct: float,
        regime: MarketRegime,
    ) -> list[str]:
        params = self._trade_filter_params()
        reasons: list[str] = []
        if confidence < params["min_confidence"]:
            reasons.append(f"confidence {confidence:.2f} is below {params['min_confidence']:.2f}")
        if quality_score < params["min_quality"]:
            reasons.append(f"quality {quality_score:.1f} is below {params['min_quality']:.1f}")
        if risk_reward_ratio < params["min_rr_ratio"]:
            reasons.append(f"R:R {risk_reward_ratio:.2f} is below {params['min_rr_ratio']:.2f}")
        if stop_distance_pct > params["max_stop_distance_pct"]:
            reasons.append(
                f"stop distance {stop_distance_pct:.2f}% is above {params['max_stop_distance_pct']:.2f}%"
            )
        if regime == MarketRegime.VOLATILE_CHOP:
            reasons.append("volatile chop is filtered out")
        return reasons

    def _collect_evidence(
        self,
        side: str,
        *,
        higher_trend: float,
        context_trend: float,
        trigger_momentum: float,
        entry_momentum: float,
        trigger_pressure: float,
        entry_pressure: float,
        trigger_volume_surge: float,
        entry_volume_surge: float,
        regime: MarketRegime,
    ) -> _EvidenceSnapshot:
        from futures_analyzer.config import load_app_config
        direction = 1.0 if side == "long" else -1.0
        checks = {
            "macro": higher_trend * direction > 0,
            "context": context_trend * direction > 0,
            "trigger": trigger_momentum * direction > 0,
            "entry": entry_momentum * direction > 0,
            "pressure": ((trigger_pressure + entry_pressure) / 2.0) * direction > load_app_config().strategy.pressure_threshold,
            "volume": max(trigger_volume_surge, entry_volume_surge) >= load_app_config().strategy.volume_surge_threshold,
            "regime": _regime_alignment(regime, side) > 0,
        }
        agreement = sum(1 for ok in checks.values() if ok)
        total = len(checks)
        aligned = ", ".join(label for label, ok in checks.items() if ok) or "none"
        blocked = ", ".join(label for label, ok in checks.items() if not ok) or "none"
        summary = f"{agreement}/{total} aligned | supporting: {aligned} | missing: {blocked}"
        return _EvidenceSnapshot(agreement=agreement, total=total, summary=summary, raw_checks=checks)

    def _apply_deliberation(
        self,
        setup: _SideMetrics,
        evidence: _EvidenceSnapshot,
        opposing_evidence: _EvidenceSnapshot,
        regime: MarketRegime,
    ) -> None:
        params = self._trade_filter_params()
        setup.evidence_agreement = evidence.agreement
        setup.evidence_total = evidence.total
        setup.deliberation_summary = evidence.summary
        if evidence.agreement < params["min_evidence_agreement"]:
            setup.tradable_reasons.append(
                f"evidence agreement {evidence.agreement}/{evidence.total} is below {params['min_evidence_agreement']}"
            )
        if (evidence.agreement - opposing_evidence.agreement) < params["min_evidence_edge"]:
            setup.tradable_reasons.append(
                f"evidence edge over opposing side is only {evidence.agreement - opposing_evidence.agreement}"
            )
        if not evidence.raw_checks.get("macro", False) and not evidence.raw_checks.get("context", False):
            setup.tradable_reasons.append("macro and context trends are not aligned for this side")
        if not evidence.raw_checks.get("trigger", False) and not evidence.raw_checks.get("entry", False):
            setup.tradable_reasons.append("trigger and entry momentum are both unsupportive")
        if evidence.agreement < params["min_evidence_agreement"]:
            setup.quality_score = min(setup.quality_score, _quality_score_cap_from_confidence(0.0))
            setup.quality_label = _quality_label(setup.quality_score)
        if regime == MarketRegime.VOLATILE_CHOP:
            setup.quality_score = min(setup.quality_score, 49.9)
            setup.quality_label = _quality_label(setup.quality_score)
        setup.is_tradable = not setup.tradable_reasons
        setup.leverage_suggestion = _leverage_suggestion(
            stop_distance_pct=setup.stop_distance_pct,
            quality_label=setup.quality_label,
            confidence=setup.confidence,
            regime=regime,
            risk_reward_ratio=setup.risk_reward_ratio,
            is_tradable=setup.is_tradable,
        )

    def analyze(
        self,
        *,
        symbol: str,
        trigger_candles: list[Candle],
        context_candles: list[Candle],
        market: MarketMeta,
        timeframe_plan: TimeframePlan,
        entry_candles: list[Candle] | None = None,
        higher_candles: list[Candle] | None = None,
    ) -> AnalysisResult:
        entry_candles = entry_candles or trigger_candles
        higher_candles = higher_candles or context_candles
        if min(len(entry_candles), len(trigger_candles), len(context_candles), len(higher_candles)) < 30:
            raise ValueError("Insufficient candle history: need at least 30 bars per timeframe")

        atr = _atr(trigger_candles)
        support, resistance = _structure(trigger_candles)
        px = market.mark_price
        higher_trend = _trend_strength(higher_candles, fast_window=4, slow_window=16)
        context_trend = _trend_strength(context_candles)
        trigger_momentum = _momentum(trigger_candles)
        entry_momentum = _momentum(entry_candles, window=12)

        # Volume profile proximity (used for stop reference + quality boost)
        vp = volume_profile_strength(trigger_candles, px)
        regime, regime_confidence = _classify_regime(context_candles, higher_candles, atr, px)
        trigger_volume_surge = _volume_surge_ratio(trigger_candles)
        entry_volume_surge = _volume_surge_ratio(entry_candles)
        trigger_pressure = _buy_sell_pressure(trigger_candles)
        entry_pressure = _buy_sell_pressure(entry_candles)
        combined_pressure = (trigger_pressure + entry_pressure) / 2.0
        combined_volume = max((trigger_volume_surge + entry_volume_surge) / 2.0, max(trigger_volume_surge, entry_volume_surge))
        long_div_penalty, short_div_penalty = _volume_divergence_penalties(
            trigger_momentum,
            combined_pressure,
            combined_volume,
        )
        long_oi_funding_bias, short_oi_funding_bias = _oi_funding_biases(
            market.funding_rate, market.open_interest_change_pct
        )
        structure_long, structure_short = _structure_biases(px, support, resistance)
        long_evidence = self._collect_evidence(
            "long",
            higher_trend=higher_trend,
            context_trend=context_trend,
            trigger_momentum=trigger_momentum,
            entry_momentum=entry_momentum,
            trigger_pressure=trigger_pressure,
            entry_pressure=entry_pressure,
            trigger_volume_surge=trigger_volume_surge,
            entry_volume_surge=entry_volume_surge,
            regime=regime,
        )
        short_evidence = self._collect_evidence(
            "short",
            higher_trend=higher_trend,
            context_trend=context_trend,
            trigger_momentum=trigger_momentum,
            entry_momentum=entry_momentum,
            trigger_pressure=trigger_pressure,
            entry_pressure=entry_pressure,
            trigger_volume_surge=trigger_volume_surge,
            entry_volume_surge=entry_volume_surge,
            regime=regime,
        )
        long_confirmation_penalty = _confirmation_penalty(
            "long",
            higher_trend=higher_trend,
            context_trend=context_trend,
            trigger_momentum=trigger_momentum,
            entry_momentum=entry_momentum,
            trigger_pressure=trigger_pressure,
            entry_pressure=entry_pressure,
            trigger_volume_surge=trigger_volume_surge,
            entry_volume_surge=entry_volume_surge,
        )
        short_confirmation_penalty = _confirmation_penalty(
            "short",
            higher_trend=higher_trend,
            context_trend=context_trend,
            trigger_momentum=trigger_momentum,
            entry_momentum=entry_momentum,
            trigger_pressure=trigger_pressure,
            entry_pressure=entry_pressure,
            trigger_volume_surge=trigger_volume_surge,
            entry_volume_surge=entry_volume_surge,
        )

        # Timeframe alignment scores for both sides
        long_alignment = _timeframe_alignment_score(
            "long",
            higher_trend=higher_trend,
            context_trend=context_trend,
            trigger_momentum=trigger_momentum,
            entry_momentum=entry_momentum,
        )
        short_alignment = _timeframe_alignment_score(
            "short",
            higher_trend=higher_trend,
            context_trend=context_trend,
            trigger_momentum=trigger_momentum,
            entry_momentum=entry_momentum,
        )

        long_setup = self._build_side(
            side="long",
            px=px,
            support=support,
            resistance=resistance,
            atr=atr,
            raw_positive={
                "higher_trend": max(higher_trend, 0.0),
                "momentum": max(trigger_momentum, 0.0),
                "trend": max(context_trend, 0.0),
                "entry_confirmation": max((entry_momentum + max(entry_pressure, 0.0)) / 2.0, 0.0),
                "structure": max(structure_long, 0.0),
                "volume_surge": max(combined_volume - 1.0, 0.0),
                "buy_sell_pressure": max(combined_pressure, 0.0),
                "oi_funding_bias": max(long_oi_funding_bias, 0.0),
                "regime_alignment": _regime_alignment(regime, "long"),
            },
            divergence_penalty=long_div_penalty,
            confirmation_penalty=long_confirmation_penalty,
            regime=regime,
            regime_penalty=_regime_penalty(regime, "long"),
            tick=market.tick_size,
            volume_profile=vp,
            alignment_score=long_alignment,
        )
        short_setup = self._build_side(
            side="short",
            px=px,
            support=support,
            resistance=resistance,
            atr=atr,
            raw_positive={
                "higher_trend": max(-higher_trend, 0.0),
                "momentum": max(-trigger_momentum, 0.0),
                "trend": max(-context_trend, 0.0),
                "entry_confirmation": max(((-entry_momentum) + max(-entry_pressure, 0.0)) / 2.0, 0.0),
                "structure": max(structure_short, 0.0),
                "volume_surge": max(combined_volume - 1.0, 0.0),
                "buy_sell_pressure": max(-combined_pressure, 0.0),
                "oi_funding_bias": max(short_oi_funding_bias, 0.0),
                "regime_alignment": _regime_alignment(regime, "short"),
            },
            divergence_penalty=short_div_penalty,
            confirmation_penalty=short_confirmation_penalty,
            regime=regime,
            regime_penalty=_regime_penalty(regime, "short"),
            tick=market.tick_size,
            volume_profile=vp,
            alignment_score=short_alignment,
        )

        self._apply_deliberation(long_setup, long_evidence, short_evidence, regime)
        self._apply_deliberation(short_setup, short_evidence, long_evidence, regime)

        tradable_setups = [setup for setup in (long_setup, short_setup) if setup.is_tradable]
        ranked_setups = tradable_setups if tradable_setups else [long_setup, short_setup]
        primary = max(
            ranked_setups,
            key=lambda item: (item.is_tradable, item.evidence_agreement, item.score, item.quality_score, item.confidence),
        )
        secondary = short_setup if primary.side == "long" else long_setup

        warnings: list[str] = []
        if not tradable_setups:
            warnings.append("No setup passed the hard trade filters and deliberation checks; treat this symbol as a no-trade.")
        if not primary.is_tradable and primary.tradable_reasons:
            warnings.append("Primary setup was filtered out: " + "; ".join(primary.tradable_reasons))
        if primary.quality_label == QualityLabel.LOW:
            warnings.append("Primary setup quality is low; treat the signal as exploratory.")
        if primary.invalidation_strength < 0.35:
            warnings.append("Primary invalidation level is weak relative to current ATR and structure.")

        # Calculate enhanced metrics
        enhanced_metrics = _calculate_enhanced_metrics(
            entry_candles=entry_candles,
            trigger_candles=trigger_candles,
            context_candles=context_candles,
            market_meta=market,
        )

        # Apply enhanced metrics boost to scoring
        self._apply_enhanced_metrics_boost(long_setup, enhanced_metrics, "long")
        self._apply_enhanced_metrics_boost(short_setup, enhanced_metrics, "short")

        # Early reversal detection
        long_reversal = _detect_early_reversal_signals(
            "long",
            trigger_candles=trigger_candles,
            entry_candles=entry_candles,
            support=support,
            resistance=resistance,
            rsi_divergence_type=enhanced_metrics.rsi_divergence_type,
            macd_histogram=enhanced_metrics.macd_histogram,
        )
        short_reversal = _detect_early_reversal_signals(
            "short",
            trigger_candles=trigger_candles,
            entry_candles=entry_candles,
            support=support,
            resistance=resistance,
            rsi_divergence_type=enhanced_metrics.rsi_divergence_type,
            macd_histogram=enhanced_metrics.macd_histogram,
        )
        _apply_reversal_penalty(long_setup, long_reversal, regime)
        _apply_reversal_penalty(short_setup, short_reversal, regime)

        # Confluence zone detection (uses vp from volume profile, already computed)
        long_confluence = _score_confluence(
            "long",
            entry=long_setup.entry,
            target=long_setup.target,
            support=support,
            resistance=resistance,
            atr=atr,
            volume_profile=vp,
        )
        short_confluence = _score_confluence(
            "short",
            entry=short_setup.entry,
            target=short_setup.target,
            support=support,
            resistance=resistance,
            atr=atr,
            volume_profile=vp,
        )
        _apply_confluence_boost(long_setup, long_confluence)
        _apply_confluence_boost(short_setup, short_confluence)

        # Recalculate quality labels after all post-build adjustments
        long_setup.quality_label = _quality_label(long_setup.quality_score)
        short_setup.quality_label = _quality_label(short_setup.quality_score)

        return AnalysisResult(
            primary_setup=TradeSetup(
                side=primary.side,
                entry_price=primary.entry,
                target_price=primary.target,
                stop_loss=primary.stop,
                leverage_suggestion=primary.leverage_suggestion,
                confidence=primary.confidence,
                quality_label=primary.quality_label,
                quality_score=primary.quality_score,
                rationale=primary.rationale,
                top_positive_contributors=primary.top_positive_contributors,
                top_negative_contributors=primary.top_negative_contributors,
                score_components=primary.components,
                structure_points=primary.structure_points,
                risk_reward_ratio=primary.risk_reward_ratio,
                stop_distance_pct=primary.stop_distance_pct,
                target_distance_pct=primary.target_distance_pct,
                atr_multiple_to_stop=primary.atr_multiple_to_stop,
                atr_multiple_to_target=primary.atr_multiple_to_target,
                invalidation_strength=primary.invalidation_strength,
                is_tradable=primary.is_tradable,
                tradable_reasons=primary.tradable_reasons,
                evidence_agreement=primary.evidence_agreement,
                evidence_total=primary.evidence_total,
                deliberation_summary=primary.deliberation_summary,
            ),
            secondary_context=TradeSetup(
                side=secondary.side,
                entry_price=secondary.entry,
                target_price=secondary.target,
                stop_loss=secondary.stop,
                leverage_suggestion=secondary.leverage_suggestion,
                confidence=secondary.confidence,
                quality_label=secondary.quality_label,
                quality_score=secondary.quality_score,
                rationale=secondary.rationale,
                top_positive_contributors=secondary.top_positive_contributors,
                top_negative_contributors=secondary.top_negative_contributors,
                score_components=secondary.components,
                structure_points=secondary.structure_points,
                risk_reward_ratio=secondary.risk_reward_ratio,
                stop_distance_pct=secondary.stop_distance_pct,
                target_distance_pct=secondary.target_distance_pct,
                atr_multiple_to_stop=secondary.atr_multiple_to_stop,
                atr_multiple_to_target=secondary.atr_multiple_to_target,
                invalidation_strength=secondary.invalidation_strength,
                is_tradable=secondary.is_tradable,
                tradable_reasons=secondary.tradable_reasons,
                evidence_agreement=secondary.evidence_agreement,
                evidence_total=secondary.evidence_total,
                deliberation_summary=secondary.deliberation_summary,
            ),
            timeframe_plan=timeframe_plan,
            market_snapshot_meta=market.model_copy(update={"symbol": symbol}),
            market_regime=regime,
            regime_confidence=regime_confidence,
            enhanced_metrics=enhanced_metrics,
            warnings=warnings,
        )

    def _apply_enhanced_metrics_boost(
        self,
        setup: _SideMetrics,
        enhanced_metrics: EnhancedMetrics,
        side: str,
    ) -> None:
        """Apply enhanced metrics adjustments to setup scoring."""

        # 1. RSI-based confidence adjustment
        rsi = enhanced_metrics.rsi_14
        if side == "long":
            if 30 < rsi < 70:
                setup.confidence *= 1.05  # Neutral zone = better
            elif rsi > 70:
                setup.confidence *= 0.85  # Overbought = risky for longs
            elif rsi < 30:
                setup.confidence *= 0.90  # Oversold = risky
        else:  # short
            if 30 < rsi < 70:
                setup.confidence *= 1.05
            elif rsi < 30:
                setup.confidence *= 0.85  # Oversold = risky for shorts
            elif rsi > 70:
                setup.confidence *= 0.90  # Overbought = risky

        # 2. MACD momentum confirmation
        if enhanced_metrics.macd_histogram > 0 and side == "long":
            setup.confidence += 0.03  # Positive momentum confirms long
        elif enhanced_metrics.macd_histogram < 0 and side == "short":
            setup.confidence += 0.03  # Negative momentum confirms short

        # 3. Bollinger Band position for entry timing
        bb_pos = enhanced_metrics.bollinger_position
        if 0.35 < bb_pos < 0.65:
            setup.quality_score += 5.0  # Middle band = better entry
        elif bb_pos < 0.2 or bb_pos > 0.8:
            setup.quality_score -= 3.0  # Extreme = risky entry

        # 4. Order book imbalance for entry strength
        if abs(enhanced_metrics.order_book_imbalance) > 0.3:
            setup.confidence += 0.02  # Strong imbalance = conviction

        # 5. Volatility-based stop adjustment
        vol_rank = enhanced_metrics.volatility_rank
        if vol_rank > 75:
            setup.stop_distance_pct *= 1.15  # High volatility = wider stops
        elif vol_rank < 25:
            setup.stop_distance_pct *= 0.85  # Low volatility = tighter stops

        # Clamp to valid ranges
        setup.confidence = max(0.0, min(1.0, setup.confidence))
        setup.quality_score = max(10.0, min(95.0, setup.quality_score))

    def _build_side(
        self,
        *,
        side: str,
        px: float,
        support: float,
        resistance: float,
        atr: float,
        raw_positive: dict[str, float],
        divergence_penalty: float,
        confirmation_penalty: float,
        regime: MarketRegime,
        regime_penalty: float,
        tick: float | None,
        volume_profile: dict | None = None,
        alignment_score: float = 0.0,
    ) -> _SideMetrics:
        from futures_analyzer.config import load_app_config
        strategy = load_app_config().strategy
        weights, penalty_multiplier, confidence_ceiling = _regime_weight_profile(regime, side)
        mode_params = self._mode_params()
        atr_buffer = max(atr * 0.5, px * 0.001)

        if side == "long":
            entry = px if px > support else support
            structure_stop = support - atr_buffer
            risk = max(entry - structure_stop, px * 0.001)
            structure_target = resistance
            fallback_target = entry + (risk * self.risk_reward)
            raw_target = structure_target if structure_target > entry + (risk * 0.8) else fallback_target
            max_target_distance = max((atr * mode_params["target_cap_atr_mult"]), px * 0.002)
            target = min(raw_target, entry + max_target_distance)
            structure_reference = support
            rationale = "Bullish confluence from macro trend, context trend, trigger structure, and entry timing."
        else:
            entry = px if px < resistance else resistance
            structure_stop = resistance + atr_buffer
            risk = max(structure_stop - entry, px * 0.001)
            structure_target = support
            fallback_target = entry - (risk * self.risk_reward)
            raw_target = structure_target if structure_target < entry - (risk * 0.8) else fallback_target
            max_target_distance = max((atr * mode_params["target_cap_atr_mult"]), px * 0.002)
            target = max(raw_target, entry - max_target_distance)
            structure_reference = resistance
            rationale = "Bearish confluence from macro trend, context trend, trigger structure, and entry timing."

        # Volume profile: use VAL/VAH as stop reference when they're closer and
        # volume-backed (stronger than a raw swing low/high).
        if volume_profile is not None:
            if side == "long":
                val = volume_profile.get("val", 0.0)
                if val > support and abs(px - val) < atr * 1.5:
                    structure_stop = val - atr_buffer
                    structure_reference = val
                    risk = max(entry - structure_stop, px * 0.001)
            else:
                vah = volume_profile.get("vah", 0.0)
                if vah > 0 and vah < resistance and abs(px - vah) < atr * 1.5:
                    structure_stop = vah + atr_buffer
                    structure_reference = vah
                    risk = max(structure_stop - entry, px * 0.001)

        reward = abs(target - entry)
        target_distance_atr = reward / max(atr, 1e-9)
        ambition_excess = max(target_distance_atr - mode_params["ambition_penalty_start_atr"], 0.0)
        ambition_penalty = ambition_excess * mode_params["ambition_penalty_slope"]

        catalog = _contributor_catalog(side, regime)
        positive_contribs: list[_Contribution] = []
        weighted_positive_total = 0.0
        components: dict[str, float] = {}

        # Inject volume POC proximity into raw_positive
        if volume_profile is not None:
            if volume_profile.get("near_poc"):
                raw_positive["volume_poc_proximity"] = 0.8
            elif side == "long" and volume_profile.get("near_val"):
                raw_positive["volume_poc_proximity"] = 0.4
            elif side == "short" and volume_profile.get("near_vah"):
                raw_positive["volume_poc_proximity"] = 0.4
            else:
                raw_positive["volume_poc_proximity"] = 0.0

        for key, value in raw_positive.items():
            impact = value * weights.get(key, 1.0)
            weighted_positive_total += impact
            components[key] = impact
            label, summary = catalog[key]
            positive_contribs.append(
                _Contribution(
                    key=key,
                    label=label,
                    value=value,
                    impact=impact,
                    direction=ContributorDirection.POSITIVE,
                    summary=summary,
                )
            )

        negative_raw = {
            "volume_divergence_penalty": divergence_penalty,
            "confirmation_penalty": confirmation_penalty,
            "target_ambition_penalty": ambition_penalty,
            "regime_penalty": regime_penalty,
        }
        negative_contribs: list[_Contribution] = []
        weighted_negative_total = 0.0
        for key, value in negative_raw.items():
            if key == "volume_divergence_penalty":
                impact = value * penalty_multiplier
            elif key == "confirmation_penalty":
                impact = value * strategy.confirmation_penalty_weight
            elif key == "target_ambition_penalty":
                impact = value * strategy.target_ambition_penalty_weight
            else:
                impact = value * strategy.regime_penalty_weight
            weighted_negative_total += impact
            components[key] = impact
            label, summary = catalog[key]
            negative_contribs.append(
                _Contribution(
                    key=key,
                    label=label,
                    value=value,
                    impact=impact,
                    direction=ContributorDirection.NEGATIVE,
                    summary=summary,
                )
            )

        score = max(weighted_positive_total - weighted_negative_total, 0.0)
        base_confidence = score / strategy.score_confidence_divisor
        if regime == MarketRegime.VOLATILE_CHOP:
            strong_agreement = sum(
                1
                for key in ("higher_trend", "trend", "momentum", "entry_confirmation")
                if raw_positive.get(key, 0.0) > strategy.volatile_chop_strong_signal_threshold
            )
            confidence_ceiling = min(
                confidence_ceiling + (strong_agreement * strategy.volatile_chop_boost_per_signal),
                strategy.volatile_chop_confidence_ceiling_cap,
            )
        confidence = _clamp(base_confidence, 0.0, confidence_ceiling)

        # Timeframe alignment multiplier
        if alignment_score >= 0.6:
            confidence = _clamp(confidence * 1.10, 0.0, confidence_ceiling)
        elif alignment_score >= 0.2:
            pass  # mostly agree — no change
        elif alignment_score >= -0.2:
            confidence = _clamp(confidence * 0.90, 0.0, confidence_ceiling)
        else:
            confidence = _clamp(confidence * 0.75, 0.0, confidence_ceiling)

        rr_ratio = reward / max(risk, 1e-9)
        stop_distance_pct = (risk / max(entry, 1e-9)) * 100.0
        target_distance_pct = (reward / max(entry, 1e-9)) * 100.0
        atr_multiple_to_stop = risk / max(atr, 1e-9)
        atr_multiple_to_target = reward / max(atr, 1e-9)
        structure_gap = abs(entry - structure_reference)
        range_context = abs(resistance - support) / max(atr, 1e-9)
        formula_penalty = 0.2 if structure_gap <= (px * 0.0015) else 0.0
        invalidation_strength = _clamp(
            ((structure_gap / max(atr, 1e-9)) * 0.35)
            + ((atr_buffer / max(atr, 1e-9)) * 0.45)
            + (min(range_context, 3.0) / 3.0 * 0.2)
            - formula_penalty,
            0.0,
            1.0,
        )

        raw_quality_score = _clamp(
            strategy.quality_base_score
            + min(weighted_positive_total, strategy.quality_positive_cap) * strategy.quality_positive_weight
            + min(rr_ratio, 3.0) * strategy.quality_rr_weight
            + invalidation_strength * strategy.quality_invalidation_weight
            - min(weighted_negative_total, strategy.quality_negative_cap) * strategy.quality_negative_weight,
            strategy.quality_min_score,
            strategy.quality_max_score,
        )
        quality_score = min(raw_quality_score, _quality_score_cap_from_confidence(confidence))
        quality_label = _quality_label(quality_score)
        tradable_reasons = self._trade_filter_reasons(
            confidence=confidence,
            quality_score=quality_score,
            risk_reward_ratio=rr_ratio,
            stop_distance_pct=stop_distance_pct,
            regime=regime,
        )
        is_tradable = not tradable_reasons
        leverage_suggestion = _leverage_suggestion(
            stop_distance_pct=stop_distance_pct,
            quality_label=quality_label,
            confidence=confidence,
            regime=regime,
            risk_reward_ratio=rr_ratio,
            is_tradable=is_tradable,
        )

        positive_contribs.sort(key=lambda item: item.impact, reverse=True)
        negative_contribs.sort(key=lambda item: item.impact, reverse=True)
        top_positive = _to_contributor_details(positive_contribs[:3])
        top_negative = _to_contributor_details([item for item in negative_contribs if item.impact > 0][:3])

        components["risk_reward_ratio"] = rr_ratio
        components["stop_distance_pct"] = stop_distance_pct
        components["target_distance_pct"] = target_distance_pct
        components["atr_multiple_to_stop"] = atr_multiple_to_stop
        components["atr_multiple_to_target"] = atr_multiple_to_target
        components["invalidation_strength"] = invalidation_strength
        components["score"] = score
        components["timeframe_alignment"] = round(alignment_score, 4)

        return _SideMetrics(
            side=side,
            score=score,
            confidence=round(confidence, 4),
            quality_label=quality_label,
            quality_score=round(quality_score, 1),
            leverage_suggestion=leverage_suggestion,
            entry=_quantize(entry, tick),
            stop=_quantize(structure_stop, tick),
            target=_quantize(target, tick),
            rationale=rationale,
            top_positive_contributors=top_positive,
            top_negative_contributors=top_negative,
            components={key: round(value, 6) for key, value in components.items()},
            structure_points={
                "support": round(support, 6),
                "resistance": round(resistance, 6),
                "atr": round(atr, 6),
                "pressure": round(raw_positive.get("buy_sell_pressure", 0.0), 6),
            },
            risk_reward_ratio=round(rr_ratio, 4),
            stop_distance_pct=round(stop_distance_pct, 4),
            target_distance_pct=round(target_distance_pct, 4),
            atr_multiple_to_stop=round(atr_multiple_to_stop, 4),
            atr_multiple_to_target=round(atr_multiple_to_target, 4),
            invalidation_strength=round(invalidation_strength, 4),
            is_tradable=is_tradable,
            tradable_reasons=tradable_reasons,
            evidence_agreement=0,
            evidence_total=0,
            deliberation_summary="",
        )


# ── Drawdown Recovery Adjuster ────────────────────────────────────────────────

class DrawdownAdjuster:
    """Applies leverage and quality penalties when recent predictions are in drawdown.

    Severity thresholds and step sizes are driven by AppConfig.drawdown so they
    can be tuned without touching code.
    """

    # Leverage step ladder used to reduce suggestions (e.g. "5x" → "3x")
    _LEVERAGE_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    @classmethod
    def apply(
        cls,
        setup: "TradeSetup",
        drawdown_state: "DrawdownState",
    ) -> "TradeSetup":
        """Return a copy of *setup* with leverage and quality adjusted for drawdown.

        No-op when severity is "none".
        """
        from futures_analyzer.history.models import DrawdownState  # local to avoid circular

        severity = drawdown_state.severity
        if severity == "none":
            return setup

        quality_penalty, confidence_bump, leverage_steps_down = {
            "mild":     (5.0,  0.05, 1),
            "moderate": (10.0, 0.10, 2),
            "severe":   (15.0, 0.15, None),  # None = floor at 1x
        }[severity]

        new_quality = _clamp(setup.quality_score - quality_penalty, 10.0, 95.0)
        new_quality_label = _quality_label(new_quality)
        new_confidence = _clamp(setup.confidence - confidence_bump, 0.0, 1.0)

        # Reduce leverage
        current_lev = int(setup.leverage_suggestion.rstrip("x") or 1)
        if leverage_steps_down is None:
            new_lev = 1
        else:
            try:
                idx = cls._LEVERAGE_STEPS.index(current_lev)
            except ValueError:
                # Find nearest step
                idx = min(
                    range(len(cls._LEVERAGE_STEPS)),
                    key=lambda i: abs(cls._LEVERAGE_STEPS[i] - current_lev),
                )
            new_lev = cls._LEVERAGE_STEPS[max(0, idx - leverage_steps_down)]

        warning = (
            f"drawdown guard ({severity}): "
            f"{drawdown_state.current_drawdown_pct:.1f}% drawdown, "
            f"{drawdown_state.consecutive_losses} consecutive loss(es) — "
            f"leverage reduced to {new_lev}x, quality −{quality_penalty:.0f} pts"
        )

        updated_reasons = list(setup.tradable_reasons)
        # Only add the warning if the setup was tradable (non-tradable already has reasons)
        if setup.is_tradable:
            updated_reasons.append(warning)

        return setup.model_copy(update={
            "quality_score": new_quality,
            "quality_label": new_quality_label,
            "confidence": new_confidence,
            "leverage_suggestion": f"{new_lev}x",
            "tradable_reasons": updated_reasons,
        })
