from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    LiquiditySweep,
    sigmoid,
    gaussian_peak,
    rsi,
    macd,
    stochastic,
    bollinger_bands,
    rsi_divergence,
    volume_profile_strength,
    adx,
    volume_profile,
    compute_vwap_bands,
    compute_market_structure,
    compute_cumulative_delta,
    compute_adx_slope,
    detect_liquidity_sweeps,
    _swing_pivots,
)
from futures_analyzer.analysis.regime import (
    classify_regime as _regime_classify,
    classify_regime_consensus,
)


@dataclass
class IndicatorBundle:
    # Per-timeframe ATR
    entry_atr: float
    trigger_atr: float
    context_atr: float
    higher_atr: float

    # Trend / momentum
    higher_trend: float        # EMA ribbon slope, normalized [-1, 1]
    context_trend: float
    trigger_momentum: float    # rate-of-change, normalized [-1, 1]
    entry_momentum: float

    # Volume
    trigger_volume_surge: float   # ratio vs rolling mean, [0, ∞)
    entry_volume_surge: float
    cumulative_delta: float       # buy_vol - sell_vol normalized [-1, 1]

    # Oscillators (entry TF)
    rsi_14: float                 # [0, 100]
    macd_histogram: float         # raw
    stoch_k: float                # [0, 100]
    bb_position: float            # [0, 1]
    bb_bandwidth_pct: float

    # Structure
    swing_highs: list[float]      # recent confirmed swing highs
    swing_lows: list[float]       # recent confirmed swing lows
    market_structure: str         # "HH_HL" | "LH_LL" | "mixed"
    liquidity_sweeps: list        # list[LiquiditySweep]

    # VWAP
    vwap: float
    vwap_upper_1sd: float
    vwap_lower_1sd: float
    vwap_upper_2sd: float
    vwap_lower_2sd: float

    # Volume profile
    poc: float
    vah: float
    val: float

    # RSI divergence
    rsi_divergence_type: str      # "bullish" | "bearish" | "none"
    rsi_divergence_strength: float

    # OI / funding
    funding_rate: float | None
    oi_change_pct: float | None
    funding_momentum: float       # slope of recent funding history [-1, 1]

    # Order book (optional)
    order_book_imbalance: float   # [-1, 1]
    bid_ask_spread_pct: float

    # Warnings accumulated during computation
    warnings: list[str] = field(default_factory=list)


@dataclass
class NormalizedSignals:
    # Each field is in [0, 1] — strength of evidence FOR that side
    higher_trend: float
    context_trend: float
    trigger_momentum: float
    entry_momentum: float
    volume_surge: float
    buy_pressure: float
    oi_funding_bias: float
    funding_momentum: float
    structure_position: float
    rsi_alignment: float
    macd_alignment: float
    bb_alignment: float
    vwap_alignment: float
    market_structure_align: float
    cumulative_delta_align: float
    volume_poc_proximity: float


@dataclass
class EvidenceVector:
    weighted_sum: float
    signal_count_above_threshold: int
    strongest_signals: list[tuple[str, float]]  # top 3 (name, strength)
    weakest_signals: list[tuple[str, float]]    # bottom 3
    regime_gate_passed: bool


@dataclass
class SwingPoints:
    recent_highs: list[float]   # last N confirmed swing highs, ascending
    recent_lows: list[float]    # last N confirmed swing lows, ascending
    nearest_high: float
    nearest_low: float
    second_high: float
    second_low: float


@dataclass
class EntryGeometry:
    entry: float
    stop: float
    target: float
    risk: float
    reward: float
    rr_ratio: float
    stop_distance_pct: float
    target_distance_pct: float
    atr_multiple_to_stop: float
    atr_multiple_to_target: float
    stop_anchor: str    # "swing_low" | "vwap_lower" | "val" | "atr_fallback" | "rr_enforced"
    target_anchor: str  # "swing_high" | "vwap_upper" | "vah" | "atr_cap" | "rr_enforced"
    invalidation_strength: float


def find_swing_points(candles: list[Candle], pivot_n: int = 3) -> SwingPoints:
    """Find confirmed swing highs and lows from candle data.

    Uses _swing_pivots on candle highs and lows separately, then sorts both
    lists ascending (most recent last).

    Returns a SwingPoints dataclass with:
    - recent_highs / recent_lows: all confirmed pivot values, sorted ascending
    - nearest_high / nearest_low: last element (most recent), or 0.0 if empty
    - second_high / second_low: second-to-last element, or nearest if only one
    """
    highs_series = [c.high for c in candles]
    lows_series = [c.low for c in candles]

    # Collect swing highs: pivots where value equals the window maximum
    raw_high_pivots = _swing_pivots(highs_series, n=pivot_n)
    highs = sorted(
        val for idx, val in raw_high_pivots
        if val == max(highs_series[max(0, idx - pivot_n): idx + pivot_n + 1])
    )

    # Collect swing lows: pivots where value equals the window minimum
    raw_low_pivots = _swing_pivots(lows_series, n=pivot_n)
    lows = sorted(
        val for idx, val in raw_low_pivots
        if val == min(lows_series[max(0, idx - pivot_n): idx + pivot_n + 1])
    )

    nearest_high = highs[-1] if highs else 0.0
    nearest_low = lows[-1] if lows else 0.0
    second_high = highs[-2] if len(highs) >= 2 else nearest_high
    second_low = lows[-2] if len(lows) >= 2 else nearest_low

    return SwingPoints(
        recent_highs=highs,
        recent_lows=lows,
        nearest_high=nearest_high,
        nearest_low=nearest_low,
        second_high=second_high,
        second_low=second_low,
    )


def select_best_stop(
    candidates: list[tuple[float, str]],
    entry: float,
    atr: float,
) -> tuple[float, str]:
    """Select the best stop candidate from a prioritized list.

    Prefers the first candidate whose anchor_label starts with "swing" and
    whose price is within 2.5 ATR of entry. Falls back to the first candidate
    if no swing anchor meets the proximity criterion.

    Args:
        candidates: list of (price, anchor_label) tuples in priority order
        entry: entry price
        atr: current ATR value

    Returns:
        (price, anchor_label) tuple
    """
    for price, label in candidates:
        if label.startswith("swing") and abs(entry - price) <= 2.5 * atr:
            return price, label
    return candidates[0]


def select_best_target(
    candidates: list[tuple[float, str]],
    entry: float,
    stop: float,
    params: dict,
) -> tuple[float, str]:
    """Select the best target candidate from a prioritized list.

    Prefers the first candidate whose anchor_label starts with "swing" and
    whose R:R ratio is >= params["min_rr_ratio"]. Falls back to the first
    candidate if no swing anchor meets the R:R criterion.

    Args:
        candidates: list of (price, anchor_label) tuples in priority order
        entry: entry price
        stop: stop price
        params: dict with "min_rr_ratio" key

    Returns:
        (price, anchor_label) tuple
    """
    min_rr = params["min_rr_ratio"]
    risk = max(abs(entry - stop), 1e-9)
    for price, label in candidates:
        if label.startswith("swing"):
            rr = abs(price - entry) / risk
            if rr >= min_rr:
                return price, label
    return candidates[0]


def place_entry_stop_target(
    side: str,
    px: float,
    swings: SwingPoints,
    bundle: IndicatorBundle,
    atr: float,
    regime: MarketRegime,
    style: StrategyStyle,
    mode_params: dict[str, float],
    tick: float | None,
) -> EntryGeometry:
    """Place entry, stop, and target using swing pivots and VWAP anchors.

    Long stop priority: (1) nearest swing low below entry minus ATR buffer,
    (2) VWAP lower 1SD, (3) VAL, (4) ATR fallback.
    Long target priority: (1) nearest swing high above entry, (2) VWAP upper 2SD,
    (3) VAH, (4) ATR cap.
    Symmetric mirror logic applies for short setups.
    Enforces minimum R:R by extending target if needed.
    Quantizes entry/stop/target to tick size.
    Substitutes max(atr, px * 0.001) when ATR is zero.
    """
    # ATR safety: substitute max(atr, px * 0.001) when ATR is zero
    atr = max(atr, px * 0.001)
    atr_buffer = atr * mode_params.get("atr_buffer_factor", 0.5)
    min_rr = mode_params.get("min_rr_ratio", 1.5)
    target_cap_atr_mult = mode_params.get("target_cap_atr_mult", 3.0)

    entry = px

    if side == "long":
        # --- Stop placement (priority order) ---
        candidate_stops: list[tuple[float, str]] = []

        # 1. Nearest swing low below entry
        valid_lows = [l for l in swings.recent_lows if l < entry - atr * 0.1]
        if valid_lows:
            nearest_low = max(valid_lows)
            candidate_stops.append((nearest_low - atr_buffer, "swing_low"))

        # 2. VWAP lower 1SD band
        if bundle.vwap_lower_1sd < entry:
            candidate_stops.append((bundle.vwap_lower_1sd - atr_buffer * 0.5, "vwap_lower"))

        # 3. Volume profile VAL
        if bundle.val < entry:
            candidate_stops.append((bundle.val - atr_buffer * 0.5, "val"))

        # 4. ATR fallback
        candidate_stops.append((entry - atr * 1.5, "atr_fallback"))

        stop, stop_anchor = select_best_stop(candidate_stops, entry, atr)

        # --- Target placement (priority order) ---
        candidate_targets: list[tuple[float, str]] = []

        # 1. Nearest swing high above entry
        valid_highs = [h for h in swings.recent_highs if h > entry + atr * 0.3]
        if valid_highs:
            nearest_high = min(valid_highs)
            candidate_targets.append((nearest_high, "swing_high"))

        # 2. VWAP upper 2SD band
        if bundle.vwap_upper_2sd > entry:
            candidate_targets.append((bundle.vwap_upper_2sd, "vwap_upper"))

        # 3. Volume profile VAH
        if bundle.vah > entry:
            candidate_targets.append((bundle.vah, "vah"))

        # 4. ATR cap
        max_target = entry + atr * target_cap_atr_mult
        candidate_targets.append((max_target, "atr_cap"))

        target, target_anchor = select_best_target(candidate_targets, entry, stop, {"min_rr_ratio": min_rr})

    else:  # short — mirror logic
        # --- Stop placement (priority order) ---
        candidate_stops = []

        # 1. Nearest swing high above entry
        valid_highs = [h for h in swings.recent_highs if h > entry + atr * 0.1]
        if valid_highs:
            nearest_high = min(valid_highs)
            candidate_stops.append((nearest_high + atr_buffer, "swing_high"))

        # 2. VWAP upper 1SD band
        if bundle.vwap_upper_1sd > entry:
            candidate_stops.append((bundle.vwap_upper_1sd + atr_buffer * 0.5, "vwap_upper"))

        # 3. Volume profile VAH
        if bundle.vah > entry:
            candidate_stops.append((bundle.vah + atr_buffer * 0.5, "vah"))

        # 4. ATR fallback
        candidate_stops.append((entry + atr * 1.5, "atr_fallback"))

        stop, stop_anchor = select_best_stop(candidate_stops, entry, atr)

        # --- Target placement (priority order) ---
        candidate_targets = []

        # 1. Nearest swing low below entry
        valid_lows = [l for l in swings.recent_lows if l < entry - atr * 0.3]
        if valid_lows:
            nearest_low = max(valid_lows)
            candidate_targets.append((nearest_low, "swing_low"))

        # 2. VWAP lower 2SD band
        if bundle.vwap_lower_2sd < entry:
            candidate_targets.append((bundle.vwap_lower_2sd, "vwap_lower"))

        # 3. Volume profile VAL
        if bundle.val < entry:
            candidate_targets.append((bundle.val, "val"))

        # 4. ATR cap (floor for short)
        min_target = entry - atr * target_cap_atr_mult
        candidate_targets.append((min_target, "atr_cap"))

        target, target_anchor = select_best_target(candidate_targets, entry, stop, {"min_rr_ratio": min_rr})

    # Enforce minimum R:R
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if reward / max(risk, 1e-9) < min_rr:
        if side == "long":
            target = entry + risk * min_rr
        else:
            target = entry - risk * min_rr
        target_anchor = "rr_enforced"

    # Quantize
    entry = _quantize(entry, tick)
    stop = _quantize(stop, tick)
    target = _quantize(target, tick)

    # Enforce ordering invariants after quantization (adjust by 1 tick if violated)
    if tick and tick > 0:
        if side == "long":
            if stop >= entry:
                stop = entry - tick
            if target <= entry:
                target = entry + tick
        else:
            if stop <= entry:
                stop = entry + tick
            if target >= entry:
                target = entry - tick

    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr_ratio = reward / max(risk, 1e-9)
    stop_distance_pct = risk / max(entry, 1e-9) * 100.0
    target_distance_pct = reward / max(entry, 1e-9) * 100.0
    atr_multiple_to_stop = risk / max(atr, 1e-9)
    atr_multiple_to_target = reward / max(atr, 1e-9)

    # Invalidation strength: how far stop is from entry in ATR multiples, clamped to [0, 1]
    invalidation_strength = _clamp(atr_multiple_to_stop / 3.0, 0.0, 1.0)

    return EntryGeometry(
        entry=entry,
        stop=stop,
        target=target,
        risk=risk,
        reward=reward,
        rr_ratio=rr_ratio,
        stop_distance_pct=stop_distance_pct,
        target_distance_pct=target_distance_pct,
        atr_multiple_to_stop=atr_multiple_to_stop,
        atr_multiple_to_target=atr_multiple_to_target,
        stop_anchor=stop_anchor,
        target_anchor=target_anchor,
        invalidation_strength=invalidation_strength,
    )


def geometry_quality_score(
    geometry: EntryGeometry,
    regime: MarketRegime,
    confluence: dict[str, float],
    regime_confidence: float,
) -> tuple[float, QualityLabel]:
    """Compute a quality score in [min_score, max_score] from setup geometry.

    All scoring parameters come from config.strategy.geometry_quality.
    Falls back to defaults module if config is unavailable.
    """
    import futures_analyzer.defaults as D
    try:
        from futures_analyzer.config import load_app_config
        gq = load_app_config().strategy.geometry_quality
        rr_weight              = gq.rr_weight
        rr_cap                 = gq.rr_cap
        stop_atr_ideal_min     = gq.stop_atr_ideal_min
        stop_atr_ideal_max     = gq.stop_atr_ideal_max
        stop_atr_bonus         = gq.stop_atr_bonus
        stop_atr_penalty       = gq.stop_atr_penalty
        anchor_swing_bonus     = gq.anchor_swing_bonus
        anchor_vwap_bonus      = gq.anchor_vwap_bonus
        anchor_vp_bonus        = gq.anchor_vp_bonus
        anchor_atr_penalty     = gq.anchor_atr_penalty
        entry_confluence_bonus = gq.entry_confluence_bonus
        target_confluence_bonus = gq.target_confluence_bonus
        base_score             = gq.base_score
        min_score              = gq.min_score
        max_score              = gq.max_score
    except Exception:
        rr_weight              = D.GQ_RR_WEIGHT
        rr_cap                 = D.GQ_RR_CAP
        stop_atr_ideal_min     = D.GQ_STOP_ATR_IDEAL_MIN
        stop_atr_ideal_max     = D.GQ_STOP_ATR_IDEAL_MAX
        stop_atr_bonus         = D.GQ_STOP_ATR_BONUS
        stop_atr_penalty       = D.GQ_STOP_ATR_PENALTY
        anchor_swing_bonus     = D.GQ_ANCHOR_SWING_BONUS
        anchor_vwap_bonus      = D.GQ_ANCHOR_VWAP_BONUS
        anchor_vp_bonus        = D.GQ_ANCHOR_VP_BONUS
        anchor_atr_penalty     = D.GQ_ANCHOR_ATR_PENALTY
        entry_confluence_bonus = D.GQ_ENTRY_CONFLUENCE_BONUS
        target_confluence_bonus = D.GQ_TARGET_CONFLUENCE_BONUS
        base_score             = D.GQ_BASE_SCORE
        min_score              = D.GQ_MIN_SCORE
        max_score              = D.GQ_MAX_SCORE

    # 1. R:R contribution (primary driver, 0–30 pts)
    rr_pts = min(geometry.rr_ratio, rr_cap) / rr_cap * rr_weight

    # 2. Stop distance quality
    atr_stop = geometry.atr_multiple_to_stop
    if stop_atr_ideal_min <= atr_stop <= stop_atr_ideal_max:
        stop_pts = stop_atr_bonus
    else:
        stop_pts = -stop_atr_penalty

    # 3. Anchor quality bonus/penalty
    def _anchor_pts(anchor: str) -> float:
        if anchor.startswith("swing"):
            return anchor_swing_bonus
        elif anchor.startswith("vwap"):
            return anchor_vwap_bonus
        elif anchor in ("val", "vah"):
            return anchor_vp_bonus
        elif anchor == "atr_fallback":
            return -anchor_atr_penalty
        else:  # rr_enforced or unknown
            return 0.0

    stop_anchor_pts   = _anchor_pts(geometry.stop_anchor)
    target_anchor_pts = _anchor_pts(geometry.target_anchor)

    # 4. Confluence bonuses (capped at 3 factors each)
    entry_c  = min(confluence.get("entry_confluence", 0.0), 3.0)
    target_c = min(confluence.get("target_confluence", 0.0), 3.0)
    entry_confluence_pts  = entry_c  * entry_confluence_bonus
    target_confluence_pts = target_c * target_confluence_bonus

    # 5. Total score
    score = (
        base_score
        + rr_pts
        + stop_pts
        + stop_anchor_pts
        + target_anchor_pts
        + entry_confluence_pts
        + target_confluence_pts
    )
    score = _clamp(score, min_score, max_score)

    return (score, _quality_label(score))


def normalize_signals(
    bundle: IndicatorBundle,
    market_meta: MarketMeta,
    side: str,
    regime: MarketRegime,
) -> NormalizedSignals:
    """Convert raw indicator values to normalized [0, 1] signal strengths per side.

    All scaling factors come from config.strategy.signal_transforms — no magic numbers.
    All outputs are clamped to [0.0, 1.0]. No NaN or infinite values are produced.
    """
    from futures_analyzer.config import load_app_config
    t = load_app_config().strategy.signal_transforms

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


def compute_graded_evidence(
    signals: NormalizedSignals,
    regime: MarketRegime,
    side: str,
    weights: dict[str, float],
) -> EvidenceVector:
    """Compute a weighted evidence strength score from normalized signals.

    Replaces the binary 7-check system with a continuous weighted sum.
    """
    import dataclasses

    # Build signal dict from NormalizedSignals fields
    signal_dict: dict[str, float] = {
        f.name: getattr(signals, f.name)
        for f in dataclasses.fields(signals)
    }

    # Weighted sum: dot product of signal values and regime-specific weights
    weighted_sum = sum(weights.get(name, 0.0) * value for name, value in signal_dict.items())

    # Count signals above 0.5 threshold
    signal_count_above_threshold = sum(1 for v in signal_dict.values() if v > 0.5)

    # Top-3 strongest and bottom-3 weakest signals
    sorted_desc = sorted(signal_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_asc = sorted(signal_dict.items(), key=lambda kv: kv[1])
    strongest_signals = sorted_desc[:3]
    weakest_signals = sorted_asc[:3]

    # Regime gate check — thresholds from config.strategy.regime_gate_thresholds
    try:
        from futures_analyzer.config import load_app_config
        config = load_app_config().strategy
        thresholds = getattr(config, "regime_gate_thresholds", {}) or {}
    except Exception:
        thresholds = {}

    def _thr(key: str, default: float) -> float:
        return thresholds.get(key, default)

    if regime in (MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND):
        regime_gate_passed = signals.higher_trend > _thr("trend", 0.4)
    elif regime == MarketRegime.BREAKOUT:
        regime_gate_passed = signals.volume_surge > _thr("breakout", 0.3)
    elif regime == MarketRegime.EXHAUSTION:
        regime_gate_passed = signals.rsi_alignment > _thr("exhaustion", 0.3)
    elif regime == MarketRegime.RANGE:
        regime_gate_passed = signals.structure_position > _thr("range", 0.3)
    elif regime == MarketRegime.VOLATILE_CHOP:
        regime_gate_passed = signals.higher_trend > _thr("volatile_chop", 0.5)
    elif regime == MarketRegime.TRANSITION:
        regime_gate_passed = signals.higher_trend > _thr("transition", 0.35)
    else:
        regime_gate_passed = True

    return EvidenceVector(
        weighted_sum=weighted_sum,
        signal_count_above_threshold=signal_count_above_threshold,
        strongest_signals=strongest_signals,
        weakest_signals=weakest_signals,
        regime_gate_passed=regime_gate_passed,
    )


def logistic_confidence(
    evidence_weighted_sum: float,
    regime: MarketRegime,
    side: str,
    config,  # StrategyConfig from load_app_config().strategy
) -> float:
    """Map evidence weighted sum to confidence via logistic sigmoid.

    Uses regime/side-specific steepness and midpoint from config.logistic_params.
    Falls back to the "default" entry in config.logistic_params, then to
    defaults.LOGISTIC_DEFAULT_STEEPNESS / LOGISTIC_DEFAULT_MIDPOINT.
    No hardcoded per-regime values — all params come from config.json.
    """
    import futures_analyzer.defaults as D

    key = f"{regime.value}_{side}"
    logistic_params = getattr(config, "logistic_params", {}) or {}
    params = (
        logistic_params.get(key)
        or logistic_params.get(regime.value)
        or logistic_params.get("default")
    )
    if params is None:
        steepness = D.LOGISTIC_DEFAULT_STEEPNESS
        midpoint  = D.LOGISTIC_DEFAULT_MIDPOINT
    else:
        steepness = params.steepness
        midpoint  = params.midpoint

    x = steepness * (evidence_weighted_sum - midpoint)
    return _clamp(sigmoid(x), 0.0, 1.0)


def logistic_confidence_from_config(
    evidence_weighted_sum: float,
    regime: MarketRegime,
    side: str,
) -> float:
    """Convenience wrapper that loads config automatically."""
    from futures_analyzer.config import load_app_config
    config = load_app_config().strategy
    return logistic_confidence(evidence_weighted_sum, regime, side, config)


def _compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Compute ATR as average of last `period` true ranges. Returns 0.0 when insufficient data."""
    if len(candles) < 2:
        return 0.0
    trs: list[float] = []
    prev_close = candles[0].close
    for c in candles[1:]:
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        prev_close = c.close
    if not trs:
        return 0.0
    tail = trs[-period:] if len(trs) >= period else trs
    return sum(tail) / len(tail)


def _ema_value(closes: list[float], period: int) -> float:
    """Compute EMA of a close series. Returns last close when insufficient data."""
    if not closes:
        return 0.0
    if len(closes) < period:
        return closes[-1]
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema


def compute_all_indicators(
    entry: list[Candle],
    trigger: list[Candle],
    context: list[Candle],
    higher: list[Candle],
    market_meta: MarketMeta,
) -> IndicatorBundle:
    """Compute all indicators in a single coordinated pass over the four timeframe candle lists.

    Returns a fully-populated IndicatorBundle. Safe neutral defaults are used for any
    indicator when candle data is insufficient; a warning string is appended to the
    returned bundle's `warnings` list.
    """
    import math
    warnings_list: list[str] = []

    # ── Per-timeframe ATR ────────────────────────────────────────────────────
    entry_atr = _compute_atr(entry)
    trigger_atr = _compute_atr(trigger)
    context_atr = _compute_atr(context)
    higher_atr = _compute_atr(higher)

    # ── Trend: EMA(fast) vs EMA(slow) slope, normalized to [-1, 1] via tanh ──
    from futures_analyzer.config import load_app_config as _lac
    _ip = _lac().strategy.indicator_params
    _st = _lac().strategy.signal_transforms
    _ema_fast = _ip.ema_fast_period
    _ema_slow = _ip.ema_slow_period
    _trend_tanh_scale = _st.trend_tanh_scale
    _momentum_tanh_scale = _st.momentum_tanh_scale
    _vol_window = _ip.volume_window
    _roc_period = _ip.roc_period

    def _trend_ema(candles: list[Candle], label: str) -> float:
        if len(candles) < _ema_slow:
            warnings_list.append(
                f"Insufficient candles for {label} trend (need {_ema_slow}, got {len(candles)}); defaulting to 0.0"
            )
            return 0.0
        closes = [c.close for c in candles]
        ema_f = _ema_value(closes, _ema_fast)
        ema_s = _ema_value(closes, _ema_slow)
        if ema_s == 0.0:
            return 0.0
        raw = (ema_f - ema_s) / ema_s
        return math.tanh(raw * _trend_tanh_scale)

    higher_trend = _trend_ema(higher, "higher")
    context_trend = _trend_ema(context, "context")

    # ── Momentum: rate of change, normalized via tanh ────────────────────────
    def _roc_momentum(candles: list[Candle], label: str) -> float:
        need = _roc_period + 1
        if len(candles) < need:
            warnings_list.append(
                f"Insufficient candles for {label} momentum (need {need}, got {len(candles)}); defaulting to 0.0"
            )
            return 0.0
        first = candles[-need].close
        last = candles[-1].close
        if first <= 0:
            return 0.0
        raw = last / first - 1.0
        return _clamp(math.tanh(raw * _momentum_tanh_scale), -1.0, 1.0)

    trigger_momentum = _roc_momentum(trigger, "trigger")
    entry_momentum = _roc_momentum(entry, "entry")

    # ── Volume surge: last volume / mean(last N volumes) ─────────────────────
    def _vol_surge(candles: list[Candle]) -> float:
        if len(candles) < 2:
            return 1.0
        window = candles[-_vol_window:] if len(candles) >= _vol_window else candles
        base = sum(c.volume for c in window) / len(window)
        if base <= 0:
            return 1.0
        return candles[-1].volume / base

    trigger_volume_surge = _vol_surge(trigger)
    entry_volume_surge = _vol_surge(entry)

    # ── Cumulative delta ──────────────────────────────────────────────────────
    if trigger:
        cumulative_delta = compute_cumulative_delta(trigger)
    else:
        cumulative_delta = 0.0
        warnings_list.append("No trigger candles for cumulative_delta; defaulting to 0.0")

    # ── Oscillators on entry TF ───────────────────────────────────────────────
    _atr_period = _ip.atr_period
    if len(entry) >= _atr_period + 1:
        rsi_14 = rsi(entry, period=_atr_period)
    else:
        rsi_14 = 50.0
        warnings_list.append(
            f"Insufficient entry candles for RSI (need {_atr_period + 1}, got {len(entry)}); defaulting to 50.0"
        )

    macd_result = macd(entry)
    macd_histogram = macd_result.histogram if macd_result.histogram is not None else 0.0

    stoch_result = stochastic(entry)
    stoch_k = stoch_result.value

    bb_result = bollinger_bands(entry)
    bb_position = bb_result.get("position", 0.5)
    bb_bandwidth_pct = bb_result.get("bandwidth", 0.0)

    # ── Swing highs/lows from trigger candles ─────────────────────────────────
    pivot_n = _ip.pivot_n

    if len(trigger) >= (2 * pivot_n + 1):
        highs = [c.high for c in trigger]
        lows = [c.low for c in trigger]
        raw_pivots = _swing_pivots(highs, n=pivot_n)
        swing_highs_raw = [
            val for idx, val in raw_pivots
            if val == max(highs[max(0, idx - pivot_n): idx + pivot_n + 1])
        ]
        raw_pivots_low = _swing_pivots(lows, n=pivot_n)
        swing_lows_raw = [
            val for idx, val in raw_pivots_low
            if val == min(lows[max(0, idx - pivot_n): idx + pivot_n + 1])
        ]
        swing_highs = sorted(swing_highs_raw)
        swing_lows = sorted(swing_lows_raw)
    else:
        swing_highs = []
        swing_lows = []
        warnings_list.append(f"Insufficient trigger candles for swing pivots (need {2 * pivot_n + 1}, got {len(trigger)}); defaulting to empty lists")

    # ── Market structure ──────────────────────────────────────────────────────
    if len(trigger) >= 7:
        market_structure = compute_market_structure(trigger)
    else:
        market_structure = "mixed"
        warnings_list.append(f"Insufficient trigger candles for market_structure (need 7, got {len(trigger)}); defaulting to 'mixed'")

    # ── Liquidity sweeps ──────────────────────────────────────────────────────
    if len(trigger) >= 7:
        liquidity_sweeps = detect_liquidity_sweeps(trigger)
    else:
        liquidity_sweeps = []
        warnings_list.append(f"Insufficient trigger candles for liquidity_sweeps (need 7, got {len(trigger)}); defaulting to []")

    # ── VWAP bands from trigger candles ──────────────────────────────────────
    if trigger:
        vwap_bands = compute_vwap_bands(trigger)
        vwap_val = vwap_bands["vwap"]
        vwap_upper_1sd = vwap_bands["upper_1sd"]
        vwap_lower_1sd = vwap_bands["lower_1sd"]
        vwap_upper_2sd = vwap_bands["upper_2sd"]
        vwap_lower_2sd = vwap_bands["lower_2sd"]
    else:
        last_close = 0.0
        vwap_val = last_close
        vwap_upper_1sd = last_close
        vwap_lower_1sd = last_close
        vwap_upper_2sd = last_close
        vwap_lower_2sd = last_close
        warnings_list.append("No trigger candles for VWAP bands; defaulting to 0.0")

    # ── Volume profile from trigger candles ───────────────────────────────────
    if trigger:
        vp = volume_profile(trigger)
        poc = vp["poc"]
        vah = vp["vah"]
        val = vp["val"]
    else:
        last_close = trigger[-1].close if trigger else 0.0
        poc = last_close
        vah = last_close
        val = last_close
        warnings_list.append("No trigger candles for volume_profile; defaulting to last close")

    # ── RSI divergence on entry TF ────────────────────────────────────────────
    min_div_bars = 14 + 30 + 3  # period + lookback + pivot_n
    if len(entry) >= min_div_bars:
        div_detected, div_type, div_strength = rsi_divergence(entry)
    else:
        div_detected, div_type, div_strength = False, "none", 0.0
        warnings_list.append(f"Insufficient entry candles for RSI divergence (need {min_div_bars}, got {len(entry)}); defaulting to none")

    rsi_divergence_type = div_type
    rsi_divergence_strength = div_strength

    # ── OI / funding from market_meta ─────────────────────────────────────────
    funding_rate = getattr(market_meta, "funding_rate", None)
    oi_change_pct = getattr(market_meta, "open_interest_change_pct", None)
    funding_momentum_val = 0.0  # placeholder — no historical funding data available

    # ── Order book ────────────────────────────────────────────────────────────
    order_book_imbalance = getattr(market_meta, "order_book_imbalance", 0.0) or 0.0
    bid_ask_spread_pct = 0.0  # placeholder

    return IndicatorBundle(
        entry_atr=entry_atr,
        trigger_atr=trigger_atr,
        context_atr=context_atr,
        higher_atr=higher_atr,
        higher_trend=higher_trend,
        context_trend=context_trend,
        trigger_momentum=trigger_momentum,
        entry_momentum=entry_momentum,
        trigger_volume_surge=trigger_volume_surge,
        entry_volume_surge=entry_volume_surge,
        cumulative_delta=cumulative_delta,
        rsi_14=rsi_14,
        macd_histogram=macd_histogram,
        stoch_k=stoch_k,
        bb_position=bb_position,
        bb_bandwidth_pct=bb_bandwidth_pct,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        market_structure=market_structure,
        liquidity_sweeps=liquidity_sweeps,
        vwap=vwap_val,
        vwap_upper_1sd=vwap_upper_1sd,
        vwap_lower_1sd=vwap_lower_1sd,
        vwap_upper_2sd=vwap_upper_2sd,
        vwap_lower_2sd=vwap_lower_2sd,
        poc=poc,
        vah=vah,
        val=val,
        rsi_divergence_type=rsi_divergence_type,
        rsi_divergence_strength=rsi_divergence_strength,
        funding_rate=funding_rate,
        oi_change_pct=oi_change_pct,
        funding_momentum=funding_momentum_val,
        order_book_imbalance=order_book_imbalance,
        bid_ask_spread_pct=bid_ask_spread_pct,
        warnings=warnings_list,
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
    stop_anchor: str = "atr_fallback"
    target_anchor: str = "atr_cap"
    signal_strengths: dict[str, float] = field(default_factory=dict)
    evidence_weighted_sum: float = 0.0
    logistic_input: float = 0.0


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


def _structure(candles: list[Candle], window: int = 60) -> tuple[float, float]:
    if not candles:
        raise ValueError("No candles available for structure detection")
    recent = candles[-window:] if len(candles) >= window else candles
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


def _funding_momentum(history: list[float], window: int = 4) -> float:
    """Compute the slope of the last `window` funding rate values.

    Returns a normalised value in roughly [-1, 1]:
    positive = funding rising (crowded longs), negative = falling (crowded shorts).
    """
    from futures_analyzer.config import load_app_config
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
    from futures_analyzer.config import load_app_config
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


def _classify_regime(
    context_candles: list[Candle],
    higher_candles: list[Candle],
    trigger_atr: float,
    px: float,
) -> tuple[MarketRegime, float]:
    return _regime_classify(context_candles, higher_candles, trigger_atr, px)


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

    if stop_distance_pct >= strategy.leverage_stop_wide:
        base = floor
    elif stop_distance_pct >= strategy.leverage_stop_mid:
        base = floor + 1
    elif stop_distance_pct >= strategy.leverage_stop_narrow:
        base = floor + 2
    elif stop_distance_pct >= strategy.leverage_stop_tight:
        base = floor + 3
    else:
        base = cap

    if confidence >= strategy.leverage_confidence_high:
        base += 1
    elif confidence < strategy.leverage_confidence_low:
        base -= 1

    if risk_reward_ratio >= strategy.leverage_rr_good:
        base += 1
    elif risk_reward_ratio < strategy.leverage_rr_poor:
        base -= 1

    if regime == MarketRegime.VOLATILE_CHOP:
        cap = min(cap, strategy.leverage_volatile_chop_cap)
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
    div_detected, div_type, _div_strength = rsi_divergence(entry_candles, period=14, lookback=30)
    
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
        "funding_momentum": ("Funding Momentum", "Funding rate trend is supporting this side — positioning is shifting in this direction."),
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

    from futures_analyzer.config import load_app_config
    s = load_app_config().strategy

    if count == 1:
        setup.confidence = _clamp(setup.confidence * s.reversal_1_confidence_mult, 0.0, 1.0)
    elif count == 2:
        setup.confidence = _clamp(setup.confidence * s.reversal_2_confidence_mult, 0.0, 1.0)
        setup.quality_score = _clamp(setup.quality_score - s.reversal_2_quality_deduction, 10.0, 95.0)
    else:
        setup.confidence = _clamp(setup.confidence * s.reversal_3_confidence_mult, 0.0, 1.0)
        setup.quality_score = _clamp(setup.quality_score - s.reversal_3_quality_deduction, 10.0, 95.0)
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

    from futures_analyzer.config import load_app_config
    s = load_app_config().strategy

    if entry_c >= 3:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_entry_3_boost, 10.0, 95.0)
    elif entry_c == 2:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_entry_2_boost, 10.0, 95.0)
    elif entry_c == 1:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_entry_1_boost, 10.0, 95.0)

    if target_c >= 2:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_target_2_boost, 10.0, 95.0)
    elif target_c == 1:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_target_1_boost, 10.0, 95.0)

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
        # Resolve config once at construction — avoids repeated imports and
        # lru_cache lookups inside hot paths (_mode_params / _trade_filter_params).
        from futures_analyzer.config import load_app_config
        self._config = load_app_config()

    def _mode_params(self) -> dict[str, float]:
        tuning = self._config.style_tuning(self.style)
        mode_settings = self._config.market_mode_settings(self.market_mode)
        return {
            "fallback_risk_reward": tuning.fallback_risk_reward,
            "target_cap_atr_mult": (
                mode_settings.target_cap_atr_mult
                if mode_settings.target_cap_atr_mult is not None
                else tuning.target_cap_atr_mult
            ),
            "ambition_penalty_start_atr": tuning.ambition_penalty_start_atr,
            "ambition_penalty_slope": tuning.ambition_penalty_slope,
            "atr_buffer_factor": tuning.atr_buffer_factor,
        }

    def _trade_filter_params(self) -> dict[str, float]:
        tuning = self._config.style_tuning(self.style)
        mode_settings = self._config.market_mode_settings(self.market_mode)

        # Priority: market_mode < style < filter_overrides (runtime)
        # market_mode contributes only fields it explicitly sets (non-None).
        # style (tuning) always has values for every field and overrides market_mode.
        # filter_overrides are runtime caller overrides and win above all.
        mode_params: dict[str, float] = {
            k: v for k, v in {
                "min_confidence": mode_settings.min_confidence,
                "max_stop_distance_pct": mode_settings.max_stop_distance_pct,
                "min_evidence_agreement": mode_settings.min_evidence_agreement,
                "min_evidence_edge": mode_settings.min_evidence_edge,
            }.items() if v is not None
        }

        style_params: dict[str, float] = {
            "min_confidence": tuning.min_confidence,
            "max_confidence": tuning.max_confidence,
            "min_quality": tuning.min_quality,
            "min_rr_ratio": tuning.min_rr_ratio,
            "max_stop_distance_pct": tuning.max_stop_distance_pct,
            "min_evidence_agreement": tuning.min_evidence_agreement,
            "min_evidence_edge": tuning.min_evidence_edge,
        }

        # style_params last → style always overrides market_mode
        params = {**mode_params, **style_params, **self._filter_overrides}
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
        if confidence > params.get("max_confidence", 1.0):
            reasons.append(f"confidence {confidence:.2f} is above {params['max_confidence']:.2f}")
        if quality_score < params["min_quality"]:
            reasons.append(f"quality {quality_score:.1f} is below {params['min_quality']:.1f}")
        if risk_reward_ratio < params["min_rr_ratio"]:
            reasons.append(f"R:R {risk_reward_ratio:.2f} is below {params['min_rr_ratio']:.2f}")
        if stop_distance_pct > params["max_stop_distance_pct"]:
            reasons.append(
                f"stop distance {stop_distance_pct:.2f}% is above {params['max_stop_distance_pct']:.2f}%"
            )
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
            "regime": _regime_alignment(regime, side) > 0.3,
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

        # EVIDENCE FILTER (regime-aware)
        if regime == MarketRegime.RANGE:
            if evidence.agreement < 2:
                setup.tradable_reasons.append(
                    f"evidence agreement {evidence.agreement}/{evidence.total} is below 2 (range)"
                )
        else:
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
            
        from futures_analyzer.config import load_app_config
        config = load_app_config()
        if hasattr(config.strategy, "allowed_regimes") and regime.value not in config.strategy.allowed_regimes:
            setup.tradable_reasons.append(f"regime {regime.value} is globally disabled via StrategyConfig")
        if hasattr(config.strategy, "enable_longs") and setup.side == "long" and not config.strategy.enable_longs:
            setup.tradable_reasons.append("long setups are globally disabled via StrategyConfig")
        if hasattr(config.strategy, "enable_shorts") and setup.side == "short" and not config.strategy.enable_shorts:
            setup.tradable_reasons.append("short setups are globally disabled via StrategyConfig")

        if regime == MarketRegime.VOLATILE_CHOP:
            setup.quality_score = min(setup.quality_score, 50.0)
            setup.quality_label = _quality_label(setup.quality_score)
            
            chop_min = max(params["min_evidence_agreement"], 3)
            if evidence.agreement < chop_min:
                setup.tradable_reasons.append(
                    f"volatile chop requires {chop_min}+ evidence signals, got {evidence.agreement}"
                )

        if regime == MarketRegime.RANGE:
            setup.quality_score = min(setup.quality_score, 52.0)
            setup.quality_label = _quality_label(setup.quality_score)

            # Range needs at least 2 signals
            range_min = 2
            if evidence.agreement < range_min:
                setup.tradable_reasons.append(
                    f"range regime requires {range_min}+ evidence signals, got {evidence.agreement}"
                )

            # RANGE: enforce only relaxed threshold (no overrides here)
            if setup.quality_score < 45:
                setup.tradable_reasons.append(
                    f"quality {setup.quality_score:.1f} below threshold (range)"
                )

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
        # Scale structure window to timeframe: scalper (5m) uses 30 bars (~2.5h),
        # intraday (15m-1h) uses 50 bars, higher TFs use 60 bars.
        _tf = timeframe_plan.trigger_timeframe
        _tf_mins = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}.get(_tf, 15)
        _structure_window = 30 if _tf_mins <= 5 else (40 if _tf_mins <= 30 else 60)
        support, resistance = _structure(trigger_candles, window=_structure_window)
        px = market.mark_price
        higher_trend = _trend_strength(higher_candles, fast_window=4, slow_window=16)
        context_trend = _trend_strength(context_candles)
        trigger_momentum = _momentum(trigger_candles)
        entry_momentum = _momentum(entry_candles, window=12)

        # Volume profile proximity (used for stop reference + quality boost)
        vp = volume_profile_strength(trigger_candles, px)
        regime_result = classify_regime_consensus(
            context_candles=context_candles,
            higher_candles=higher_candles,
            trigger_candles=trigger_candles,
            trigger_atr=atr,
            px=px,
        )
        regime = regime_result.regime
        regime_confidence = regime_result.confidence
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
            market.funding_rate, market.open_interest_change_pct, market.funding_rate_history
        )
        # Funding momentum as a separate directional signal
        funding_mom = _funding_momentum(market.funding_rate_history)
        # Rising funding (positive momentum) → tailwind for shorts, headwind for longs
        long_funding_momentum = max(-funding_mom, 0.0)
        short_funding_momentum = max(funding_mom, 0.0)
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
                "funding_momentum": long_funding_momentum,
                "regime_alignment": _regime_alignment(regime, "long"),
            },
            divergence_penalty=long_div_penalty,
            confirmation_penalty=long_confirmation_penalty,
            regime=regime,
            regime_penalty=_regime_penalty(regime, "long"),
            tick=market.tick_size,
            volume_profile=vp,
            alignment_score=long_alignment,
            entry_candles=entry_candles,
            trigger_candles=trigger_candles,
            context_candles=context_candles,
            higher_candles=higher_candles,
            market_meta=market,
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
                "funding_momentum": short_funding_momentum,
                "regime_alignment": _regime_alignment(regime, "short"),
            },
            divergence_penalty=short_div_penalty,
            confirmation_penalty=short_confirmation_penalty,
            regime=regime,
            regime_penalty=_regime_penalty(regime, "short"),
            tick=market.tick_size,
            volume_profile=vp,
            alignment_score=short_alignment,
            entry_candles=entry_candles,
            trigger_candles=trigger_candles,
            context_candles=context_candles,
            higher_candles=higher_candles,
            market_meta=market,
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

        # Compute signal TTL for decay tracking
        from futures_analyzer.analysis.decay import compute_ttl
        from datetime import UTC, datetime as _dt
        _now = _dt.now(UTC)
        _ttl = compute_ttl(timeframe_plan.trigger_timeframe)
        _ttl_seconds = _ttl.total_seconds()
        _valid_until = _now + _ttl

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
                valid_until=_valid_until,
                ttl_seconds=_ttl_seconds,
                stop_anchor=primary.stop_anchor,
                target_anchor=primary.target_anchor,
                regime_state=regime.value,
                signal_strengths=primary.signal_strengths,
                evidence_weighted_sum=primary.evidence_weighted_sum,
                logistic_input=primary.logistic_input,
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
                valid_until=_valid_until,
                ttl_seconds=_ttl_seconds,
                stop_anchor=secondary.stop_anchor,
                target_anchor=secondary.target_anchor,
                regime_state=regime.value,
                signal_strengths=secondary.signal_strengths,
                evidence_weighted_sum=secondary.evidence_weighted_sum,
                logistic_input=secondary.logistic_input,
            ),
            timeframe_plan=timeframe_plan,
            market_snapshot_meta=market.model_copy(update={"symbol": symbol}),
            market_regime=regime_result.regime,
            regime_confidence=regime_result.confidence,
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
        from futures_analyzer.config import load_app_config
        s = load_app_config().strategy

        # 1. RSI-based confidence adjustment — directionally aware
        rsi_val = enhanced_metrics.rsi_14
        if side == "long":
            if 40 < rsi_val < 65:  # healthy momentum zone for longs
                setup.confidence *= s.rsi_neutral_confidence_mult
            elif rsi_val >= 75:  # overbought — bad long entry
                setup.confidence *= s.rsi_extreme_confidence_mult
            elif rsi_val <= 25:  # oversold — good long entry
                setup.confidence *= s.rsi_mild_extreme_confidence_mult
            elif rsi_val >= 65:  # mildly overbought
                setup.confidence *= s.rsi_mild_extreme_confidence_mult
        else:  # short
            if 35 < rsi_val < 60:  # healthy momentum zone for shorts
                setup.confidence *= s.rsi_neutral_confidence_mult
            elif rsi_val <= 25:  # oversold — bad short entry
                setup.confidence *= s.rsi_extreme_confidence_mult
            elif rsi_val >= 75:  # overbought — good short entry
                setup.confidence *= s.rsi_mild_extreme_confidence_mult
            elif rsi_val <= 35:  # mildly oversold
                setup.confidence *= s.rsi_mild_extreme_confidence_mult

        # 2. MACD momentum confirmation
        if enhanced_metrics.macd_histogram > 0 and side == "long":
            setup.confidence += s.macd_confirm_confidence_delta
        elif enhanced_metrics.macd_histogram < 0 and side == "short":
            setup.confidence += s.macd_confirm_confidence_delta

        # 3. Bollinger Band position for entry timing
        bb_pos = enhanced_metrics.bollinger_position
        if s.bb_mid_low < bb_pos < s.bb_mid_high:
            setup.quality_score += s.bb_mid_quality_boost
        elif bb_pos < s.bb_extreme_low or bb_pos > s.bb_extreme_high:
            setup.quality_score -= s.bb_extreme_quality_penalty

        # 4. Order book imbalance for entry strength
        if abs(enhanced_metrics.order_book_imbalance) > s.ob_imbalance_threshold:
            setup.confidence += s.ob_imbalance_confidence_delta

        # 5. Volatility-based stop adjustment
        vol_rank = enhanced_metrics.volatility_rank
        if vol_rank > s.vol_rank_high_threshold:
            setup.stop_distance_pct *= s.vol_rank_high_stop_mult
        elif vol_rank < s.vol_rank_low_threshold:
            setup.stop_distance_pct *= s.vol_rank_low_stop_mult

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
        # New signal pipeline inputs
        entry_candles: list[Candle] | None = None,
        trigger_candles: list[Candle] | None = None,
        context_candles: list[Candle] | None = None,
        higher_candles: list[Candle] | None = None,
        market_meta: MarketMeta | None = None,
    ) -> _SideMetrics:
        from futures_analyzer.config import load_app_config
        strategy = load_app_config().strategy
        weights, penalty_multiplier, confidence_ceiling = _regime_weight_profile(regime, side)
        mode_params = self._mode_params()
        atr_buffer = max(atr * mode_params["atr_buffer_factor"], px * strategy.min_risk_px_factor)

        # ── New signal pipeline ───────────────────────────────────────────────
        # When candle data and market_meta are provided, use the new pipeline to
        # compute confidence via normalize_signals → compute_graded_evidence →
        # logistic_confidence_from_config. Otherwise fall back to the legacy path.
        _use_new_pipeline = (
            entry_candles is not None
            and trigger_candles is not None
            and context_candles is not None
            and higher_candles is not None
            and market_meta is not None
        )
        _bundle: IndicatorBundle | None = None
        _signals: NormalizedSignals | None = None
        _evidence: EvidenceVector | None = None
        _new_confidence: float | None = None
        _geometry: EntryGeometry | None = None
        _logistic_input = 0.0

        if _use_new_pipeline:
            _bundle = compute_all_indicators(
                entry_candles,  # type: ignore[arg-type]
                trigger_candles,  # type: ignore[arg-type]
                context_candles,  # type: ignore[arg-type]
                higher_candles,  # type: ignore[arg-type]
                market_meta,  # type: ignore[arg-type]
            )
            _signals = normalize_signals(_bundle, market_meta, side, regime)  # type: ignore[arg-type]
            # Use regime-specific weights from config when available (Phase 5 will
            # populate these with NormalizedSignals-compatible keys summing to 1.0).
            # Fall back to built-in defaults that match NormalizedSignals field names.
            _cfg_weights, _penalty_multiplier, _confidence_ceiling = _regime_weight_profile(regime, side)
            _NORMALIZED_SIGNAL_FIELDS = {
                "higher_trend", "context_trend", "trigger_momentum", "entry_momentum",
                "volume_surge", "buy_pressure", "oi_funding_bias", "funding_momentum",
                "structure_position", "rsi_alignment", "macd_alignment", "bb_alignment",
                "vwap_alignment", "market_structure_align", "cumulative_delta_align",
                "volume_poc_proximity",
            }
            # Check if config weights use the new field names (sum ≈ 1.0 and keys match)
            _cfg_new_keys = {k: v for k, v in _cfg_weights.items() if k in _NORMALIZED_SIGNAL_FIELDS}
            _cfg_new_sum = sum(_cfg_new_keys.values())
            if abs(_cfg_new_sum - 1.0) < 0.1 and len(_cfg_new_keys) >= 12:
                # Config has been updated with new field names — use them
                _signal_weights = _cfg_new_keys
            else:
                # Fall back to default_signal_weights from config (no hardcoded dict)
                _signal_weights = dict(strategy.default_signal_weights) or {
                    k: 1.0 / len(_NORMALIZED_SIGNAL_FIELDS)
                    for k in _NORMALIZED_SIGNAL_FIELDS
                }
            _evidence = compute_graded_evidence(_signals, regime, side, _signal_weights)
            _new_confidence = logistic_confidence_from_config(_evidence.weighted_sum, regime, side)
            _new_confidence = min(_new_confidence, _confidence_ceiling)
            _strategy_logistic = getattr(strategy, "logistic_params", {}) or {}
            _logistic_params = (
                _strategy_logistic.get(f"{regime.value}_{side}")
                or _strategy_logistic.get(regime.value)
                or _strategy_logistic.get("default")
            )
            if isinstance(_logistic_params, dict):
                _steepness = _logistic_params.get("steepness", 6.0)
                _midpoint = _logistic_params.get("midpoint", 0.5)
            elif _logistic_params is not None:
                _steepness = getattr(_logistic_params, "steepness", 6.0)
                _midpoint = getattr(_logistic_params, "midpoint", 0.5)
            else:
                _steepness = 6.0
                _midpoint = 0.5
            _logistic_input = _steepness * (_evidence.weighted_sum - _midpoint)

            # Extract raw signal values from bundle for _confirmation_penalty and
            # _timeframe_alignment_score (these helpers expect the original [-1,1] scale).
            _higher_trend_raw = _bundle.higher_trend
            _context_trend_raw = _bundle.context_trend
            _trigger_momentum_raw = _bundle.trigger_momentum
            _entry_momentum_raw = _bundle.entry_momentum
            # Use cumulative_delta as proxy for pressure signals
            _trigger_pressure_raw = _bundle.cumulative_delta
            _entry_pressure_raw = _bundle.cumulative_delta
            _trigger_volume_surge_raw = _bundle.trigger_volume_surge
            _entry_volume_surge_raw = _bundle.entry_volume_surge

            # Recompute confirmation_penalty and alignment_score from bundle values
            confirmation_penalty = _confirmation_penalty(
                side,
                higher_trend=_higher_trend_raw,
                context_trend=_context_trend_raw,
                trigger_momentum=_trigger_momentum_raw,
                entry_momentum=_entry_momentum_raw,
                trigger_pressure=_trigger_pressure_raw,
                entry_pressure=_entry_pressure_raw,
                trigger_volume_surge=_trigger_volume_surge_raw,
                entry_volume_surge=_entry_volume_surge_raw,
            )
            alignment_score = _timeframe_alignment_score(
                side,
                higher_trend=_higher_trend_raw,
                context_trend=_context_trend_raw,
                trigger_momentum=_trigger_momentum_raw,
                entry_momentum=_entry_momentum_raw,
            )

        if _use_new_pipeline and _bundle is not None:
            # ── Geometry engine path ──────────────────────────────────────────
            _swings = find_swing_points(trigger_candles)  # type: ignore[arg-type]
            _geometry = place_entry_stop_target(
                side,
                px,
                _swings,
                _bundle,
                atr,
                regime,
                self.style,
                {**mode_params, "min_rr_ratio": self._trade_filter_params().get("min_rr_ratio", 1.5)},
                tick,
            )
            entry = _geometry.entry
            structure_stop = _geometry.stop
            target = _geometry.target
            risk = _geometry.risk
            reward = _geometry.reward
            structure_reference = support if side == "long" else resistance
            rationale = "Bullish confluence from macro trend, context trend, trigger structure, and entry timing." if side == "long" else "Bearish confluence from macro trend, context trend, trigger structure, and entry timing."
        elif side == "long":
            # Enter at current price — don't chase support if price is already above it
            entry = px
            structure_stop = support - atr_buffer
            risk = max(entry - structure_stop, px * strategy.min_risk_px_factor)
            structure_target = resistance
            fallback_target = entry + (risk * self.risk_reward)
            raw_target = structure_target if structure_target > entry + (risk * 0.8) else fallback_target
            max_target_distance = max((atr * mode_params["target_cap_atr_mult"]), px * 0.002)
            target = min(raw_target, entry + max_target_distance)
            structure_reference = support
            rationale = "Bullish confluence from macro trend, context trend, trigger structure, and entry timing."
        else:
            # Enter at current price — don't chase resistance if price is already below it
            entry = px
            structure_stop = resistance + atr_buffer
            risk = max(structure_stop - entry, px * strategy.min_risk_px_factor)
            structure_target = support
            fallback_target = entry - (risk * self.risk_reward)
            raw_target = structure_target if structure_target < entry - (risk * 0.8) else fallback_target
            max_target_distance = max((atr * mode_params["target_cap_atr_mult"]), px * 0.002)
            target = max(raw_target, entry - max_target_distance)
            structure_reference = resistance
            rationale = "Bearish confluence from macro trend, context trend, trigger structure, and entry timing."

        # Volume profile: use VAL/VAH as stop reference when they're closer and
        # volume-backed (stronger than a raw swing low/high) — legacy path only.
        if not _use_new_pipeline and volume_profile is not None:
            if side == "long":
                val = volume_profile.get("val", 0.0)
                if val > support and abs(px - val) < atr * strategy.volume_profile_atr_proximity:
                    structure_stop = val - atr_buffer
                    structure_reference = val
                    risk = max(entry - structure_stop, px * strategy.min_risk_px_factor)
            else:
                vah = volume_profile.get("vah", 0.0)
                if vah > 0 and vah < resistance and abs(px - vah) < atr * strategy.volume_profile_atr_proximity:
                    structure_stop = vah + atr_buffer
                    structure_reference = vah
                    risk = max(structure_stop - entry, px * strategy.min_risk_px_factor)

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
                raw_positive["volume_poc_proximity"] = strategy.volume_poc_near_score
            elif side == "long" and volume_profile.get("near_val"):
                raw_positive["volume_poc_proximity"] = strategy.volume_poc_side_score
            elif side == "short" and volume_profile.get("near_vah"):
                raw_positive["volume_poc_proximity"] = strategy.volume_poc_side_score
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

        # ── Confidence computation ────────────────────────────────────────────
        # When the new signal pipeline was used, confidence comes from the
        # logistic sigmoid over the evidence weighted sum (already capped by
        # confidence_ceiling above). Otherwise fall back to the legacy formula.
        if _new_confidence is not None:
            confidence = _new_confidence
        else:
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

            # Timeframe alignment multiplier (legacy path only)
            if alignment_score >= strategy.alignment_strong_threshold:
                confidence = _clamp(confidence * strategy.alignment_strong_confidence_mult, 0.0, confidence_ceiling)
            elif alignment_score >= strategy.alignment_neutral_high:
                pass  # mostly agree — no change
            elif alignment_score >= strategy.alignment_neutral_low:
                confidence = _clamp(confidence * strategy.alignment_weak_confidence_mult, 0.0, confidence_ceiling)
            else:
                confidence = _clamp(confidence * strategy.alignment_opposed_confidence_mult, 0.0, confidence_ceiling)

        if _use_new_pipeline and _geometry is not None:
            rr_ratio = _geometry.rr_ratio
            stop_distance_pct = _geometry.stop_distance_pct
            target_distance_pct = _geometry.target_distance_pct
            atr_multiple_to_stop = _geometry.atr_multiple_to_stop
            atr_multiple_to_target = _geometry.atr_multiple_to_target
            invalidation_strength = _geometry.invalidation_strength
        else:
            rr_ratio = reward / max(risk, 1e-9)
            stop_distance_pct = (risk / max(entry, 1e-9)) * 100.0
            target_distance_pct = (reward / max(entry, 1e-9)) * 100.0
            atr_multiple_to_stop = risk / max(atr, 1e-9)
            atr_multiple_to_target = reward / max(atr, 1e-9)
            structure_gap = abs(entry - structure_reference)
            range_context = abs(resistance - support) / max(atr, 1e-9)
            formula_penalty = strategy.invalidation_formula_penalty if structure_gap <= (px * strategy.invalidation_proximity_threshold_pct) else 0.0
            invalidation_strength = _clamp(
                ((structure_gap / max(atr, 1e-9)) * strategy.invalidation_structure_gap_weight)
                + ((atr_buffer / max(atr, 1e-9)) * strategy.invalidation_atr_buffer_weight)
                + (min(range_context, strategy.invalidation_range_context_cap) / strategy.invalidation_range_context_cap * strategy.invalidation_range_context_weight)
                - formula_penalty,
                0.0,
                1.0,
            )

        raw_quality_score = _clamp(
            strategy.quality_base_score
            + min(weighted_positive_total, strategy.quality_positive_cap) * strategy.quality_positive_weight
            + min(rr_ratio, strategy.quality_rr_cap) * strategy.quality_rr_weight
            + invalidation_strength * strategy.quality_invalidation_weight
            - min(weighted_negative_total, strategy.quality_negative_cap) * strategy.quality_negative_weight,
            strategy.quality_min_score,
            strategy.quality_max_score,
        )
        if _use_new_pipeline and _geometry is not None:
            # Build confluence_dict from the score_confluence result (computed later in
            # analyze(), but we need it here — use a placeholder that will be overridden
            # by _apply_confluence_boost in analyze()).
            _confluence_dict: dict[str, float] = {"entry_confluence": 0.0, "target_confluence": 0.0}
            geo_quality_score, geo_quality_label = geometry_quality_score(
                _geometry, regime, _confluence_dict, 0.5
            )
            quality_score = geo_quality_score
            quality_label = geo_quality_label
        else:
            quality_score = min(raw_quality_score, _quality_score_cap_from_confidence(confidence))
            quality_label = _quality_label(quality_score)
        # ── Primary trade filter ─────────────────────────────────────────────
        # Centralised gate evaluated BEFORE _SideMetrics is constructed.
        # Downstream reversal penalties may append further reasons and
        # re-evaluate is_tradable via _apply_reversal_penalty.
        _filter_params = self._trade_filter_params()

        reasons: list[str] = []

        # --- Confidence floor (regime-aware) ---
        if regime == MarketRegime.RANGE:
            if confidence < 0.55:
                reasons.append(f"confidence {confidence:.2f} below threshold (range)")
        else:
            if confidence < _filter_params["min_confidence"]:
                reasons.append(f"confidence {confidence:.2f} below threshold")

        # --- Quality floor (regime-aware) ---
        if regime == MarketRegime.RANGE:
            if quality_score < 48:
                reasons.append(f"quality {quality_score:.1f} below threshold (range)")
        else:
            if quality_score < _filter_params["min_quality"]:
                reasons.append(f"quality {quality_score:.1f} below threshold")

        # --- Evidence agreement (new pipeline only) ---
        if _evidence is not None:
            _min_ev = _filter_params.get("min_evidence_agreement", 0)
            if _evidence.signal_count_above_threshold < _min_ev:
                reasons.append(
                    f"evidence agreement {_evidence.signal_count_above_threshold} below {_min_ev}"
                )

        # --- Regime filter ---
        if hasattr(strategy, "allowed_regimes") and regime.value not in strategy.allowed_regimes:
            reasons.append(f"regime {regime.value} disabled")

        # --- Trend alignment (RELAXED — macro OR context must agree, new pipeline only) ---
        if _bundle is not None:
            direction = 1 if side == "long" else -1
            macro_ok = _bundle.higher_trend * direction > 0
            context_ok = _bundle.context_trend * direction > 0
            if not (macro_ok or context_ok):
                reasons.append("macro and context trends are not aligned")

        # --- Legacy filters (R:R, stop distance, max confidence) ---
        if _filter_params.get("max_confidence", 1.0) and confidence > _filter_params.get("max_confidence", 1.0):
            reasons.append(f"confidence {confidence:.2f} is above {_filter_params['max_confidence']:.2f}")
        if rr_ratio < _filter_params["min_rr_ratio"]:
            reasons.append(f"R:R {rr_ratio:.2f} is below {_filter_params['min_rr_ratio']:.2f}")
        if stop_distance_pct > _filter_params["max_stop_distance_pct"]:
            reasons.append(
                f"stop distance {stop_distance_pct:.2f}% is above {_filter_params['max_stop_distance_pct']:.2f}%"
            )

        tradable_reasons = reasons
        is_tradable = len(reasons) == 0
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

        # ── Evidence agreement from new pipeline ─────────────────────────────
        # When the new signal pipeline was used, populate evidence_agreement from
        # EvidenceVector.signal_count_above_threshold and evidence_total = 16
        # (total number of NormalizedSignals fields).
        if _evidence is not None:
            _evidence_agreement = _evidence.signal_count_above_threshold
            _evidence_total = 16  # total NormalizedSignals fields
        else:
            _evidence_agreement = 0
            _evidence_total = 0

        _signal_strengths = {
            key: round(value, 6)
            for key, value in (asdict(_signals).items() if _signals is not None else [])
        }

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
            evidence_agreement=_evidence_agreement,
            evidence_total=_evidence_total,
            deliberation_summary="",
            stop_anchor=_geometry.stop_anchor if _geometry is not None else "atr_fallback",
            target_anchor=_geometry.target_anchor if _geometry is not None else "atr_cap",
            signal_strengths=_signal_strengths,
            evidence_weighted_sum=round(_evidence.weighted_sum, 6) if _evidence is not None else 0.0,
            logistic_input=round(_logistic_input, 6),
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
