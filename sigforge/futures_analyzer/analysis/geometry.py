"""Geometry module — entry/stop/target placement and quality scoring.

All functions are pure: inputs are prices, ATR, swing points, and VWAP/VP
structure from IndicatorBundle. No dependency on evidence, confidence, or
filter layers.
"""
from __future__ import annotations

from futures_analyzer.analysis.models import (
    Candle,
    EntryGeometry,
    IndicatorBundle,
    MarketRegime,
    QualityLabel,
    StrategyStyle,
    SwingPoints,
)
from futures_analyzer.analysis.indicators import _swing_pivots
from futures_analyzer.analysis.scoring.utils import _clamp, _quantize, _quality_label
from futures_analyzer.config import AppConfig, load_app_config

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

    Returns the first structure anchor (swing, vwap, or volume-profile) found,
    regardless of R:R ratio. Falls back to the first candidate (atr_cap) only
    when no structure anchor exists in the list.

    Args:
        candidates: list of (price, anchor_label) tuples in priority order
        entry: entry price
        stop: stop price
        params: dict (kept for interface compatibility)

    Returns:
        (price, anchor_label) tuple
    """
    _structure_anchors = {"swing_high_sweep", "swing_high", "swing_low_sweep",
                          "swing_low", "vwap_upper", "vwap_lower", "vah", "val"}
    for price, label in candidates:
        if label in _structure_anchors:
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

    Long stop priority:   (1) swing low sweep, (2) VWAP lower 1SD, (3) VAL, (4) ATR fallback.
    Long target priority: (1) swing high sweep, (2) VWAP upper 2SD, (3) VAH, (4) ATR cap.
    Short stop priority:  (1) swing high, (2) VWAP upper 1SD, (3) VAH, (4) ATR fallback.
    Short target priority:(1) swing low, (2) VWAP lower 2SD, (3) VAL, (4) ATR cap.

    Structure targets (swing/vwap/vp) are always used as-is — no R:R check.
    rr_enforced fires ONLY when no structure target exists (atr_cap fallback only).
    SHORT structure targets are capped at 2.5× risk to prevent extreme exits.
    Stops are never modified. Quantizes to tick size. ATR floor: max(atr, px*0.001).
    """

    # ATR safety: substitute max(atr, px * 0.001) when ATR is zero

    atr = max(atr, px * 0.001)

    atr_buffer = atr * mode_params.get("atr_buffer_factor", 0.5)

    min_rr = mode_params.get("min_rr_ratio", 1.5)

    target_cap_atr_mult = mode_params.get("target_cap_atr_mult", 3.0)



    entry = px

    if side == "long":

        # Bait entry: offset below market price to require a pullback before fill.
        entry = px - (atr * 0.15)

        # --- Stop placement (priority order) ---

        candidate_stops: list[tuple[float, str]] = []



        # 1. Nearest swing low below entry

        valid_lows = [l for l in swings.recent_lows if l < entry - atr * 0.1]



        if valid_lows:

            nearest_low = max(valid_lows)

            sweep_buffer = atr * 0.2

            candidate_stops.append(

                (nearest_low - atr_buffer - sweep_buffer, "swing_low_sweep")

            )



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

            candidate_targets.append((nearest_high, "swing_high_sweep"))



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



    else:  # short ??? mirror logic

        # --- Stop placement (priority order) ---

        candidate_stops = []



        # 1. Nearest swing high above entry

        valid_highs = [h for h in swings.recent_highs if h > entry + atr * 0.1]

        if valid_highs:

            nearest_high = min(valid_highs)

            candidate_stops.append((nearest_high + atr * 0.15, "swing_high"))



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



    # rr_enforced fires ONLY when no structure target exists (atr_cap fallback).
    # Structure targets are always used as-is regardless of R:R ratio.
    # The final safety gate (rr_ratio < 0.8) in scorer.py handles absolute floors.
    # For longs (bait entries), enforce a minimum 2.0 R/R from the discounted entry.
    risk = abs(entry - stop)
    reward = abs(target - entry)
    _structure_anchors = {"swing_high_sweep", "swing_high", "swing_low_sweep",
                          "swing_low", "vwap_upper", "vwap_lower", "vah", "val"}
    _has_structure_target = target_anchor in _structure_anchors
    _long_min_rr = 2.0 if side == "long" else min_rr
    if not _has_structure_target and reward / max(risk, 1e-9) < _long_min_rr:
        if side == "long":
            target = entry + risk * _long_min_rr
        else:
            target = entry - risk * min_rr
        target_anchor = "rr_enforced"

    # SHORT TP cap — prevents unrealistic distant structure targets.
    # Cap at 2.5× risk. LONG cap deferred (future use).
    risk = abs(entry - stop)
    if _has_structure_target and side == "short":
        _max_reward = risk * 2.5
        if abs(target - entry) > _max_reward:
            target = entry - _max_reward
            target_anchor = "tp_capped"



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

    config: AppConfig | None = None,

) -> tuple[float, QualityLabel]:

    """Compute a quality score in [min_score, max_score] from setup geometry.



    All scoring parameters come from config.strategy.geometry_quality.

    """

    cfg = config or load_app_config()

    gq = cfg.strategy.geometry_quality

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



    # 1. R:R contribution (primary driver, 0???30 pts)

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
