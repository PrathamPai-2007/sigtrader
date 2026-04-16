"""Enhanced LONG entry filters to improve entry timing and confirmation.

This module implements optimized filters for LONG trades to address:
1. Early/weak entries (MAE > MFE issue)
2. Insufficient momentum expansion confirmation
3. Poor entry timing (entering too early in consolidation)
4. Weak structural confirmation

The filters focus on delayed entry with strong confirmation rather than
immediate entry on signal appearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from futures_analyzer.analysis.indicators import adx
from futures_analyzer.analysis.models import Candle, MarketRegime

if TYPE_CHECKING:
    from futures_analyzer.config import AppConfig


@dataclass
class MomentumExpansionResult:
    """Result of momentum expansion analysis."""
    has_strong_momentum: bool
    trigger_momentum: float
    volume_surge: float
    momentum_threshold_met: bool
    volume_threshold_met: bool
    expansion_score: float
    reason: str


@dataclass
class DelayedEntryResult:
    """Result of delayed entry confirmation."""
    allow_immediate_entry: bool
    needs_confirmation: bool
    candles_since_signal: int
    price_continuation: bool
    momentum_sustained: bool
    confirmation_score: float
    reason: str


@dataclass
class StructureConfirmationResult:
    """Result of enhanced structure confirmation analysis."""
    has_strong_confirmation: bool
    higher_high_break: bool
    higher_low_confirmed: bool
    structure_shift: bool
    volume_expansion: bool
    confirmation_count: int
    reason: str


@dataclass
class TrendDominanceResult:
    """Result of trend dominance analysis."""
    is_strongly_bullish: bool
    ema_separation_pct: float
    adx_value: float
    plus_di: float
    minus_di: float
    dominance_score: float
    reason: str


@dataclass
class PullbackTrapResult:
    """Result of pullback trap detection."""
    is_pullback_trap: bool
    short_term_bullish: bool
    higher_tf_bearish: bool
    momentum_divergence: float
    reason: str


@dataclass
class LongEntryFilterResult:
    """Combined result of all enhanced LONG entry filters."""
    allow_long: bool
    momentum_expansion: MomentumExpansionResult
    delayed_entry: DelayedEntryResult
    structure_confirmation: StructureConfirmationResult
    trend_dominance: TrendDominanceResult
    pullback_trap: PullbackTrapResult
    overall_reason: str


def momentum_expansion_check(
    trigger_momentum: float,
    volume_surge: float,
    config: AppConfig | None = None,
) -> MomentumExpansionResult:
    """Check if momentum is strong and expanding (CRITICAL filter).
    
    Only allow LONG if:
    1. trigger_momentum > threshold (strong bullish momentum)
    2. volume_surge > threshold (volume confirmation)
    
    Rejects slow drift and weak candles.
    
    Args:
        trigger_momentum: Current momentum reading [-1, 1]
        volume_surge: Volume surge ratio (1.0 = average)
        config: Optional config for thresholds
        
    Returns:
        MomentumExpansionResult with analysis details
    """
    from futures_analyzer.config import load_app_config
    
    cfg = config or load_app_config()
    
    # Get thresholds from config
    momentum_cfg = cfg.strategy.long_entry_filters.momentum_expansion
    min_momentum_threshold = momentum_cfg.min_momentum_threshold
    min_volume_surge = momentum_cfg.min_volume_surge
    
    # Convert normalized momentum [-1,1] to percentage
    momentum_pct = trigger_momentum * 100.0
    
    # Check thresholds
    momentum_threshold_met = trigger_momentum > min_momentum_threshold
    volume_threshold_met = volume_surge > min_volume_surge
    
    # Calculate expansion score (0-1)
    momentum_score = max(0.0, (trigger_momentum - min_momentum_threshold) / (1.0 - min_momentum_threshold))
    volume_score = max(0.0, (volume_surge - min_volume_surge) / (2.0 - min_volume_surge))
    expansion_score = (momentum_score + volume_score) / 2.0
    
    has_strong_momentum = momentum_threshold_met and volume_threshold_met
    
    if not momentum_threshold_met:
        reason = f"Momentum {momentum_pct:.1f}% below threshold {min_momentum_threshold*100:.1f}%"
    elif not volume_threshold_met:
        reason = f"Volume surge {volume_surge:.2f}x below threshold {min_volume_surge:.2f}x"
    else:
        reason = f"Strong momentum {momentum_pct:.1f}% with volume {volume_surge:.2f}x"
    
    return MomentumExpansionResult(
        has_strong_momentum=has_strong_momentum,
        trigger_momentum=trigger_momentum,
        volume_surge=volume_surge,
        momentum_threshold_met=momentum_threshold_met,
        volume_threshold_met=volume_threshold_met,
        expansion_score=expansion_score,
        reason=reason,
    )


def delayed_entry_confirmation(
    trigger_candles: list[Candle],
    current_price: float,
    trigger_momentum: float,
    config: AppConfig | None = None,
) -> DelayedEntryResult:
    """Implement delayed entry logic (VERY IMPORTANT).
    
    For backtest compatibility, we simulate the delay by checking if:
    1. Recent momentum has been building (not just current bar)
    2. Price action shows sustained bullish pressure
    3. Current momentum is strong enough to sustain
    
    Args:
        trigger_candles: Recent trigger timeframe candles
        current_price: Current market price
        trigger_momentum: Current momentum reading
        config: Optional config for thresholds
        
    Returns:
        DelayedEntryResult with confirmation analysis
    """
    from futures_analyzer.config import load_app_config
    
    cfg = config or load_app_config()
    
    # Get delay settings from config
    delay_cfg = cfg.strategy.long_entry_filters.delayed_entry
    min_price_continuation = delay_cfg.min_price_continuation_pct
    min_momentum_sustain = delay_cfg.min_momentum_sustain
    
    if len(trigger_candles) < 5:
        return DelayedEntryResult(
            allow_immediate_entry=False,
            needs_confirmation=True,
            candles_since_signal=0,
            price_continuation=False,
            momentum_sustained=False,
            confirmation_score=0.0,
            reason="Insufficient candle history for delayed entry analysis",
        )
    
    # Check sustained momentum over last 2-3 candles (simulating delay)
    recent_candles = trigger_candles[-3:]  # Last 3 candles
    
    # Calculate price progression over recent candles
    price_changes = []
    for i in range(1, len(recent_candles)):
        change_pct = ((recent_candles[i].close - recent_candles[i-1].close) / recent_candles[i-1].close) * 100.0
        price_changes.append(change_pct)
    
    # Check if price has been generally rising
    avg_price_change = sum(price_changes) / len(price_changes) if price_changes else 0.0
    price_continuation = avg_price_change > min_price_continuation
    
    # Check if current momentum is sustained (not fading)
    momentum_sustained = trigger_momentum > min_momentum_sustain
    
    # Additional check: ensure we're not in a choppy/sideways period
    price_volatility = max(price_changes) - min(price_changes) if price_changes else 0.0
    low_chop = price_volatility < 2.0  # Less than 2% volatility range
    
    # Calculate confirmation score
    price_score = max(0.0, avg_price_change / (min_price_continuation * 2.0))
    momentum_score = max(0.0, (trigger_momentum - min_momentum_sustain) / (1.0 - min_momentum_sustain))
    chop_score = 1.0 if low_chop else 0.5
    confirmation_score = (price_score + momentum_score + chop_score) / 3.0
    
    allow_immediate_entry = price_continuation and momentum_sustained and low_chop
    
    if not price_continuation:
        reason = f"Price trend {avg_price_change:.2f}% below {min_price_continuation:.2f}%"
    elif not momentum_sustained:
        reason = f"Momentum {trigger_momentum:.3f} below sustain threshold {min_momentum_sustain:.3f}"
    elif not low_chop:
        reason = f"High volatility {price_volatility:.2f}% indicates choppy conditions"
    else:
        reason = f"Confirmed: sustained trend +{avg_price_change:.2f}%, momentum {trigger_momentum:.3f}"
    
    return DelayedEntryResult(
        allow_immediate_entry=allow_immediate_entry,
        needs_confirmation=False,
        candles_since_signal=len(recent_candles),
        price_continuation=price_continuation,
        momentum_sustained=momentum_sustained,
        confirmation_score=confirmation_score,
        reason=reason,
    )


def structure_confirmation_check(
    trigger_candles: list[Candle],
    market_structure: str,
    swing_highs: list[float],
    current_price: float,
    volume_surge: float,
    config: AppConfig | None = None,
) -> StructureConfirmationResult:
    """Enhanced structure confirmation requiring stronger bullish evidence.
    
    Instead of single higher high, require:
    1. Higher high AND higher low, OR
    2. Confirmed structure break with volume
    
    Args:
        trigger_candles: Recent trigger timeframe candles
        market_structure: Current market structure pattern
        swing_highs: Recent swing high levels
        current_price: Current market price
        volume_surge: Current volume surge ratio
        config: Optional config for thresholds
        
    Returns:
        StructureConfirmationResult with analysis details
    """
    from futures_analyzer.config import load_app_config
    
    cfg = config or load_app_config()
    
    # Get structure settings from config
    struct_cfg = cfg.strategy.long_entry_filters.structure_confirmation
    min_volume_expansion = struct_cfg.min_volume_expansion
    higher_high_buffer_pct = struct_cfg.higher_high_buffer_pct
    min_confirmations_required = struct_cfg.min_confirmations_required
    
    if len(trigger_candles) < 10 or not swing_highs:
        return StructureConfirmationResult(
            has_strong_confirmation=False,
            higher_high_break=False,
            higher_low_confirmed=False,
            structure_shift=False,
            volume_expansion=False,
            confirmation_count=0,
            reason="Insufficient data for structure analysis",
        )
    
    # Check 1: Higher high break
    recent_high = max(swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1]
    higher_high_threshold = recent_high * (1.0 + higher_high_buffer_pct)
    higher_high_break = current_price > higher_high_threshold
    
    # Check 2: Higher low confirmed (last 5 candles vs previous 5)
    recent_lows = [c.low for c in trigger_candles[-5:]]
    previous_lows = [c.low for c in trigger_candles[-10:-5]]
    
    if recent_lows and previous_lows:
        recent_low_avg = sum(recent_lows) / len(recent_lows)
        previous_low_avg = sum(previous_lows) / len(previous_lows)
        higher_low_confirmed = recent_low_avg > previous_low_avg
    else:
        higher_low_confirmed = False
    
    # Check 3: Structure shift to bullish
    structure_shift = market_structure in ["HH_HL", "mixed_bullish"]
    
    # Check 4: Volume expansion
    volume_expansion = volume_surge >= min_volume_expansion
    
    # Count confirmations
    confirmations = [
        higher_high_break,
        higher_low_confirmed,
        structure_shift,
        volume_expansion,
    ]
    confirmation_count = sum(confirmations)
    
    has_strong_confirmation = confirmation_count >= min_confirmations_required
    
    # Build reason
    reasons = []
    if higher_high_break:
        reasons.append(f"HH break above {recent_high:.2f}")
    if higher_low_confirmed:
        reasons.append("HL confirmed")
    if structure_shift:
        reasons.append(f"structure: {market_structure}")
    if volume_expansion:
        reasons.append(f"volume: {volume_surge:.2f}x")
    
    if has_strong_confirmation:
        reason = f"Strong confirmation ({confirmation_count}/{len(confirmations)}): " + ", ".join(reasons)
    else:
        reason = f"Weak confirmation ({confirmation_count}/{min_confirmations_required}): " + ", ".join(reasons)
    
    return StructureConfirmationResult(
        has_strong_confirmation=has_strong_confirmation,
        higher_high_break=higher_high_break,
        higher_low_confirmed=higher_low_confirmed,
        structure_shift=structure_shift,
        volume_expansion=volume_expansion,
        confirmation_count=confirmation_count,
        reason=reason,
    )


def check_trend_dominance(
    higher_candles: list[Candle],
    config: AppConfig | None = None,
) -> TrendDominanceResult:
    """Check if higher timeframe trend is STRONGLY bullish.
    
    Requirements for strong bullish trend:
    1. EMA fast > EMA slow with sufficient separation (>= 1.5%)
    OR
    2. ADX above threshold (>= 25) AND DI+ > DI- with margin (>= 5 points)
    
    Args:
        higher_candles: Higher timeframe candle data
        config: Optional config for thresholds
        
    Returns:
        TrendDominanceResult with analysis details
    """
    from futures_analyzer.config import load_app_config
    
    cfg = config or load_app_config()
    
    # Get thresholds from config
    trend_cfg = cfg.strategy.long_entry_filters.trend_dominance
    min_ema_separation_pct = trend_cfg.min_ema_separation_pct
    min_adx_threshold = trend_cfg.min_adx_threshold
    min_di_margin = trend_cfg.min_di_margin
    
    if len(higher_candles) < 50:  # Need sufficient history for reliable EMA
        return TrendDominanceResult(
            is_strongly_bullish=False,
            ema_separation_pct=0.0,
            adx_value=0.0,
            plus_di=0.0,
            minus_di=0.0,
            dominance_score=0.0,
            reason="Insufficient candle history for trend analysis"
        )
    
    # Calculate EMA separation
    closes = [c.close for c in higher_candles]
    ema_fast_period = 20
    ema_slow_period = 50
    
    # Simple EMA calculation
    def _ema(values: list[float], period: int) -> float:
        if len(values) < period:
            return values[-1] if values else 0.0
        k = 2.0 / (period + 1)
        ema = sum(values[:period]) / period
        for val in values[period:]:
            ema = val * k + ema * (1 - k)
        return ema
    
    ema_fast = _ema(closes, ema_fast_period)
    ema_slow = _ema(closes, ema_slow_period)
    
    ema_separation_pct = ((ema_fast - ema_slow) / ema_slow * 100.0) if ema_slow > 0 else 0.0
    
    # Calculate ADX and DI values
    adx_result = adx(higher_candles)
    adx_value = adx_result["adx"]
    plus_di = adx_result["plus_di"]
    minus_di = adx_result["minus_di"]
    
    # Check conditions
    ema_condition = ema_separation_pct >= min_ema_separation_pct
    adx_condition = (adx_value >= min_adx_threshold and 
                     plus_di > minus_di and 
                     (plus_di - minus_di) >= min_di_margin)
    
    is_strongly_bullish = ema_condition or adx_condition
    
    # Calculate dominance score (0-100)
    ema_score = min(ema_separation_pct / min_ema_separation_pct * 50, 50)
    adx_score = 0
    if adx_value >= min_adx_threshold:
        di_margin = plus_di - minus_di
        adx_score = min((adx_value / 50.0) * 25 + (di_margin / 20.0) * 25, 50)
    
    dominance_score = max(ema_score, adx_score)
    
    # Generate reason
    if is_strongly_bullish:
        if ema_condition and adx_condition:
            reason = f"Strong bullish: EMA sep {ema_separation_pct:.1f}% + ADX {adx_value:.1f} (+DI {plus_di:.1f} > -DI {minus_di:.1f})"
        elif ema_condition:
            reason = f"Strong bullish: EMA separation {ema_separation_pct:.1f}% (>= {min_ema_separation_pct}%)"
        else:
            reason = f"Strong bullish: ADX {adx_value:.1f} with +DI {plus_di:.1f} > -DI {minus_di:.1f}"
    else:
        reason = f"Weak trend: EMA sep {ema_separation_pct:.1f}%, ADX {adx_value:.1f}, DI margin {plus_di - minus_di:.1f}"
    
    return TrendDominanceResult(
        is_strongly_bullish=is_strongly_bullish,
        ema_separation_pct=ema_separation_pct,
        adx_value=adx_value,
        plus_di=plus_di,
        minus_di=minus_di,
        dominance_score=dominance_score,
        reason=reason
    )


def detect_pullback_trap(
    trigger_candles: list[Candle],
    higher_candles: list[Candle],
    config: AppConfig | None = None,
) -> PullbackTrapResult:
    """Detect pullback trap scenarios where short-term momentum is bullish
    but higher timeframe trend is bearish.
    
    This prevents entering LONG during bearish pullbacks (false reversals).
    
    Args:
        trigger_candles: Trigger timeframe candles
        higher_candles: Higher timeframe candles
        config: Optional config for thresholds
        
    Returns:
        PullbackTrapResult with analysis details
    """
    from futures_analyzer.analysis.scorer import _momentum
    from futures_analyzer.config import load_app_config
    
    cfg = config or load_app_config()
    
    # Get thresholds from config
    trap_cfg = cfg.strategy.long_entry_filters.pullback_trap
    short_term_threshold = trap_cfg.short_term_momentum_threshold
    higher_tf_threshold = trap_cfg.higher_tf_bearish_threshold
    min_divergence = trap_cfg.min_momentum_divergence
    
    if len(trigger_candles) < 20 or len(higher_candles) < 20:
        return PullbackTrapResult(
            is_pullback_trap=False,
            short_term_bullish=False,
            higher_tf_bearish=False,
            momentum_divergence=0.0,
            reason="Insufficient data for pullback analysis"
        )
    
    # Check short-term momentum (trigger timeframe)
    trigger_momentum = _momentum(trigger_candles, window=14)
    short_term_bullish = trigger_momentum > short_term_threshold
    
    # Check higher timeframe trend
    higher_momentum = _momentum(higher_candles, window=14)
    higher_tf_bearish = higher_momentum < higher_tf_threshold
    
    # Calculate momentum divergence
    momentum_divergence = trigger_momentum - higher_momentum
    
    # Pullback trap occurs when:
    # 1. Short-term is bullish
    # 2. Higher timeframe is bearish
    # 3. Significant momentum divergence
    is_pullback_trap = (short_term_bullish and 
                       higher_tf_bearish and 
                       momentum_divergence > min_divergence)
    
    if is_pullback_trap:
        reason = f"Pullback trap: trigger +{trigger_momentum*100:.1f}% vs higher {higher_momentum*100:.1f}%"
    else:
        reason = f"No trap: trigger {trigger_momentum*100:.1f}%, higher {higher_momentum*100:.1f}%"
    
    return PullbackTrapResult(
        is_pullback_trap=is_pullback_trap,
        short_term_bullish=short_term_bullish,
        higher_tf_bearish=higher_tf_bearish,
        momentum_divergence=momentum_divergence,
        reason=reason
    )


def check_structure_confirmation(
    trigger_candles: list[Candle],
    market_structure: str,
    volume_surge: float,
    swing_highs: list[float],
    current_price: float,
    config: AppConfig | None = None,
) -> StructureConfirmationResult:
    """Check for structural confirmation before allowing LONG entry.
    
    Requires at least one of:
    1. Break of previous high (higher high)
    2. Market structure shift to bullish (HH_HL)
    3. Strong volume expansion (>= 1.5x)
    
    Args:
        trigger_candles: Trigger timeframe candles
        market_structure: Current market structure ("HH_HL", "LH_LL", "mixed")
        volume_surge: Current volume surge ratio
        swing_highs: Recent swing high levels
        current_price: Current market price
        config: Optional config for thresholds
        
    Returns:
        StructureConfirmationResult with analysis details
    """
    from futures_analyzer.config import load_app_config
    
    cfg = config or load_app_config()
    
    # Get thresholds from config
    struct_cfg = cfg.strategy.long_entry_filters.structure_confirmation
    min_volume_expansion = struct_cfg.min_volume_expansion
    hh_buffer_pct = struct_cfg.higher_high_buffer_pct
    
    # Check for higher high break
    higher_high_break = False
    if swing_highs:
        recent_high = max(swing_highs[-2:]) if len(swing_highs) >= 2 else swing_highs[-1]
        higher_high_break = current_price > recent_high * (1.0 + hh_buffer_pct)
    
    # Check for bullish structure shift
    structure_shift = market_structure == "HH_HL"
    
    # Check for volume expansion
    volume_expansion = volume_surge >= min_volume_expansion
    
    # Count confirmations
    confirmations = [higher_high_break, structure_shift, volume_expansion]
    confirmation_count = sum(confirmations)
    has_confirmation = confirmation_count >= 1
    
    # Generate reason
    reasons = []
    if higher_high_break:
        reasons.append("higher high break")
    if structure_shift:
        reasons.append("bullish structure")
    if volume_expansion:
        reasons.append(f"volume surge {volume_surge:.1f}x")
    
    if has_confirmation:
        reason = f"Confirmed: {', '.join(reasons)} ({confirmation_count}/3)"
    else:
        reason = f"No confirmation: structure={market_structure}, volume={volume_surge:.1f}x, no HH break"
    
    return StructureConfirmationResult(
        has_confirmation=has_confirmation,
        higher_high_break=higher_high_break,
        structure_shift=structure_shift,
        volume_expansion=volume_expansion,
        confirmation_count=confirmation_count,
        reason=reason
    )


def apply_long_entry_filters(
    higher_candles: list[Candle],
    trigger_candles: list[Candle],
    market_structure: str,
    volume_surge: float,
    swing_highs: list[float],
    current_price: float,
    regime: MarketRegime,
    trigger_momentum: float,
    config: AppConfig | None = None,
) -> LongEntryFilterResult:
    """Apply enhanced LONG entry filters for improved timing and confirmation.
    
    Implements the complete optimized filter stack:
    1. Momentum expansion check (CRITICAL) - prevents weak entries
    2. Delayed entry confirmation (VERY IMPORTANT) - prevents early entries
    3. Enhanced structure confirmation - requires stronger evidence
    4. Trend dominance validation - existing filter
    5. Pullback trap detection - existing filter
    
    Args:
        higher_candles: Higher timeframe candle data
        trigger_candles: Trigger timeframe candle data
        market_structure: Current market structure pattern
        volume_surge: Current volume surge ratio
        swing_highs: Recent swing high levels
        current_price: Current market price
        regime: Current market regime
        trigger_momentum: Current trigger momentum reading
        config: Optional config for thresholds
        
    Returns:
        LongEntryFilterResult with comprehensive analysis
    """
    from futures_analyzer.config import load_app_config
    
    cfg = config or load_app_config()
    
    # Check if enhanced filters are enabled
    if not cfg.strategy.long_entry_filters.enable_enhanced_filters:
        # Return permissive result if filters are disabled
        return LongEntryFilterResult(
            allow_long=True,
            momentum_expansion=MomentumExpansionResult(
                has_strong_momentum=True,
                trigger_momentum=trigger_momentum,
                volume_surge=volume_surge,
                momentum_threshold_met=True,
                volume_threshold_met=True,
                expansion_score=1.0,
                reason="Enhanced filters disabled"
            ),
            delayed_entry=DelayedEntryResult(
                allow_immediate_entry=True,
                needs_confirmation=False,
                candles_since_signal=0,
                price_continuation=True,
                momentum_sustained=True,
                confirmation_score=1.0,
                reason="Enhanced filters disabled"
            ),
            structure_confirmation=StructureConfirmationResult(
                has_strong_confirmation=True,
                higher_high_break=False,
                higher_low_confirmed=False,
                structure_shift=False,
                volume_expansion=False,
                confirmation_count=0,
                reason="Enhanced filters disabled"
            ),
            trend_dominance=TrendDominanceResult(
                is_strongly_bullish=True,
                ema_separation_pct=0.0,
                adx_value=0.0,
                plus_di=0.0,
                minus_di=0.0,
                dominance_score=0.0,
                reason="Enhanced filters disabled"
            ),
            pullback_trap=PullbackTrapResult(
                is_pullback_trap=False,
                short_term_bullish=False,
                higher_tf_bearish=False,
                momentum_divergence=0.0,
                reason="Enhanced filters disabled"
            ),
            overall_reason="Enhanced LONG filters disabled - allowing all LONG trades"
        )
    
    # 1. Momentum Expansion Check (CRITICAL)
    momentum_result = momentum_expansion_check(
        trigger_momentum=trigger_momentum,
        volume_surge=volume_surge,
        config=cfg,
    )
    
    # 2. Delayed Entry Confirmation (VERY IMPORTANT)
    delayed_result = delayed_entry_confirmation(
        trigger_candles=trigger_candles,
        current_price=current_price,
        trigger_momentum=trigger_momentum,
        config=cfg,
    )
    
    # 3. Enhanced Structure Confirmation
    structure_result = structure_confirmation_check(
        trigger_candles=trigger_candles,
        market_structure=market_structure,
        swing_highs=swing_highs,
        current_price=current_price,
        volume_surge=volume_surge,
        config=cfg,
    )
    
    # 4. Trend Dominance (existing)
    trend_result = check_trend_dominance(
        higher_candles=higher_candles,
        config=cfg,
    )
    
    # 5. Pullback Trap Detection (existing)
    pullback_result = detect_pullback_trap(
        trigger_candles=trigger_candles,
        higher_candles=higher_candles,
        config=cfg,
    )
    
    # Combine results - ALL must pass for LONG entry
    blocking_reasons = []
    
    if not momentum_result.has_strong_momentum:
        blocking_reasons.append(f"Momentum: {momentum_result.reason}")
    
    if not delayed_result.allow_immediate_entry:
        if delayed_result.needs_confirmation:
            blocking_reasons.append(f"Timing: {delayed_result.reason}")
        else:
            blocking_reasons.append(f"Expired: {delayed_result.reason}")
    
    if not structure_result.has_strong_confirmation:
        blocking_reasons.append(f"Structure: {structure_result.reason}")
    
    if not trend_result.is_strongly_bullish:
        blocking_reasons.append(f"Trend: {trend_result.reason}")
    
    if pullback_result.is_pullback_trap:
        blocking_reasons.append(f"Pullback: {pullback_result.reason}")
    
    allow_long = len(blocking_reasons) == 0
    
    if allow_long:
        overall_reason = "All filters passed - strong LONG setup confirmed"
    else:
        overall_reason = " | ".join(blocking_reasons)
    
    return LongEntryFilterResult(
        allow_long=allow_long,
        momentum_expansion=momentum_result,
        delayed_entry=delayed_result,
        structure_confirmation=structure_result,
        trend_dominance=trend_result,
        pullback_trap=pullback_result,
        overall_reason=overall_reason,
    )


def adjust_long_confidence_threshold(
    base_confidence: float,
    regime: MarketRegime,
    filter_result: LongEntryFilterResult,
    config: AppConfig | None = None,
) -> float:
    """Adjust LONG confidence threshold to be more strict than SHORT.
    
    LONG trades require higher confidence (0.6-0.65 midpoint) vs SHORT trades.
    Additional penalties applied based on filter results.
    
    Args:
        base_confidence: Original confidence score
        regime: Current market regime
        filter_result: Result from LONG entry filters
        config: Optional config
        
    Returns:
        Adjusted confidence score (potentially reduced)
    """
    from futures_analyzer.config import load_app_config
    
    cfg = config or load_app_config()
    
    # Get penalty configuration
    adj_cfg = cfg.strategy.long_entry_filters.confidence_adjustments
    
    # Base adjustment - LONG requires higher threshold
    long_penalty = adj_cfg.base_long_penalty
    
    # Additional penalties based on filter strength
    if not filter_result.trend_dominance.is_strongly_bullish:
        long_penalty += adj_cfg.weak_trend_penalty
    
    if filter_result.pullback_trap.momentum_divergence > 0.02:
        long_penalty += adj_cfg.momentum_divergence_penalty
    
    if filter_result.structure_confirmation.confirmation_count == 0:
        long_penalty += adj_cfg.no_confirmation_penalty
    
    # Regime-specific adjustments
    if regime == MarketRegime.BEARISH_TREND:
        long_penalty += adj_cfg.bearish_regime_penalty
    elif regime == MarketRegime.VOLATILE_CHOP:
        long_penalty += adj_cfg.volatile_chop_penalty
    
    # Apply penalty (multiplicative to preserve relative differences)
    adjusted_confidence = base_confidence * (1.0 - long_penalty)
    
    return max(adjusted_confidence, 0.0)  # Ensure non-negative