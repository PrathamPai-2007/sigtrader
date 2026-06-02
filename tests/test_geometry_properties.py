import math
from hypothesis import given, strategies as st, assume
from analyzer.analysis.geometry import place_entry_stop_target
from analyzer.analysis.models import (
    IndicatorBundle,
    MarketRegime,
    StrategyStyle,
    SwingPoints,
)

# Shared strategy for generating realistic prices and indicators
prices = st.floats(min_value=0.001, max_value=1000000.0, allow_nan=False, allow_infinity=False)
small_positive = st.floats(min_value=1e-6, max_value=1000.0, allow_nan=False, allow_infinity=False)

@st.composite
def geometry_inputs(draw):
    px = draw(prices)
    atr = draw(st.floats(min_value=0.0, max_value=px * 0.2, allow_nan=False, allow_infinity=False))
    
    # Generate swing points around px
    lows = sorted(draw(st.lists(st.floats(min_value=px * 0.8, max_value=px * 1.1), min_size=0, max_size=5)))
    highs = sorted(draw(st.lists(st.floats(min_value=px * 0.9, max_value=px * 1.2), min_size=0, max_size=5)))
    
    swings = SwingPoints(
        recent_highs=highs,
        recent_lows=lows,
        nearest_high=highs[-1] if highs else 0.0,
        nearest_low=lows[-1] if lows else 0.0,
        second_high=highs[-2] if len(highs) >= 2 else (highs[-1] if highs else 0.0),
        second_low=lows[-2] if len(lows) >= 2 else (lows[-1] if lows else 0.0),
    )
    
    bundle = IndicatorBundle(
        entry_atr=atr,
        trigger_atr=atr,
        context_atr=atr,
        higher_atr=atr,
        higher_trend=0.0,
        context_trend=0.0,
        trigger_momentum=0.0,
        entry_momentum=0.0,
        trigger_volume_surge=1.0,
        entry_volume_surge=1.0,
        cumulative_delta=0.0,
        rsi_14=50.0,
        macd_histogram=0.0,
        stoch_k=50.0,
        bb_position=0.5,
        bb_bandwidth_pct=1.0,
        swing_highs=highs,
        swing_lows=lows,
        market_structure="mixed",
        liquidity_sweeps=[],
        vwap=draw(prices),
        vwap_upper_1sd=draw(prices),
        vwap_lower_1sd=draw(prices),
        vwap_upper_2sd=draw(prices),
        vwap_lower_2sd=draw(prices),
        val=draw(prices),
        vah=draw(prices),
        poc=draw(prices),
        rsi_divergence_type="none",
        rsi_divergence_strength=0.0,
        funding_rate=0.0,
        oi_change_pct=0.0,
        funding_momentum=0.0,
        order_book_imbalance=0.0,
        bid_ask_spread_pct=0.0,
        ema_20=draw(prices),
    )
    
    return px, swings, bundle, atr

@given(side=st.sampled_from(["long", "short"]), inputs=geometry_inputs())
def test_geometry_invariants(side, inputs):
    px, swings, bundle, atr = inputs
    tick = 0.01 # common tick size
    mode_params = {"atr_buffer_factor": 0.5, "min_rr_ratio": 1.5, "target_cap_atr_mult": 3.0}
    
    geo = place_entry_stop_target(
        side=side,
        px=px,
        swings=swings,
        bundle=bundle,
        atr=atr,
        regime=MarketRegime.RANGE,
        style=StrategyStyle.CONSERVATIVE,
        mode_params=mode_params,
        tick=tick
    )
    
    # Invariant 1: Ordering
    if side == "long":
        assert geo.stop < geo.entry, f"Long stop {geo.stop} must be below entry {geo.entry}"
        assert geo.target > geo.entry, f"Long target {geo.target} must be above entry {geo.entry}"
    else:
        assert geo.stop > geo.entry, f"Short stop {geo.stop} must be above entry {geo.entry}"
        assert geo.target < geo.entry, f"Short target {geo.target} must be below entry {geo.entry}"
    
    # Invariant 2: Positivity
    assert geo.risk > 0, "Risk must be positive"
    assert geo.reward > 0, "Reward must be positive"
    assert geo.rr_ratio > 0, "R:R ratio must be positive"
    
    # Invariant 3: Quantization
    if tick:
        # Use a more robust check for floating point modulo
        def is_multiple(val, divisor):
            return math.isclose(val % divisor, 0, abs_tol=1e-7) or math.isclose(val % divisor, divisor, abs_tol=1e-7)
        
        assert is_multiple(geo.entry, tick), f"Entry {geo.entry} not quantized to {tick}"
        assert is_multiple(geo.stop, tick), f"Stop {geo.stop} not quantized to {tick}"
        assert is_multiple(geo.target, tick), f"Target {geo.target} not quantized to {tick}"

@given(inputs=geometry_inputs())
def test_geometry_atr_clamping(inputs):
    px, swings, bundle, _ = inputs
    # Force ATR to 0 to test clamping
    atr = 0.0
    
    geo = place_entry_stop_target(
        side="long",
        px=px,
        swings=swings,
        bundle=bundle,
        atr=atr,
        regime=MarketRegime.RANGE,
        style=StrategyStyle.CONSERVATIVE,
        mode_params={"atr_buffer_factor": 0.5, "min_rr_ratio": 1.5, "target_cap_atr_mult": 3.0},
        tick=0.01
    )
    
    # Should not crash and should have positive risk
    assert geo.risk > 0, "Risk must be positive even with 0 ATR"
    assert geo.reward > 0, "Reward must be positive even with 0 ATR"
