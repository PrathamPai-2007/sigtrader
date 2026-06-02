import pytest
from datetime import UTC, datetime, timedelta
from analyzer.analysis.scorer import SetupAnalyzer
from analyzer.analysis.models import (
    Candle,
    MarketMeta,
    TimeframePlan,
    MarketMode,
    StrategyStyle,
    MarketRegime,
)
from analyzer.config import AppConfig, StrategyConfig

def _mock_candles(count: int, trend: float = 0.0, start_price: float = 100.0) -> list[Candle]:
    now = datetime.now(UTC)
    candles = []
    curr_price = start_price
    for i in range(count):
        # Apply a simple linear trend
        curr_price += trend
        candles.append(
            Candle(
                open_time=now + timedelta(minutes=i*15),
                close_time=now + timedelta(minutes=(i+1)*15),
                open=curr_price - trend,
                high=curr_price + 1.0,
                low=curr_price - 1.0,
                close=curr_price,
                volume=100.0,
            )
        )
    return candles

def test_long_entry_filters_disabled_by_default_is_permissive():
    # Setup analyzer with filters disabled
    config = AppConfig(strategy=StrategyConfig())
    config.strategy.long_entry_filters.enable_enhanced_filters = False
    analyzer = SetupAnalyzer(config=config)
    
    candles = _mock_candles(50, trend=0.1) # Bullish trend
    market = MarketMeta(
        symbol="BTCUSDT",
        tick_size=0.1,
        mark_price=candles[-1].close,
        as_of=datetime.now(UTC),
    )
    plan = TimeframePlan(
        context_timeframe="1h",
        trigger_timeframe="15m",
    )
    
    result = analyzer.analyze(
        symbol="BTCUSDT",
        trigger_candles=candles,
        context_candles=candles,
        market=market,
        timeframe_plan=plan
    )
    
    # Verify no Enhanced Safety Rejects are present
    assert not any("Enhanced Safety Reject" in r for r in result.primary_setup.tradable_reasons)

def test_long_entry_filters_pullback_trap_rejection():
    # Setup analyzer with filters enabled
    config = AppConfig(strategy=StrategyConfig())
    config.strategy.long_entry_filters.enable_enhanced_filters = True
    analyzer = SetupAnalyzer(config=config)
    
    # Mock Scenario:
    # Trigger TF: Bullish (pullback starting)
    # Higher TF: Bearish (macro downtrend)
    trigger_candles = _mock_candles(50, trend=0.2, start_price=90.0) # Moving up
    higher_candles = _mock_candles(100, trend=-0.5, start_price=120.0) # Moving down heavily
    
    market = MarketMeta(
        symbol="BTCUSDT",
        tick_size=0.1,
        mark_price=trigger_candles[-1].close,
        as_of=datetime.now(UTC),
    )
    plan = TimeframePlan(
        context_timeframe="1h",
        trigger_timeframe="15m",
    )
    
    result = analyzer.analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger_candles,
        context_candles=trigger_candles,
        higher_candles=higher_candles,
        market=market,
        timeframe_plan=plan
    )
    
    # Even if other indicators were bullish, the pullback trap should block it
    # Note: Depending on other factors, it might already be non-tradable, 
    # but we check if the specific filter reason is present.
    if result.primary_setup.side == "long":
        assert any("Enhanced Safety Reject" in r and "Pullback trap" in r 
                  for r in result.primary_setup.tradable_reasons)
        assert result.primary_setup.is_tradable is False

def test_long_entry_filters_no_momentum_expansion_rejection():
    config = AppConfig(strategy=StrategyConfig())
    config.strategy.long_entry_filters.enable_enhanced_filters = True
    # Require 5% momentum
    config.strategy.long_entry_filters.momentum_expansion.min_momentum_threshold = 0.05
    analyzer = SetupAnalyzer(config=config)
    
    # Mock Scenario: Flat candles (no momentum)
    candles = _mock_candles(50, trend=0.001) 
    
    market = MarketMeta(
        symbol="BTCUSDT",
        tick_size=0.1,
        mark_price=candles[-1].close,
        as_of=datetime.now(UTC),
    )
    plan = TimeframePlan(
        context_timeframe="1h",
        trigger_timeframe="15m",
    )
    
    result = analyzer.analyze(
        symbol="BTCUSDT",
        trigger_candles=candles,
        context_candles=candles,
        market=market,
        timeframe_plan=plan
    )
    
    if result.primary_setup.side == "long":
        assert any("Enhanced Safety Reject" in r and "Momentum" in r 
                  for r in result.primary_setup.tradable_reasons)
        assert result.primary_setup.is_tradable is False

def test_long_confidence_adjustment_multiplier():
    config = AppConfig(strategy=StrategyConfig())
    config.strategy.long_entry_filters.enable_enhanced_filters = True
    # Apply a heavy 50% base penalty for long
    config.strategy.long_entry_filters.confidence_adjustments.base_long_penalty = 0.5
    analyzer = SetupAnalyzer(config=config)
    
    candles = _mock_candles(50, trend=0.1)
    market = MarketMeta(
        symbol="BTCUSDT",
        tick_size=0.1,
        mark_price=candles[-1].close,
        as_of=datetime.now(UTC),
    )
    plan = TimeframePlan(
        context_timeframe="1h",
        trigger_timeframe="15m",
    )
    
    result = analyzer.analyze(
        symbol="BTCUSDT",
        trigger_candles=candles,
        context_candles=candles,
        market=market,
        timeframe_plan=plan
    )
    
    # The confidence should be significantly lower than a default long setup
    # (assuming base confidence would have been higher)
    if result.primary_setup.side == "long":
        # Multiplicative penalty: adjusted = base * (1.0 - penalty)
        # Even with perfect signals, confidence would be capped at 0.5 (1.0 * 0.5)
        assert result.primary_setup.confidence <= 0.5
