import pytest
from datetime import UTC, datetime
from analyzer.analysis.scorer import SetupAnalyzer
from analyzer.analysis.models import (
    Candle,
    MarketMeta,
    TimeframePlan,
    MarketMode,
    StrategyStyle,
)

def test_analyze_handles_missing_market_data():
    # Setup mock data with missing funding/OI
    now = datetime.now(UTC)
    candles = [
        Candle(open_time=now, close_time=now, open=100, high=101, low=99, close=100.5, volume=100)
        for _ in range(50)
    ]
    
    market = MarketMeta.model_construct(
        symbol="BTCUSDT",
        tick_size=0.1,
        step_size=0.001,
        mark_price=100.0,
        funding_rate=None, # Missing
        funding_rate_history=None, # Missing - This is what causes the crash if not guarded
        open_interest=None, # Missing
        open_interest_change_pct=None, # Missing
        as_of=now,
    )
    
    plan = TimeframePlan(
        profile_name="test",
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        entry_timeframe="5m",
        trigger_timeframe="15m",
        context_timeframe="1h",
        higher_timeframe="4h",
        lookback_bars=100,
    )
    
    analyzer = SetupAnalyzer()
    
    # This should NOT crash
    result = analyzer.analyze(
        symbol="BTCUSDT",
        trigger_candles=candles,
        context_candles=candles,
        entry_candles=candles,
        higher_candles=candles,
        market=market,
        timeframe_plan=plan
    )
    
    # It should contain the Partial Evidence warning
    assert any("Partial Evidence" in w for w in result.warnings)
    # Result should still be valid
    assert result.primary_setup.confidence >= 0
