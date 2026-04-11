from futures_analyzer.analysis import MarketMode, StrategyStyle, build_timeframe_plan



def test_timeframe_plan_auto_profile() -> None:
    plan = build_timeframe_plan()
    assert plan.profile_name == "intraday_core"
    assert plan.style == StrategyStyle.CONSERVATIVE
    assert plan.market_mode == MarketMode.INTRADAY
    assert plan.entry_timeframe == "5m"
    assert plan.trigger_timeframe == "15m"
    assert plan.context_timeframe == "1h"
    assert plan.higher_timeframe == "4h"
    assert plan.lookback_bars == 600
