from __future__ import annotations

import asyncio
from datetime import datetime, UTC, timedelta
from unittest.mock import MagicMock
import pytest

from analyzer.analysis.models import Candle, MarketMode, StrategyStyle, MarketMeta
from backtest.models import BacktestConfig, BacktestReport, BacktestTrade
from backtest.evaluator import (
    evaluation_window_for_timeframe,
    resolve_outcome,
)


def test_evaluation_window_for_timeframe():
    # Scalper (< 5m)
    assert evaluation_window_for_timeframe("1m", 100) == 25
    assert evaluation_window_for_timeframe("1m", 600) == 120  # capped at 120
    assert evaluation_window_for_timeframe("1m", 40) == 20   # floored at 20
    
    # Intraday (15m - 1h)
    assert evaluation_window_for_timeframe("15m", 200) == 50
    assert evaluation_window_for_timeframe("15m", 1000) == 200  # capped at 200
    
    # Swing (4h - 1d)
    assert evaluation_window_for_timeframe("4h", 200) == 50
    assert evaluation_window_for_timeframe("4h", 600) == 100  # capped at 100


def test_resolve_outcome_empty_candles():
    trade = BacktestTrade(
        bar_time=datetime.now(UTC),
        side="long",
        entry=100.0,
        target=110.0,
        stop=95.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bullish_trend",
    )
    
    res = resolve_outcome(trade, [])
    assert res.outcome == "neither"
    assert res.exit_reason == "neither"


def test_resolve_outcome_target_and_stop_hit():
    trade = BacktestTrade(
        bar_time=datetime.now(UTC),
        side="long",
        entry=100.0,
        target=110.0,
        stop=95.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bullish_trend",
    )
    
    # Target Hit Long
    candles_target = [
        Candle(open_time=datetime.now(UTC), close_time=datetime.now(UTC), open=100.0, high=112.0, low=98.0, close=108.0, volume=100)
    ]
    res_target = resolve_outcome(trade, candles_target)
    assert res_target.outcome == "target_hit"
    assert res_target.exit_reason == "target_hit"
    assert res_target.pnl_pct == 10.0
    
    # Stop Hit Long
    trade.outcome = "unresolved"
    candles_stop = [
        Candle(open_time=datetime.now(UTC), close_time=datetime.now(UTC), open=100.0, high=102.0, low=94.0, close=96.0, volume=100)
    ]
    res_stop = resolve_outcome(trade, candles_stop)
    assert res_stop.outcome == "stop_hit"
    assert res_stop.exit_reason == "stop_hit"
    assert res_stop.pnl_pct == -5.0

    # Ambiguous Same Bar (both high >= target and low <= stop)
    trade.outcome = "unresolved"
    candles_both = [
        Candle(open_time=datetime.now(UTC), close_time=datetime.now(UTC), open=100.0, high=115.0, low=90.0, close=100.0, volume=100)
    ]
    res_both = resolve_outcome(trade, candles_both)
    assert res_both.outcome == "ambiguous_same_bar"


def test_resolve_outcome_short_exit(monkeypatch):
    import backtest.evaluator
    monkeypatch.setattr(backtest.evaluator, "_load_runtime_config_dict", lambda: {})

    trade = BacktestTrade(
        bar_time=datetime.now(UTC),
        side="short",
        entry=100.0,
        target=90.0,
        stop=105.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bearish_trend",
    )
    
    # Target Hit Short
    candles = [
        Candle(open_time=datetime.now(UTC), close_time=datetime.now(UTC), open=100.0, high=101.0, low=89.0, close=91.0, volume=100)
    ]
    res = resolve_outcome(trade, candles)
    assert res.outcome == "target_hit"
    assert res.pnl_pct == 10.0


def test_resolve_outcome_short_exit_scaled(monkeypatch):
    import backtest.evaluator
    monkeypatch.setattr(
        backtest.evaluator,
        "_load_runtime_config_dict",
        lambda: {
            "exit": {
                "mode": "scaled",
                "range": {
                    "tp1_rr": 1.0,
                    "tp2_rr": 2.0,
                    "tp1_size": 0.5,
                    "move_stop_to_be_after_tp1": False
                }
            }
        }
    )

    trade = BacktestTrade(
        bar_time=datetime.now(UTC),
        side="short",
        entry=100.0,
        target=90.0,
        stop=105.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bearish_trend",
    )
    
    # Two-candle sequence to hit TP1 first, then TP2
    candles = [
        # Candle 1: hits TP1 (95.0) but not TP2 (90.0)
        Candle(open_time=datetime.now(UTC), close_time=datetime.now(UTC), open=100.0, high=101.0, low=93.0, close=94.0, volume=100),
        # Candle 2: hits TP2 (90.0)
        Candle(open_time=datetime.now(UTC), close_time=datetime.now(UTC), open=94.0, high=95.0, low=89.0, close=90.0, volume=100),
    ]
    res = resolve_outcome(trade, candles)
    assert res.outcome == "target_hit"
    assert res.tp1_hit is True
    assert res.tp1_price == 95.0
    assert res.tp2_price == 90.0
    assert res.pnl_pct == 7.5


def test_resolve_outcome_time_stop_and_soft_stop(monkeypatch):
    import backtest.evaluator
    monkeypatch.setattr(backtest.evaluator, "_load_runtime_config_dict", lambda: {})

    now = datetime.now(UTC)
    trade = BacktestTrade(
        bar_time=now,
        side="long",
        entry=100.0,
        target=120.0,
        stop=80.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bullish_trend",
    )
    
    # Soft stop: unrealized_pnl <= -6.0 after 5 bars
    # Note: Because momentum_failure is checked earlier in the loop body (at -0.8%),
    # any trade that reaches -6.0% will mathematically trigger momentum_failure first.
    candles_soft = [
        Candle(open_time=now, close_time=now, open=100.0, high=101.0, low=99.0, close=99.0, volume=100)
        for _ in range(4)
    ] + [
        Candle(open_time=now, close_time=now, open=99.0, high=100.0, low=93.0, close=93.5, volume=100)
    ]
    res_soft = resolve_outcome(trade, candles_soft)
    assert res_soft.outcome == "momentum_failure"
    
    # Time stop: duration >= 30 and no open profit
    trade.outcome = "unresolved"
    candles_time = [
        Candle(open_time=now, close_time=now, open=100.0, high=100.5, low=99.5, close=99.9, volume=100)
        for _ in range(35)
    ]
    res_time = resolve_outcome(trade, candles_time)
    assert res_time.outcome == "time_stop"
    assert res_time.exit_reason == "time_stop"


def test_resolve_outcome_momentum_failure():
    now = datetime.now(UTC)
    trade = BacktestTrade(
        bar_time=now,
        side="long",
        entry=100.0,
        target=120.0,
        stop=80.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bullish_trend",
    )
    
    # Momentum failure: entry_momentum < -0.008 after 5 bars
    candles = [
        Candle(open_time=now, close_time=now, open=100.0, high=101.0, low=99.5, close=100.1, volume=100)
        for _ in range(4)
    ] + [
        Candle(open_time=now, close_time=now, open=100.1, high=100.2, low=98.5, close=99.0, volume=100)
    ]
    res = resolve_outcome(trade, candles)
    assert res.outcome == "momentum_failure"


def test_backtest_report_aggregates():
    config = BacktestConfig(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 10),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
    )
    
    # Empty Report
    report_empty = BacktestReport(config=config)
    report_empty.compute_aggregates()
    assert report_empty.total_trades == 0
    
    # Report with mixed trades
    t1 = BacktestTrade(
        bar_time=datetime(2026, 1, 2),
        side="long",
        entry=100.0,
        target=110.0,
        stop=95.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bullish_trend",
        outcome="target_hit",
        pnl_pct=10.0,
        mfe_pct=11.0,
        mae_pct=1.0,
    )
    t2 = BacktestTrade(
        bar_time=datetime(2026, 1, 3),
        side="long",
        entry=100.0,
        target=110.0,
        stop=95.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bullish_trend",
        outcome="stop_hit",
        pnl_pct=-5.0,
        mfe_pct=2.0,
        mae_pct=6.0,
    )
    t3 = BacktestTrade(
        bar_time=datetime(2026, 1, 4),
        side="long",
        entry=100.0,
        target=110.0,
        stop=95.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bullish_trend",
        outcome="neither",
        pnl_pct=1.0,
        mfe_pct=3.0,
        mae_pct=2.0,
    )
    
    report = BacktestReport(config=config, trades=[t1, t2, t3])
    report.compute_aggregates()
    
    assert report.total_trades == 3
    assert report.target_hit_rate == pytest.approx(1/3)
    assert report.stop_hit_rate == pytest.approx(1/3)
    assert report.win_rate == pytest.approx(2/3)  # t1 (+10) and t3 (+1) are wins
    assert report.avg_pnl_pct == pytest.approx(2.0)  # (10 - 5 + 1) / 3 = 2.0
    assert report.avg_mfe_pct == pytest.approx(16/3)  # (11 + 2 + 3) / 3 = 5.333
    assert report.avg_mae_pct == pytest.approx(3.0)  # (1 + 6 + 2) / 3 = 3.0
    assert report.expectancy == pytest.approx((2/3)*5.5 + (1/3)*(-5.0)) # wins = 10, 1 -> avg = 5.5. losses = -5 -> avg = -5. (2/3 * 5.5) + (1/3 * -5) = 3.666 - 1.666 = 2.0
    assert report.max_drawdown_pct == pytest.approx(5.0)  # DD is peak (10.0) - t2 cumulative (10 - 5 = 5.0) = 5.0
    assert report.sharpe_approx >= 0.0


def test_backtest_runner_in_memory(monkeypatch):
    from backtest.runner import BacktestRunner
    from analyzer.analysis.models import TimeframePlan, AnalysisResult, TradeSetup, QualityLabel, MarketRegime
    
    # 1. Setup mock config
    config = BacktestConfig(
        symbol="BTCUSDT",
        start=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 30, 11, 0, tzinfo=UTC),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        order_size_usd=1000.0,
        daily_volume_usd=10_000_000.0,
    )
    
    # 2. Setup mock candles (at least 35 to satisfy _MIN_BARS = 30)
    now = datetime(2026, 5, 30, 7, 0, tzinfo=UTC)
    candles = [
        Candle(open_time=now + timedelta(minutes=5*i), close_time=now + timedelta(minutes=5*i+5), open=100.0, high=101.0, low=99.0, close=100.0, volume=100)
        for i in range(60)
    ]
    
    # 3. Setup mock analyzer
    setup = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.8,
        quality_score=80.0,
        quality_label=QualityLabel.HIGH,
        rationale="test",
        is_tradable=True,
    )
    
    mock_result = AnalysisResult(
        primary_setup=setup,
        secondary_context=setup,
        timeframe_plan=TimeframePlan(profile_name="auto_core", style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY, entry_timeframe="5m", trigger_timeframe="15m", context_timeframe="1h", higher_timeframe="4h"),
        market_snapshot_meta=MarketMeta(symbol="BTCUSDT", mark_price=100.0),
        market_regime=MarketRegime.BULLISH_TREND,
    )
    
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = mock_result
    mock_analyzer.get_execution_params.return_value = {"tp_rr": 2.0}
    
    # 4. Mock the config dict inside runner.py
    monkeypatch.setattr(
        "backtest.runner._load_runtime_config_dict",
        lambda: {
            "entry_filters": {"enable_confirmation": False},
            "position_sizing": {"mode": "confidence_scaled"},
            "slippage": {"default_model": "moderate", "volatility_multipliers": {"normal": 1.0}}
        }
    )
    
    # Mock portfolio get_position_size
    monkeypatch.setattr("portfolio.get_position_size", lambda *args, **kwargs: 1.5)
    
    # 5. Run in-memory
    tfp = TimeframePlan(profile_name="test", style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY, entry_timeframe="5m", trigger_timeframe="5m", context_timeframe="5m", higher_timeframe="5m")
    runner = BacktestRunner(config)
    report = runner._run_in_memory(
        tfp=tfp,
        entry_candles=candles,
        trigger_candles=candles,
        context_candles=candles,
        higher_candles=candles,
        _analyzer=mock_analyzer
    )
    
    assert report.total_trades > 0
    assert len(report.trades) > 0
    assert report.trades[0].side == "long"
    assert report.trades[0].position_size == 1.5


def test_backtest_reporter():
    from backtest.reporter import render_backtest_text, render_walk_forward_text
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 10),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
    )
    
    t1 = BacktestTrade(
        bar_time=datetime(2026, 1, 2),
        side="long",
        entry=100.0,
        target=110.0,
        stop=95.0,
        confidence=0.8,
        quality_score=75.0,
        quality_label="high",
        regime="bullish_trend",
        outcome="target_hit",
        pnl_pct=10.0,
        mfe_pct=11.0,
        mae_pct=1.0,
    )
    # Trade with lower confidence (< 0.45)
    t2 = BacktestTrade(
        bar_time=datetime(2026, 1, 3),
        side="short",
        entry=100.0,
        target=90.0,
        stop=105.0,
        confidence=0.3,
        quality_score=50.0,
        quality_label="medium",
        regime="volatile_chop",
        outcome="stop_hit",
        pnl_pct=-5.0,
        mfe_pct=2.0,
        mae_pct=5.0,
    )
    # Trade with medium confidence (0.45 to 0.70)
    t3 = BacktestTrade(
        bar_time=datetime(2026, 1, 4),
        side="long",
        entry=100.0,
        target=110.0,
        stop=95.0,
        confidence=0.6,
        quality_score=60.0,
        quality_label="medium",
        regime="bullish_trend",
        outcome="neither",
        pnl_pct=0.0,
        mfe_pct=1.0,
        mae_pct=1.0,
    )
    
    report = BacktestReport(
        config=config, 
        trades=[t1, t2, t3],
        rejection_reasons={"rsi_overbought": 3, "min_confidence": 2},
        stop_anchor_counts={"atr_2.0": 2, "recent_low": 1},
        target_anchor_counts={"atr_3.0": 2, "recent_high": 1},
    )
    report.compute_aggregates()
    
    # Render backtest text report
    text = render_backtest_text(report, verbose=True)
    assert "Backtest Report" in text
    assert "BTCUSDT" in text
    assert "target_hit" in text
    assert "Top Rejection Reasons" in text
    assert "rsi_overbought" in text
    assert "Stop Anchors" in text
    assert "atr_2.0" in text
    assert "Target Anchors" in text
    assert "atr_3.0" in text
    
    # Render walk forward report
    wf_text = render_walk_forward_text([report])
    assert "Walk-Forward Results" in wf_text
    assert "Combined (all folds)" in wf_text


def test_fetch_range_paging(monkeypatch):
    from backtest.runner import _fetch_range
    from analyzer.analysis.models import Candle

    monkeypatch.setattr("backtest.runner._MAX_PER_REQUEST", 2)

    class MockProvider:
        def __init__(self):
            self.calls = []

        async def fetch_klines(self, symbol, interval, limit, start_time, end_time, min_required_candles):
            self.calls.append((start_time, end_time))
            if len(self.calls) == 1:
                return [
                    Candle(open_time=start_time, close_time=start_time + timedelta(minutes=5), open=100.0, high=101.0, low=99.0, close=100.0, volume=100),
                    Candle(open_time=start_time + timedelta(minutes=5), close_time=start_time + timedelta(minutes=10), open=100.0, high=101.0, low=99.0, close=100.0, volume=100)
                ]
            elif len(self.calls) == 2:
                return [
                    Candle(open_time=start_time + timedelta(milliseconds=999), close_time=start_time + timedelta(minutes=5), open=100.0, high=101.0, low=99.0, close=100.0, volume=100)
                ]
            return []

    provider = MockProvider()
    
    async def run():
        start = datetime(2026, 5, 30, 10, 0, tzinfo=UTC)
        end = datetime(2026, 5, 30, 11, 0, tzinfo=UTC)
        candles = await _fetch_range(
            provider,  # type: ignore
            symbol="BTCUSDT",
            interval="5m",
            start=start,
            end=end
        )
        assert len(candles) == 3
        assert len(provider.calls) == 2

    asyncio.run(run())


def test_backtest_runner_run(monkeypatch):
    from backtest.runner import BacktestRunner
    from backtest.models import BacktestConfig
    from analyzer.analysis.models import StrategyStyle, MarketMode, Candle
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        start=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 30, 11, 0, tzinfo=UTC),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
    )
    
    runner = BacktestRunner(config)
    
    now = datetime(2026, 5, 30, 7, 0, tzinfo=UTC)
    mock_candles = [
        Candle(open_time=now + timedelta(minutes=5*i), close_time=now + timedelta(minutes=5*i+5), open=100.0, high=101.0, low=99.0, close=100.0, volume=100)
        for i in range(60)
    ]
    
    async def mock_fetch_range_fn(*args, **kwargs):
        return mock_candles
        
    monkeypatch.setattr("backtest.runner._fetch_range", mock_fetch_range_fn)
    
    from analyzer.analysis.models import AnalysisResult, TradeSetup, QualityLabel, MarketRegime, TimeframePlan, MarketMeta
    setup = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.8,
        quality_score=80.0,
        quality_label=QualityLabel.HIGH,
        rationale="test",
        is_tradable=True,
    )
    
    mock_result = AnalysisResult(
        primary_setup=setup,
        secondary_context=setup,
        timeframe_plan=TimeframePlan(profile_name="auto_core", style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY, entry_timeframe="5m", trigger_timeframe="15m", context_timeframe="1h", higher_timeframe="4h"),
        market_snapshot_meta=MarketMeta(symbol="BTCUSDT", mark_price=100.0),
        market_regime=MarketRegime.BULLISH_TREND,
    )
    
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance.analyze.return_value = mock_result
    mock_analyzer_instance.get_execution_params.return_value = {"tp_rr": 2.0}
    
    monkeypatch.setattr("backtest.runner.SetupAnalyzer", lambda *args, **kwargs: mock_analyzer_instance)
    
    async def mock_aclose(self):
        pass
    monkeypatch.setattr("providers.binance_futures.BinanceFuturesProvider.aclose", mock_aclose)
    
    async def run():
        report = await runner.run(progress=True)
        assert report.total_trades > 0
        
    asyncio.run(run())


def test_backtest_runner_run_exception(monkeypatch):
    from backtest.runner import BacktestRunner
    from backtest.models import BacktestConfig
    from analyzer.analysis.models import StrategyStyle, MarketMode
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        start=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 30, 11, 0, tzinfo=UTC),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
    )
    
    runner = BacktestRunner(config)
    
    async def mock_fetch_range_error(*args, **kwargs):
        raise ValueError("Simulated network/fetch failure")
        
    monkeypatch.setattr("backtest.runner._fetch_range", mock_fetch_error_helper(mock_fetch_range_error))
    
    async def mock_aclose(self):
        pass
    monkeypatch.setattr("providers.binance_futures.BinanceFuturesProvider.aclose", mock_aclose)
    
    async def run():
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            await runner.run(progress=True)
            
    asyncio.run(run())


def mock_fetch_error_helper(func):
    return func


def test_backtest_runner_walk_forward(monkeypatch):
    from backtest.runner import BacktestRunner
    from backtest.models import BacktestConfig
    from analyzer.analysis.models import StrategyStyle, MarketMode, Candle
    
    config = BacktestConfig(
        symbol="BTCUSDT",
        start=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 30, 11, 0, tzinfo=UTC),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
    )
    
    runner = BacktestRunner(config)
    
    now = datetime(2026, 5, 30, 7, 0, tzinfo=UTC)
    mock_candles = [
        Candle(open_time=now + timedelta(minutes=5*i), close_time=now + timedelta(minutes=5*i+5), open=100.0, high=101.0, low=99.0, close=100.0, volume=100)
        for i in range(60)
    ]
    
    async def mock_fetch_range_fn(*args, **kwargs):
        return mock_candles
        
    monkeypatch.setattr("backtest.runner._fetch_range", mock_fetch_range_fn)
    
    from analyzer.analysis.models import AnalysisResult, TradeSetup, QualityLabel, MarketRegime, TimeframePlan, MarketMeta
    setup = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.8,
        quality_score=80.0,
        quality_label=QualityLabel.HIGH,
        rationale="test",
        is_tradable=True,
    )
    
    mock_result = AnalysisResult(
        primary_setup=setup,
        secondary_context=setup,
        timeframe_plan=TimeframePlan(profile_name="auto_core", style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY, entry_timeframe="5m", trigger_timeframe="15m", context_timeframe="1h", higher_timeframe="4h"),
        market_snapshot_meta=MarketMeta(symbol="BTCUSDT", mark_price=100.0),
        market_regime=MarketRegime.BULLISH_TREND,
    )
    
    mock_analyzer_instance = MagicMock()
    mock_analyzer_instance.analyze.return_value = mock_result
    mock_analyzer_instance.get_execution_params.return_value = {"tp_rr": 2.0}
    
    monkeypatch.setattr("backtest.runner.SetupAnalyzer", lambda *args, **kwargs: mock_analyzer_instance)
    
    async def mock_aclose(self):
        pass
    monkeypatch.setattr("providers.binance_futures.BinanceFuturesProvider.aclose", mock_aclose)
    
    async def run():
        reports = await runner.walk_forward(folds=3, progress=True)
        assert len(reports) == 3
        assert all(r.total_trades > 0 for r in reports)
        
        # Test degenerate fold length skip
        skipped_reports = await runner.walk_forward(folds=100, progress=True)
        assert len(skipped_reports) == 0
        
        # Test degenerate fold exception
        with pytest.raises(ValueError, match="requires at least 2 folds"):
            await runner.walk_forward(folds=1)
            
    asyncio.run(run())


def test_backtest_runner_edge_cases(monkeypatch):
    from backtest.runner import BacktestRunner, _slice_to_anchor
    from backtest.models import BacktestConfig
    from analyzer.analysis.models import StrategyStyle, MarketMode, Candle
    import pathlib
    
    # 1. Test _load_runtime_config_dict exception logic
    original_read_text = pathlib.Path.read_text
    def mock_read_text(*args, **kwargs):
        raise ValueError("Simulated read error")
    monkeypatch.setattr(pathlib.Path, "read_text", mock_read_text)
    
    from backtest.runner import _load_runtime_config_dict
    _load_runtime_config_dict.cache_clear()
    assert _load_runtime_config_dict() == {}
    
    # Mock runtime config dictionary
    _load_runtime_config_dict.cache_clear()
    monkeypatch.setattr(pathlib.Path, "read_text", lambda *args, **kwargs: '{"entry_filters": {"enable_confirmation": true, "confirmation_candles": "invalid_int"}, "position_sizing": {"mode": "confidence_scaled"}}')
    
    # 2. Test presets and preset exception
    config_valid_preset = BacktestConfig(
        symbol="BTCUSDT",
        start=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 30, 11, 0, tzinfo=UTC),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        preset="scalper"
    )
    runner_valid_preset = BacktestRunner(config_valid_preset)
    tfp_valid = runner_valid_preset._build_timeframe_plan()
    assert tfp_valid.profile_name == "Scalper"
    
    config_invalid_preset = BacktestConfig(
        symbol="BTCUSDT",
        start=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
        end=datetime(2026, 5, 30, 11, 0, tzinfo=UTC),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        preset="invalid_preset_name"
    )
    runner_invalid_preset = BacktestRunner(config_invalid_preset)
    tfp_invalid = runner_invalid_preset._build_timeframe_plan()
    assert tfp_invalid.profile_name != "invalid_preset_name"

    # 3. Test _slice_to_anchor where end <= 0
    t = datetime(2026, 5, 30, 10, 0, tzinfo=UTC)
    assert _slice_to_anchor([], [], t, 10) == []
    
    # 4. Standard running of run in memory with various specific mock setups to hit edge cases:
    # Setup runner config with start close to first candle to trigger history skip
    now = datetime(2026, 5, 30, 7, 0, tzinfo=UTC)
    config = BacktestConfig(
        symbol="BTCUSDT",
        start=now + timedelta(minutes=5),  # very early, triggers insufficient history skip
        end=now + timedelta(hours=4),
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        order_size_usd=1000.0,
    )
    runner = BacktestRunner(config)
    
    # 60 candles (each 5 min)
    candles = [
        Candle(open_time=now + timedelta(minutes=5*i), close_time=now + timedelta(minutes=5*i+5), open=100.0, high=101.0, low=99.0, close=100.0, volume=100)
        for i in range(60)
    ]
    
    from analyzer.analysis.models import AnalysisResult, TradeSetup, QualityLabel, MarketRegime, TimeframePlan, MarketMeta
    
    analyze_calls = 0
    def custom_analyze(symbol, entry_candles, trigger_candles, context_candles, higher_candles, market, timeframe_plan, long_log):
        nonlocal analyze_calls
        analyze_calls += 1
        
        # Call 1: raise exception to cover bars_skipped_analysis_error
        if analyze_calls == 1:
            raise RuntimeError("Analysis failure for testing")
            
        # Call 2: is_tradable = False, side = "long"
        if analyze_calls == 2:
            setup = TradeSetup(
                side="long",
                entry_price=100.0,
                target_price=110.0,
                stop_loss=95.0,
                confidence=0.8,
                quality_score=80.0,
                quality_label=QualityLabel.HIGH,
                rationale="test",
                is_tradable=False,
                tradable_reasons=["spread_too_wide"]
            )
        # Call 3: side = "short", confirmation check (will fail because close times are flat/equal)
        elif analyze_calls == 3:
            setup = TradeSetup(
                side="short",
                entry_price=100.0,
                target_price=90.0,
                stop_loss=105.0,
                confidence=0.8,
                quality_score=80.0,
                quality_label=QualityLabel.HIGH,
                rationale="test",
                is_tradable=True,
            )
        # Call 4: get_position_size throws exception, and capped target applies, and slippage calculation throws
        else:
            setup = TradeSetup(
                side="long",
                entry_price=100.0,
                target_price=200.0, # large target to trigger capped_target
                stop_loss=95.0,
                confidence=0.8,
                quality_score=80.0,
                quality_label=QualityLabel.HIGH,
                rationale="test",
                is_tradable=True,
            )
            
        return AnalysisResult(
            primary_setup=setup,
            secondary_context=setup,
            timeframe_plan=TimeframePlan(profile_name="auto_core", style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY, entry_timeframe="5m", trigger_timeframe="15m", context_timeframe="1h", higher_timeframe="4h"),
            market_snapshot_meta=MarketMeta(symbol="BTCUSDT", mark_price=100.0),
            market_regime=MarketRegime.BULLISH_TREND,
        )
        
    mock_analyzer = MagicMock()
    mock_analyzer.analyze = custom_analyze
    mock_analyzer.get_execution_params.return_value = {"tp_rr": 2.0}
    
    # Mock get_position_size to raise an exception
    def mock_get_position_size(*args, **kwargs):
        raise ValueError("Position size error")
    monkeypatch.setattr("portfolio.get_position_size", mock_get_position_size)
    
    # Mock SlippageCalculator.calculate_slippage to raise an exception
    from history.evaluation import SlippageCalculator
    def mock_calculate_slippage(*args, **kwargs):
        raise ValueError("Slippage error")
    monkeypatch.setattr(SlippageCalculator, "calculate_slippage", mock_calculate_slippage)
    
    tfp = TimeframePlan(profile_name="test", style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY, entry_timeframe="5m", trigger_timeframe="5m", context_timeframe="5m", higher_timeframe="5m")
    
    report = runner._run_in_memory(
        tfp=tfp,
        entry_candles=candles,
        trigger_candles=candles,
        context_candles=candles,
        higher_candles=candles,
        progress=True,
        _analyzer=mock_analyzer
    )
    
    assert report.bars_skipped_insufficient_history > 0
    assert report.bars_skipped_analysis_error == 1
    assert "spread_too_wide" in report.rejection_reasons
    assert report.rejection_reasons["confirmation_failed"] == 1
    
    # 5. Test walk_forward exception in one fold fetch
    async def mock_fetch_range_error(*args, **kwargs):
        raise ValueError("Walk forward fetch error")
    
    monkeypatch.setattr("backtest.runner._fetch_range", mock_fetch_range_error)
    async def mock_aclose(self):
        pass
    monkeypatch.setattr("providers.binance_futures.BinanceFuturesProvider.aclose", mock_aclose)
    
    async def run_wf():
        with pytest.raises(RuntimeError, match="Failed to fetch"):
            await runner.walk_forward(folds=3)
            
    asyncio.run(run_wf())


