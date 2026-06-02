from __future__ import annotations

import asyncio
from datetime import datetime, UTC
import pytest
from unittest.mock import AsyncMock, MagicMock

from analyzer.analysis.models import (
    Candle,
    MarketMeta,
    QualityLabel,
    TradeSetup,
)
from history.evaluation import SlippageModel
from market.models import CorrelationPair, CorrelationReport
from market.correlation import (
    _pearson,
    _log_returns,
    _align_series,
    _cluster,
    CorrelationAnalyzer,
)
from market.slippage import SlippageAdvisor, _raw_rr


def test_pearson_correlation():
    # Identical series -> 1.0
    assert pytest.approx(_pearson([1, 2, 3], [1, 2, 3])) == 1.0
    # Negative series -> -1.0
    assert pytest.approx(_pearson([1, 2, 3], [3, 2, 1])) == -1.0
    # Neutral/Zero -> 0.0
    assert pytest.approx(_pearson([1, 1, 1], [2, 3, 4])) == 0.0
    # Length < 2 -> 0.0
    assert _pearson([1], [2]) == 0.0


def test_log_returns():
    closes = [100.0, 110.0, 99.0]
    # log(110/100) = 0.09531, log(99/110) = -0.10536
    rets = _log_returns(closes)
    assert len(rets) == 2
    assert pytest.approx(rets[0], rel=1e-4) == 0.09531
    assert pytest.approx(rets[1], rel=1e-4) == -0.10536


def test_align_series():
    t1 = datetime(2026, 1, 1)
    t2 = datetime(2026, 1, 2)
    t3 = datetime(2026, 1, 3)
    
    series = {
        "BTC": [(t1, 100.0), (t2, 101.0), (t3, 102.0)],
        "ETH": [(t2, 50.0), (t3, 51.0), (t1, 49.0)],
    }
    
    aligned = _align_series(series)
    assert len(aligned["BTC"]) == 3
    assert len(aligned["ETH"]) == 3
    assert aligned["BTC"] == [100.0, 101.0, 102.0]
    assert aligned["ETH"] == [49.0, 50.0, 51.0]  # sorted by timestamp t1, t2, t3


def test_clustering():
    symbols = ["BTC", "ETH", "SOL", "ADA"]
    pairs = [
        CorrelationPair(symbol_a="BTC", symbol_b="ETH", correlation=0.85, window_bars=10, interval="1h"),
        CorrelationPair(symbol_a="ETH", symbol_b="SOL", correlation=0.80, window_bars=10, interval="1h"),
        CorrelationPair(symbol_a="ADA", symbol_b="SOL", correlation=0.20, window_bars=10, interval="1h"),
    ]
    clusters = _cluster(symbols, pairs, threshold=0.75)
    # BTC, ETH, SOL should be clustered together; ADA should be isolated (omitted from output groups since len < 2)
    assert len(clusters) == 1
    assert set(clusters[0]) == {"BTC", "ETH", "SOL"}


def test_correlation_analyzer_analyze(monkeypatch):
    analyzer = CorrelationAnalyzer(window_bars=5)
    
    t1 = datetime(2026, 1, 1)
    t2 = datetime(2026, 1, 2)
    t3 = datetime(2026, 1, 3)
    
    # Mock _fetch_series instead of hitting network
    async def mock_fetch(symbols, provider):
        return {
            "BTCUSDT": [(t1, 100.0), (t2, 101.0), (t3, 102.0)],
            "ETHUSDT": [(t1, 10.0), (t2, 10.1), (t3, 10.2)],
        }
    monkeypatch.setattr(analyzer, "_fetch_series", mock_fetch)
    
    async def run():
        report = await analyzer.analyze(["BTCUSDT", "ETHUSDT"])
        assert len(report.symbols) == 2
        assert len(report.pairs) == 1
        assert report.pairs[0].symbol_a == "BTCUSDT"
        assert report.pairs[0].symbol_b == "ETHUSDT"
        assert report.pairs[0].correlation > 0.9  # Highly correlated returns
        
    asyncio.run(run())


def test_warn_concentrated_setups():
    analyzer = CorrelationAnalyzer()
    report = CorrelationReport(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        interval="1h",
        window_bars=100,
        pairs=[
            CorrelationPair(symbol_a="BTCUSDT", symbol_b="ETHUSDT", correlation=0.85, window_bars=100, interval="1h"),
            CorrelationPair(symbol_a="BTCUSDT", symbol_b="SOLUSDT", correlation=-0.70, window_bars=100, interval="1h"),
        ]
    )
    
    warns = analyzer.warn_concentrated_setups(report, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert len(warns) == 2
    assert any("warning" in w for w in warns)  # for positive correlation
    assert any("Hedge" in w for w in warns)    # for negative correlation


def test_raw_rr():
    setup_long = TradeSetup(
        side="long", entry_price=100.0, target_price=110.0, stop_loss=95.0, confidence=0.5, rationale=""
    )
    setup_short = TradeSetup(
        side="short", entry_price=100.0, target_price=90.0, stop_loss=105.0, confidence=0.5, rationale=""
    )
    assert _raw_rr(setup_long) == 2.0
    assert _raw_rr(setup_short) == 2.0


def test_slippage_advisor_estimate(monkeypatch):
    advisor = SlippageAdvisor(model=SlippageModel.MODERATE, min_viable_rr=1.5)
    
    setup = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.5,
        quality_score=50.0,
        quality_label=QualityLabel.MEDIUM,
        rationale="",
    )
    
    market = MarketMeta(symbol="BTCUSDT", mark_price=100.0)
    
    # Mock the EnhancedDataProvider
    mock_ob = MagicMock()
    mock_ob.spread_pct = 0.02
    mock_ob.bid_volume = 1000.0
    mock_ob.ask_volume = 1000.0
    
    mock_enhanced = AsyncMock()
    mock_enhanced.fetch_order_book_snapshot.return_value = mock_ob
    
    # Patch EnhancedDataProvider class at its source import location
    monkeypatch.setattr("providers.microstructure.EnhancedDataProvider", lambda *args, **kwargs: mock_enhanced)
    
    async def run():
        report = await advisor.estimate(setup, market, order_size_usd=1000.0)
        assert report.symbol == "BTCUSDT"
        assert report.is_still_viable is True
        assert report.raw_rr == 2.0
        assert report.adj_entry > 100.0  # entry is slipped up for long
        assert report.adj_target < 110.0  # target is slipped down
        assert report.adj_stop < 95.0  # stop is slipped down

    asyncio.run(run())


def test_estimate_from_params(monkeypatch):
    advisor = SlippageAdvisor()
    
    mock_ob = MagicMock()
    mock_ob.spread_pct = 0.01
    mock_ob.bid_volume = 5000.0
    mock_ob.ask_volume = 5000.0
    
    mock_enhanced = AsyncMock()
    mock_enhanced.fetch_order_book_snapshot.return_value = mock_ob
    monkeypatch.setattr("providers.microstructure.EnhancedDataProvider", lambda *args, **kwargs: mock_enhanced)
    
    async def run():
        report = await advisor.estimate_from_params(
            symbol="BTCUSDT",
            side="long",
            entry=100.0,
            target=110.0,
            stop=95.0,
            order_size_usd=1000.0,
        )
        assert report.symbol == "BTCUSDT"
        assert report.raw_entry == 100.0
        assert report.adj_rr < report.raw_rr

    asyncio.run(run())


def test_vol_multiplier():
    advisor = SlippageAdvisor()
    assert advisor._vol_multiplier("low") == 0.7
    assert advisor._vol_multiplier("normal") == 1.0
    assert advisor._vol_multiplier("high") == 1.4
    assert advisor._vol_multiplier("extreme") == 2.0
    assert advisor._vol_multiplier("unknown") == 1.0


def test_slippage_advisor_estimate_order_book_failure(monkeypatch):
    advisor = SlippageAdvisor(model=SlippageModel.MODERATE, min_viable_rr=1.5)
    
    setup = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.5,
        quality_score=50.0,
        quality_label=QualityLabel.MEDIUM,
        rationale="",
    )
    market = MarketMeta(symbol="BTCUSDT", mark_price=100.0)
    
    mock_enhanced = AsyncMock()
    mock_enhanced.fetch_order_book_snapshot.side_effect = Exception("Simulated order book error")
    
    monkeypatch.setattr("providers.microstructure.EnhancedDataProvider", lambda *args, **kwargs: mock_enhanced)
    
    async def run():
        report = await advisor.estimate(setup, market, order_size_usd=1000.0)
        assert report.symbol == "BTCUSDT"
        assert report.spread_pct == 0.05  # fallback spread
        assert report.liquidity_score == 50.0  # fallback liquidity score
        assert report.volatility_regime == "normal"
        
    asyncio.run(run())


def test_slippage_advisor_estimate_with_candles_success(monkeypatch):
    advisor = SlippageAdvisor(model=SlippageModel.MODERATE, min_viable_rr=1.0)
    
    setup = TradeSetup(
        side="short",
        entry_price=100.0,
        target_price=90.0,
        stop_loss=105.0,
        confidence=0.5,
        quality_score=50.0,
        quality_label=QualityLabel.MEDIUM,
        rationale="",
    )
    market = MarketMeta(symbol="BTCUSDT", mark_price=100.0)
    
    mock_ob = MagicMock()
    mock_ob.spread_pct = 0.02
    mock_ob.bid_volume = 1000.0
    mock_ob.ask_volume = 1000.0
    
    mock_vol = MagicMock()
    mock_vol.volatility_regime = "high"
    
    mock_liq = MagicMock()
    mock_liq.liquidity_score = 75.0
    mock_liq.bid_ask_spread_pct = 0.015
    
    mock_enhanced = AsyncMock()
    mock_enhanced.fetch_order_book_snapshot.return_value = mock_ob
    mock_enhanced.fetch_volatility_metrics.return_value = mock_vol
    mock_enhanced.fetch_liquidity_metrics.return_value = mock_liq
    
    monkeypatch.setattr("providers.microstructure.EnhancedDataProvider", lambda *args, **kwargs: mock_enhanced)
    
    async def run():
        mock_candles = [MagicMock()]
        report = await advisor.estimate(
            setup, 
            market, 
            order_size_usd=1000.0, 
            daily_volume_usd=5_000_000.0, 
            candles=mock_candles
        )
        assert report.symbol == "BTCUSDT"
        assert report.volatility_regime == "high"
        assert report.liquidity_score == 75.0
        assert report.spread_pct == 0.015
        
    asyncio.run(run())


def test_slippage_advisor_estimate_with_candles_failure(monkeypatch):
    advisor = SlippageAdvisor(model=SlippageModel.MODERATE, min_viable_rr=1.0)
    
    setup = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.5,
        quality_score=50.0,
        quality_label=QualityLabel.MEDIUM,
        rationale="",
    )
    market = MarketMeta(symbol="BTCUSDT", mark_price=100.0)
    
    mock_ob = MagicMock()
    mock_ob.spread_pct = 0.02
    mock_ob.bid_volume = 1000.0
    mock_ob.ask_volume = 1000.0
    
    mock_enhanced = AsyncMock()
    mock_enhanced.fetch_order_book_snapshot.return_value = mock_ob
    mock_enhanced.fetch_volatility_metrics.side_effect = Exception("Simulated volatility error")
    
    monkeypatch.setattr("providers.microstructure.EnhancedDataProvider", lambda *args, **kwargs: mock_enhanced)
    
    async def run():
        mock_candles = [MagicMock()]
        report = await advisor.estimate(
            setup, 
            market, 
            order_size_usd=1000.0, 
            daily_volume_usd=5_000_000.0, 
            candles=mock_candles
        )
        assert report.symbol == "BTCUSDT"
        # Since fetch_volatility_metrics failed, it should fall back to default regime
        assert report.volatility_regime == "normal"
        
    asyncio.run(run())

