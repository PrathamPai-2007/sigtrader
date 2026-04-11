import asyncio
from datetime import UTC, datetime, timedelta

from futures_analyzer.analysis import MarketMode, StrategyStyle
from futures_analyzer.analysis.models import AnalysisResult, Candle, MarketMeta, MarketRegime, QualityLabel, TimeframePlan, TradeSetup
from futures_analyzer.analysis.replay import find_latest_tradable_chart_timestamp


def _build_candles(start: datetime, *, interval_minutes: int, count: int, base_price: float) -> list[Candle]:
    candles: list[Candle] = []
    for idx in range(count):
        open_time = start + timedelta(minutes=interval_minutes * idx)
        candles.append(
            Candle(
                open_time=open_time,
                close_time=open_time + timedelta(minutes=interval_minutes),
                open=base_price + (idx * 0.1),
                high=base_price + (idx * 0.1) + 0.5,
                low=base_price + (idx * 0.1) - 0.5,
                close=base_price + (idx * 0.1) + 0.2,
                volume=100.0 + idx,
            )
        )
    return candles


def _result(tradable: bool, timeframe_plan: TimeframePlan, market: MarketMeta) -> AnalysisResult:
    primary = TradeSetup(
        side="long",
        entry_price=market.mark_price,
        target_price=market.mark_price * 1.02,
        stop_loss=market.mark_price * 0.99,
        confidence=0.8,
        quality_label=QualityLabel.HIGH,
        quality_score=80.0,
        rationale="replay test",
        risk_reward_ratio=2.0,
        stop_distance_pct=1.0,
        target_distance_pct=2.0,
        atr_multiple_to_stop=1.0,
        atr_multiple_to_target=2.0,
        invalidation_strength=0.8,
        is_tradable=tradable,
        evidence_agreement=6,
        evidence_total=7,
        deliberation_summary="6/7 aligned",
    )
    secondary = primary.model_copy(update={"side": "short", "is_tradable": False, "quality_label": QualityLabel.MEDIUM})
    return AnalysisResult(
        primary_setup=primary,
        secondary_context=secondary,
        timeframe_plan=timeframe_plan,
        market_snapshot_meta=market,
        market_regime=MarketRegime.BULLISH_TREND,
        regime_confidence=0.8,
    )


class _ReplayProvider:
    def __init__(self, candles_by_interval: dict[str, list[Candle]]) -> None:
        self.candles_by_interval = candles_by_interval
        self.context_requests: list[tuple[str, datetime, str]] = []

    async def fetch_klines(self, *, symbol: str, interval: str, limit: int = 300, start_time=None, end_time=None, min_required_candles: int = 30):
        candles = self.candles_by_interval[interval]
        if end_time is not None:
            candles = [candle for candle in candles if candle.open_time <= end_time]
        return candles[-limit:]

    async def fetch_historical_market_context(self, *, symbol: str, as_of: datetime, interval: str = "5m"):
        self.context_requests.append((symbol, as_of, interval))
        return (0.0002, 2000.0, 1.5)


def test_replay_returns_latest_tradable_trigger_timestamp(monkeypatch) -> None:
    timeframe_plan = TimeframePlan(
        profile_name="intraday_core",
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        entry_timeframe="5m",
        trigger_timeframe="15m",
        context_timeframe="1h",
        higher_timeframe="4h",
        lookback_bars=40,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger_candles = _build_candles(start, interval_minutes=15, count=70, base_price=100.0)
    target_anchor = trigger_candles[-4].close_time
    provider = _ReplayProvider(
        {
            "5m": _build_candles(start - timedelta(hours=12), interval_minutes=5, count=160, base_price=99.0),
            "15m": trigger_candles,
            "1h": _build_candles(start - timedelta(hours=35), interval_minutes=60, count=45, base_price=98.0),
            "4h": _build_candles(start - timedelta(hours=132), interval_minutes=240, count=35, base_price=97.0),
        }
    )
    market = MarketMeta(symbol="BTCUSDT", tick_size=0.1, step_size=0.001, mark_price=100.0, as_of=trigger_candles[-1].close_time)

    def fake_analyze(self, *, symbol: str, trigger_candles, context_candles, market, timeframe_plan, entry_candles=None, higher_candles=None):
        return _result(market.as_of <= target_anchor, timeframe_plan, market)

    monkeypatch.setattr("futures_analyzer.analysis.replay.SetupAnalyzer.analyze", fake_analyze)

    replay_time = asyncio.run(
        find_latest_tradable_chart_timestamp(
            provider=provider,
            symbol="BTCUSDT",
            market=market,
            timeframe_plan=timeframe_plan,
            risk_reward=0.7,
            style=StrategyStyle.CONSERVATIVE,
            market_mode=MarketMode.INTRADAY,
            replay_trigger_bars=40,
        )
    )

    assert replay_time == target_anchor
    assert provider.context_requests
    assert provider.context_requests[0][2] == "15m"


def test_replay_passes_historical_market_context_into_analyzer(monkeypatch) -> None:
    timeframe_plan = TimeframePlan(
        profile_name="intraday_core",
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        entry_timeframe="5m",
        trigger_timeframe="15m",
        context_timeframe="1h",
        higher_timeframe="4h",
        lookback_bars=40,
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    provider = _ReplayProvider(
        {
            "5m": _build_candles(start - timedelta(hours=12), interval_minutes=5, count=160, base_price=99.0),
            "15m": _build_candles(start, interval_minutes=15, count=70, base_price=100.0),
            "1h": _build_candles(start - timedelta(hours=35), interval_minutes=60, count=45, base_price=98.0),
            "4h": _build_candles(start - timedelta(hours=132), interval_minutes=240, count=35, base_price=97.0),
        }
    )
    market = MarketMeta(symbol="BTCUSDT", tick_size=0.1, step_size=0.001, mark_price=100.0, as_of=start)
    seen_market: dict[str, float | None] = {}

    def fake_analyze(self, *, symbol: str, trigger_candles, context_candles, market, timeframe_plan, entry_candles=None, higher_candles=None):
        seen_market["funding_rate"] = market.funding_rate
        seen_market["open_interest"] = market.open_interest
        seen_market["open_interest_change_pct"] = market.open_interest_change_pct
        return _result(True, timeframe_plan, market)

    monkeypatch.setattr("futures_analyzer.analysis.replay.SetupAnalyzer.analyze", fake_analyze)

    replay_time = asyncio.run(
        find_latest_tradable_chart_timestamp(
            provider=provider,
            symbol="BTCUSDT",
            market=market,
            timeframe_plan=timeframe_plan,
            risk_reward=0.7,
            style=StrategyStyle.CONSERVATIVE,
            market_mode=MarketMode.INTRADAY,
            replay_trigger_bars=40,
        )
    )

    assert replay_time is not None
    assert seen_market["funding_rate"] == 0.0002
    assert seen_market["open_interest"] == 2000.0
    assert seen_market["open_interest_change_pct"] == 1.5
