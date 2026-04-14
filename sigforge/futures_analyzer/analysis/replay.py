from __future__ import annotations

import asyncio
from bisect import bisect_right
from datetime import UTC, datetime, timedelta
import math

from futures_analyzer.analysis.scorer import SetupAnalyzer
from futures_analyzer.analysis.models import Candle, MarketMeta, MarketMode, StrategyStyle, TimeframePlan
from futures_analyzer.config import load_app_config, AppConfig
from futures_analyzer.providers import BinanceFuturesProvider

_MIN_BARS_PER_TIMEFRAME = 30
_MAX_KLINES_PER_REQUEST = 1500
_INTERVAL_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
}


def _required_bars_for_interval(*, trigger_interval: str, target_interval: str, replay_trigger_bars: int) -> int:
    trigger_minutes = _INTERVAL_MINUTES[trigger_interval]
    target_minutes = _INTERVAL_MINUTES[target_interval]
    span_minutes = max(replay_trigger_bars - 1, 0) * trigger_minutes
    return max(math.ceil(span_minutes / target_minutes) + _MIN_BARS_PER_TIMEFRAME, _MIN_BARS_PER_TIMEFRAME)


async def _fetch_recent_candles(
    provider: BinanceFuturesProvider,
    *,
    symbol: str,
    interval: str,
    count: int,
) -> list[Candle]:
    remaining = max(count, 0)
    if remaining <= 0:
        return []
    batches: list[list[Candle]] = []
    end_time: datetime | None = None
    while remaining > 0:
        chunk_limit = min(remaining, _MAX_KLINES_PER_REQUEST)
        candles = await provider.fetch_klines(
            symbol=symbol,
            interval=interval,
            limit=chunk_limit,
            end_time=end_time,
            min_required_candles=1,
        )
        if not candles:
            break
        batches.append(candles)
        remaining -= len(candles)
        if len(candles) < chunk_limit:
            break
        end_time = candles[0].open_time - timedelta(milliseconds=1)

    merged: list[Candle] = []
    seen: set[datetime] = set()
    for batch in reversed(batches):
        for candle in batch:
            if candle.open_time in seen:
                continue
            seen.add(candle.open_time)
            merged.append(candle)
    merged.sort(key=lambda candle: candle.open_time)
    return merged[-count:]


def _slice_to_anchor(candles: list[Candle], close_times: list[datetime], anchor: datetime, lookback_bars: int) -> list[Candle]:
    end = bisect_right(close_times, anchor)
    if end <= 0:
        return []
    start = max(0, end - lookback_bars)
    return candles[start:end]


async def find_latest_tradable_chart_timestamp(
    *,
    provider: BinanceFuturesProvider,
    symbol: str,
    market: MarketMeta,
    timeframe_plan: TimeframePlan,
    risk_reward: float,
    style: StrategyStyle,
    market_mode: MarketMode,
    replay_trigger_bars: int | None = None,
    config: AppConfig | None = None,
    preset: str = "position_trader",
) -> datetime | None:
    replay_trigger_bars = max(replay_trigger_bars or timeframe_plan.lookback_bars, 1)
    required_counts = {
        "entry": _required_bars_for_interval(
            trigger_interval=timeframe_plan.trigger_timeframe,
            target_interval=timeframe_plan.entry_timeframe,
            replay_trigger_bars=replay_trigger_bars,
        ),
        "trigger": _required_bars_for_interval(
            trigger_interval=timeframe_plan.trigger_timeframe,
            target_interval=timeframe_plan.trigger_timeframe,
            replay_trigger_bars=replay_trigger_bars,
        ),
        "context": _required_bars_for_interval(
            trigger_interval=timeframe_plan.trigger_timeframe,
            target_interval=timeframe_plan.context_timeframe,
            replay_trigger_bars=replay_trigger_bars,
        ),
        "higher": _required_bars_for_interval(
            trigger_interval=timeframe_plan.trigger_timeframe,
            target_interval=timeframe_plan.higher_timeframe,
            replay_trigger_bars=replay_trigger_bars,
        ),
    }
    entry_candles, trigger_candles, context_candles, higher_candles = await asyncio.gather(
        _fetch_recent_candles(
            provider,
            symbol=symbol,
            interval=timeframe_plan.entry_timeframe,
            count=required_counts["entry"],
        ),
        _fetch_recent_candles(
            provider,
            symbol=symbol,
            interval=timeframe_plan.trigger_timeframe,
            count=required_counts["trigger"],
        ),
        _fetch_recent_candles(
            provider,
            symbol=symbol,
            interval=timeframe_plan.context_timeframe,
            count=required_counts["context"],
        ),
        _fetch_recent_candles(
            provider,
            symbol=symbol,
            interval=timeframe_plan.higher_timeframe,
            count=required_counts["higher"],
        ),
    )

    if min(len(entry_candles), len(trigger_candles), len(context_candles), len(higher_candles)) < _MIN_BARS_PER_TIMEFRAME:
        return None

    cfg = config or load_app_config()
    replay_cap = cfg.cache.replay_lookback_cap

    analyzer = SetupAnalyzer(risk_reward=risk_reward, style=style, market_mode=market_mode, config=cfg, preset=preset)
    entry_closes = [candle.close_time for candle in entry_candles]
    trigger_closes = [candle.close_time for candle in trigger_candles]
    context_closes = [candle.close_time for candle in context_candles]
    higher_closes = [candle.close_time for candle in higher_candles]

    candidate_bars = trigger_candles[_MIN_BARS_PER_TIMEFRAME - 1:][-replay_cap:]

    # Pre-fetch all historical market contexts in parallel — avoids sequential
    # HTTP calls inside the loop (the biggest replay bottleneck).
    context_results = await asyncio.gather(
        *[
            provider.fetch_historical_market_context(
                symbol=symbol,
                as_of=bar.close_time,
                interval=timeframe_plan.trigger_timeframe,
            )
            for bar in candidate_bars
        ],
        return_exceptions=True,
    )
    context_map: dict[datetime, tuple[float | None, float | None, float | None]] = {
        bar.close_time: (ctx if not isinstance(ctx, Exception) else (None, None, None))
        for bar, ctx in zip(candidate_bars, context_results)
    }

    for trigger_candle in reversed(candidate_bars):
        anchor = trigger_candle.close_time
        entry_slice = _slice_to_anchor(entry_candles, entry_closes, anchor, timeframe_plan.lookback_bars)
        trigger_slice = _slice_to_anchor(trigger_candles, trigger_closes, anchor, timeframe_plan.lookback_bars)
        context_slice = _slice_to_anchor(context_candles, context_closes, anchor, timeframe_plan.lookback_bars)
        higher_slice = _slice_to_anchor(higher_candles, higher_closes, anchor, timeframe_plan.lookback_bars)
        if min(len(entry_slice), len(trigger_slice), len(context_slice), len(higher_slice)) < _MIN_BARS_PER_TIMEFRAME:
            continue
        funding_rate, open_interest, oi_change_pct = context_map[anchor]

        replay_market = MarketMeta(
            symbol=market.symbol,
            tick_size=market.tick_size,
            step_size=market.step_size,
            mark_price=trigger_slice[-1].close,
            funding_rate=funding_rate,
            open_interest=open_interest,
            open_interest_change_pct=oi_change_pct,
            as_of=anchor.astimezone(UTC),
        )
        result = analyzer.analyze(
            symbol=symbol,
            entry_candles=entry_slice,
            trigger_candles=trigger_slice,
            context_candles=context_slice,
            higher_candles=higher_slice,
            market=replay_market,
            timeframe_plan=timeframe_plan,
        )
        if result.primary_setup.is_tradable:
            return anchor.astimezone(UTC)
    return None
