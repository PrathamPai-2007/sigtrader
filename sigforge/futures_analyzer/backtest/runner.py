from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections import Counter
from datetime import UTC, datetime, timedelta
from functools import lru_cache
import json
import math
import traceback

from futures_analyzer.analysis.models import (
    Candle,
    MarketMeta,
    TimeframePlan,
)
from futures_analyzer.analysis.scorer import SetupAnalyzer, build_timeframe_plan
from futures_analyzer.backtest.evaluator import resolve_outcome
from futures_analyzer.backtest.models import BacktestConfig, BacktestReport, BacktestTrade
from futures_analyzer.config import DEFAULT_CONFIG_PATH, load_app_config
from futures_analyzer.logging import get_logger
from futures_analyzer.providers import BinanceFuturesProvider

log = get_logger(__name__)

_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080,
}
_MIN_BARS = 30
_MAX_PER_REQUEST = 1500


@lru_cache(maxsize=1)
def _load_runtime_config_dict() -> dict:
    try:
        return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _minutes(interval: str) -> int:
    return _INTERVAL_MINUTES.get(interval, 5)


def _slice_to_anchor(
    candles: list[Candle],
    close_times: list[datetime],
    anchor: datetime,
    lookback: int,
) -> list[Candle]:
    end = bisect_right(close_times, anchor)
    if end <= 0:
        return []
    start = max(0, end - lookback)
    return candles[start:end]


async def _fetch_range(
    provider: BinanceFuturesProvider,
    *,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> list[Candle]:
    """Fetch all candles in [start, end] for the given interval, paging as needed."""
    all_candles: list[Candle] = []
    seen: set[datetime] = set()
    current_start = start

    while current_start < end:
        batch = await provider.fetch_klines(
            symbol=symbol,
            interval=interval,
            limit=_MAX_PER_REQUEST,
            start_time=current_start,
            end_time=end,
            min_required_candles=1,
        )
        if not batch:
            break
        added = 0
        for candle in batch:
            if candle.open_time in seen:
                continue
            seen.add(candle.open_time)
            all_candles.append(candle)
            added += 1
        if added == 0 or len(batch) < _MAX_PER_REQUEST:
            break
        current_start = batch[-1].close_time + timedelta(milliseconds=1)

    all_candles.sort(key=lambda c: c.open_time)
    return all_candles


class BacktestRunner:
    """Runs the SetupAnalyzer over a historical date range and collects trades."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._app_config = load_app_config()

    def _build_timeframe_plan(self) -> TimeframePlan:
        from futures_analyzer.config_presets import get_preset, StrategyPreset
        from futures_analyzer.analysis.models import TimeframePlan as TFP
        if self.config.preset:
            try:
                preset_cfg = get_preset(StrategyPreset(self.config.preset))
                return TFP(
                    profile_name=preset_cfg.name,
                    style=self.config.style,
                    market_mode=self.config.market_mode,
                    entry_timeframe=preset_cfg.entry_timeframe,
                    trigger_timeframe=preset_cfg.trigger_timeframe,
                    context_timeframe=preset_cfg.context_timeframe,
                    higher_timeframe=preset_cfg.higher_timeframe,
                    lookback_bars=preset_cfg.lookback_bars,
                )
            except Exception:
                pass
        return build_timeframe_plan(config=self._app_config, style=self.config.style, market_mode=self.config.market_mode)

    def _warmup_start(self, tfp: TimeframePlan) -> datetime:
        """Return a start time that includes enough warmup bars before config.start."""
        trigger_mins = _minutes(tfp.trigger_timeframe)
        warmup_minutes = tfp.lookback_bars * trigger_mins
        return self.config.start - timedelta(minutes=warmup_minutes)

    async def run(self, *, progress: bool = False) -> BacktestReport:
        """Single-pass backtest over config.start → config.end."""
        tfp = self._build_timeframe_plan()
        warmup_start = self._warmup_start(tfp)
        fetch_end = self.config.end

        if progress:
            print(f"[backtest] Fetching candles for {self.config.symbol} "
                  f"{warmup_start.date()} → {fetch_end.date()} ...")

        provider = BinanceFuturesProvider()
        try:
            # Fetch all timeframes in parallel with error handling
            results = await asyncio.gather(
                _fetch_range(provider, symbol=self.config.symbol, interval=tfp.entry_timeframe,
                             start=warmup_start, end=fetch_end),
                _fetch_range(provider, symbol=self.config.symbol, interval=tfp.trigger_timeframe,
                             start=warmup_start, end=fetch_end),
                _fetch_range(provider, symbol=self.config.symbol, interval=tfp.context_timeframe,
                             start=warmup_start, end=fetch_end),
                _fetch_range(provider, symbol=self.config.symbol, interval=tfp.higher_timeframe,
                             start=warmup_start, end=fetch_end),
                return_exceptions=True,
            )
            
            # Check for exceptions in results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    timeframes = [tfp.entry_timeframe, tfp.trigger_timeframe, tfp.context_timeframe, tfp.higher_timeframe]
                    raise RuntimeError(f"Failed to fetch {timeframes[i]} candles: {result}") from result
            
            entry_candles, trigger_candles, context_candles, higher_candles = results
        finally:
            await provider.aclose()

        return self._run_in_memory(
            tfp=tfp,
            entry_candles=entry_candles,
            trigger_candles=trigger_candles,
            context_candles=context_candles,
            higher_candles=higher_candles,
            progress=progress,
        )

    def _run_in_memory(
        self,
        *,
        tfp: TimeframePlan,
        entry_candles: list[Candle],
        trigger_candles: list[Candle],
        context_candles: list[Candle],
        higher_candles: list[Candle],
        progress: bool = False,
        _analyzer: "SetupAnalyzer | None" = None,
    ) -> BacktestReport:
        """Walk trigger bars and collect trades — no I/O."""
        from futures_analyzer.config_presets import get_preset, StrategyPreset
        from futures_analyzer.analysis.models import TimeframePlan as TFP

        from futures_analyzer.backtest.evaluator import evaluation_window_for_timeframe

        analyzer = _analyzer or SetupAnalyzer(
            risk_reward=self.config.risk_reward,
            style=self.config.style,
            market_mode=self.config.market_mode,
            config=self._app_config,
        )

        entry_closes = [c.close_time for c in entry_candles]
        trigger_closes = [c.close_time for c in trigger_candles]
        context_closes = [c.close_time for c in context_candles]
        higher_closes = [c.close_time for c in higher_candles]

        # Evaluation window: scale with trigger timeframe via shared helper.
        eval_window = evaluation_window_for_timeframe(tfp.trigger_timeframe, tfp.lookback_bars)

        # Only walk bars within the requested date range
        candidate_bars = [
            c for c in trigger_candles
            if self.config.start <= c.close_time.astimezone(UTC) <= self.config.end
        ]

        # Estimate daily quote volume from candle data (bars per day × avg bar quote volume).
        # Falls back to config.daily_volume_usd if provided, or 10M as a last resort.
        def _estimate_daily_vol(candles: list[Candle], interval_mins: int) -> float:
            if self.config.daily_volume_usd is not None:
                return self.config.daily_volume_usd
            if not candles:
                return 10_000_000.0
            bars_per_day = max(1, int(1440 / interval_mins))
            recent = candles[-min(bars_per_day * 7, len(candles)):]
            if not recent:
                return 10_000_000.0
            avg_bar_quote_vol = sum(c.volume * c.close for c in recent) / len(recent)
            return max(avg_bar_quote_vol * bars_per_day, 1.0)

        trigger_mins = _minutes(tfp.trigger_timeframe)
        estimated_daily_vol = _estimate_daily_vol(trigger_candles, trigger_mins)

        # Hoist slippage config once — avoids repeated lru_cache lookups inside the hot loop
        _slip_model = None
        _slip_vol_mult = 1.0
        if self.config.order_size_usd is not None:
            from futures_analyzer.history.evaluation import SlippageModel
            _slip_cfg = load_app_config().slippage
            _slip_model = SlippageModel(_slip_cfg.default_model)
            _slip_vol_mult = _slip_cfg.volatility_multipliers.get("normal", 1.0)

        runtime_cfg = _load_runtime_config_dict()
        entry_filters = runtime_cfg.get("entry_filters") if isinstance(runtime_cfg, dict) else None
        if not isinstance(entry_filters, dict):
            entry_filters = {}
        enable_confirmation = bool(entry_filters.get("enable_confirmation", False))
        try:
            confirmation_candles = int(entry_filters.get("confirmation_candles") or 1)
        except Exception:
            confirmation_candles = 1
        confirmation_candles = max(1, confirmation_candles)

        trades: list[BacktestTrade] = []
        rejection_reasons: Counter[str] = Counter()
        stop_anchor_counts: Counter[str] = Counter()
        target_anchor_counts: Counter[str] = Counter()
        bars_skipped_insufficient_history = 0
        bars_skipped_analysis_error = 0
        printed_first_analysis_error = False
        total = len(candidate_bars)

        for i, trigger_bar in enumerate(candidate_bars):
            if progress and i % 200 == 0:
                print(f"[backtest] {i}/{total} bars processed, {len(trades)} trades so far")

            anchor = trigger_bar.close_time
            entry_sl = _slice_to_anchor(entry_candles, entry_closes, anchor, tfp.lookback_bars)
            trigger_sl = _slice_to_anchor(trigger_candles, trigger_closes, anchor, tfp.lookback_bars)
            context_sl = _slice_to_anchor(context_candles, context_closes, anchor, tfp.lookback_bars)
            higher_sl = _slice_to_anchor(higher_candles, higher_closes, anchor, tfp.lookback_bars)

            if min(len(entry_sl), len(trigger_sl), len(context_sl), len(higher_sl)) < _MIN_BARS:
                bars_skipped_insufficient_history += 1
                continue

            market = MarketMeta(
                symbol=self.config.symbol,
                mark_price=trigger_sl[-1].close,
                as_of=anchor.astimezone(UTC),
            )

            try:
                result = analyzer.analyze(
                    symbol=self.config.symbol,
                    entry_candles=entry_sl,
                    trigger_candles=trigger_sl,
                    context_candles=context_sl,
                    higher_candles=higher_sl,
                    market=market,
                    timeframe_plan=tfp,
                )
            except Exception as exc:
                if not printed_first_analysis_error:
                    printed_first_analysis_error = True
                    print(f"[backtest] First analysis error at {anchor.isoformat()}: {exc}")
                    print(traceback.format_exc())
                log.debug("backtest.bar_skipped", bar=anchor.isoformat(), error=str(exc))
                bars_skipped_analysis_error += 1
                continue

            stop_anchor_counts[result.primary_setup.stop_anchor] += 1
            target_anchor_counts[result.primary_setup.target_anchor] += 1

            if not result.primary_setup.is_tradable:
                for reason in result.primary_setup.tradable_reasons:
                    rejection_reasons[reason] += 1
                continue

            setup = result.primary_setup
            if enable_confirmation and setup.side == "short":
                required = confirmation_candles + 1
                if len(entry_sl) < required:
                    rejection_reasons["confirmation_insufficient_candles"] += 1
                    continue
                ok = True
                for j in range(1, confirmation_candles + 1):
                    if not (entry_sl[-j].close < entry_sl[-j - 1].close):
                        ok = False
                        break
                if not ok:
                    rejection_reasons["confirmation_failed"] += 1
                    continue

            position_size = 1.0
            ps = runtime_cfg.get("position_sizing") if isinstance(runtime_cfg, dict) else None
            if isinstance(ps, dict) and ps.get("mode") == "confidence_scaled":
                try:
                    from futures_analyzer.portfolio import get_position_size
                    position_size = float(get_position_size(setup.confidence, runtime_cfg))
                except Exception:
                    position_size = 1.0

            trade = BacktestTrade(
                bar_time=anchor.astimezone(UTC),
                side=setup.side,
                entry=setup.entry_price,
                target=setup.target_price,
                stop=setup.stop_loss,
                confidence=setup.confidence,
                quality_score=setup.quality_score,
                quality_label=setup.quality_label.value,
                regime=result.market_regime.value,
                position_size=position_size,
            )

            # Optional slippage adjustment (in-memory, no I/O)
            if self.config.order_size_usd is not None:
                try:
                    from futures_analyzer.history.evaluation import SlippageCalculator, SlippageModel
                    slip_model = _slip_model
                    vol_mult = _slip_vol_mult
                    # Approximate: use spread from enhanced metrics if available, else 0.05%
                    spread = result.enhanced_metrics.bid_ask_spread_pct or 0.05
                    liq = result.enhanced_metrics.liquidity_score or 50.0
                    order_pct = (self.config.order_size_usd / estimated_daily_vol) * 100.0
                    base_slip = SlippageCalculator.calculate_slippage(
                        order_size_pct=order_pct,
                        bid_ask_spread_pct=spread,
                        liquidity_score=liq,
                        model=slip_model,
                    ) * vol_mult
                    exec_m = SlippageCalculator.adjust_prices(
                        trade.entry, trade.target, trade.stop,
                        base_slip, base_slip * 0.6, base_slip * 0.6,
                        trade.side,
                    )
                    trade.entry = exec_m.adjusted_entry_price
                    trade.target = exec_m.adjusted_target_price
                    trade.stop = exec_m.adjusted_stop_price
                except Exception:
                    pass

            # Forward candles for outcome resolution
            bar_idx = bisect_right(trigger_closes, anchor)
            forward = trigger_candles[bar_idx: bar_idx + eval_window]
            resolve_outcome(trade, forward, evaluation_window=eval_window)
            trades.append(trade)
        report = BacktestReport(
            config=self.config,
            trades=trades,
            rejection_reasons=dict(rejection_reasons),
            stop_anchor_counts=dict(stop_anchor_counts),
            target_anchor_counts=dict(target_anchor_counts),
            bars_skipped_insufficient_history=bars_skipped_insufficient_history,
            bars_skipped_analysis_error=bars_skipped_analysis_error,
        )
        report.compute_aggregates()
        return report

    async def walk_forward(self, *, folds: int = 5, progress: bool = False) -> list[BacktestReport]:
        """Split the date range into folds and run a backtest on each fold.

        Returns one BacktestReport per fold (test window only).
        """
        if folds < 2:
            raise ValueError("walk_forward requires at least 2 folds")

        total_seconds = (self.config.end - self.config.start).total_seconds()
        fold_seconds = total_seconds / folds

        tfp = self._build_timeframe_plan()
        warmup_start = self._warmup_start(tfp)

        if progress:
            print(f"[backtest] Fetching full range for walk-forward ({folds} folds) ...")

        provider = BinanceFuturesProvider()
        try:
            results = await asyncio.gather(
                _fetch_range(provider, symbol=self.config.symbol, interval=tfp.entry_timeframe,
                             start=warmup_start, end=self.config.end),
                _fetch_range(provider, symbol=self.config.symbol, interval=tfp.trigger_timeframe,
                             start=warmup_start, end=self.config.end),
                _fetch_range(provider, symbol=self.config.symbol, interval=tfp.context_timeframe,
                             start=warmup_start, end=self.config.end),
                _fetch_range(provider, symbol=self.config.symbol, interval=tfp.higher_timeframe,
                             start=warmup_start, end=self.config.end),
                return_exceptions=True,
            )
            
            # Check for exceptions in results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    timeframes = [tfp.entry_timeframe, tfp.trigger_timeframe, tfp.context_timeframe, tfp.higher_timeframe]
                    raise RuntimeError(f"Failed to fetch {timeframes[i]} candles: {result}") from result
            
            entry_candles, trigger_candles, context_candles, higher_candles = results
        finally:
            await provider.aclose()

        reports: list[BacktestReport] = []
        # Instantiate once — SetupAnalyzer is stateless, no need to recreate per fold
        shared_analyzer = SetupAnalyzer(
            risk_reward=self.config.risk_reward,
            style=self.config.style,
            market_mode=self.config.market_mode,
            config=self._app_config,
        )
        for fold_idx in range(folds):
            fold_start = self.config.start + timedelta(seconds=fold_idx * fold_seconds)
            fold_end = self.config.start + timedelta(seconds=(fold_idx + 1) * fold_seconds)
            if fold_idx == folds - 1:
                fold_end = self.config.end

            # Skip degenerate folds with zero or near-zero duration
            if (fold_end - fold_start).total_seconds() < 60:
                if progress:
                    print(f"[backtest] Walk-forward fold {fold_idx + 1}/{folds}: skipped (zero-length window)")
                continue

            fold_config = BacktestConfig(
                symbol=self.config.symbol,
                start=fold_start,
                end=fold_end,
                style=self.config.style,
                market_mode=self.config.market_mode,
                risk_reward=self.config.risk_reward,
                preset=self.config.preset,
            )
            fold_runner = BacktestRunner(fold_config)
            if progress:
                print(f"[backtest] Walk-forward fold {fold_idx + 1}/{folds}: "
                      f"{fold_start.date()} → {fold_end.date()}")
            report = fold_runner._run_in_memory(
                tfp=tfp,
                entry_candles=entry_candles,
                trigger_candles=trigger_candles,
                context_candles=context_candles,
                higher_candles=higher_candles,
                progress=progress,
                _analyzer=shared_analyzer,
            )
            reports.append(report)

        return reports
