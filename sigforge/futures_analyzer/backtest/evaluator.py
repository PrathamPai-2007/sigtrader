from __future__ import annotations

from futures_analyzer.analysis.models import Candle
from futures_analyzer.backtest.models import BacktestTrade
from futures_analyzer.history.models import EvaluationOutcome

_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1d": 1440,
}

# (min_bars, max_bars) caps per timeframe tier
_WINDOW_CAPS: dict[str, tuple[int, int]] = {
    "scalper":   (20, 120),   # 1m–5m: 20–120 bars (was 60 — too short to reach targets)
    "intraday":  (20, 200),   # 15m–1h: 20–200 bars
    "swing":     (10, 100),   # 4h–1d: 10–100 bars
}


def evaluation_window_for_timeframe(interval: str, lookback_bars: int = 200) -> int:
    """Return the number of forward bars to use when evaluating a trade outcome.

    Scales with the trigger timeframe so that:
    - 1m  setups are evaluated over ~30 min  (30 bars)
    - 15m setups are evaluated over ~6h      (24 bars)
    - 4h  setups are evaluated over ~5 days  (30 bars)

    Formula: int(lookback_bars * 0.25), capped per timeframe tier.
    The cap ensures scalper windows stay tight and swing windows don't balloon.
    """
    minutes = _INTERVAL_MINUTES.get(interval, 15)
    raw = max(int(lookback_bars * 0.25), 10)
    if minutes <= 5:
        tier = "scalper"
    elif minutes <= 60:
        tier = "intraday"
    else:
        tier = "swing"
    lo, hi = _WINDOW_CAPS[tier]
    return max(lo, min(hi, raw))


def resolve_outcome(
    trade: BacktestTrade,
    forward_candles: list[Candle],
    evaluation_window: int = 96,
) -> BacktestTrade:
    """Resolve a trade's outcome from candles that follow the entry bar.

    Mirrors the logic in HistoryService._evaluate_snapshot but operates
    entirely in-memory — no API calls needed.

    Args:
        trade: The trade to resolve (mutated in-place and returned).
        forward_candles: Candles starting at or after the entry bar.
        evaluation_window: Max bars to scan before falling back to close PnL.

    Returns:
        The same trade object with outcome, mfe_pct, mae_pct, pnl_pct set.
    """
    candles = forward_candles[:evaluation_window]
    if not candles:
        trade.outcome = EvaluationOutcome.NEITHER.value
        return trade

    entry = trade.entry
    target = trade.target
    stop = trade.stop
    side = trade.side

    outcome = EvaluationOutcome.NEITHER.value

    if side == "long":
        mfe = max(((c.high - entry) / entry) * 100.0 for c in candles)
        mae = max(((entry - c.low) / entry) * 100.0 for c in candles)
        for candle in candles:
            hit_target = candle.high >= target
            hit_stop = candle.low <= stop
            if hit_target and hit_stop:
                outcome = EvaluationOutcome.AMBIGUOUS_SAME_BAR.value
                break
            if hit_target:
                outcome = EvaluationOutcome.TARGET_HIT.value
                break
            if hit_stop:
                outcome = EvaluationOutcome.STOP_HIT.value
                break
        pnl = ((candles[-1].close - entry) / entry) * 100.0
    else:
        mfe = max(((entry - c.low) / entry) * 100.0 for c in candles)
        mae = max(((c.high - entry) / entry) * 100.0 for c in candles)
        for candle in candles:
            hit_target = candle.low <= target
            hit_stop = candle.high >= stop
            if hit_target and hit_stop:
                outcome = EvaluationOutcome.AMBIGUOUS_SAME_BAR.value
                break
            if hit_target:
                outcome = EvaluationOutcome.TARGET_HIT.value
                break
            if hit_stop:
                outcome = EvaluationOutcome.STOP_HIT.value
                break
        pnl = ((entry - candles[-1].close) / entry) * 100.0

    # For clean target/stop hits use the actual price, not close
    if outcome == EvaluationOutcome.TARGET_HIT.value:
        pnl = abs((target - entry) / entry) * 100.0
    elif outcome == EvaluationOutcome.STOP_HIT.value:
        pnl = -abs((entry - stop) / entry) * 100.0

    trade.outcome = outcome
    trade.mfe_pct = round(mfe, 4)
    trade.mae_pct = round(mae, 4)
    trade.pnl_pct = round(pnl, 4)
    return trade
