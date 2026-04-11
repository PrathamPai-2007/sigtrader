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

    Exit priority (checked before hard SL/TP each bar):
      1. Momentum failure  — entry_momentum (close-to-close) flips against trade
      2. Time stop         — bars_in_trade >= 12 with no open profit
      3. Soft stop         — unrealized PnL <= -2.5%
      4. Breakeven stop    — once +1% floated, floor/ceil stop to entry
      5. Hard stop / target (existing logic, unchanged)

    Args:
        trade: The trade to resolve (mutated in-place and returned).
        forward_candles: Candles starting at or after the entry bar.
        evaluation_window: Max bars to scan before falling back to close PnL.

    Returns:
        The same trade object with outcome, exit_reason, mfe_pct, mae_pct,
        pnl_pct set.
    """
    candles = forward_candles[:evaluation_window]
    if not candles:
        trade.outcome = EvaluationOutcome.NEITHER.value
        trade.exit_reason = "neither"
        return trade

    entry = trade.entry
    target = trade.target
    stop = trade.stop          # may be adjusted intra-loop (breakeven)
    side = trade.side

    outcome = EvaluationOutcome.NEITHER.value
    exit_reason = ""
    exit_price: float | None = None

    mfe_running = 0.0
    mae_running = 0.0

    prev_close = entry  # seed for momentum calculation

    for bars_in_trade, candle in enumerate(candles, start=1):
        # ── Running MFE / MAE ────────────────────────────────────────────────
        if side == "long":
            bar_mfe = ((candle.high - entry) / entry) * 100.0
            bar_mae = ((entry - candle.low) / entry) * 100.0
            unrealized_pnl = ((candle.close - entry) / entry) * 100.0
        else:
            bar_mfe = ((entry - candle.low) / entry) * 100.0
            bar_mae = ((candle.high - entry) / entry) * 100.0
            unrealized_pnl = ((entry - candle.close) / entry) * 100.0

        mfe_running = max(mfe_running, bar_mfe)
        mae_running = max(mae_running, bar_mae)

        # ── 1. Momentum failure (VERY conservative) ─────────────
        entry_momentum = (candle.close - entry) / max(entry, 1e-9)

        if bars_in_trade >= 5:
            if side == "long" and entry_momentum < -0.008:
                outcome = "momentum_failure"
                exit_reason = "momentum_failure"
                exit_price = candle.close
                break
            if side == "short" and entry_momentum > 0.008:
                outcome = "momentum_failure"
                exit_reason = "momentum_failure"
                exit_price = candle.close
                break

        # ── 2. Time-based exit (give trade time) ────────────────
        if bars_in_trade >= 30 and unrealized_pnl <= 0:
            outcome = "time_stop"
            exit_reason = "time_stop"
            exit_price = candle.close
            break

        # ── 3. Breakeven (disabled for now) ───────────────────────────
        # if unrealized_pnl >= 1.5:
        #     if side == "long":
        #         stop = max(stop, entry)
        #     else:
        #         stop = min(stop, entry)

        # ── 3. Soft stop (only real losers) ────────────────────
        if bars_in_trade >= 5 and unrealized_pnl <= -6.0:
            outcome = "soft_stop"
            exit_reason = "soft_stop"
            exit_price = candle.close
            break

        # ── 5. Hard stop / target (existing logic, unchanged) ────────────────
        if side == "long":
            hit_target = candle.high >= target
            hit_stop = candle.low <= stop
        else:
            hit_target = candle.low <= target
            hit_stop = candle.high >= stop

        if hit_target and hit_stop:
            outcome = EvaluationOutcome.AMBIGUOUS_SAME_BAR.value
            exit_reason = "ambiguous_same_bar"
            break
        if hit_target:
            outcome = EvaluationOutcome.TARGET_HIT.value
            exit_reason = "target_hit"
            exit_price = target
            break
        if hit_stop:
            outcome = EvaluationOutcome.STOP_HIT.value
            exit_reason = "stop_hit"
            exit_price = stop
            break

        prev_close = candle.close

    # ── PnL resolution ───────────────────────────────────────────────────────
    if exit_price is not None:
        if side == "long":
            pnl = ((exit_price - entry) / entry) * 100.0
        else:
            pnl = ((entry - exit_price) / entry) * 100.0
    elif outcome == EvaluationOutcome.TARGET_HIT.value:
        pnl = abs((target - entry) / entry) * 100.0
    elif outcome == EvaluationOutcome.STOP_HIT.value:
        pnl = -abs((entry - stop) / entry) * 100.0
    else:
        # NEITHER / window exhausted — use final close
        last_close = candles[-1].close
        if side == "long":
            pnl = ((last_close - entry) / entry) * 100.0
        else:
            pnl = ((entry - last_close) / entry) * 100.0
        if not exit_reason:
            exit_reason = "neither"

    trade.outcome = outcome
    trade.exit_reason = exit_reason
    trade.mfe_pct = round(mfe_running, 4)
    trade.mae_pct = round(mae_running, 4)
    trade.pnl_pct = round(pnl, 4)
    return trade
