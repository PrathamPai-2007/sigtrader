from __future__ import annotations

from futures_analyzer.analysis.models import Candle
from futures_analyzer.backtest.models import BacktestTrade
from futures_analyzer.history.models import EvaluationOutcome


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
