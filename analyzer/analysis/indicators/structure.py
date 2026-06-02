"""Swing pivot and market structure detection."""
from __future__ import annotations
from analyzer.analysis.models import Candle
from analyzer.analysis.indicators._models import LiquiditySweep


def _swing_pivots(
    values: list[float],
    n: int = 3,
) -> list[tuple[int, float]]:
    """Return confirmed swing pivot indices and values.

    A pivot at index i requires values[i] to be the min or max of the
    surrounding 2*n+1 window. Returns (index, value) tuples ascending.
    """
    pivots: list[tuple[int, float]] = []
    length = len(values)
    for i in range(n, length - n):
        window = values[i - n: i + n + 1]
        v = values[i]
        if v == min(window) or v == max(window):
            pivots.append((i, v))
    return pivots


def compute_market_structure(candles: list[Candle], pivot_n: int = 2) -> str:
    """Classify recent swing pivot sequence as HH_HL, LH_LL, or mixed."""
    if not candles:
        return "mixed"
    closes = [c.close for c in candles]
    if len(closes) >= 2:
        if all(closes[i] > closes[i - 1] for i in range(1, len(closes))):
            return "HH_HL"
        if all(closes[i] < closes[i - 1] for i in range(1, len(closes))):
            return "LH_LL"
    pivots = _swing_pivots(closes, n=pivot_n)
    pivot_highs: list[tuple[int, float]] = []
    pivot_lows: list[tuple[int, float]] = []
    for idx, val in pivots:
        window = closes[max(0, idx - pivot_n): idx + pivot_n + 1]
        if val == max(window):
            pivot_highs.append((idx, val))
        if val == min(window):
            pivot_lows.append((idx, val))
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return "mixed"
    (_, h1), (_, h2) = pivot_highs[-2], pivot_highs[-1]
    (_, l1), (_, l2) = pivot_lows[-2], pivot_lows[-1]
    if h2 > h1 and l2 > l1:
        return "HH_HL"
    if h2 < h1 and l2 < l1:
        return "LH_LL"
    return "mixed"


def detect_liquidity_sweeps(candles: list[Candle], swing_n: int = 2) -> list[LiquiditySweep]:
    """Detect bars that exceed a prior swing high/low and optionally recover."""
    if not candles:
        return []
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    swing_high_pivots: list[tuple[int, float]] = []
    swing_low_pivots: list[tuple[int, float]] = []
    for idx, val in _swing_pivots(highs, n=swing_n):
        window = highs[max(0, idx - swing_n): idx + swing_n + 1]
        if val == max(window):
            swing_high_pivots.append((idx, val))
    for idx, val in _swing_pivots(lows, n=swing_n):
        window = lows[max(0, idx - swing_n): idx + swing_n + 1]
        if val == min(window):
            swing_low_pivots.append((idx, val))
    sweeps: list[LiquiditySweep] = []
    for i, bar in enumerate(candles):
        for sh_idx, sh_val in swing_high_pivots:
            if sh_idx >= i:
                continue
            if bar.high > sh_val:
                sweeps.append(LiquiditySweep(
                    direction="above", swept_level=sh_val,
                    bar_index=i, recovered=bar.close < sh_val,
                ))
        for sl_idx, sl_val in swing_low_pivots:
            if sl_idx >= i:
                continue
            if bar.low < sl_val:
                sweeps.append(LiquiditySweep(
                    direction="below", swept_level=sl_val,
                    bar_index=i, recovered=bar.close > sl_val,
                ))
    sweeps.sort(key=lambda s: s.bar_index)
    return sweeps
