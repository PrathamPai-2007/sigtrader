"""Momentum and oscillator indicators: RSI, MACD, Stochastic, RSI divergence."""
from __future__ import annotations
from analyzer.analysis.models import Candle
from analyzer.analysis.indicators._models import IndicatorResult
from analyzer.analysis.indicators._math import _ema
from analyzer.analysis.indicators.structure import _swing_pivots


def rsi(candles: list[Candle], period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI) on a 0-100 scale."""
    if len(candles) < period + 1:
        return 50.0
    deltas = [candles[i].close - candles[i - 1].close for i in range(1, len(candles))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(candles: list[Candle], fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorResult:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if len(candles) < slow + signal:
        return IndicatorResult(value=0.0, signal=0.0, histogram=0.0)
    closes = [c.close for c in candles]
    k_fast = 2.0 / (fast + 1)
    k_slow = 2.0 / (slow + 1)
    ema_f = sum(closes[:fast]) / fast
    ema_s = sum(closes[:slow]) / slow
    for c in closes[fast:slow]:
        ema_f = c * k_fast + ema_f * (1 - k_fast)
    macd_values: list[float] = []
    for c in closes[slow:]:
        ema_f = c * k_fast + ema_f * (1 - k_fast)
        ema_s = c * k_slow + ema_s * (1 - k_slow)
        macd_values.append(ema_f - ema_s)
    if len(macd_values) < signal:
        return IndicatorResult(value=macd_values[-1] if macd_values else 0.0, signal=0.0, histogram=0.0)
    signal_line = _ema(macd_values, signal)
    histogram = macd_values[-1] - signal_line
    return IndicatorResult(value=macd_values[-1], signal=signal_line, histogram=histogram)


def stochastic(candles: list[Candle], period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> IndicatorResult:
    """Calculate Stochastic Oscillator."""
    if len(candles) < period:
        return IndicatorResult(value=50.0, signal=50.0)
    recent = candles[-period:]
    low = min(c.low for c in recent)
    high = max(c.high for c in recent)
    close = candles[-1].close
    k_percent = 50.0 if high == low else 100.0 * (close - low) / (high - low)
    k_values = []
    for i in range(period, len(candles)):
        window = candles[i - period + 1:i + 1]
        lo = min(c.low for c in window)
        hi = max(c.high for c in window)
        cl = candles[i].close
        k_values.append(50.0 if hi == lo else 100.0 * (cl - lo) / (hi - lo))
    if len(k_values) < smooth_k:
        return IndicatorResult(value=k_percent, signal=k_percent)
    k_smooth = sum(k_values[-smooth_k:]) / smooth_k
    d_smooth = sum(k_values[-smooth_d:]) / smooth_d if len(k_values) >= smooth_d else k_smooth
    return IndicatorResult(value=k_smooth, signal=d_smooth)


def rsi_divergence(
    candles: list[Candle],
    period: int = 14,
    lookback: int = 30,
    pivot_n: int = 2,
    min_price_move_pct: float = 0.5,
) -> tuple[bool, str, float]:
    """Detect RSI divergence using confirmed swing pivots.

    Returns (divergence_detected, divergence_type, strength).
    divergence_type: 'bullish' | 'bearish' | 'none'.
    """
    min_required = period + lookback + pivot_n
    if len(candles) < min_required:
        return False, "none", 0.0
    window = candles[-lookback:]
    offset = len(candles) - lookback
    prices = [c.close for c in window]
    all_closes = [c.close for c in candles]
    deltas = [all_closes[i] - all_closes[i - 1] for i in range(1, len(all_closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    _all_rsi: list[float] = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            _all_rsi.append(100.0 if avg_gain > 0 else 50.0)
        else:
            rs = avg_gain / avg_loss
            _all_rsi.append(100.0 - (100.0 / (1.0 + rs)))
    rsi_vals: list[float] = []
    for i in range(len(window)):
        candle_idx = offset + i
        rsi_idx = candle_idx - period - 1
        rsi_vals.append(_all_rsi[rsi_idx] if rsi_idx >= 0 else 50.0)
    price_lows = [
        (i, v) for i, v in _swing_pivots(prices, n=pivot_n)
        if prices[i] == min(prices[max(0, i - pivot_n): i + pivot_n + 1])
    ]
    if len(price_lows) >= 2:
        (i1, p1), (i2, p2) = price_lows[-2], price_lows[-1]
        r1, r2 = rsi_vals[i1], rsi_vals[i2]
        if p2 < p1 and r2 > r1 and abs(p2 - p1) / max(p1, 1e-9) * 100 >= min_price_move_pct:
            return True, "bullish", round(r2 - r1, 2)
    price_highs = [
        (i, v) for i, v in _swing_pivots(prices, n=pivot_n)
        if prices[i] == max(prices[max(0, i - pivot_n): i + pivot_n + 1])
    ]
    if len(price_highs) >= 2:
        (i1, p1), (i2, p2) = price_highs[-2], price_highs[-1]
        r1, r2 = rsi_vals[i1], rsi_vals[i2]
        if p2 > p1 and r2 < r1 and abs(p2 - p1) / max(p1, 1e-9) * 100 >= min_price_move_pct:
            return True, "bearish", round(r1 - r2, 2)
    return False, "none", 0.0
