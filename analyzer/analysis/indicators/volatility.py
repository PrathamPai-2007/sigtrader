"""Volatility indicators: Bollinger Bands, ADX, ADX slope."""
from __future__ import annotations
from analyzer.analysis.models import Candle


def bollinger_bands(candles: list[Candle], period: int = 20, std_dev: float = 2.0) -> dict[str, float]:
    """Calculate Bollinger Bands — upper, middle (SMA), lower, bandwidth, position."""
    if len(candles) < period:
        close = candles[-1].close if candles else 0.0
        return {"upper": close, "middle": close, "lower": close, "bandwidth": 0.0, "position": 0.5}
    closes = [c.close for c in candles[-period:]]
    sma = sum(closes) / len(closes)
    variance = sum((c - sma) ** 2 for c in closes) / len(closes)
    std = variance ** 0.5
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    bandwidth = ((upper - lower) / sma * 100) if sma != 0 else 0.0
    position = (candles[-1].close - lower) / (upper - lower) if upper != lower else 0.5
    return {
        "upper": upper,
        "middle": sma,
        "lower": lower,
        "bandwidth": bandwidth,
        "position": max(0.0, min(1.0, position)),
    }


def adx(candles: list[Candle], period: int = 14) -> dict[str, float]:
    """Calculate Average Directional Index (ADX) using Wilder's smoothing."""
    _zero = {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}
    if len(candles) < 2 * period:
        return _zero
    tr_list: list[float] = []
    plus_dm_list: list[float] = []
    minus_dm_list: list[float] = []
    for i in range(1, len(candles)):
        prev = candles[i - 1]
        curr = candles[i]
        tr = max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        up_move = curr.high - prev.high
        down_move = prev.low - curr.low
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
    smoothed_tr = sum(tr_list[:period])
    smoothed_plus_dm = sum(plus_dm_list[:period])
    smoothed_minus_dm = sum(minus_dm_list[:period])

    def _dx(s_tr: float, s_pdm: float, s_mdm: float) -> float:
        if s_tr == 0.0:
            return 0.0
        pdi = 100.0 * s_pdm / s_tr
        mdi = 100.0 * s_mdm / s_tr
        denom = pdi + mdi
        return 100.0 * abs(pdi - mdi) / denom if denom != 0.0 else 0.0

    dx_list: list[float] = [_dx(smoothed_tr, smoothed_plus_dm, smoothed_minus_dm)]
    k = 1.0 / period
    for i in range(period, len(tr_list)):
        smoothed_tr = smoothed_tr * (1.0 - k) + tr_list[i]
        smoothed_plus_dm = smoothed_plus_dm * (1.0 - k) + plus_dm_list[i]
        smoothed_minus_dm = smoothed_minus_dm * (1.0 - k) + minus_dm_list[i]
        dx_list.append(_dx(smoothed_tr, smoothed_plus_dm, smoothed_minus_dm))
    if len(dx_list) < period:
        return _zero
    adx_val = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx_val = adx_val * (1.0 - k) + dx * k
    plus_di = (100.0 * smoothed_plus_dm / smoothed_tr) if smoothed_tr != 0.0 else 0.0
    minus_di = (100.0 * smoothed_minus_dm / smoothed_tr) if smoothed_tr != 0.0 else 0.0
    return {"adx": float(adx_val), "plus_di": float(plus_di), "minus_di": float(minus_di)}


def compute_adx_slope(candles: list[Candle], lookback: int = 5) -> float:
    """Compute the linear regression slope of ADX values over the last `lookback` bars."""
    if len(candles) < lookback:
        return 0.0
    n = lookback
    start = len(candles) - lookback + 1
    adx_values: list[float] = [adx(candles[:i])["adx"] for i in range(start, len(candles) + 1)]
    x = list(range(n))
    sum_x = sum(x)
    sum_y = sum(adx_values)
    sum_xy = sum(x[i] * adx_values[i] for i in range(n))
    sum_x2 = sum(xi * xi for xi in x)
    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom
