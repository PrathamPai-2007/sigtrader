"""Advanced technical indicators for enhanced prediction accuracy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from futures_analyzer.analysis.models import Candle


@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""
    value: float
    signal: float | None = None
    histogram: float | None = None
    divergence: bool = False


def rsi(candles: list[Candle], period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI).
    
    Measures momentum on a scale of 0-100. Values above 70 indicate overbought,
    below 30 indicate oversold.
    """
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
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Returns MACD line, signal line, and histogram for momentum analysis.
    """
    if len(candles) < slow + signal:
        return IndicatorResult(value=0.0, signal=0.0, histogram=0.0)
    
    closes = [c.close for c in candles]
    
    # Calculate EMAs
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = _ema([macd_line], signal) if isinstance(macd_line, (int, float)) else _ema([macd_line], signal)
    
    # Recalculate for proper signal line
    macd_values = []
    for i in range(slow - 1, len(closes)):
        fast_ema = _ema(closes[:i + 1], fast)
        slow_ema = _ema(closes[:i + 1], slow)
        macd_values.append(fast_ema - slow_ema)
    
    if len(macd_values) < signal:
        return IndicatorResult(value=macd_values[-1] if macd_values else 0.0, signal=0.0, histogram=0.0)
    
    signal_line = _ema(macd_values, signal)
    histogram = macd_values[-1] - signal_line
    
    return IndicatorResult(value=macd_values[-1], signal=signal_line, histogram=histogram)


def stochastic(candles: list[Candle], period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> IndicatorResult:
    """Calculate Stochastic Oscillator.
    
    Measures momentum by comparing closing price to price range over a period.
    """
    if len(candles) < period:
        return IndicatorResult(value=50.0, signal=50.0)
    
    recent = candles[-period:]
    low = min(c.low for c in recent)
    high = max(c.high for c in recent)
    close = candles[-1].close
    
    if high == low:
        k_percent = 50.0
    else:
        k_percent = 100.0 * (close - low) / (high - low)
    
    # Smooth K
    k_values = []
    for i in range(period, len(candles)):
        window = candles[i - period + 1:i + 1]
        low = min(c.low for c in window)
        high = max(c.high for c in window)
        close = candles[i].close
        if high == low:
            k_values.append(50.0)
        else:
            k_values.append(100.0 * (close - low) / (high - low))
    
    if len(k_values) < smooth_k:
        return IndicatorResult(value=k_percent, signal=k_percent)
    
    k_smooth = sum(k_values[-smooth_k:]) / smooth_k
    d_smooth = sum(k_values[-smooth_d:]) / smooth_d if len(k_values) >= smooth_d else k_smooth
    
    return IndicatorResult(value=k_smooth, signal=d_smooth)


def bollinger_bands(candles: list[Candle], period: int = 20, std_dev: float = 2.0) -> dict[str, float]:
    """Calculate Bollinger Bands.
    
    Returns upper band, middle band (SMA), lower band, and bandwidth percentage.
    """
    if len(candles) < period:
        close = candles[-1].close if candles else 0.0
        return {"upper": close, "middle": close, "lower": close, "bandwidth": 0.0}
    
    closes = [c.close for c in candles[-period:]]
    sma = sum(closes) / len(closes)
    
    variance = sum((c - sma) ** 2 for c in closes) / len(closes)
    std = variance ** 0.5
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    bandwidth = ((upper - lower) / sma * 100) if sma != 0 else 0.0
    
    position = (candles[-1].close - lower) / (upper - lower) if upper != lower else 0.5
    position = max(0.0, min(1.0, position))  # Clamp to [0.0, 1.0]
    
    return {
        "upper": upper,
        "middle": sma,
        "lower": lower,
        "bandwidth": bandwidth,
        "position": position,
    }


def volume_profile(candles: list[Candle], bins: int = 20) -> dict[str, Any]:
    """Calculate volume profile across price levels.
    
    Identifies support/resistance levels based on volume concentration.
    """
    if not candles:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0, "profile": {}}
    
    low = min(c.low for c in candles)
    high = max(c.high for c in candles)
    
    if high == low:
        return {"poc": low, "vah": low, "val": low, "profile": {}}
    
    bin_size = (high - low) / bins
    profile = {i: 0.0 for i in range(bins)}
    
    for candle in candles:
        bin_idx = min(int((candle.close - low) / bin_size), bins - 1)
        profile[bin_idx] += candle.volume
    
    max_volume = max(profile.values()) if profile else 0.0
    poc_bin = max(profile, key=profile.get) if profile else 0
    poc = low + (poc_bin + 0.5) * bin_size
    
    # Calculate VAH (Value Area High) and VAL (Value Area Low)
    sorted_bins = sorted(profile.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0.0
    total_volume = sum(profile.values())
    va_threshold = total_volume * 0.68
    
    va_bins = []
    for bin_idx, vol in sorted_bins:
        cumulative += vol
        va_bins.append(bin_idx)
        if cumulative >= va_threshold:
            break
    
    if va_bins:
        vah = low + (max(va_bins) + 1) * bin_size
        val = low + (min(va_bins)) * bin_size
    else:
        vah = poc
        val = poc
    
    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "profile": profile,
        "max_volume": max_volume,
    }


def volume_profile_strength(
    candles: list[Candle],
    current_price: float,
    *,
    bins: int = 20,
    near_threshold_pct: float = 1.5,
) -> dict[str, Any]:
    """Compute proximity of current_price to POC, VAH, and VAL.

    Returns a dict with the raw levels plus boolean 'near_*' flags.
    near_threshold_pct: price is considered 'near' a level if within this %.
    """
    vp = volume_profile(candles, bins=bins)
    poc = vp["poc"]
    vah = vp["vah"]
    val = vp["val"]

    def dist_pct(level: float) -> float:
        return abs(current_price - level) / max(level, 1e-9) * 100.0

    return {
        "poc": poc,
        "vah": vah,
        "val": val,
        "dist_to_poc_pct": dist_pct(poc),
        "dist_to_vah_pct": dist_pct(vah),
        "dist_to_val_pct": dist_pct(val),
        "near_poc": dist_pct(poc) < near_threshold_pct,
        "near_vah": dist_pct(vah) < near_threshold_pct,
        "near_val": dist_pct(val) < near_threshold_pct,
    }


def rsi_divergence(candles: list[Candle], period: int = 14, lookback: int = 20) -> tuple[bool, str]:
    """Detect RSI divergence patterns.
    
    Returns (divergence_detected, divergence_type) where type is 'bullish' or 'bearish'.
    """
    if len(candles) < lookback + period:
        return False, "none"
    
    recent = candles[-lookback:]
    rsi_values = [rsi(candles[:i + 1], period) for i in range(len(candles) - lookback, len(candles))]
    
    prices = [c.close for c in recent]
    
    # Find local lows and highs
    if len(prices) < 3:
        return False, "none"
    
    # Bullish divergence: lower lows in price, higher lows in RSI
    price_low_idx = prices.index(min(prices))
    rsi_low_idx = rsi_values.index(min(rsi_values))
    
    if price_low_idx > 0 and rsi_low_idx > 0:
        prev_price_low = min(prices[:price_low_idx])
        prev_rsi_low = min(rsi_values[:rsi_low_idx])
        
        if prices[price_low_idx] < prev_price_low and rsi_values[rsi_low_idx] > prev_rsi_low:
            return True, "bullish"
    
    # Bearish divergence: higher highs in price, lower highs in RSI
    price_high_idx = prices.index(max(prices))
    rsi_high_idx = rsi_values.index(max(rsi_values))
    
    if price_high_idx > 0 and rsi_high_idx > 0:
        prev_price_high = max(prices[:price_high_idx])
        prev_rsi_high = max(rsi_values[:rsi_high_idx])
        
        if prices[price_high_idx] > prev_price_high and rsi_values[rsi_high_idx] < prev_rsi_high:
            return True, "bearish"
    
    return False, "none"


def _ema(values: list[float], period: int) -> float:
    """Calculate Exponential Moving Average."""
    if not values or period < 1:
        return values[-1] if values else 0.0
    
    if len(values) < period:
        return sum(values) / len(values)
    
    multiplier = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    
    for value in values[period:]:
        ema = value * multiplier + ema * (1 - multiplier)
    
    return ema
