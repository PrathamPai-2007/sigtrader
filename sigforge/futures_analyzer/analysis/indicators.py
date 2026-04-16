"""Advanced technical indicators for enhanced prediction accuracy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from futures_analyzer.analysis.models import Candle, IndicatorBundle, MarketMeta
from futures_analyzer.config import AppConfig, load_app_config


@dataclass
class LiquiditySweep:
    """Records a price excursion beyond a prior swing high/low that closes back inside."""

    direction: str   # "above" | "below"
    swept_level: float
    bar_index: int
    recovered: bool  # True if price closed back inside the range (stop hunt)


@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""
    value: float
    signal: float | None = None
    histogram: float | None = None
    divergence: bool = False


def sigmoid(x: float) -> float:
    """Compute the logistic sigmoid: 1 / (1 + exp(-x)), clamped to [0.0, 1.0].

    Handles overflow for very large positive x (returns 1.0) and very large
    negative x (returns 0.0) via try/except OverflowError.
    """
    try:
        result = 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        # exp(-x) overflows when x is very large negative → sigmoid → 0.0
        # exp(-x) underflows (no error) when x is very large positive → sigmoid → 1.0
        result = 0.0 if x < 0 else 1.0
    return max(0.0, min(1.0, result))


def gaussian_peak(x: float, center: float, width: float) -> float:
    """Compute a Gaussian bell curve: exp(-0.5 * ((x - center) / width) ** 2).

    Clamped to [0.0, 1.0].  When width is 0 (or effectively zero), returns 1.0
    if x == center, else 0.0.
    """
    if width == 0.0 or abs(width) < 1e-12:
        return 1.0 if x == center else 0.0
    result = math.exp(-0.5 * ((x - center) / width) ** 2)
    return max(0.0, min(1.0, result))


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

    Computes the full fast/slow EMA series in a single O(n) forward pass,
    then applies _ema once on the resulting MACD values for the signal line.
    Returns MACD line, signal line, and histogram for momentum analysis.
    """
    if len(candles) < slow + signal:
        return IndicatorResult(value=0.0, signal=0.0, histogram=0.0)

    closes = [c.close for c in candles]
    k_fast = 2.0 / (fast + 1)
    k_slow = 2.0 / (slow + 1)

    # Seed both EMAs with their respective SMA over the first `slow` bars
    ema_f = sum(closes[:slow]) / slow
    ema_s = sum(closes[:slow]) / slow

    # Warm up the fast EMA independently over its own seed window, then
    # continue from bar `slow` onward so both series stay in sync.
    ema_f = sum(closes[:fast]) / fast
    for c in closes[fast:slow]:
        ema_f = c * k_fast + ema_f * (1 - k_fast)

    # Build the MACD series in a single forward pass from bar `slow` onward
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


def _swing_pivots(
    values: list[float],
    n: int = 3,
) -> list[tuple[int, float]]:
    """Return confirmed swing pivot indices and values.

    A pivot low at index i requires values[i] to be the minimum of the
    surrounding 2*n+1 window (n bars to the left and n bars to the right).
    A pivot high is the maximum of the same window.

    Returns a list of (index, value) tuples for *all* pivots (both highs and
    lows) in ascending index order.  Callers filter by direction.
    """
    pivots: list[tuple[int, float]] = []
    length = len(values)
    for i in range(n, length - n):
        window = values[i - n: i + n + 1]
        v = values[i]
        if v == min(window) or v == max(window):
            pivots.append((i, v))
    return pivots


def rsi_divergence(
    candles: list[Candle],
    period: int = 14,
    lookback: int = 30,
    pivot_n: int = 2,
    min_price_move_pct: float = 0.5,
) -> tuple[bool, str, float]:
    """Detect RSI divergence using confirmed swing pivots.

    Replaces the previous global-extremes approach which fired on almost any
    trending series.  This version:
    - Identifies local swing lows/highs via an N-bar left/right comparison.
    - Compares only the two most recent confirmed pivots of the same type.
    - Requires a minimum price move between pivots to filter noise.
    - Returns divergence strength (absolute RSI distance between the two
      pivot RSI values) as the third element.

    Returns:
        (divergence_detected, divergence_type, strength)
        divergence_type: 'bullish' | 'bearish' | 'none'
        strength: RSI-point distance between the two pivot RSI values (0.0 when
                  no divergence).
    """
    # Need enough bars for RSI warm-up + lookback + pivot wings
    min_required = period + lookback + pivot_n
    if len(candles) < min_required:
        return False, "none", 0.0

    # Work on the lookback window only, but compute RSI over the full series
    # so the warm-up is always satisfied.
    window = candles[-lookback:]
    offset = len(candles) - lookback  # index of window[0] in candles

    prices = [c.close for c in window]

    # Compute RSI once over the full candle series in a single O(n) pass,
    # then slice out the values that correspond to the lookback window.
    # This avoids the previous O(n²) approach of calling rsi() per bar.
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
    # _all_rsi[i] corresponds to candles[period + i + 1] (1-indexed delta offset)
    # Map each window position to its RSI value; fall back to 50 if not yet warmed up.
    rsi_vals: list[float] = []
    for i in range(len(window)):
        candle_idx = offset + i          # absolute index in candles[]
        rsi_idx = candle_idx - period - 1  # index into _all_rsi
        rsi_vals.append(_all_rsi[rsi_idx] if rsi_idx >= 0 else 50.0)

    # --- Bullish divergence: lower price low, higher RSI low ---
    # Keep only confirmed lows (value equals the window minimum in its neighbourhood)
    price_lows = [
        (i, v) for i, v in _swing_pivots(prices, n=pivot_n)
        if prices[i] == min(prices[max(0, i - pivot_n): i + pivot_n + 1])
    ]

    if len(price_lows) >= 2:
        (i1, p1), (i2, p2) = price_lows[-2], price_lows[-1]
        r1, r2 = rsi_vals[i1], rsi_vals[i2]
        price_move_pct = abs(p2 - p1) / max(p1, 1e-9) * 100.0
        if (
            p2 < p1                        # lower price low
            and r2 > r1                    # higher RSI low
            and price_move_pct >= min_price_move_pct
        ):
            return True, "bullish", round(r2 - r1, 2)

    # --- Bearish divergence: higher price high, lower RSI high ---
    price_highs = [
        (i, v) for i, v in _swing_pivots(prices, n=pivot_n)
        if prices[i] == max(prices[max(0, i - pivot_n): i + pivot_n + 1])
    ]

    if len(price_highs) >= 2:
        (i1, p1), (i2, p2) = price_highs[-2], price_highs[-1]
        r1, r2 = rsi_vals[i1], rsi_vals[i2]
        price_move_pct = abs(p2 - p1) / max(p1, 1e-9) * 100.0
        if (
            p2 > p1                        # higher price high
            and r2 < r1                    # lower RSI high
            and price_move_pct >= min_price_move_pct
        ):
            return True, "bearish", round(r1 - r2, 2)

    return False, "none", 0.0


def adx(candles: list[Candle], period: int = 14) -> dict[str, float]:
    """Calculate Average Directional Index (ADX) using Wilder's smoothing method.

    Returns {"adx": float, "plus_di": float, "minus_di": float}.
    Falls back to all-zero dict when len(candles) < 2 * period.
    """
    _zero = {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}
    if len(candles) < 2 * period:
        return _zero

    # Step 1: compute TR, +DM, -DM for each bar (starting from index 1)
    tr_list: list[float] = []
    plus_dm_list: list[float] = []
    minus_dm_list: list[float] = []

    for i in range(1, len(candles)):
        prev = candles[i - 1]
        curr = candles[i]

        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close),
        )

        up_move = curr.high - prev.high
        down_move = prev.low - curr.low

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    # Step 2: seed first smoothed values with simple sum over first `period` bars
    smoothed_tr = sum(tr_list[:period])
    smoothed_plus_dm = sum(plus_dm_list[:period])
    smoothed_minus_dm = sum(minus_dm_list[:period])

    # Step 3: compute first period DX values (seeding ADX)
    dx_list: list[float] = []

    def _dx(s_tr: float, s_pdm: float, s_mdm: float) -> float:
        if s_tr == 0.0:
            return 0.0
        pdi = 100.0 * s_pdm / s_tr
        mdi = 100.0 * s_mdm / s_tr
        denom = pdi + mdi
        return 100.0 * abs(pdi - mdi) / denom if denom != 0.0 else 0.0

    dx_list.append(_dx(smoothed_tr, smoothed_plus_dm, smoothed_minus_dm))

    # Step 4: Wilder-smooth through remaining bars, collecting DX values
    k = 1.0 / period
    for i in range(period, len(tr_list)):
        smoothed_tr = smoothed_tr * (1.0 - k) + tr_list[i]
        smoothed_plus_dm = smoothed_plus_dm * (1.0 - k) + plus_dm_list[i]
        smoothed_minus_dm = smoothed_minus_dm * (1.0 - k) + minus_dm_list[i]
        dx_list.append(_dx(smoothed_tr, smoothed_plus_dm, smoothed_minus_dm))

    # Step 5: seed ADX as mean of first `period` DX values, then Wilder-smooth
    if len(dx_list) < period:
        return _zero

    adx_val = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx_val = adx_val * (1.0 - k) + dx * k

    # Final +DI / -DI from last smoothed values
    plus_di = (100.0 * smoothed_plus_dm / smoothed_tr) if smoothed_tr != 0.0 else 0.0
    minus_di = (100.0 * smoothed_minus_dm / smoothed_tr) if smoothed_tr != 0.0 else 0.0

    return {
        "adx": float(adx_val),
        "plus_di": float(plus_di),
        "minus_di": float(minus_di),
    }


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


def compute_market_structure(candles: list[Candle], pivot_n: int = 2) -> str:
    """Classify recent swing pivot sequence as HH_HL, LH_LL, or mixed.

    Uses close prices to identify swing pivots via _swing_pivots.
    Returns "HH_HL" when the last two pivot highs are higher highs and the
    last two pivot lows are higher lows.  Returns "LH_LL" for the inverse.
    Returns "mixed" when there are fewer than 2 pivot highs or 2 pivot lows,
    or when the pattern does not match either trend definition.

    Special case: a monotonically rising close series (each close strictly
    greater than the previous) is treated as "HH_HL"; a monotonically falling
    series is treated as "LH_LL".  This handles the case where _swing_pivots
    finds no interior extrema in a perfectly trending series.
    """
    if not candles:
        return "mixed"

    closes = [c.close for c in candles]

    # Fast path: detect monotonic series before pivot search
    if len(closes) >= 2:
        if all(closes[i] > closes[i - 1] for i in range(1, len(closes))):
            return "HH_HL"
        if all(closes[i] < closes[i - 1] for i in range(1, len(closes))):
            return "LH_LL"

    pivots = _swing_pivots(closes, n=pivot_n)

    # Separate into highs and lows using the surrounding window
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

    higher_highs = h2 > h1
    higher_lows = l2 > l1
    lower_highs = h2 < h1
    lower_lows = l2 < l1

    if higher_highs and higher_lows:
        return "HH_HL"
    if lower_highs and lower_lows:
        return "LH_LL"
    return "mixed"


def compute_cumulative_delta(candles: list[Candle], window: int = 20) -> float:
    """Compute net buy/sell pressure over the last `window` candles, normalized to [-1, 1].

    Uses (close - open) / max(high - low, 1e-9) as a proxy for buy/sell volume
    when tick data is unavailable.  Sums the per-candle deltas over the window
    and divides by window size to normalize.  Result is clamped to [-1, 1].

    Returns 0.0 on empty input.
    """
    if not candles:
        return 0.0

    recent = candles[-window:]
    total = 0.0
    for c in recent:
        rng = max(c.high - c.low, 1e-9)
        total += (c.close - c.open) / rng

    normalized = total / window
    return max(-1.0, min(1.0, normalized))


def compute_adx_slope(candles: list[Candle], lookback: int = 5) -> float:
    """Compute the linear regression slope of ADX values over the last `lookback` bars.

    Calls adx(candles[:i]) for each of the last `lookback` bars to build the ADX
    series, then fits a simple linear regression (OLS) slope.

    Returns 0.0 when:
    - len(candles) < lookback
    - the denominator of the slope formula is zero (flat x-series, impossible in
      practice but guarded for safety)
    """
    if len(candles) < lookback:
        return 0.0

    n = lookback
    start = len(candles) - lookback + 1  # first slice ends here (exclusive)

    # Build ADX series: adx(candles[:i]) for i in [start, len(candles)+1)
    adx_values: list[float] = []
    for i in range(start, len(candles) + 1):
        result = adx(candles[:i])
        adx_values.append(result["adx"])

    # x = [0, 1, ..., n-1], y = adx_values
    x = list(range(n))
    sum_x = sum(x)
    sum_y = sum(adx_values)
    sum_xy = sum(x[i] * adx_values[i] for i in range(n))
    sum_x2 = sum(xi * xi for xi in x)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0.0:
        return 0.0

    return (n * sum_xy - sum_x * sum_y) / denom


def detect_liquidity_sweeps(candles: list[Candle], swing_n: int = 2) -> list[LiquiditySweep]:
    """Detect liquidity sweeps: bars that exceed a prior swing high/low.

    A sweep above is when bar.high > a prior swing high.  If bar.close < that
    swing high the price recovered (stop hunt, recovered=True); otherwise
    recovered=False.

    A sweep below is when bar.low < a prior swing low.  If bar.close > that
    swing low the price recovered (recovered=True); otherwise recovered=False.

    Returns sweeps in chronological order (ascending bar_index).
    """
    if not candles:
        return []

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    # Collect confirmed swing highs and lows (index, value)
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
        # Check sweeps above prior swing highs
        for sh_idx, sh_val in swing_high_pivots:
            if sh_idx >= i:
                # Only consider pivots that are strictly prior to bar i
                continue
            if bar.high > sh_val:
                recovered = bar.close < sh_val
                sweeps.append(LiquiditySweep(
                    direction="above",
                    swept_level=sh_val,
                    bar_index=i,
                    recovered=recovered,
                ))

        # Check sweeps below prior swing lows
        for sl_idx, sl_val in swing_low_pivots:
            if sl_idx >= i:
                continue
            if bar.low < sl_val:
                recovered = bar.close > sl_val
                sweeps.append(LiquiditySweep(
                    direction="below",
                    swept_level=sl_val,
                    bar_index=i,
                    recovered=recovered,
                ))

    # Sort chronologically (ascending bar_index)
    sweeps.sort(key=lambda s: s.bar_index)
    return sweeps


def compute_vwap_bands(candles: list[Candle]) -> dict[str, float]:
    """Compute VWAP with ±1 and ±2 standard deviation bands.

    Typical price = (high + low + close) / 3.
    VWAP = sum(typical_price * volume) / sum(volume).
    Standard deviation is volume-weighted: sqrt(sum(volume * (tp - vwap)^2) / sum(volume)).

    Safe defaults: when volume is zero or candles are insufficient, all bands
    are set to the last close (or 0.0 if no candles), so the strict ordering
    invariant is NOT required to hold for zero-volume data.

    For any candles with positive volume the returned dict satisfies:
        lower_2sd < lower_1sd < vwap < upper_1sd < upper_2sd
    """
    if not candles:
        return {
            "vwap": 0.0,
            "upper_1sd": 0.0,
            "lower_1sd": 0.0,
            "upper_2sd": 0.0,
            "lower_2sd": 0.0,
        }

    last_close = candles[-1].close

    total_volume = sum(c.volume for c in candles)
    if total_volume <= 0.0:
        return {
            "vwap": last_close,
            "upper_1sd": last_close,
            "lower_1sd": last_close,
            "upper_2sd": last_close,
            "lower_2sd": last_close,
        }

    # Compute VWAP
    tp_vol_sum = sum((c.high + c.low + c.close) / 3.0 * c.volume for c in candles)
    vwap = tp_vol_sum / total_volume

    # Volume-weighted variance of typical price around VWAP
    variance = sum(c.volume * ((c.high + c.low + c.close) / 3.0 - vwap) ** 2 for c in candles) / total_volume
    sd = variance ** 0.5

    # Ensure strict ordering even when sd is very small but positive
    # (sd == 0 only when all typical prices are identical, which still satisfies
    # the invariant because the bands would all equal vwap — but the requirement
    # says strictly less/greater, so we use a tiny epsilon when sd is zero)
    if sd == 0.0:
        # All typical prices are the same; bands collapse to vwap.
        # The ordering invariant requires strict inequality, so we use a minimal
        # epsilon derived from the price scale to separate the bands.
        sd = max(vwap * 1e-9, 1e-12)

    return {
        "vwap": vwap,
        "upper_1sd": vwap + sd,
        "lower_1sd": vwap - sd,
        "upper_2sd": vwap + 2.0 * sd,
        "lower_2sd": vwap - 2.0 * sd,
    }


def _compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Compute ATR as average of last `period` true ranges. Returns 0.0 when insufficient data."""
    if len(candles) < 2:
        return 0.0
    trs: list[float] = []
    prev_close = candles[0].close
    for c in candles[1:]:
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        prev_close = c.close
    if not trs:
        return 0.0
    tail = trs[-period:] if len(trs) >= period else trs
    return sum(tail) / len(tail)


def _ema_value(closes: list[float], period: int) -> float:
    """Compute EMA of a close series. Returns last close when insufficient data."""
    if not closes:
        return 0.0
    if len(closes) < period:
        return closes[-1]
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema


def _funding_momentum_local(history: list[float], window: int = 4) -> float:
    """Compute slope of last `window` funding rate values, normalised to [-1, 1]."""
    from futures_analyzer.analysis.scoring.utils import _clamp as _cl
    slope_scale = load_app_config().strategy.funding_momentum_slope_scale
    tail = history[-window:] if len(history) >= window else history
    n = len(tail)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(tail) / n
    num = sum((xs[i] - x_mean) * (tail[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    return _cl(slope * slope_scale, -1.0, 1.0)


def compute_all_indicators(
    entry: list[Candle],
    trigger: list[Candle],
    context: list[Candle],
    higher: list[Candle],
    market_meta: MarketMeta,
    indicator_params=None,
    signal_transforms=None,
    config: AppConfig | None = None,
) -> IndicatorBundle:
    """Compute all indicators in a single coordinated pass over the four timeframe candle lists.

    Returns a fully-populated IndicatorBundle. Safe neutral defaults are used for any
    indicator when candle data is insufficient; a warning string is appended to the
    returned bundle's `warnings` list.
    """
    from futures_analyzer.analysis.scoring.utils import _clamp as _cl
    warnings_list: list[str] = []

    # --- Per-timeframe ATR ---
    entry_atr = _compute_atr(entry)
    trigger_atr = _compute_atr(trigger)
    context_atr = _compute_atr(context)
    higher_atr = _compute_atr(higher)

    # --- Config resolution ---
    if indicator_params is None or signal_transforms is None:
        cfg = config or load_app_config()
        _strategy = cfg.strategy
        _ip = indicator_params if indicator_params is not None else _strategy.indicator_params
        _st = signal_transforms if signal_transforms is not None else _strategy.signal_transforms
    else:
        _ip = indicator_params
        _st = signal_transforms
    _ema_fast = _ip.ema_fast_period
    _ema_slow = _ip.ema_slow_period
    _trend_tanh_scale = _st.trend_tanh_scale
    _momentum_tanh_scale = _st.momentum_tanh_scale
    _vol_window = _ip.volume_window
    _roc_period = _ip.roc_period

    # --- Trend: EMA(fast) vs EMA(slow) slope, normalised via tanh ---
    def _trend_ema(candles: list[Candle], label: str) -> float:
        if len(candles) < _ema_slow:
            warnings_list.append(
                f"Insufficient candles for {label} trend (need {_ema_slow}, got {len(candles)}); defaulting to 0.0"
            )
            return 0.0
        closes = [c.close for c in candles]
        ema_f = _ema_value(closes, _ema_fast)
        ema_s = _ema_value(closes, _ema_slow)
        if ema_s == 0.0:
            return 0.0
        raw = (ema_f - ema_s) / ema_s
        return math.tanh(raw * _trend_tanh_scale)

    higher_trend = _trend_ema(higher, "higher")
    context_trend = _trend_ema(context, "context")

    # --- Momentum: rate of change, normalised via tanh ---
    def _roc_momentum(candles: list[Candle], label: str) -> float:
        need = _roc_period + 1
        if len(candles) < need:
            warnings_list.append(
                f"Insufficient candles for {label} momentum (need {need}, got {len(candles)}); defaulting to 0.0"
            )
            return 0.0
        first = candles[-need].close
        last = candles[-1].close
        if first <= 0:
            return 0.0
        raw = last / first - 1.0
        return _cl(math.tanh(raw * _momentum_tanh_scale), -1.0, 1.0)

    trigger_momentum = _roc_momentum(trigger, "trigger")
    entry_momentum = _roc_momentum(entry, "entry")

    # --- Volume surge: last volume / mean(last N volumes) ---
    def _vol_surge(candles: list[Candle]) -> float:
        if len(candles) < 2:
            return 1.0
        window = candles[-_vol_window:] if len(candles) >= _vol_window else candles
        base = sum(c.volume for c in window) / len(window)
        if base <= 0:
            return 1.0
        return candles[-1].volume / base

    trigger_volume_surge = _vol_surge(trigger)
    entry_volume_surge = _vol_surge(entry)

    # --- Cumulative delta ---
    if trigger:
        cumulative_delta = compute_cumulative_delta(trigger)
    else:
        cumulative_delta = 0.0
        warnings_list.append("No trigger candles for cumulative_delta; defaulting to 0.0")

    # --- Oscillators on entry TF ---
    _atr_period = _ip.atr_period
    if len(entry) >= _atr_period + 1:
        rsi_14 = rsi(entry, period=_atr_period)
    else:
        rsi_14 = 50.0
        warnings_list.append(
            f"Insufficient entry candles for RSI (need {_atr_period + 1}, got {len(entry)}); defaulting to 50.0"
        )

    macd_result = macd(entry)
    macd_histogram = macd_result.histogram if macd_result.histogram is not None else 0.0

    stoch_result = stochastic(entry)
    stoch_k = stoch_result.value

    bb_result = bollinger_bands(entry)
    bb_position = bb_result.get("position", 0.5)
    bb_bandwidth_pct = bb_result.get("bandwidth", 0.0)

    # --- Swing highs/lows from trigger candles ---
    pivot_n = _ip.pivot_n

    if len(trigger) >= (2 * pivot_n + 1):
        highs = [c.high for c in trigger]
        lows = [c.low for c in trigger]
        raw_pivots = _swing_pivots(highs, n=pivot_n)
        swing_highs_raw = [
            val for idx, val in raw_pivots
            if val == max(highs[max(0, idx - pivot_n): idx + pivot_n + 1])
        ]
        raw_pivots_low = _swing_pivots(lows, n=pivot_n)
        swing_lows_raw = [
            val for idx, val in raw_pivots_low
            if val == min(lows[max(0, idx - pivot_n): idx + pivot_n + 1])
        ]
        swing_highs = sorted(swing_highs_raw)
        swing_lows = sorted(swing_lows_raw)
    else:
        swing_highs = []
        swing_lows = []
        warnings_list.append(
            f"Insufficient trigger candles for swing pivots (need {2 * pivot_n + 1}, got {len(trigger)}); defaulting to empty lists"
        )

    # --- Market structure ---
    if len(trigger) >= 7:
        market_structure = compute_market_structure(trigger)
    else:
        market_structure = "mixed"
        warnings_list.append(
            f"Insufficient trigger candles for market_structure (need 7, got {len(trigger)}); defaulting to 'mixed'"
        )

    # --- Liquidity sweeps ---
    if len(trigger) >= 7:
        liquidity_sweeps = detect_liquidity_sweeps(trigger)
    else:
        liquidity_sweeps = []
        warnings_list.append(
            f"Insufficient trigger candles for liquidity_sweeps (need 7, got {len(trigger)}); defaulting to []"
        )

    # --- VWAP bands from trigger candles ---
    if trigger:
        vwap_bands = compute_vwap_bands(trigger)
        vwap_val = vwap_bands["vwap"]
        vwap_upper_1sd = vwap_bands["upper_1sd"]
        vwap_lower_1sd = vwap_bands["lower_1sd"]
        vwap_upper_2sd = vwap_bands["upper_2sd"]
        vwap_lower_2sd = vwap_bands["lower_2sd"]
    else:
        last_close = 0.0
        vwap_val = last_close
        vwap_upper_1sd = last_close
        vwap_lower_1sd = last_close
        vwap_upper_2sd = last_close
        vwap_lower_2sd = last_close
        warnings_list.append("No trigger candles for VWAP bands; defaulting to 0.0")

    # --- Volume profile from trigger candles ---
    if trigger:
        vp = volume_profile(trigger)
        poc = vp["poc"]
        vah = vp["vah"]
        val = vp["val"]
    else:
        last_close = trigger[-1].close if trigger else 0.0
        poc = last_close
        vah = last_close
        val = last_close
        warnings_list.append("No trigger candles for volume_profile; defaulting to last close")

    # --- RSI divergence on entry TF ---
    min_div_bars = 14 + 30 + 3  # period + lookback + pivot_n
    if len(entry) >= min_div_bars:
        div_detected, div_type, div_strength = rsi_divergence(entry)
    else:
        div_detected, div_type, div_strength = False, "none", 0.0
        warnings_list.append(
            f"Insufficient entry candles for RSI divergence (need {min_div_bars}, got {len(entry)}); defaulting to none"
        )

    rsi_divergence_type = div_type
    rsi_divergence_strength = div_strength

    # --- OI / funding from market_meta ---
    funding_rate = getattr(market_meta, "funding_rate", None)
    oi_change_pct = getattr(market_meta, "open_interest_change_pct", None)
    funding_history = getattr(market_meta, "funding_rate_history", None) or []
    funding_momentum_val = _funding_momentum_local(funding_history)

    # --- Order book ---
    order_book_imbalance = getattr(market_meta, "order_book_imbalance", 0.0) or 0.0
    bid_ask_spread_pct = 0.0  # placeholder

    return IndicatorBundle(
        entry_atr=entry_atr,
        trigger_atr=trigger_atr,
        context_atr=context_atr,
        higher_atr=higher_atr,
        higher_trend=higher_trend,
        context_trend=context_trend,
        trigger_momentum=trigger_momentum,
        entry_momentum=entry_momentum,
        trigger_volume_surge=trigger_volume_surge,
        entry_volume_surge=entry_volume_surge,
        cumulative_delta=cumulative_delta,
        rsi_14=rsi_14,
        macd_histogram=macd_histogram,
        stoch_k=stoch_k,
        bb_position=bb_position,
        bb_bandwidth_pct=bb_bandwidth_pct,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        market_structure=market_structure,
        liquidity_sweeps=liquidity_sweeps,
        vwap=vwap_val,
        vwap_upper_1sd=vwap_upper_1sd,
        vwap_lower_1sd=vwap_lower_1sd,
        vwap_upper_2sd=vwap_upper_2sd,
        vwap_lower_2sd=vwap_lower_2sd,
        poc=poc,
        vah=vah,
        val=val,
        rsi_divergence_type=rsi_divergence_type,
        rsi_divergence_strength=rsi_divergence_strength,
        funding_rate=funding_rate,
        oi_change_pct=oi_change_pct,
        funding_momentum=funding_momentum_val,
        order_book_imbalance=order_book_imbalance,
        bid_ask_spread_pct=bid_ask_spread_pct,
        warnings=warnings_list,
    )


def _compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Compute ATR as average of last `period` true ranges. Returns 0.0 when insufficient data."""
    if len(candles) < 2:
        return 0.0
    trs: list[float] = []
    prev_close = candles[0].close
    for c in candles[1:]:
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
        prev_close = c.close
    if not trs:
        return 0.0
    tail = trs[-period:] if len(trs) >= period else trs
    return sum(tail) / len(tail)


def _ema_value(closes: list[float], period: int) -> float:
    """Compute EMA of a close series. Returns last close when insufficient data."""
    if not closes:
        return 0.0
    if len(closes) < period:
        return closes[-1]
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema


def _funding_momentum_local(history: list[float], window: int = 4) -> float:
    """Compute slope of last `window` funding rate values, normalised to [-1, 1]."""
    from futures_analyzer.analysis.scoring.utils import _clamp as _cl
    slope_scale = load_app_config().strategy.funding_momentum_slope_scale
    tail = history[-window:] if len(history) >= window else history
    n = len(tail)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(tail) / n
    num = sum((xs[i] - x_mean) * (tail[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    return _cl(slope * slope_scale, -1.0, 1.0)


def compute_all_indicators(
    entry: list[Candle],
    trigger: list[Candle],
    context: list[Candle],
    higher: list[Candle],
    market_meta: MarketMeta,
    indicator_params=None,
    signal_transforms=None,
    config: AppConfig | None = None,
) -> IndicatorBundle:
    """Compute all indicators in a single coordinated pass over the four timeframe candle lists.

    Returns a fully-populated IndicatorBundle. Safe neutral defaults are used for any
    indicator when candle data is insufficient; a warning string is appended to the
    returned bundle's `warnings` list.
    """
    from futures_analyzer.analysis.scoring.utils import _clamp as _cl
    warnings_list: list[str] = []

    # --- Per-timeframe ATR ---
    entry_atr = _compute_atr(entry)
    trigger_atr = _compute_atr(trigger)
    context_atr = _compute_atr(context)
    higher_atr = _compute_atr(higher)

    # --- Config resolution ---
    if indicator_params is None or signal_transforms is None:
        cfg = config or load_app_config()
        _strategy = cfg.strategy
        _ip = indicator_params if indicator_params is not None else _strategy.indicator_params
        _st = signal_transforms if signal_transforms is not None else _strategy.signal_transforms
    else:
        _ip = indicator_params
        _st = signal_transforms
    _ema_fast = _ip.ema_fast_period
    _ema_slow = _ip.ema_slow_period
    _trend_tanh_scale = _st.trend_tanh_scale
    _momentum_tanh_scale = _st.momentum_tanh_scale
    _vol_window = _ip.volume_window
    _roc_period = _ip.roc_period

    # --- Trend: EMA(fast) vs EMA(slow) slope, normalised via tanh ---
    def _trend_ema(candles: list[Candle], label: str) -> float:
        if len(candles) < _ema_slow:
            warnings_list.append(
                f"Insufficient candles for {label} trend (need {_ema_slow}, got {len(candles)}); defaulting to 0.0"
            )
            return 0.0
        closes = [c.close for c in candles]
        ema_f = _ema_value(closes, _ema_fast)
        ema_s = _ema_value(closes, _ema_slow)
        if ema_s == 0.0:
            return 0.0
        raw = (ema_f - ema_s) / ema_s
        return math.tanh(raw * _trend_tanh_scale)

    higher_trend = _trend_ema(higher, "higher")
    context_trend = _trend_ema(context, "context")

    # --- Momentum: rate of change, normalised via tanh ---
    def _roc_momentum(candles: list[Candle], label: str) -> float:
        need = _roc_period + 1
        if len(candles) < need:
            warnings_list.append(
                f"Insufficient candles for {label} momentum (need {need}, got {len(candles)}); defaulting to 0.0"
            )
            return 0.0
        first = candles[-need].close
        last = candles[-1].close
        if first <= 0:
            return 0.0
        raw = last / first - 1.0
        return _cl(math.tanh(raw * _momentum_tanh_scale), -1.0, 1.0)

    trigger_momentum = _roc_momentum(trigger, "trigger")
    entry_momentum = _roc_momentum(entry, "entry")

    # --- Volume surge: last volume / mean(last N volumes) ---
    def _vol_surge(candles: list[Candle]) -> float:
        if len(candles) < 2:
            return 1.0
        window = candles[-_vol_window:] if len(candles) >= _vol_window else candles
        base = sum(c.volume for c in window) / len(window)
        if base <= 0:
            return 1.0
        return candles[-1].volume / base

    trigger_volume_surge = _vol_surge(trigger)
    entry_volume_surge = _vol_surge(entry)

    # --- Cumulative delta ---
    if trigger:
        cumulative_delta = compute_cumulative_delta(trigger)
    else:
        cumulative_delta = 0.0
        warnings_list.append("No trigger candles for cumulative_delta; defaulting to 0.0")

    # --- Oscillators on entry TF ---
    _atr_period = _ip.atr_period
    if len(entry) >= _atr_period + 1:
        rsi_14 = rsi(entry, period=_atr_period)
    else:
        rsi_14 = 50.0
        warnings_list.append(
            f"Insufficient entry candles for RSI (need {_atr_period + 1}, got {len(entry)}); defaulting to 50.0"
        )

    macd_result = macd(entry)
    macd_histogram = macd_result.histogram if macd_result.histogram is not None else 0.0

    stoch_result = stochastic(entry)
    stoch_k = stoch_result.value

    bb_result = bollinger_bands(entry)
    bb_position = bb_result.get("position", 0.5)
    bb_bandwidth_pct = bb_result.get("bandwidth", 0.0)

    # --- Swing highs/lows from trigger candles ---
    pivot_n = _ip.pivot_n

    if len(trigger) >= (2 * pivot_n + 1):
        highs = [c.high for c in trigger]
        lows = [c.low for c in trigger]
        raw_pivots = _swing_pivots(highs, n=pivot_n)
        swing_highs_raw = [
            val for idx, val in raw_pivots
            if val == max(highs[max(0, idx - pivot_n): idx + pivot_n + 1])
        ]
        raw_pivots_low = _swing_pivots(lows, n=pivot_n)
        swing_lows_raw = [
            val for idx, val in raw_pivots_low
            if val == min(lows[max(0, idx - pivot_n): idx + pivot_n + 1])
        ]
        swing_highs = sorted(swing_highs_raw)
        swing_lows = sorted(swing_lows_raw)
    else:
        swing_highs = []
        swing_lows = []
        warnings_list.append(
            f"Insufficient trigger candles for swing pivots (need {2 * pivot_n + 1}, got {len(trigger)}); defaulting to empty lists"
        )

    # --- Market structure ---
    if len(trigger) >= 7:
        market_structure = compute_market_structure(trigger)
    else:
        market_structure = "mixed"
        warnings_list.append(
            f"Insufficient trigger candles for market_structure (need 7, got {len(trigger)}); defaulting to 'mixed'"
        )

    # --- Liquidity sweeps ---
    if len(trigger) >= 7:
        liquidity_sweeps = detect_liquidity_sweeps(trigger)
    else:
        liquidity_sweeps = []
        warnings_list.append(
            f"Insufficient trigger candles for liquidity_sweeps (need 7, got {len(trigger)}); defaulting to []"
        )

    # --- VWAP bands from trigger candles ---
    if trigger:
        vwap_bands = compute_vwap_bands(trigger)
        vwap_val = vwap_bands["vwap"]
        vwap_upper_1sd = vwap_bands["upper_1sd"]
        vwap_lower_1sd = vwap_bands["lower_1sd"]
        vwap_upper_2sd = vwap_bands["upper_2sd"]
        vwap_lower_2sd = vwap_bands["lower_2sd"]
    else:
        last_close = 0.0
        vwap_val = last_close
        vwap_upper_1sd = last_close
        vwap_lower_1sd = last_close
        vwap_upper_2sd = last_close
        vwap_lower_2sd = last_close
        warnings_list.append("No trigger candles for VWAP bands; defaulting to 0.0")

    # --- Volume profile from trigger candles ---
    if trigger:
        vp = volume_profile(trigger)
        poc = vp["poc"]
        vah = vp["vah"]
        val = vp["val"]
    else:
        last_close = trigger[-1].close if trigger else 0.0
        poc = last_close
        vah = last_close
        val = last_close
        warnings_list.append("No trigger candles for volume_profile; defaulting to last close")

    # --- RSI divergence on entry TF ---
    min_div_bars = 14 + 30 + 3  # period + lookback + pivot_n
    if len(entry) >= min_div_bars:
        div_detected, div_type, div_strength = rsi_divergence(entry)
    else:
        div_detected, div_type, div_strength = False, "none", 0.0
        warnings_list.append(
            f"Insufficient entry candles for RSI divergence (need {min_div_bars}, got {len(entry)}); defaulting to none"
        )

    rsi_divergence_type = div_type
    rsi_divergence_strength = div_strength

    # --- OI / funding from market_meta ---
    funding_rate = getattr(market_meta, "funding_rate", None)
    oi_change_pct = getattr(market_meta, "open_interest_change_pct", None)
    funding_history = getattr(market_meta, "funding_rate_history", None) or []
    funding_momentum_val = _funding_momentum_local(funding_history)

    # --- Order book ---
    order_book_imbalance = getattr(market_meta, "order_book_imbalance", 0.0) or 0.0
    bid_ask_spread_pct = 0.0  # placeholder

    return IndicatorBundle(
        entry_atr=entry_atr,
        trigger_atr=trigger_atr,
        context_atr=context_atr,
        higher_atr=higher_atr,
        higher_trend=higher_trend,
        context_trend=context_trend,
        trigger_momentum=trigger_momentum,
        entry_momentum=entry_momentum,
        trigger_volume_surge=trigger_volume_surge,
        entry_volume_surge=entry_volume_surge,
        cumulative_delta=cumulative_delta,
        rsi_14=rsi_14,
        macd_histogram=macd_histogram,
        stoch_k=stoch_k,
        bb_position=bb_position,
        bb_bandwidth_pct=bb_bandwidth_pct,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        market_structure=market_structure,
        liquidity_sweeps=liquidity_sweeps,
        vwap=vwap_val,
        vwap_upper_1sd=vwap_upper_1sd,
        vwap_lower_1sd=vwap_lower_1sd,
        vwap_upper_2sd=vwap_upper_2sd,
        vwap_lower_2sd=vwap_lower_2sd,
        poc=poc,
        vah=vah,
        val=val,
        rsi_divergence_type=rsi_divergence_type,
        rsi_divergence_strength=rsi_divergence_strength,
        funding_rate=funding_rate,
        oi_change_pct=oi_change_pct,
        funding_momentum=funding_momentum_val,
        order_book_imbalance=order_book_imbalance,
        bid_ask_spread_pct=bid_ask_spread_pct,
        warnings=warnings_list,
    )
