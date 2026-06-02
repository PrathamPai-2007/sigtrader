"""Volume-based indicators: volume profile, cumulative delta, VWAP bands."""
from __future__ import annotations
from typing import Any
from analyzer.analysis.models import Candle


def volume_profile(candles: list[Candle], bins: int = 20) -> dict[str, Any]:
    """Calculate volume profile — POC, VAH, VAL across binned price levels."""
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
    sorted_bins = sorted(profile.items(), key=lambda x: x[1], reverse=True)
    cumulative = 0.0
    total_volume = sum(profile.values())
    va_threshold = total_volume * 0.68
    va_bins: list[int] = []
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
    return {"poc": poc, "vah": vah, "val": val, "profile": profile, "max_volume": max_volume}


def volume_profile_strength(
    candles: list[Candle],
    current_price: float,
    *,
    bins: int = 20,
    near_threshold_pct: float = 1.5,
) -> dict[str, Any]:
    """Compute proximity of current_price to POC, VAH, and VAL."""
    vp = volume_profile(candles, bins=bins)
    poc, vah, val = vp["poc"], vp["vah"], vp["val"]

    def dist_pct(level: float) -> float:
        return abs(current_price - level) / max(level, 1e-9) * 100.0

    return {
        "poc": poc, "vah": vah, "val": val,
        "dist_to_poc_pct": dist_pct(poc),
        "dist_to_vah_pct": dist_pct(vah),
        "dist_to_val_pct": dist_pct(val),
        "near_poc": dist_pct(poc) < near_threshold_pct,
        "near_vah": dist_pct(vah) < near_threshold_pct,
        "near_val": dist_pct(val) < near_threshold_pct,
    }


def compute_cumulative_delta(candles: list[Candle], window: int = 20) -> float:
    """Compute net buy/sell pressure over last `window` candles, normalized to [-1, 1]."""
    if not candles:
        return 0.0
    recent = candles[-window:]
    total = sum((c.close - c.open) / max(c.high - c.low, 1e-9) for c in recent)
    return max(-1.0, min(1.0, total / window))


def compute_vwap_bands(candles: list[Candle]) -> dict[str, float]:
    """Compute VWAP with ±1 and ±2 standard deviation bands.

    For any candles with positive volume the returned dict satisfies:
        lower_2sd < lower_1sd < vwap < upper_1sd < upper_2sd
    """
    if not candles:
        return {"vwap": 0.0, "upper_1sd": 0.0, "lower_1sd": 0.0, "upper_2sd": 0.0, "lower_2sd": 0.0}
    last_close = candles[-1].close
    total_volume = sum(c.volume for c in candles)
    if total_volume <= 0.0:
        return {"vwap": last_close, "upper_1sd": last_close, "lower_1sd": last_close,
                "upper_2sd": last_close, "lower_2sd": last_close}
    tp_vol_sum = sum((c.high + c.low + c.close) / 3.0 * c.volume for c in candles)
    vwap = tp_vol_sum / total_volume
    variance = sum(c.volume * ((c.high + c.low + c.close) / 3.0 - vwap) ** 2 for c in candles) / total_volume
    sd = variance ** 0.5
    if sd == 0.0:
        sd = max(vwap * 1e-9, 1e-12)
    return {
        "vwap": vwap,
        "upper_1sd": vwap + sd,
        "lower_1sd": vwap - sd,
        "upper_2sd": vwap + 2.0 * sd,
        "lower_2sd": vwap - 2.0 * sd,
    }
