"""Pure signal-detection functions for reversal and confluence scoring.

These functions are side-effect-free and operate only on scalar/candle inputs.
Functions that mutate _SideMetrics (_apply_reversal_penalty, _apply_confluence_boost)
remain in scorer.py because they depend on scorer-internal state.
"""
from __future__ import annotations

from analyzer.analysis.models import Candle
from analyzer.analysis.indicators import macd
from analyzer.analysis.indicators_compute import _momentum


def _detect_early_reversal_signals(
    side: str,
    *,
    trigger_candles: list[Candle],
    entry_candles: list[Candle],
    support: float,
    resistance: float,
    rsi_divergence_type: str,
    macd_histogram: float,
) -> dict[str, bool]:
    """Return a dict of fired reversal signals for the given side.

    True = that signal is warning against entering this side right now.
    """
    signals: dict[str, bool] = {}

    # 1. RSI divergence against the side
    if side == "long":
        signals["rsi_divergence"] = rsi_divergence_type == "bearish"
    else:
        signals["rsi_divergence"] = rsi_divergence_type == "bullish"

    # 2. MACD histogram flip against the side (histogram crossed zero)
    # Guard: suppress micro-flips unless the move has had time to develop.
    if len(trigger_candles) >= 2:
        prev_hist = macd(trigger_candles[:-1]).histogram or 0.0
        _raw_flip_long = prev_hist > 0 and macd_histogram < 0
        _raw_flip_short = prev_hist < 0 and macd_histogram > 0
        _raw_flip = _raw_flip_long if side == "long" else _raw_flip_short
        if _raw_flip:
            _bars_since_cross = 0
            _cross_price = trigger_candles[-1].close
            for _i in range(len(trigger_candles) - 2, max(len(trigger_candles) - 20, -1), -1):
                _h = macd(trigger_candles[:_i + 1]).histogram or 0.0
                if (side == "long" and _h > 0) or (side == "short" and _h < 0):
                    _bars_since_cross = len(trigger_candles) - 1 - _i
                    _cross_price = trigger_candles[_i].close
                    break
            _price_move_pct = abs(trigger_candles[-1].close - _cross_price) / max(_cross_price, 1e-9) * 100.0
            _min_bars = 8 if side == "long" else 5
            _escape_pct = 0.25 if side == "long" else 0.1
            _confirmed = _bars_since_cross >= _min_bars or _price_move_pct >= _escape_pct
            signals["macd_flip"] = _confirmed
        else:
            signals["macd_flip"] = False

    # 3. Momentum divergence: trigger TF going one way but entry TF reversed
    trigger_mom = _momentum(trigger_candles)
    entry_mom = _momentum(entry_candles, window=12)
    if side == "long":
        signals["momentum_divergence"] = trigger_mom > 0 and entry_mom < -0.002
    else:
        signals["momentum_divergence"] = trigger_mom < 0 and entry_mom > 0.002

    # 4. Structure break: price already through the key level (0.2% buffer)
    px = trigger_candles[-1].close
    if side == "long":
        signals["structure_break"] = px < support * 0.998
    else:
        signals["structure_break"] = px > resistance * 1.002

    return signals


def _round_number_proximity(price: float, threshold_pct: float = 0.5) -> bool:
    """True if price is within threshold_pct% of a psychologically significant round number.

    Checks multiples of the largest power of 10 that fits in the price,
    and half that magnitude (e.g. for a 5-digit price: 10000s and 5000s).
    """
    if price <= 0:
        return False
    digits = len(str(int(price)))
    magnitude = 10 ** max(digits - 1, 0)
    for step in (magnitude, magnitude // 2 if magnitude >= 2 else magnitude):
        if step <= 0:
            continue
        nearest = round(price / step) * step
        if nearest > 0 and abs(price - nearest) / nearest * 100 < threshold_pct:
            return True
    return False


def _score_confluence(
    side: str,
    *,
    entry: float,
    target: float,
    support: float,
    resistance: float,
    atr: float,
    volume_profile: dict | None = None,
    near_threshold_pct: float = 1.0,
) -> dict[str, float]:
    """Count independent factors aligning at entry and target price levels.

    Returns entry_confluence and target_confluence counts (0-4 each).
    volume_profile is optional — skipped when None.
    """
    def near(price: float, level: float) -> bool:
        return level > 0 and abs(price - level) / level * 100 < near_threshold_pct

    entry_factors = 0
    target_factors = 0

    if side == "long":
        if near(entry, support):
            entry_factors += 1
        if volume_profile and near(entry, volume_profile.get("val", 0.0)):
            entry_factors += 1
        if volume_profile and near(entry, volume_profile.get("poc", 0.0)):
            entry_factors += 1
        if _round_number_proximity(entry):
            entry_factors += 1

        if near(target, resistance):
            target_factors += 1
        if volume_profile and near(target, volume_profile.get("vah", 0.0)):
            target_factors += 1
        if volume_profile and near(target, volume_profile.get("poc", 0.0)):
            target_factors += 1
        if _round_number_proximity(target):
            target_factors += 1
    else:
        if near(entry, resistance):
            entry_factors += 1
        if volume_profile and near(entry, volume_profile.get("vah", 0.0)):
            entry_factors += 1
        if volume_profile and near(entry, volume_profile.get("poc", 0.0)):
            entry_factors += 1
        if _round_number_proximity(entry):
            entry_factors += 1

        if near(target, support):
            target_factors += 1
        if volume_profile and near(target, volume_profile.get("val", 0.0)):
            target_factors += 1
        if volume_profile and near(target, volume_profile.get("poc", 0.0)):
            target_factors += 1
        if _round_number_proximity(target):
            target_factors += 1

    return {
        "entry_confluence": float(entry_factors),
        "target_confluence": float(target_factors),
    }
