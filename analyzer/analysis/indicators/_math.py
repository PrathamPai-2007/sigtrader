"""Pure math helpers used across indicator modules."""
from __future__ import annotations
import math


def sigmoid(x: float) -> float:
    """Compute the logistic sigmoid: 1 / (1 + exp(-x)), clamped to [0.0, 1.0].

    Handles overflow for very large positive x (returns 1.0) and very large
    negative x (returns 0.0) via try/except OverflowError.
    """
    try:
        result = 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        result = 0.0 if x < 0 else 1.0
    return max(0.0, min(1.0, result))


def gaussian_peak(x: float, center: float, width: float) -> float:
    """Compute a Gaussian bell curve: exp(-0.5 * ((x - center) / width) ** 2).

    Clamped to [0.0, 1.0]. When width is 0 (or effectively zero), returns 1.0
    if x == center, else 0.0.
    """
    if width == 0.0 or abs(width) < 1e-12:
        return 1.0 if x == center else 0.0
    result = math.exp(-0.5 * ((x - center) / width) ** 2)
    return max(0.0, min(1.0, result))


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
