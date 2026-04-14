"""Pure utility functions used throughout the scoring pipeline.

These are simple, self-contained helpers with no dependencies on other
scoring modules. They can be safely imported anywhere without circular
dependency concerns.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from futures_analyzer.analysis.models import QualityLabel
from futures_analyzer.config import load_app_config


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high] range."""
    return max(low, min(value, high))


def _quantize(price: float, tick: float | None) -> float:
    """Quantize a price to the exchange tick size.
    
    Args:
        price: price value to quantize
        tick: tick size (e.g., 0.01 for cents), or None to round to 6 decimals
        
    Returns:
        Quantized price value
    """
    if tick is None or tick <= 0:
        return round(price, 6)
    try:
        tick_decimal = Decimal(str(tick)).normalize()
        price_decimal = Decimal(str(price))
    except InvalidOperation:
        return round(price, 10)
    if tick_decimal <= 0:
        return round(price, 10)
    units = (price_decimal / tick_decimal).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    quantized = units * tick_decimal
    precision = max(0, -tick_decimal.as_tuple().exponent)
    return float(format(quantized, f".{min(precision, 16)}f"))


def _quality_label(quality_score: float) -> QualityLabel:
    """Convert numeric quality score to QualityLabel enum.
    
    Args:
        quality_score: score in [0, 100]
        
    Returns:
        QualityLabel.HIGH if >= 75, MEDIUM if >= 55, else LOW
    """
    if quality_score >= 75:
        return QualityLabel.HIGH
    if quality_score >= 55:
        return QualityLabel.MEDIUM
    return QualityLabel.LOW


def _quality_score_cap_from_confidence(confidence: float) -> float:
    """Get quality score cap based on confidence level.
    
    Lower confidence setups have stricter quality caps to prevent
    low-confidence, low-quality trades from being considered tradable.
    
    Args:
        confidence: confidence in [0, 1]
        
    Returns:
        Maximum allowed quality score for this confidence level
    """
    caps = load_app_config().strategy.confidence_quality_caps
    if confidence < 0.45:
        return caps["below_0_45"]
    if confidence < 0.7:
        return caps["below_0_70"]
    return caps["default"]
