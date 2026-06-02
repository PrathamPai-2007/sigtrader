"""Shared dataclasses for indicator results."""
from __future__ import annotations
from dataclasses import dataclass


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
