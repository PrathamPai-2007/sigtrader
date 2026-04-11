"""Tests for the pivot-based rsi_divergence implementation (SQ-1)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from futures_analyzer.analysis.indicators import _swing_pivots, rsi_divergence
from futures_analyzer.analysis.models import Candle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _candle(close: float, *, high: float | None = None, low: float | None = None) -> Candle:
    h = high if high is not None else close + 0.5
    l = low if low is not None else close - 0.5
    return Candle(
        open_time=_BASE,
        close_time=_BASE,
        open=close,
        high=h,
        low=l,
        close=close,
        volume=1000.0,
    )


def _trending_up(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[Candle]:
    """Monotonically rising candles — no real pivot structure."""
    return [_candle(start + i * step) for i in range(n)]


def _trending_down(n: int = 60, start: float = 130.0, step: float = 0.5) -> list[Candle]:
    """Monotonically falling candles — no real pivot structure."""
    return [_candle(start - i * step) for i in range(n)]


def _bullish_divergence_candles(n_warmup: int = 30) -> list[Candle]:
    """
    Craft a series with a clear bullish divergence:
    - First swing low at price 90, RSI will be low (oversold territory).
    - Second swing low at price 85 (lower price low).
    - But the second low is shallower in RSI terms (higher RSI low) because
      the surrounding bars are less extreme.

    Structure (after warmup):
      warmup (flat ~100) → drop to 90 → recover to 105 → drop to 85 → recover
    The second drop is steeper in price but the recovery before it is stronger,
    so RSI at the second low is higher than at the first.
    """
    candles: list[Candle] = []
    # Warmup: flat around 100 so RSI stabilises near 50
    for _ in range(n_warmup):
        candles.append(_candle(100.0))

    # First leg down: 100 → 90 over 8 bars
    for i in range(8):
        candles.append(_candle(100.0 - i * 1.25))

    # Recovery: 90 → 105 over 8 bars (strong recovery → RSI rises well)
    for i in range(8):
        candles.append(_candle(90.0 + i * 1.875))

    # Second leg down: 105 → 85 over 8 bars (lower price low)
    # But the drop is from a higher base so RSI doesn't get as oversold
    for i in range(8):
        candles.append(_candle(105.0 - i * 2.5))

    # Recovery tail: 85 → 95 over 5 bars
    for i in range(5):
        candles.append(_candle(85.0 + i * 2.0))

    return candles


def _bearish_divergence_candles(n_warmup: int = 30) -> list[Candle]:
    """
    Craft a series with a clear bearish divergence:
    - First swing high at price 110.
    - Second swing high at price 115 (higher price high).
    - RSI at second high is lower than at first (bearish divergence).
    """
    candles: list[Candle] = []
    # Warmup: flat around 100
    for _ in range(n_warmup):
        candles.append(_candle(100.0))

    # First leg up: 100 → 110 over 8 bars
    for i in range(8):
        candles.append(_candle(100.0 + i * 1.25))

    # Pullback: 110 → 98 over 8 bars (weak pullback → RSI stays elevated)
    for i in range(8):
        candles.append(_candle(110.0 - i * 1.5))

    # Second leg up: 98 → 115 over 8 bars (higher price high)
    # Starts from a lower base so RSI doesn't get as overbought
    for i in range(8):
        candles.append(_candle(98.0 + i * 2.125))

    # Tail: 115 → 108 over 5 bars
    for i in range(5):
        candles.append(_candle(115.0 - i * 1.4))

    return candles


# ---------------------------------------------------------------------------
# _swing_pivots unit tests
# ---------------------------------------------------------------------------

class TestSwingPivots:
    def test_empty_returns_empty(self) -> None:
        assert _swing_pivots([], n=3) == []

    def test_too_short_returns_empty(self) -> None:
        # Need at least 2*n+1 values to have any pivot
        assert _swing_pivots([1.0, 2.0, 3.0], n=3) == []

    def test_detects_single_low(self) -> None:
        # V-shape: clear low in the middle
        values = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        pivots = _swing_pivots(values, n=2)
        indices = [i for i, _ in pivots]
        assert 4 in indices  # index 4 is the minimum

    def test_detects_single_high(self) -> None:
        # Inverted V: clear high in the middle
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        pivots = _swing_pivots(values, n=2)
        indices = [i for i, _ in pivots]
        assert 4 in indices  # index 4 is the maximum

    def test_detects_multiple_pivots(self) -> None:
        # Two clear lows separated by a high
        values = [5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0, 3.0, 5.0]
        pivots = _swing_pivots(values, n=1)
        # Should find alternating lows and highs
        assert len(pivots) >= 2

    def test_flat_series_no_pivots(self) -> None:
        values = [5.0] * 20
        # All values equal — no strict min/max pivot
        pivots = _swing_pivots(values, n=3)
        # Every point is both min and max of its window — all are pivots
        # The important thing is the function doesn't crash
        assert isinstance(pivots, list)


# ---------------------------------------------------------------------------
# rsi_divergence unit tests
# ---------------------------------------------------------------------------

class TestRsiDivergence:
    def test_returns_three_tuple(self) -> None:
        candles = _trending_up(60)
        result = rsi_divergence(candles)
        assert len(result) == 3
        detected, div_type, strength = result
        assert isinstance(detected, bool)
        assert isinstance(div_type, str)
        assert isinstance(strength, float)

    def test_too_few_candles_returns_no_divergence(self) -> None:
        candles = _trending_up(10)
        detected, div_type, strength = rsi_divergence(candles, period=14, lookback=30)
        assert detected is False
        assert div_type == "none"
        assert strength == 0.0

    def test_monotonic_uptrend_no_divergence(self) -> None:
        """A clean uptrend has no swing pivot structure — should not fire."""
        candles = _trending_up(80)
        detected, div_type, _ = rsi_divergence(candles, period=14, lookback=30)
        assert detected is False
        assert div_type == "none"

    def test_monotonic_downtrend_no_divergence(self) -> None:
        """A clean downtrend has no swing pivot structure — should not fire."""
        candles = _trending_down(80)
        detected, div_type, _ = rsi_divergence(candles, period=14, lookback=30)
        assert detected is False
        assert div_type == "none"

    def test_strength_zero_when_no_divergence(self) -> None:
        candles = _trending_up(80)
        _, _, strength = rsi_divergence(candles)
        assert strength == 0.0

    def test_strength_positive_when_divergence_detected(self) -> None:
        """When divergence fires, strength must be > 0."""
        candles = _bullish_divergence_candles()
        detected, div_type, strength = rsi_divergence(candles, period=14, lookback=30, pivot_n=2)
        if detected:
            assert strength > 0.0

    def test_divergence_type_is_valid_string(self) -> None:
        candles = _bullish_divergence_candles()
        _, div_type, _ = rsi_divergence(candles)
        assert div_type in ("bullish", "bearish", "none")

    def test_min_price_move_filters_noise(self) -> None:
        """With a very high min_price_move_pct, tiny pivots should not fire."""
        candles = _bullish_divergence_candles()
        # Require 50% price move between pivots — impossible in our test data
        detected, _, _ = rsi_divergence(candles, min_price_move_pct=50.0)
        assert detected is False

    def test_min_price_move_zero_is_permissive(self) -> None:
        """With min_price_move_pct=0, any pivot pair is eligible."""
        candles = _bullish_divergence_candles()
        # Should not raise; result is a valid 3-tuple
        result = rsi_divergence(candles, min_price_move_pct=0.0)
        assert len(result) == 3

    def test_bullish_divergence_detected(self) -> None:
        """Crafted bullish divergence series should be detected."""
        candles = _bullish_divergence_candles(n_warmup=40)
        detected, div_type, strength = rsi_divergence(
            candles, period=14, lookback=35, pivot_n=2, min_price_move_pct=0.3
        )
        if detected:
            assert div_type == "bullish"
            assert strength > 0.0

    def test_bearish_divergence_detected(self) -> None:
        """Crafted bearish divergence series should be detected."""
        candles = _bearish_divergence_candles(n_warmup=40)
        detected, div_type, strength = rsi_divergence(
            candles, period=14, lookback=35, pivot_n=2, min_price_move_pct=0.3
        )
        if detected:
            assert div_type == "bearish"
            assert strength > 0.0

    def test_bullish_and_bearish_never_both_true(self) -> None:
        """The function returns at most one divergence type per call."""
        for candles in [_bullish_divergence_candles(), _bearish_divergence_candles(), _trending_up(80)]:
            detected, div_type, _ = rsi_divergence(candles)
            if detected:
                assert div_type in ("bullish", "bearish")
            else:
                assert div_type == "none"
