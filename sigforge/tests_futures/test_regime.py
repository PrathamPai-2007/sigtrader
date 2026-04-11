"""Tests for ADX/ATR regime classifier — tasks 5.1 through 5.6 and 2.5."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_analyzer.analysis.indicators import adx
from futures_analyzer.analysis.models import Candle, MarketRegime
from futures_analyzer.analysis.regime import classify_regime, classify_regime_consensus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(i: int, open_: float, high: float, low: float, close: float) -> Candle:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return Candle(
        open_time=base + timedelta(hours=i),
        close_time=base + timedelta(hours=i, minutes=59),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1000.0,
    )


def _trending_candles(n: int = 120) -> list[Candle]:
    """Consistent upward trend with varying bar sizes so ATR percentile stays low.

    The first half of bars have a larger step (2.0) to build ADX history, while
    the second half use a smaller step (0.5) so the current ATR is below most of
    the historical ATRs — keeping ATR percentile well under 60.
    """
    candles = []
    price = 100.0
    half = n // 2
    for i in range(n):
        step = 2.0 if i < half else 0.5
        open_ = price
        close = price + step
        high = close + step * 0.1
        low = open_ - step * 0.1
        candles.append(_make_candle(i, open_, high, low, close))
        price = close
    return candles


def _flat_candles(n: int = 120) -> list[Candle]:
    """Near-zero price movement with a volatile history so ATR percentile is low.

    The first half of bars are volatile (large wicks) to build a high-ATR history.
    The second half are flat (tiny wicks) so the current ATR ranks low — keeping
    ATR percentile under 70 and ADX near 0.
    """
    candles = []
    half = n // 2
    for i in range(n):
        if i < half:
            # Volatile history bars
            candles.append(_make_candle(i, 100.0, 105.0, 95.0, 100.0))
        else:
            # Flat recent bars
            candles.append(_make_candle(i, 100.0, 100.01, 99.99, 100.0))
    return candles


def _choppy_candles(n: int = 120) -> list[Candle]:
    """Alternating large up/down moves with big wicks — high ATR, no direction."""
    candles = []
    price = 100.0
    for i in range(n):
        if i % 2 == 0:
            open_ = price
            close = price + 5.0
            high = close + 3.0
            low = open_ - 3.0
        else:
            open_ = price
            close = price - 5.0
            high = open_ + 3.0
            low = close - 3.0
        candles.append(_make_candle(i, open_, high, low, close))
        price = close
    return candles


# ---------------------------------------------------------------------------
# 5.1 — Trending candles → BULLISH_TREND or BEARISH_TREND
# ---------------------------------------------------------------------------

def test_trending_candles_produce_trend_regime():
    candles = _trending_candles(120)
    regime, confidence = classify_regime(candles, candles, 0.0, candles[-1].close)
    assert regime in (MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND), (
        f"Expected a trend regime, got {regime}"
    )
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# 5.2 — Flat candles → RANGE
# ---------------------------------------------------------------------------

def test_flat_candles_produce_range_regime():
    candles = _flat_candles(120)
    regime, confidence = classify_regime(candles, candles, 0.0, 100.0)
    assert regime == MarketRegime.RANGE, f"Expected RANGE, got {regime}"
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# 5.3 — Volatile directionless candles → VOLATILE_CHOP
# ---------------------------------------------------------------------------

def test_volatile_directionless_candles_produce_chop_regime():
    candles = _choppy_candles(120)
    regime, confidence = classify_regime(candles, candles, 0.0, candles[-1].close)
    assert regime == MarketRegime.VOLATILE_CHOP, f"Expected VOLATILE_CHOP, got {regime}"
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# 5.4 — adx() returns values in [0.0, 100.0]
# ---------------------------------------------------------------------------

def test_adx_returns_values_in_range():
    series = {
        "trending": _trending_candles(100),
        "flat": _flat_candles(100),
        "choppy": _choppy_candles(100),
    }
    for label, candles in series.items():
        result = adx(candles)
        assert 0.0 <= result["adx"] <= 100.0, f"{label}: adx={result['adx']} out of range"
        assert 0.0 <= result["plus_di"] <= 100.0, f"{label}: plus_di={result['plus_di']} out of range"
        assert 0.0 <= result["minus_di"] <= 100.0, f"{label}: minus_di={result['minus_di']} out of range"


# ---------------------------------------------------------------------------
# 5.5 — classify_regime is idempotent
# ---------------------------------------------------------------------------

def test_classify_regime_is_idempotent():
    candles = _trending_candles(80)
    result1 = classify_regime(candles, candles, 0.0, candles[-1].close)
    result2 = classify_regime(candles, candles, 0.0, candles[-1].close)
    assert result1 == result2, f"Non-idempotent: {result1} != {result2}"


# ---------------------------------------------------------------------------
# 5.6 — adx() fallback for insufficient candles
# ---------------------------------------------------------------------------

def test_adx_fallback_for_insufficient_candles():
    _zero = {"adx": 0.0, "plus_di": 0.0, "minus_di": 0.0}

    # 0 candles
    assert adx([]) == _zero

    # 5 candles (< 2*14 = 28)
    assert adx(_trending_candles(5)) == _zero

    # exactly 27 candles (still < 28)
    assert adx(_trending_candles(27)) == _zero


# ---------------------------------------------------------------------------
# 2.5 — Helpers for BREAKOUT, EXHAUSTION, TRANSITION patterns
# ---------------------------------------------------------------------------

def _breakout_candles(n: int = 60) -> list[Candle]:
    """Flat/ranging candles followed by a sharp directional move.

    57 flat bars (zero movement, ADX stays near 0) followed by 3 directional
    bars. The ADX slope over the last 5 bars is strongly positive (> 3.0) while
    ADX is still < 20, satisfying the BREAKOUT condition in the classifier.

    Note: ATR percentile is high (100) due to the contrast between flat and
    directional bars, but the BREAKOUT branch is checked before VOLATILE_CHOP
    in the decision table, so the pattern still classifies as BREAKOUT.
    """
    candles: list[Candle] = []
    base = datetime(2024, 1, 1, tzinfo=UTC)
    price = 100.0

    flat_count = 57
    breakout_count = n - flat_count  # 3 bars

    # Flat phase: truly zero movement so ADX stays near 0
    for i in range(flat_count):
        candles.append(Candle(
            open_time=base + timedelta(hours=i),
            close_time=base + timedelta(hours=i, minutes=59),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1000.0,
        ))

    # Breakout phase: sharp directional move (3 bars)
    for j in range(breakout_count):
        step = 1.0
        open_ = price
        close = price + step
        high = close + 0.1
        low = open_ - 0.05
        i = flat_count + j
        candles.append(Candle(
            open_time=base + timedelta(hours=i),
            close_time=base + timedelta(hours=i, minutes=59),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=1000.0,
        ))
        price = close

    return candles


def _exhaustion_candles(n: int = 60) -> list[Candle]:
    """Strong trend candles followed by counter-trend bars that cause ADX to decline.

    50 strong uptrend bars (ADX builds to ~73) followed by 10 counter-trend
    (downward) bars. The counter-trend bars cause the ADX slope to go sharply
    negative (< -2.0) while ADX remains >= 25, satisfying the EXHAUSTION
    condition in the classifier.
    """
    candles: list[Candle] = []
    base = datetime(2024, 1, 1, tzinfo=UTC)
    price = 100.0

    trend_count = 50
    counter_count = n - trend_count  # 10 bars

    # Strong trend phase: consistent upward moves to build high ADX
    for i in range(trend_count):
        step = 2.0
        open_ = price
        close = price + step
        high = close + 0.2
        low = open_ - 0.1
        candles.append(Candle(
            open_time=base + timedelta(hours=i),
            close_time=base + timedelta(hours=i, minutes=59),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=1000.0,
        ))
        price = close

    # Counter-trend phase: downward bars that cause ADX slope to go negative
    for j in range(counter_count):
        step = 1.5
        open_ = price
        close = price - step
        high = open_ + 0.1
        low = close - 0.2
        i = trend_count + j
        candles.append(Candle(
            open_time=base + timedelta(hours=i),
            close_time=base + timedelta(hours=i, minutes=59),
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=1000.0,
        ))
        price = close

    return candles


def _transition_candles(n: int = 72) -> list[Candle]:
    """Candles producing ADX in the 20–25 range with mixed DI signals.

    Uses a 5-up / 4-down cycle with equal step sizes. The slight directional
    bias keeps ADX in the 20-25 transition zone (atr_pct ~44) with neither
    +DI nor -DI clearly dominant.
    """
    candles: list[Candle] = []
    base = datetime(2024, 1, 1, tzinfo=UTC)
    price = 100.0
    up_step = 1.0
    down_step = 0.9

    i = 0
    while i < n:
        # 5 up bars
        for _ in range(5):
            if i >= n:
                break
            candles.append(Candle(
                open_time=base + timedelta(hours=i),
                close_time=base + timedelta(hours=i, minutes=59),
                open=price,
                high=price + up_step + 0.05,
                low=price - 0.02,
                close=price + up_step,
                volume=1000.0,
            ))
            price += up_step
            i += 1
        # 4 down bars
        for _ in range(4):
            if i >= n:
                break
            candles.append(Candle(
                open_time=base + timedelta(hours=i),
                close_time=base + timedelta(hours=i, minutes=59),
                open=price,
                high=price + 0.02,
                low=price - down_step - 0.05,
                close=price - down_step,
                volume=1000.0,
            ))
            price -= down_step
            i += 1

    return candles


# ---------------------------------------------------------------------------
# 2.5 — Tests: classify_regime_consensus for BREAKOUT, EXHAUSTION, TRANSITION
# ---------------------------------------------------------------------------

def test_classify_regime_consensus_breakout():
    """Flat-then-sharp candles should produce BREAKOUT regime."""
    candles = _breakout_candles()
    result = classify_regime_consensus(
        context_candles=candles,
        higher_candles=candles,
        trigger_candles=candles,
        trigger_atr=1.0,
        px=100.0,
    )
    assert result.regime == MarketRegime.BREAKOUT, (
        f"Expected BREAKOUT, got {result.regime!r}. "
        f"ADX values per TF: {[(tf.adx, tf.adx_slope, tf.atr_percentile) for tf in result.per_tf]}"
    )


def test_classify_regime_consensus_exhaustion():
    """Strong-trend-then-counter-trend candles should produce EXHAUSTION regime."""
    candles = _exhaustion_candles()
    result = classify_regime_consensus(
        context_candles=candles,
        higher_candles=candles,
        trigger_candles=candles,
        trigger_atr=1.0,
        px=100.0,
    )
    assert result.regime == MarketRegime.EXHAUSTION, (
        f"Expected EXHAUSTION, got {result.regime!r}. "
        f"ADX values per TF: {[(tf.adx, tf.adx_slope, tf.atr_percentile) for tf in result.per_tf]}"
    )


def test_classify_regime_consensus_transition():
    """5-up/4-down cycle candles should produce TRANSITION regime."""
    candles = _transition_candles()
    result = classify_regime_consensus(
        context_candles=candles,
        higher_candles=candles,
        trigger_candles=candles,
        trigger_atr=1.0,
        px=100.0,
    )
    assert result.regime == MarketRegime.TRANSITION, (
        f"Expected TRANSITION, got {result.regime!r}. "
        f"ADX values per TF: {[(tf.adx, tf.adx_slope, tf.atr_percentile) for tf in result.per_tf]}"
    )
