"""Tests for the enhanced metrics boost integration in SetupAnalyzer."""
import pytest
from datetime import datetime, timezone

from futures_analyzer.analysis.scorer import SetupAnalyzer
from futures_analyzer.analysis.models import (
    Candle,
    MarketMeta,
    TimeframePlan,
)


def _make_candles(count: int = 50, base_price: float = 100.0) -> list[Candle]:
    base_time = datetime.now(timezone.utc)
    return [
        Candle(
            open_time=base_time,
            close_time=base_time,
            open=base_price + (i * 0.1),
            high=base_price + 0.5 + (i * 0.1),
            low=base_price - 0.5 + (i * 0.1),
            close=base_price + 0.2 + (i * 0.1),
            volume=1000 + (i * 10),
        )
        for i in range(count)
    ]


def _run_analysis(candles: list[Candle]) -> object:
    analyzer = SetupAnalyzer()
    market = MarketMeta(
        symbol="TESTUSDT",
        tick_size=0.01,
        step_size=0.001,
        mark_price=candles[-1].close,
    )
    timeframe_plan = TimeframePlan(
        entry_timeframe="5m",
        context_timeframe="1h",
        trigger_timeframe="15m",
        higher_timeframe="4h",
        lookback_bars=50,
    )
    return analyzer.analyze(
        symbol="TESTUSDT",
        entry_candles=candles,
        trigger_candles=candles,
        context_candles=candles,
        higher_candles=candles,
        market=market,
        timeframe_plan=timeframe_plan,
    )


def test_analysis_runs_without_error():
    """Smoke test: enhanced metrics boost should not break analysis."""
    result = _run_analysis(_make_candles())
    assert result.primary_setup is not None
    assert result.enhanced_metrics is not None


def test_confidence_clamped_to_valid_range():
    """Confidence must stay within [0.0, 1.0] after boost."""
    result = _run_analysis(_make_candles())
    assert 0.0 <= result.primary_setup.confidence <= 1.0
    assert 0.0 <= result.secondary_context.confidence <= 1.0


def test_quality_score_clamped_to_valid_range():
    """Quality score must stay within [10.0, 95.0] after boost."""
    result = _run_analysis(_make_candles())
    assert 10.0 <= result.primary_setup.quality_score <= 95.0
    assert 10.0 <= result.secondary_context.quality_score <= 95.0


def test_quality_label_consistent_with_score():
    """Quality label should match the quality score after recalculation."""
    from futures_analyzer.analysis.models import QualityLabel

    result = _run_analysis(_make_candles())
    score = result.primary_setup.quality_score
    label = result.primary_setup.quality_label

    # Thresholds match _quality_label() in scorer.py
    if score >= 75:
        assert label == QualityLabel.HIGH
    elif score >= 55:
        assert label == QualityLabel.MEDIUM
    else:
        assert label == QualityLabel.LOW


def test_rsi_overbought_reduces_long_confidence():
    """When RSI > 70, long confidence should be reduced by the boost."""
    analyzer = SetupAnalyzer()
    from futures_analyzer.analysis.scorer import _SideMetrics, _quality_label
    from futures_analyzer.analysis.models import EnhancedMetrics, QualityLabel
    from dataclasses import dataclass

    # Build a minimal _SideMetrics with known confidence
    side = _SideMetrics(
        side="long",
        score=0.5,
        confidence=0.6,
        quality_label=QualityLabel.MEDIUM,
        quality_score=60.0,
        leverage_suggestion="2x",
        entry=100.0,
        stop=98.0,
        target=104.0,
        rationale="test",
        top_positive_contributors=[],
        top_negative_contributors=[],
        components={},
        structure_points={},
        risk_reward_ratio=2.0,
        stop_distance_pct=2.0,
        target_distance_pct=4.0,
        atr_multiple_to_stop=1.0,
        atr_multiple_to_target=2.0,
        invalidation_strength=0.5,
        is_tradable=True,
        tradable_reasons=[],
        evidence_agreement=3,
        evidence_total=5,
        deliberation_summary="",
    )

    metrics = EnhancedMetrics(
        rsi_14=75.0,  # Overbought
        macd_histogram=0.0,
        bollinger_position=0.5,
        order_book_imbalance=0.0,
        volatility_rank=50.0,
    )

    original_confidence = side.confidence
    analyzer._apply_enhanced_metrics_boost(side, metrics, "long")

    assert side.confidence < original_confidence, (
        "Overbought RSI should reduce long confidence"
    )


def test_rsi_neutral_zone_boosts_confidence():
    """When RSI is in neutral zone (30-70), confidence should increase."""
    analyzer = SetupAnalyzer()
    from futures_analyzer.analysis.scorer import _SideMetrics
    from futures_analyzer.analysis.models import EnhancedMetrics, QualityLabel

    side = _SideMetrics(
        side="long",
        score=0.5,
        confidence=0.5,
        quality_label=QualityLabel.MEDIUM,
        quality_score=60.0,
        leverage_suggestion="2x",
        entry=100.0,
        stop=98.0,
        target=104.0,
        rationale="test",
        top_positive_contributors=[],
        top_negative_contributors=[],
        components={},
        structure_points={},
        risk_reward_ratio=2.0,
        stop_distance_pct=2.0,
        target_distance_pct=4.0,
        atr_multiple_to_stop=1.0,
        atr_multiple_to_target=2.0,
        invalidation_strength=0.5,
        is_tradable=True,
        tradable_reasons=[],
        evidence_agreement=3,
        evidence_total=5,
        deliberation_summary="",
    )

    metrics = EnhancedMetrics(
        rsi_14=50.0,  # Neutral
        macd_histogram=0.0,
        bollinger_position=0.5,
        order_book_imbalance=0.0,
        volatility_rank=50.0,
    )

    original_confidence = side.confidence
    analyzer._apply_enhanced_metrics_boost(side, metrics, "long")

    assert side.confidence > original_confidence, (
        "Neutral RSI should boost confidence"
    )


def test_bollinger_middle_band_improves_quality():
    """Bollinger position in middle band should add quality score."""
    analyzer = SetupAnalyzer()
    from futures_analyzer.analysis.scorer import _SideMetrics
    from futures_analyzer.analysis.models import EnhancedMetrics, QualityLabel

    side = _SideMetrics(
        side="long",
        score=0.5,
        confidence=0.5,
        quality_label=QualityLabel.MEDIUM,
        quality_score=60.0,
        leverage_suggestion="2x",
        entry=100.0,
        stop=98.0,
        target=104.0,
        rationale="test",
        top_positive_contributors=[],
        top_negative_contributors=[],
        components={},
        structure_points={},
        risk_reward_ratio=2.0,
        stop_distance_pct=2.0,
        target_distance_pct=4.0,
        atr_multiple_to_stop=1.0,
        atr_multiple_to_target=2.0,
        invalidation_strength=0.5,
        is_tradable=True,
        tradable_reasons=[],
        evidence_agreement=3,
        evidence_total=5,
        deliberation_summary="",
    )

    metrics = EnhancedMetrics(
        rsi_14=50.0,
        macd_histogram=0.0,
        bollinger_position=0.5,  # Middle band
        order_book_imbalance=0.0,
        volatility_rank=50.0,
    )

    original_quality = side.quality_score
    analyzer._apply_enhanced_metrics_boost(side, metrics, "long")

    assert side.quality_score > original_quality, (
        "Middle Bollinger Band should improve quality score"
    )


def test_high_volatility_widens_stop_distance():
    """Volatility rank > 75 should widen stop_distance_pct."""
    analyzer = SetupAnalyzer()
    from futures_analyzer.analysis.scorer import _SideMetrics
    from futures_analyzer.analysis.models import EnhancedMetrics, QualityLabel

    side = _SideMetrics(
        side="long",
        score=0.5,
        confidence=0.5,
        quality_label=QualityLabel.MEDIUM,
        quality_score=60.0,
        leverage_suggestion="2x",
        entry=100.0,
        stop=98.0,
        target=104.0,
        rationale="test",
        top_positive_contributors=[],
        top_negative_contributors=[],
        components={},
        structure_points={},
        risk_reward_ratio=2.0,
        stop_distance_pct=2.0,
        target_distance_pct=4.0,
        atr_multiple_to_stop=1.0,
        atr_multiple_to_target=2.0,
        invalidation_strength=0.5,
        is_tradable=True,
        tradable_reasons=[],
        evidence_agreement=3,
        evidence_total=5,
        deliberation_summary="",
    )

    metrics = EnhancedMetrics(
        rsi_14=50.0,
        macd_histogram=0.0,
        bollinger_position=0.5,
        order_book_imbalance=0.0,
        volatility_rank=80.0,  # High volatility
    )

    original_stop = side.stop_distance_pct
    analyzer._apply_enhanced_metrics_boost(side, metrics, "long")

    assert side.stop_distance_pct > original_stop, (
        "High volatility should widen stop distance"
    )


def test_low_volatility_tightens_stop_distance():
    """Volatility rank < 25 should tighten stop_distance_pct."""
    analyzer = SetupAnalyzer()
    from futures_analyzer.analysis.scorer import _SideMetrics
    from futures_analyzer.analysis.models import EnhancedMetrics, QualityLabel

    side = _SideMetrics(
        side="long",
        score=0.5,
        confidence=0.5,
        quality_label=QualityLabel.MEDIUM,
        quality_score=60.0,
        leverage_suggestion="2x",
        entry=100.0,
        stop=98.0,
        target=104.0,
        rationale="test",
        top_positive_contributors=[],
        top_negative_contributors=[],
        components={},
        structure_points={},
        risk_reward_ratio=2.0,
        stop_distance_pct=2.0,
        target_distance_pct=4.0,
        atr_multiple_to_stop=1.0,
        atr_multiple_to_target=2.0,
        invalidation_strength=0.5,
        is_tradable=True,
        tradable_reasons=[],
        evidence_agreement=3,
        evidence_total=5,
        deliberation_summary="",
    )

    metrics = EnhancedMetrics(
        rsi_14=50.0,
        macd_histogram=0.0,
        bollinger_position=0.5,
        order_book_imbalance=0.0,
        volatility_rank=10.0,  # Low volatility
    )

    original_stop = side.stop_distance_pct
    analyzer._apply_enhanced_metrics_boost(side, metrics, "long")

    assert side.stop_distance_pct < original_stop, (
        "Low volatility should tighten stop distance"
    )
