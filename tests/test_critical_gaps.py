from __future__ import annotations

from datetime import UTC, datetime
import pytest

from analyzer.analysis.models import (
    MarketRegime,
    NormalizedSignals,
)
from analyzer.analysis.evidence import compute_graded_evidence
from analyzer.analysis.confidence import (
    logistic_confidence,
    logistic_confidence_from_config,
)
from analyzer.analysis.long_pipeline_log import LongPipelineLog
from analyzer.config import AppConfig, StrategyConfig


def test_evidence_vector_computation():
    signals = NormalizedSignals(
        higher_trend=0.8,
        context_trend=0.6,
        trigger_momentum=0.7,
        entry_momentum=0.5,
        volume_surge=0.9,
        buy_pressure=0.6,
        oi_funding_bias=0.5,
        funding_momentum=0.5,
        structure_position=0.4,
        rsi_alignment=0.6,
        macd_alignment=0.7,
        bb_alignment=0.5,
        vwap_alignment=0.5,
        market_structure_align=0.5,
        cumulative_delta_align=0.5,
        volume_poc_proximity=0.5,
    )
    
    weights = {
        "higher_trend": 0.4,
        "volume_surge": 0.6,
    }
    
    # Check default behavior and weighted sum calculation
    # (0.8 - 0.5)*2 * 0.4 + (0.9 - 0.5)*2 * 0.6 = 0.3*2*0.4 + 0.4*2*0.6 = 0.24 + 0.48 = 0.72
    result = compute_graded_evidence(
        signals,
        MarketRegime.BULLISH_TREND,
        side="long",
        weights=weights,
        debug=True,
        regime_gate_thresholds={"trend": 0.5},
    )
    
    assert pytest.approx(result.weighted_sum, rel=1e-5) == 0.72
    assert result.signal_count_above_threshold == 7
    assert result.regime_gate_passed is True  # higher_trend (0.8) > threshold (0.5)
    assert result.debug is not None
    assert pytest.approx(result.debug["raw_score"], rel=1e-5) == 0.72
    assert len(result.strongest_signals) == 3
    assert len(result.weakest_signals) == 3


def test_evidence_regime_gates():
    signals = NormalizedSignals(
        higher_trend=0.3,
        context_trend=0.5,
        trigger_momentum=0.5,
        entry_momentum=0.5,
        volume_surge=0.2,
        buy_pressure=0.5,
        oi_funding_bias=0.5,
        funding_momentum=0.5,
        structure_position=0.2,
        rsi_alignment=0.2,
        macd_alignment=0.5,
        bb_alignment=0.5,
        vwap_alignment=0.5,
        market_structure_align=0.5,
        cumulative_delta_align=0.5,
        volume_poc_proximity=0.5,
    )
    
    weights = {"higher_trend": 1.0}
    
    thresholds = {
        "trend": 0.4,
        "breakout": 0.3,
        "exhaustion": 0.3,
        "range": 0.3,
        "volatile_chop": 0.5,
        "transition": 0.35,
    }
    
    # 1. Trend gate
    res = compute_graded_evidence(signals, MarketRegime.BULLISH_TREND, "long", weights, regime_gate_thresholds=thresholds)
    assert res.regime_gate_passed is False
    
    # 2. Breakout gate
    res = compute_graded_evidence(signals, MarketRegime.BREAKOUT, "long", weights, regime_gate_thresholds=thresholds)
    assert res.regime_gate_passed is False
    
    # 3. Exhaustion gate
    res = compute_graded_evidence(signals, MarketRegime.EXHAUSTION, "long", weights, regime_gate_thresholds=thresholds)
    assert res.regime_gate_passed is False
    
    # 4. Range gate
    res = compute_graded_evidence(signals, MarketRegime.RANGE, "long", weights, regime_gate_thresholds=thresholds)
    assert res.regime_gate_passed is False
    
    # 5. Volatile Chop gate
    res = compute_graded_evidence(signals, MarketRegime.VOLATILE_CHOP, "long", weights, regime_gate_thresholds=thresholds)
    assert res.regime_gate_passed is False
    
    # 6. Transition gate
    res = compute_graded_evidence(signals, MarketRegime.TRANSITION, "long", weights, regime_gate_thresholds=thresholds)
    assert res.regime_gate_passed is False
    
    # Passed scenario
    signals_pass = NormalizedSignals(
        higher_trend=0.6,
        context_trend=0.5,
        trigger_momentum=0.5,
        entry_momentum=0.5,
        volume_surge=0.4,
        buy_pressure=0.5,
        oi_funding_bias=0.5,
        funding_momentum=0.5,
        structure_position=0.4,
        rsi_alignment=0.4,
        macd_alignment=0.5,
        bb_alignment=0.5,
        vwap_alignment=0.5,
        market_structure_align=0.5,
        cumulative_delta_align=0.5,
        volume_poc_proximity=0.5,
    )
    res = compute_graded_evidence(signals_pass, MarketRegime.BULLISH_TREND, "long", weights, regime_gate_thresholds=thresholds)
    assert res.regime_gate_passed is True


def test_confidence_sigmoid_mapping():
    # Setup mock strategy config
    strategy = StrategyConfig(
        logistic_params={
            "bullish_trend_long": {"steepness": 8.0},
            "range": {"steepness": 4.0},
        }
    )
    
    # Test configured mapping
    conf_trend = logistic_confidence(0.5, MarketRegime.BULLISH_TREND, "long", strategy)
    assert conf_trend > 0.5
    
    # Test range mapping
    conf_range = logistic_confidence(0.5, MarketRegime.RANGE, "long", strategy)
    assert conf_range > 0.5
    assert conf_trend > conf_range  # Trend steepness (8.0) > Range (4.0)

    # Test fallback
    conf_fallback = logistic_confidence(0.5, MarketRegime.EXHAUSTION, "long", strategy)
    assert conf_fallback > 0.5


def test_confidence_sigmoid_with_config():
    config = AppConfig(
        strategy=StrategyConfig(
            logistic_params={
                "default": {"steepness": 6.0}
            }
        )
    )
    val = logistic_confidence_from_config(0.2, MarketRegime.RANGE, "long", config=config)
    assert 0.0 <= val <= 1.0


def test_long_pipeline_log_stages():
    log = LongPipelineLog()
    
    log.record_generated()
    log.record_passed_macro()
    log.record_rejected_macro("bad_trend")
    log.record_passed_regime()
    log.record_rejected_regime("bad_chop")
    log.record_passed_evidence()
    log.record_rejected_evidence("low_evidence")
    log.record_passed_logistic()
    log.record_rejected_logistic("low_confidence")
    log.record_passed_long_filters()
    log.record_rejected_long_filters("high_slippage")
    log.record_entered()
    log.record_rejected_final("risk_exceeded")
    
    summary = log.summary()
    assert "LONG PIPELINE DEBUG SUMMARY" in summary
    assert "Macro rejections" in summary
    
    dct = log.as_dict()
    assert dct["LONG_PIPELINE_DEBUG"]["generated"] == 1
    assert dct["LONG_PIPELINE_DEBUG"]["after_macro"] == 1
    assert dct["LONG_PIPELINE_DEBUG"]["entered"] == 1
    assert dct["rejection_reasons"]["macro"]["bad_trend"] == 1


def test_evidence_private_helpers():
    from analyzer.analysis.evidence import (
        _regime_weight_profile,
        _regime_alignment,
        _regime_penalty,
    )
    
    weights, multiplier, ceiling = _regime_weight_profile(MarketRegime.BULLISH_TREND, "long")
    assert isinstance(weights, dict)
    assert multiplier >= 0
    assert ceiling >= 0
    
    alignment = _regime_alignment(MarketRegime.BULLISH_TREND, "long")
    assert isinstance(alignment, float)
    
    penalty = _regime_penalty(MarketRegime.BULLISH_TREND, "long")
    assert isinstance(penalty, float)


def test_evidence_gate_none_config(monkeypatch):
    import analyzer.analysis.evidence
    
    signals = NormalizedSignals(
        higher_trend=0.8,
        context_trend=0.5,
        trigger_momentum=0.5,
        entry_momentum=0.5,
        volume_surge=0.5,
        buy_pressure=0.5,
        oi_funding_bias=0.5,
        funding_momentum=0.5,
        structure_position=0.5,
        rsi_alignment=0.5,
        macd_alignment=0.5,
        bb_alignment=0.5,
        vwap_alignment=0.5,
        market_structure_align=0.5,
        cumulative_delta_align=0.5,
        volume_poc_proximity=0.5,
    )
    
    # 1. Test loading default config successfully
    res = compute_graded_evidence(signals, MarketRegime.BULLISH_TREND, "long", {}, regime_gate_thresholds=None, config=None)
    assert res.regime_gate_passed is True
    
    # 2. Test configuration loading exception fallback
    monkeypatch.setattr(analyzer.analysis.evidence, "load_app_config", lambda: 1/0)
    res_exc = compute_graded_evidence(signals, MarketRegime.BULLISH_TREND, "long", {}, regime_gate_thresholds=None, config=None)
    assert res_exc.regime_gate_passed is True


def test_signal_decay():
    from analyzer.analysis.decay import compute_ttl, apply_decay
    from analyzer.analysis.models import TradeSetup, QualityLabel
    from datetime import datetime, UTC, timedelta
    
    # 1. Test compute_ttl
    assert compute_ttl("15m") == timedelta(minutes=45)
    assert compute_ttl("1h", multiplier=2.0) == timedelta(hours=2)
    assert compute_ttl("unknown_tf") == timedelta(minutes=45)
    
    # 2. Test apply_decay with valid_until is None
    setup_none = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.8,
        quality_score=80.0,
        quality_label=QualityLabel.HIGH,
        rationale="test",
        is_tradable=True,
        valid_until=None
    )
    assert apply_decay(setup_none, datetime.now(UTC)) is setup_none
    
    # 3. Test apply_decay with valid_until set and ttl_seconds > 0
    now = datetime(2026, 5, 30, 12, 0, 0, tzinfo=UTC)
    valid_until = now + timedelta(minutes=30)
    setup_with_ttl = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.8,
        quality_score=80.0,
        quality_label=QualityLabel.HIGH,
        rationale="test",
        is_tradable=True,
        valid_until=valid_until,
    )
    setup_with_ttl.ttl_seconds = 1800.0  # 30 minutes
    
    # 15 minutes remaining -> 50% freshness
    decayed_15 = apply_decay(setup_with_ttl, now + timedelta(minutes=15))
    assert decayed_15.confidence == 0.4
    assert decayed_15.is_stale is False
    
    # 0 minutes remaining -> 0% freshness (stale)
    decayed_stale = apply_decay(setup_with_ttl, now + timedelta(minutes=30))
    assert decayed_stale.confidence == 0.0
    assert decayed_stale.is_stale is True
    
    # Timezone naive conversion check
    naive_now = datetime(2026, 5, 30, 12, 15, 0)
    naive_valid = datetime(2026, 5, 30, 12, 30, 0)
    setup_naive = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.8,
        quality_score=80.0,
        quality_label=QualityLabel.HIGH,
        rationale="test",
        is_tradable=True,
        valid_until=naive_valid,
    )
    setup_naive.ttl_seconds = 1800.0
    decayed_naive = apply_decay(setup_naive, naive_now)
    assert decayed_naive.confidence == 0.4
    
    # 4. Test fallback: no ttl_seconds or <= 0
    setup_no_ttl_attr = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=0.8,
        quality_score=80.0,
        quality_label=QualityLabel.HIGH,
        rationale="test",
        is_tradable=True,
        valid_until=valid_until,
    )
    # remaining > 0 -> fresh (1.0 freshness)
    decayed_fresh = apply_decay(setup_no_ttl_attr, now + timedelta(minutes=15))
    assert decayed_fresh.confidence == 0.8
    assert decayed_fresh.is_stale is False
    
    # remaining <= 0 -> stale (0.0 freshness)
    decayed_stale_fallback = apply_decay(setup_no_ttl_attr, now + timedelta(minutes=45))
    assert decayed_stale_fallback.confidence == 0.0
    assert decayed_stale_fallback.is_stale is True


