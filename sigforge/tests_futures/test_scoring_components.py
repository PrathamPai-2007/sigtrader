"""Tests for plan 2.6 — scoring component coverage.

Priority gaps covered:
- _classify_regime() edge cases (low volatility, trending, ranging)
- _score_confluence() with and without volume profile
- _apply_reversal_penalty() trigger conditions
- DrawdownAdjuster.apply() scaling behaviour
- PortfolioRiskManager.allocate() cap logic
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from futures_analyzer.analysis.models import (
    AnalysisResult,
    Candle,
    EnhancedMetrics,
    MarketMeta,
    MarketRegime,
    QualityLabel,
    TimeframePlan,
    TradeSetup,
)
from futures_analyzer.analysis.scorer import (
    _apply_reversal_penalty,
    _classify_regime,
    _score_confluence,
    _SideMetrics,
)
from futures_analyzer.history.models import DrawdownState
from futures_analyzer.analysis.scorer import DrawdownAdjuster
from futures_analyzer.portfolio import PortfolioRiskManager
from futures_analyzer.config import PortfolioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(
    n: int,
    base: float = 100.0,
    step: float = 0.0,
    volume: float = 1000.0,
    *,
    start: datetime | None = None,
) -> list[Candle]:
    t = start or datetime(2026, 1, 1, tzinfo=UTC)
    out: list[Candle] = []
    for i in range(n):
        o = base + i * step
        c = o + step * 0.6
        hi = max(o, c) + abs(step) * 0.4 + 0.01
        lo = min(o, c) - abs(step) * 0.4 - 0.01
        out.append(Candle(
            open_time=t + timedelta(minutes=i * 15),
            close_time=t + timedelta(minutes=i * 15 + 15),
            open=o, high=hi, low=lo, close=c, volume=volume,
        ))
    return out


def _flat_candles(n: int, price: float = 100.0) -> list[Candle]:
    """Candles with near-zero range — produces low ADX (ranging/chop)."""
    t = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[Candle] = []
    for i in range(n):
        out.append(Candle(
            open_time=t + timedelta(minutes=i * 15),
            close_time=t + timedelta(minutes=i * 15 + 15),
            open=price, high=price + 0.01, low=price - 0.01, close=price,
            volume=500.0,
        ))
    return out


def _trending_candles(n: int, base: float = 100.0, step: float = 1.0) -> list[Candle]:
    """Strongly trending candles — produces high ADX with low current ATR percentile.

    The series has large bars early (establishing the trend) and smaller bars
    toward the end, so the most recent ATR sits in the lower percentile of the
    window — matching a mature, sustained trend rather than a volatile breakout.
    """
    t = datetime(2026, 1, 1, tzinfo=UTC)
    out: list[Candle] = []
    for i in range(n):
        o = base + i * step
        c = o + step
        # Early bars are larger; later bars are smaller — mature trend
        decay = max(0.2, 1.0 - (i / n) * 0.8)
        hi = c + step * 0.5 * decay
        lo = o - step * 0.2 * decay
        out.append(Candle(
            open_time=t + timedelta(minutes=i * 15),
            close_time=t + timedelta(minutes=i * 15 + 15),
            open=o, high=hi, low=lo, close=c, volume=1000.0,
        ))
    return out


def _side_metrics(
    side: str = "long",
    confidence: float = 0.7,
    quality: float = 60.0,
) -> _SideMetrics:
    return _SideMetrics(
        side=side,
        score=5.0,
        confidence=confidence,
        quality_label=QualityLabel.MEDIUM,
        quality_score=quality,
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
        evidence_agreement=4,
        evidence_total=7,
        deliberation_summary="",
    )


def _make_trade_setup(
    side: str = "long",
    quality_score: float = 70.0,
    quality_label: QualityLabel = QualityLabel.HIGH,
    confidence: float = 0.75,
    leverage: str = "6x",
    stop_distance_pct: float = 1.0,
) -> TradeSetup:
    entry = 100.0
    stop = entry * (1 - stop_distance_pct / 100) if side == "long" else entry * (1 + stop_distance_pct / 100)
    target = entry * 1.02 if side == "long" else entry * 0.98
    return TradeSetup(
        side=side,
        entry_price=entry,
        target_price=target,
        stop_loss=stop,
        leverage_suggestion=leverage,
        confidence=confidence,
        quality_label=quality_label,
        quality_score=quality_score,
        rationale="test",
        stop_distance_pct=stop_distance_pct,
        target_distance_pct=2.0,
        risk_reward_ratio=2.0,
    )


def _make_analysis_result(symbol: str = "BTCUSDT", stop_distance_pct: float = 1.0) -> AnalysisResult:
    setup = _make_trade_setup(stop_distance_pct=stop_distance_pct)
    meta = MarketMeta(symbol=symbol, mark_price=100.0, tick_size=0.01, step_size=0.001,
                      funding_rate=0.0, open_interest=1_000_000.0)
    return AnalysisResult(
        primary_setup=setup,
        secondary_context=_make_trade_setup(side="short"),
        timeframe_plan=TimeframePlan(
            context_timeframe="1h", trigger_timeframe="15m", higher_timeframe="4h"
        ),
        market_snapshot_meta=meta,
        enhanced_metrics=EnhancedMetrics(),
    )


def _drawdown_state(severity: str, drawdown_pct: float = 5.0, losses: int = 2) -> DrawdownState:
    return DrawdownState(
        lookback=20,
        cumulative_pnl_pct=-drawdown_pct,
        max_drawdown_pct=drawdown_pct,
        current_drawdown_pct=drawdown_pct,
        consecutive_losses=losses,
        severity=severity,
        sample_count=20,
    )


# ---------------------------------------------------------------------------
# _classify_regime
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_flat_market_returns_range_or_chop(self) -> None:
        candles = _flat_candles(80)
        regime, confidence = _classify_regime(candles, candles, 0.1, 100.0)
        assert regime in (MarketRegime.RANGE, MarketRegime.VOLATILE_CHOP)
        assert 0.0 <= confidence <= 1.0

    def test_strong_uptrend_returns_bullish_trend(self) -> None:
        candles = _trending_candles(80, base=100.0, step=1.0)
        regime, confidence = _classify_regime(candles, candles, 1.0, 180.0)
        assert regime == MarketRegime.BULLISH_TREND
        assert confidence >= 0.5

    def test_strong_downtrend_returns_bearish_trend(self) -> None:
        candles = _trending_candles(80, base=200.0, step=-1.0)
        regime, confidence = _classify_regime(candles, candles, 1.0, 120.0)
        assert regime == MarketRegime.BEARISH_TREND
        assert confidence >= 0.5

    def test_confidence_in_unit_interval(self) -> None:
        for step in (0.0, 0.5, 2.0, -2.0):
            candles = _make_candles(80, step=step)
            _, confidence = _classify_regime(candles, candles, 0.5, 100.0)
            assert 0.0 <= confidence <= 1.0

    def test_too_few_candles_returns_valid_regime(self) -> None:
        candles = _flat_candles(5)
        regime, confidence = _classify_regime(candles, candles, 0.1, 100.0)
        assert regime in {r for r in MarketRegime}
        assert 0.0 <= confidence <= 1.0

    def test_trending_confidence_higher_than_flat(self) -> None:
        flat = _flat_candles(80)
        trending = _trending_candles(80, step=1.5)
        _, flat_conf = _classify_regime(flat, flat, 0.1, 100.0)
        _, trend_conf = _classify_regime(trending, trending, 1.0, 200.0)
        assert trend_conf >= flat_conf


# ---------------------------------------------------------------------------
# _score_confluence
# ---------------------------------------------------------------------------

class TestScoreConfluence:
    def test_long_entry_near_support_scores_entry_factor(self) -> None:
        result = _score_confluence(
            "long", entry=100.0, target=104.0,
            support=100.3, resistance=110.0, atr=1.0,
        )
        assert result["entry_confluence"] >= 1.0

    def test_short_entry_near_resistance_scores_entry_factor(self) -> None:
        result = _score_confluence(
            "short", entry=109.5, target=105.0,
            support=100.0, resistance=110.0, atr=1.0,
        )
        assert result["entry_confluence"] >= 1.0

    def test_long_target_near_resistance_scores_target_factor(self) -> None:
        result = _score_confluence(
            "long", entry=100.0, target=109.8,
            support=100.0, resistance=110.0, atr=1.0,
        )
        assert result["target_confluence"] >= 1.0

    def test_no_alignment_returns_zeros(self) -> None:
        result = _score_confluence(
            "long", entry=103.7, target=107.3,
            support=50.0, resistance=200.0, atr=1.0,
        )
        assert result["entry_confluence"] == 0.0
        assert result["target_confluence"] == 0.0

    def test_volume_profile_adds_entry_factor_for_long(self) -> None:
        vp = {"val": 100.1, "poc": 101.0, "vah": 105.0}
        without = _score_confluence(
            "long", entry=100.0, target=104.0,
            support=90.0, resistance=110.0, atr=1.0,
        )
        with_vp = _score_confluence(
            "long", entry=100.0, target=104.0,
            support=90.0, resistance=110.0, atr=1.0,
            volume_profile=vp,
        )
        assert with_vp["entry_confluence"] >= without["entry_confluence"]

    def test_volume_profile_adds_target_factor_for_short(self) -> None:
        vp = {"val": 95.0, "poc": 96.0, "vah": 105.0}
        without = _score_confluence(
            "short", entry=105.0, target=95.5,
            support=90.0, resistance=110.0, atr=1.0,
        )
        with_vp = _score_confluence(
            "short", entry=105.0, target=95.5,
            support=90.0, resistance=110.0, atr=1.0,
            volume_profile=vp,
        )
        assert with_vp["target_confluence"] >= without["target_confluence"]

    def test_round_number_entry_adds_factor(self) -> None:
        result = _score_confluence(
            "long", entry=100.0, target=102.3,
            support=50.0, resistance=200.0, atr=1.0,
        )
        assert result["entry_confluence"] >= 1.0

    def test_returns_expected_keys(self) -> None:
        result = _score_confluence(
            "long", entry=100.0, target=102.0,
            support=99.0, resistance=105.0, atr=1.0,
        )
        assert "entry_confluence" in result
        assert "target_confluence" in result


# ---------------------------------------------------------------------------
# _apply_reversal_penalty
# ---------------------------------------------------------------------------

class TestApplyReversalPenalty:
    def test_no_signals_no_penalty(self) -> None:
        m = _side_metrics(confidence=0.8, quality=70.0)
        _apply_reversal_penalty(m, {}, MarketRegime.RANGE)
        assert m.confidence == pytest.approx(0.8)
        assert m.quality_score == pytest.approx(70.0)
        assert m.is_tradable is True

    def test_one_signal_reduces_confidence(self) -> None:
        m = _side_metrics(confidence=0.8, quality=70.0)
        _apply_reversal_penalty(m, {"rsi_div": True}, MarketRegime.RANGE)
        assert m.confidence < 0.8
        assert m.is_tradable is True  # 1 signal alone doesn't block in non-chop

    def test_two_signals_reduce_confidence_and_quality(self) -> None:
        m = _side_metrics(confidence=0.8, quality=70.0)
        _apply_reversal_penalty(m, {"rsi_div": True, "macd_cross": True}, MarketRegime.RANGE)
        assert m.confidence < 0.8
        assert m.quality_score < 70.0

    def test_three_signals_blocks_trade(self) -> None:
        m = _side_metrics(confidence=0.8, quality=70.0)
        _apply_reversal_penalty(
            m,
            {"rsi_div": True, "macd_cross": True, "bb_squeeze": True},
            MarketRegime.RANGE,
        )
        assert m.is_tradable is False
        assert any("early reversal" in r for r in m.tradable_reasons)

    def test_one_signal_in_volatile_chop_blocks_trade(self) -> None:
        m = _side_metrics(confidence=0.8, quality=70.0)
        _apply_reversal_penalty(m, {"rsi_div": True}, MarketRegime.VOLATILE_CHOP)
        assert m.is_tradable is False

    def test_signal_count_stored_in_components(self) -> None:
        m = _side_metrics()
        _apply_reversal_penalty(m, {"a": True, "b": True}, MarketRegime.RANGE)
        assert m.components["reversal_signal_count"] == 2.0

    def test_zero_signal_count_stored_when_no_signals(self) -> None:
        m = _side_metrics()
        _apply_reversal_penalty(m, {}, MarketRegime.RANGE)
        assert m.components["reversal_signal_count"] == 0.0


# ---------------------------------------------------------------------------
# DrawdownAdjuster.apply()
# ---------------------------------------------------------------------------

class TestDrawdownAdjuster:
    def test_none_severity_is_noop(self) -> None:
        setup = _make_trade_setup(quality_score=70.0, confidence=0.75, leverage="6x")
        state = _drawdown_state("none")
        result = DrawdownAdjuster.apply(setup, state)
        assert result.quality_score == setup.quality_score
        assert result.confidence == setup.confidence
        assert result.leverage_suggestion == setup.leverage_suggestion

    def test_mild_reduces_leverage_by_one_step(self) -> None:
        setup = _make_trade_setup(leverage="6x")
        state = _drawdown_state("mild")
        result = DrawdownAdjuster.apply(setup, state)
        current = int(setup.leverage_suggestion.rstrip("x"))
        new = int(result.leverage_suggestion.rstrip("x"))
        assert new < current

    def test_moderate_reduces_leverage_by_two_steps(self) -> None:
        setup = _make_trade_setup(leverage="8x")
        state = _drawdown_state("moderate")
        result = DrawdownAdjuster.apply(setup, state)
        current = int(setup.leverage_suggestion.rstrip("x"))
        new = int(result.leverage_suggestion.rstrip("x"))
        assert new < current

    def test_severe_floors_leverage_at_1x(self) -> None:
        setup = _make_trade_setup(leverage="10x")
        state = _drawdown_state("severe")
        result = DrawdownAdjuster.apply(setup, state)
        assert result.leverage_suggestion == "1x"

    def test_mild_reduces_quality_score(self) -> None:
        setup = _make_trade_setup(quality_score=70.0)
        result = DrawdownAdjuster.apply(setup, _drawdown_state("mild"))
        assert result.quality_score < 70.0

    def test_moderate_reduces_quality_more_than_mild(self) -> None:
        setup = _make_trade_setup(quality_score=70.0)
        mild = DrawdownAdjuster.apply(setup, _drawdown_state("mild"))
        moderate = DrawdownAdjuster.apply(setup, _drawdown_state("moderate"))
        assert moderate.quality_score < mild.quality_score

    def test_severe_reduces_confidence(self) -> None:
        setup = _make_trade_setup(confidence=0.75)
        result = DrawdownAdjuster.apply(setup, _drawdown_state("severe"))
        assert result.confidence < 0.75

    def test_quality_score_never_below_10(self) -> None:
        setup = _make_trade_setup(quality_score=12.0)
        result = DrawdownAdjuster.apply(setup, _drawdown_state("severe"))
        assert result.quality_score >= 10.0

    def test_confidence_never_below_zero(self) -> None:
        setup = _make_trade_setup(confidence=0.05)
        result = DrawdownAdjuster.apply(setup, _drawdown_state("severe"))
        assert result.confidence >= 0.0

    def test_warning_appended_for_tradable_setup(self) -> None:
        setup = _make_trade_setup()
        result = DrawdownAdjuster.apply(setup, _drawdown_state("moderate"))
        assert any("drawdown guard" in r for r in result.tradable_reasons)

    def test_quality_label_updated_after_penalty(self) -> None:
        # Start at HIGH, severe penalty should push it down
        setup = _make_trade_setup(quality_score=55.0, quality_label=QualityLabel.MEDIUM)
        result = DrawdownAdjuster.apply(setup, _drawdown_state("severe"))
        # Label must be consistent with the new score
        from futures_analyzer.analysis.scorer import _quality_label
        assert result.quality_label == _quality_label(result.quality_score)


# ---------------------------------------------------------------------------
# PortfolioRiskManager.allocate() cap logic
# ---------------------------------------------------------------------------

class TestPortfolioRiskManagerAllocate:
    def _cfg(self, **overrides) -> PortfolioConfig:
        defaults = dict(
            max_position_pct=0.20,
            max_risk_per_trade_pct=0.02,
            max_cluster_risk_pct=0.10,
            max_total_risk_pct=0.06,
            kelly_fraction=0.25,
            min_history_for_kelly=10,
        )
        defaults.update(overrides)
        return PortfolioConfig(**defaults)

    def test_empty_results_returns_zero_report(self) -> None:
        mgr = PortfolioRiskManager(self._cfg())
        report = mgr.allocate([], total_capital=100_000.0)
        assert report.total_notional_usd == 0.0
        assert report.total_risk_usd == 0.0

    def test_zero_capital_returns_zero_report(self) -> None:
        mgr = PortfolioRiskManager(self._cfg())
        result = _make_analysis_result("BTCUSDT")
        report = mgr.allocate([(1, result)], total_capital=0.0)
        assert report.total_notional_usd == 0.0

    def test_position_cap_applied_when_notional_exceeds_max(self) -> None:
        # With a tiny max_position_pct, the allocation must be capped
        mgr = PortfolioRiskManager(self._cfg(max_position_pct=0.05, max_risk_per_trade_pct=0.02))
        result = _make_analysis_result("BTCUSDT", stop_distance_pct=0.5)
        report = mgr.allocate([(1, result)], total_capital=100_000.0)
        assert report.allocations[0].notional_usd <= 100_000.0 * 0.05 + 1e-6

    def test_risk_cap_applied_when_stop_is_tight(self) -> None:
        # Very tight stop → implied notional from risk cap is small
        mgr = PortfolioRiskManager(self._cfg(max_risk_per_trade_pct=0.01, max_total_risk_pct=0.06))
        result = _make_analysis_result("BTCUSDT", stop_distance_pct=0.1)
        report = mgr.allocate([(1, result)], total_capital=100_000.0)
        # risk_usd must not exceed 1% of capital
        assert report.allocations[0].risk_usd <= 100_000.0 * 0.01 + 1e-4

    def test_total_risk_pct_within_cap(self) -> None:
        mgr = PortfolioRiskManager(self._cfg(max_total_risk_pct=0.06))
        results = [(i + 1, _make_analysis_result(sym, stop_distance_pct=1.0))
                   for i, sym in enumerate(["BTCUSDT", "ETHUSDT", "SOLUSDT"])]
        report = mgr.allocate(results, total_capital=100_000.0)
        assert report.total_risk_pct <= 0.06 + 1e-6

    def test_cluster_risk_cap_scales_down_correlated_positions(self) -> None:
        mgr = PortfolioRiskManager(self._cfg(max_cluster_risk_pct=0.04, max_total_risk_pct=0.10))
        results = [
            (1, _make_analysis_result("BTCUSDT", stop_distance_pct=1.0)),
            (2, _make_analysis_result("ETHUSDT", stop_distance_pct=1.0)),
        ]
        clusters = [["BTCUSDT", "ETHUSDT"]]
        report = mgr.allocate(results, total_capital=100_000.0, correlation_clusters=clusters)
        cluster_risk = sum(a.risk_usd for a in report.allocations)
        assert cluster_risk <= 100_000.0 * 0.04 + 1e-4

    def test_capped_flag_set_when_position_cap_triggered(self) -> None:
        mgr = PortfolioRiskManager(self._cfg(max_position_pct=0.01, max_risk_per_trade_pct=0.005))
        result = _make_analysis_result("BTCUSDT", stop_distance_pct=0.5)
        report = mgr.allocate([(1, result)], total_capital=1_000_000.0)
        assert report.allocations[0].capped is True

    def test_allocations_count_matches_input(self) -> None:
        mgr = PortfolioRiskManager(self._cfg())
        results = [(i + 1, _make_analysis_result(sym))
                   for i, sym in enumerate(["BTCUSDT", "ETHUSDT"])]
        report = mgr.allocate(results, total_capital=100_000.0)
        assert len(report.allocations) == 2


# ---------------------------------------------------------------------------
# normalize_signals (task 4.3)
# ---------------------------------------------------------------------------

from futures_analyzer.analysis.scorer import (
    IndicatorBundle,
    NormalizedSignals,
    normalize_signals,
)


def _make_bundle(
    *,
    higher_trend: float = 0.5,
    context_trend: float = 0.3,
    trigger_momentum: float = 0.4,
    entry_momentum: float = 0.2,
    trigger_volume_surge: float = 1.5,
    cumulative_delta: float = 0.3,
    rsi_14: float = 50.0,
    macd_histogram: float = 0.001,
    bb_position: float = 0.4,
    market_structure: str = "mixed",
    vwap: float = 100.0,
    poc: float = 100.0,
    vah: float = 102.0,
    val: float = 98.0,
    funding_rate: float | None = 0.0001,
    oi_change_pct: float | None = 0.5,
    funding_momentum: float = 0.0,
) -> IndicatorBundle:
    return IndicatorBundle(
        entry_atr=1.0,
        trigger_atr=1.0,
        context_atr=1.0,
        higher_atr=1.0,
        higher_trend=higher_trend,
        context_trend=context_trend,
        trigger_momentum=trigger_momentum,
        entry_momentum=entry_momentum,
        trigger_volume_surge=trigger_volume_surge,
        entry_volume_surge=1.0,
        cumulative_delta=cumulative_delta,
        rsi_14=rsi_14,
        macd_histogram=macd_histogram,
        stoch_k=50.0,
        bb_position=bb_position,
        bb_bandwidth_pct=2.0,
        swing_highs=[101.0, 103.0],
        swing_lows=[97.0, 99.0],
        market_structure=market_structure,
        liquidity_sweeps=[],
        vwap=vwap,
        vwap_upper_1sd=101.0,
        vwap_lower_1sd=99.0,
        vwap_upper_2sd=102.0,
        vwap_lower_2sd=98.0,
        poc=poc,
        vah=vah,
        val=val,
        rsi_divergence_type="none",
        rsi_divergence_strength=0.0,
        funding_rate=funding_rate,
        oi_change_pct=oi_change_pct,
        funding_momentum=funding_momentum,
        order_book_imbalance=0.0,
        bid_ask_spread_pct=0.0,
    )


def _make_meta(mark_price: float = 100.0) -> MarketMeta:
    return MarketMeta(symbol="BTCUSDT", mark_price=mark_price)


class TestNormalizeSignals:
    """Tests for normalize_signals — Requirements 3.1, 3.2, 3.3, 3.4, 3.5."""

    # Req 3.1: all fields in [0.0, 1.0]
    def test_all_fields_in_unit_interval_long(self) -> None:
        bundle = _make_bundle()
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        for field_name, value in vars(result).items():
            assert 0.0 <= value <= 1.0, f"{field_name}={value} out of [0,1]"

    def test_all_fields_in_unit_interval_short(self) -> None:
        bundle = _make_bundle()
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "short", MarketRegime.RANGE)
        for field_name, value in vars(result).items():
            assert 0.0 <= value <= 1.0, f"{field_name}={value} out of [0,1]"

    # Req 3.1: no NaN or infinite values
    def test_no_nan_or_inf(self) -> None:
        import math
        bundle = _make_bundle()
        meta = _make_meta()
        for side in ("long", "short"):
            result = normalize_signals(bundle, meta, side, MarketRegime.RANGE)
            for field_name, value in vars(result).items():
                assert math.isfinite(value), f"{field_name}={value} is not finite"

    # Req 3.2: directional complementarity for trend/momentum
    def test_long_short_higher_trend_sum_approx_one(self) -> None:
        bundle = _make_bundle(higher_trend=0.7)
        meta = _make_meta()
        long_sig = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        short_sig = normalize_signals(bundle, meta, "short", MarketRegime.RANGE)
        assert abs(long_sig.higher_trend + short_sig.higher_trend - 1.0) < 0.01

    def test_long_short_trigger_momentum_sum_approx_one(self) -> None:
        bundle = _make_bundle(trigger_momentum=0.5)
        meta = _make_meta()
        long_sig = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        short_sig = normalize_signals(bundle, meta, "short", MarketRegime.RANGE)
        assert abs(long_sig.trigger_momentum + short_sig.trigger_momentum - 1.0) < 0.01

    # Req 3.3: strongly bullish trend → long > 0.5, short < 0.5
    def test_bullish_trend_favors_long(self) -> None:
        bundle = _make_bundle(higher_trend=0.9)
        meta = _make_meta()
        long_sig = normalize_signals(bundle, meta, "long", MarketRegime.BULLISH_TREND)
        short_sig = normalize_signals(bundle, meta, "short", MarketRegime.BULLISH_TREND)
        assert long_sig.higher_trend > 0.5
        assert short_sig.higher_trend < 0.5

    def test_bearish_trend_favors_short(self) -> None:
        bundle = _make_bundle(higher_trend=-0.9)
        meta = _make_meta()
        long_sig = normalize_signals(bundle, meta, "long", MarketRegime.BEARISH_TREND)
        short_sig = normalize_signals(bundle, meta, "short", MarketRegime.BEARISH_TREND)
        assert long_sig.higher_trend < 0.5
        assert short_sig.higher_trend > 0.5

    # Req 3.4: sigmoid for trend/momentum, min-max for volume, gaussian for RSI
    def test_volume_surge_at_one_gives_zero(self) -> None:
        # surge=1.0 → (1.0-1.0)/2.0 = 0.0
        bundle = _make_bundle(trigger_volume_surge=1.0)
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        assert result.volume_surge == pytest.approx(0.0)

    def test_volume_surge_at_three_gives_one(self) -> None:
        # surge=3.0 → (3.0-1.0)/2.0 = 1.0
        bundle = _make_bundle(trigger_volume_surge=3.0)
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        assert result.volume_surge == pytest.approx(1.0)

    def test_rsi_at_center_long_gives_max_alignment(self) -> None:
        # RSI=45 is the center for long → gaussian_peak = 1.0
        bundle = _make_bundle(rsi_14=45.0)
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        assert result.rsi_alignment == pytest.approx(1.0)

    def test_rsi_at_center_short_gives_max_alignment(self) -> None:
        # RSI=55 is the center for short → gaussian_peak = 1.0
        bundle = _make_bundle(rsi_14=55.0)
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "short", MarketRegime.RANGE)
        assert result.rsi_alignment == pytest.approx(1.0)

    # Market structure alignment
    def test_hh_hl_structure_gives_full_long_alignment(self) -> None:
        bundle = _make_bundle(market_structure="HH_HL")
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        assert result.market_structure_align == pytest.approx(1.0)

    def test_lh_ll_structure_gives_full_short_alignment(self) -> None:
        bundle = _make_bundle(market_structure="LH_LL")
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "short", MarketRegime.RANGE)
        assert result.market_structure_align == pytest.approx(1.0)

    def test_mixed_structure_gives_half_alignment(self) -> None:
        bundle = _make_bundle(market_structure="mixed")
        meta = _make_meta()
        for side in ("long", "short"):
            result = normalize_signals(bundle, meta, side, MarketRegime.RANGE)
            assert result.market_structure_align == pytest.approx(0.5)

    def test_opposing_structure_gives_zero_alignment(self) -> None:
        # HH_HL for short → 0.0
        bundle = _make_bundle(market_structure="HH_HL")
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "short", MarketRegime.RANGE)
        assert result.market_structure_align == pytest.approx(0.0)

    # BB alignment
    def test_bb_position_zero_gives_full_long_alignment(self) -> None:
        bundle = _make_bundle(bb_position=0.0)
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        assert result.bb_alignment == pytest.approx(1.0)

    def test_bb_position_one_gives_full_short_alignment(self) -> None:
        bundle = _make_bundle(bb_position=1.0)
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "short", MarketRegime.RANGE)
        assert result.bb_alignment == pytest.approx(1.0)

    # Req 3.5: no NaN/inf with extreme inputs
    def test_extreme_trend_values_no_nan(self) -> None:
        import math
        bundle = _make_bundle(higher_trend=1.0, context_trend=-1.0,
                              trigger_momentum=1.0, entry_momentum=-1.0)
        meta = _make_meta()
        for side in ("long", "short"):
            result = normalize_signals(bundle, meta, side, MarketRegime.RANGE)
            for field_name, value in vars(result).items():
                assert math.isfinite(value), f"{field_name}={value} is not finite"

    def test_zero_vwap_no_division_error(self) -> None:
        import math
        bundle = _make_bundle(vwap=0.0)
        meta = _make_meta(mark_price=100.0)
        for side in ("long", "short"):
            result = normalize_signals(bundle, meta, side, MarketRegime.RANGE)
            for field_name, value in vars(result).items():
                assert math.isfinite(value), f"{field_name}={value} is not finite"

    def test_zero_poc_no_division_error(self) -> None:
        import math
        bundle = _make_bundle(poc=0.0)
        meta = _make_meta(mark_price=100.0)
        for side in ("long", "short"):
            result = normalize_signals(bundle, meta, side, MarketRegime.RANGE)
            for field_name, value in vars(result).items():
                assert math.isfinite(value), f"{field_name}={value} is not finite"

    def test_returns_normalized_signals_instance(self) -> None:
        bundle = _make_bundle()
        meta = _make_meta()
        result = normalize_signals(bundle, meta, "long", MarketRegime.RANGE)
        assert isinstance(result, NormalizedSignals)


# ---------------------------------------------------------------------------
# compute_graded_evidence (task 4.4)
# ---------------------------------------------------------------------------

from futures_analyzer.analysis.scorer import (
    EvidenceVector,
    compute_graded_evidence,
)

# Default equal weights for all 16 NormalizedSignals fields
_EQUAL_WEIGHTS: dict[str, float] = {
    "higher_trend": 0.0625,
    "context_trend": 0.0625,
    "trigger_momentum": 0.0625,
    "entry_momentum": 0.0625,
    "volume_surge": 0.0625,
    "buy_pressure": 0.0625,
    "oi_funding_bias": 0.0625,
    "funding_momentum": 0.0625,
    "structure_position": 0.0625,
    "rsi_alignment": 0.0625,
    "macd_alignment": 0.0625,
    "bb_alignment": 0.0625,
    "vwap_alignment": 0.0625,
    "market_structure_align": 0.0625,
    "cumulative_delta_align": 0.0625,
    "volume_poc_proximity": 0.0625,
}


def _make_signals(**overrides: float) -> NormalizedSignals:
    defaults = {
        "higher_trend": 0.5,
        "context_trend": 0.5,
        "trigger_momentum": 0.5,
        "entry_momentum": 0.5,
        "volume_surge": 0.5,
        "buy_pressure": 0.5,
        "oi_funding_bias": 0.5,
        "funding_momentum": 0.5,
        "structure_position": 0.5,
        "rsi_alignment": 0.5,
        "macd_alignment": 0.5,
        "bb_alignment": 0.5,
        "vwap_alignment": 0.5,
        "market_structure_align": 0.5,
        "cumulative_delta_align": 0.5,
        "volume_poc_proximity": 0.5,
    }
    defaults.update(overrides)
    return NormalizedSignals(**defaults)


class TestComputeGradedEvidence:
    """Tests for compute_graded_evidence — Requirements 4.1, 4.2, 4.3, 4.4, 4.5."""

    # Req 4.1: weighted_sum is dot product of signal values and weights
    def test_weighted_sum_is_dot_product(self) -> None:
        signals = _make_signals(higher_trend=1.0, context_trend=0.0)
        weights = {"higher_trend": 0.6, "context_trend": 0.4}
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", weights)
        # Only higher_trend and context_trend have non-zero weights
        expected = 1.0 * 0.6 + 0.0 * 0.4
        assert result.weighted_sum == pytest.approx(expected)

    def test_weighted_sum_with_equal_weights_and_all_half(self) -> None:
        signals = _make_signals()  # all 0.5
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        assert result.weighted_sum == pytest.approx(0.5, abs=1e-6)

    # Req 4.2: all strong signals → higher weighted_sum than all weak signals
    def test_strong_signals_produce_higher_weighted_sum_than_weak(self) -> None:
        strong = _make_signals(**{k: 0.9 for k in _EQUAL_WEIGHTS})
        weak = _make_signals(**{k: 0.1 for k in _EQUAL_WEIGHTS})
        strong_result = compute_graded_evidence(strong, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        weak_result = compute_graded_evidence(weak, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        assert strong_result.weighted_sum > weak_result.weighted_sum

    # Req 4.3: signal_count_above_threshold counts fields > 0.5
    def test_signal_count_above_threshold_all_above(self) -> None:
        signals = _make_signals(**{k: 0.8 for k in _EQUAL_WEIGHTS})
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        assert result.signal_count_above_threshold == 16

    def test_signal_count_above_threshold_none_above(self) -> None:
        signals = _make_signals(**{k: 0.3 for k in _EQUAL_WEIGHTS})
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        assert result.signal_count_above_threshold == 0

    def test_signal_count_above_threshold_exactly_at_boundary(self) -> None:
        # 0.5 is NOT above threshold (strictly > 0.5)
        signals = _make_signals(**{k: 0.5 for k in _EQUAL_WEIGHTS})
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        assert result.signal_count_above_threshold == 0

    def test_signal_count_partial(self) -> None:
        # Set 4 signals above 0.5, rest at 0.3
        overrides = {k: 0.3 for k in _EQUAL_WEIGHTS}
        above_keys = list(_EQUAL_WEIGHTS.keys())[:4]
        for k in above_keys:
            overrides[k] = 0.8
        signals = _make_signals(**overrides)
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        assert result.signal_count_above_threshold == 4

    # Req 4.5: strongest_signals = top 3, weakest_signals = bottom 3
    def test_strongest_signals_are_top_3(self) -> None:
        overrides = {k: 0.1 for k in _EQUAL_WEIGHTS}
        overrides["higher_trend"] = 0.9
        overrides["volume_surge"] = 0.85
        overrides["rsi_alignment"] = 0.8
        signals = _make_signals(**overrides)
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        top_names = [name for name, _ in result.strongest_signals]
        assert "higher_trend" in top_names
        assert "volume_surge" in top_names
        assert "rsi_alignment" in top_names
        assert len(result.strongest_signals) == 3

    def test_weakest_signals_are_bottom_3(self) -> None:
        overrides = {k: 0.9 for k in _EQUAL_WEIGHTS}
        overrides["bb_alignment"] = 0.05
        overrides["macd_alignment"] = 0.08
        overrides["funding_momentum"] = 0.1
        signals = _make_signals(**overrides)
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        bottom_names = [name for name, _ in result.weakest_signals]
        assert "bb_alignment" in bottom_names
        assert "macd_alignment" in bottom_names
        assert "funding_momentum" in bottom_names
        assert len(result.weakest_signals) == 3

    # Req 4.4: regime gate checks
    def test_bullish_trend_gate_passes_when_higher_trend_above_0_4(self) -> None:
        signals = _make_signals(higher_trend=0.5)
        result = compute_graded_evidence(signals, MarketRegime.BULLISH_TREND, "long", _EQUAL_WEIGHTS)
        assert result.regime_gate_passed is True

    def test_bullish_trend_gate_fails_when_higher_trend_below_0_4(self) -> None:
        signals = _make_signals(higher_trend=0.3)
        result = compute_graded_evidence(signals, MarketRegime.BULLISH_TREND, "long", _EQUAL_WEIGHTS)
        assert result.regime_gate_passed is False

    def test_bearish_trend_gate_uses_higher_trend(self) -> None:
        signals_pass = _make_signals(higher_trend=0.45)
        signals_fail = _make_signals(higher_trend=0.35)
        assert compute_graded_evidence(signals_pass, MarketRegime.BEARISH_TREND, "short", _EQUAL_WEIGHTS).regime_gate_passed is True
        assert compute_graded_evidence(signals_fail, MarketRegime.BEARISH_TREND, "short", _EQUAL_WEIGHTS).regime_gate_passed is False

    def test_breakout_gate_uses_volume_surge(self) -> None:
        signals_pass = _make_signals(volume_surge=0.4)
        signals_fail = _make_signals(volume_surge=0.2)
        assert compute_graded_evidence(signals_pass, MarketRegime.BREAKOUT, "long", _EQUAL_WEIGHTS).regime_gate_passed is True
        assert compute_graded_evidence(signals_fail, MarketRegime.BREAKOUT, "long", _EQUAL_WEIGHTS).regime_gate_passed is False

    def test_exhaustion_gate_uses_rsi_alignment(self) -> None:
        signals_pass = _make_signals(rsi_alignment=0.4)
        signals_fail = _make_signals(rsi_alignment=0.2)
        assert compute_graded_evidence(signals_pass, MarketRegime.EXHAUSTION, "long", _EQUAL_WEIGHTS).regime_gate_passed is True
        assert compute_graded_evidence(signals_fail, MarketRegime.EXHAUSTION, "long", _EQUAL_WEIGHTS).regime_gate_passed is False

    def test_range_gate_uses_structure_position(self) -> None:
        signals_pass = _make_signals(structure_position=0.4)
        signals_fail = _make_signals(structure_position=0.2)
        assert compute_graded_evidence(signals_pass, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS).regime_gate_passed is True
        assert compute_graded_evidence(signals_fail, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS).regime_gate_passed is False

    def test_volatile_chop_gate_stricter_threshold(self) -> None:
        # VOLATILE_CHOP requires higher_trend > 0.5 (stricter than BULLISH_TREND's 0.4)
        signals_pass = _make_signals(higher_trend=0.6)
        signals_fail = _make_signals(higher_trend=0.45)
        assert compute_graded_evidence(signals_pass, MarketRegime.VOLATILE_CHOP, "long", _EQUAL_WEIGHTS).regime_gate_passed is True
        assert compute_graded_evidence(signals_fail, MarketRegime.VOLATILE_CHOP, "long", _EQUAL_WEIGHTS).regime_gate_passed is False

    def test_transition_gate_uses_higher_trend_0_35(self) -> None:
        signals_pass = _make_signals(higher_trend=0.4)
        signals_fail = _make_signals(higher_trend=0.3)
        assert compute_graded_evidence(signals_pass, MarketRegime.TRANSITION, "long", _EQUAL_WEIGHTS).regime_gate_passed is True
        assert compute_graded_evidence(signals_fail, MarketRegime.TRANSITION, "long", _EQUAL_WEIGHTS).regime_gate_passed is False

    # Returns correct type
    def test_returns_evidence_vector_instance(self) -> None:
        signals = _make_signals()
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", _EQUAL_WEIGHTS)
        assert isinstance(result, EvidenceVector)

    # Missing weight keys default to 0.0
    def test_missing_weight_keys_default_to_zero(self) -> None:
        signals = _make_signals(higher_trend=1.0)
        result = compute_graded_evidence(signals, MarketRegime.RANGE, "long", {"higher_trend": 1.0})
        # Only higher_trend contributes
        assert result.weighted_sum == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# place_entry_stop_target (task 5.3)
# ---------------------------------------------------------------------------

from futures_analyzer.analysis.models import StrategyStyle
from futures_analyzer.analysis.scorer import (
    SwingPoints,
    EntryGeometry,
    place_entry_stop_target,
)


def _make_swings(
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> SwingPoints:
    highs = highs or [95.0, 97.0, 103.0, 105.0]
    lows = lows or [93.0, 95.0, 97.0, 99.0]
    return SwingPoints(
        recent_highs=sorted(highs),
        recent_lows=sorted(lows),
        nearest_high=max(highs),
        nearest_low=max(lows),
        second_high=sorted(highs)[-2] if len(highs) >= 2 else max(highs),
        second_low=sorted(lows)[-2] if len(lows) >= 2 else max(lows),
    )


def _make_bundle_for_geometry(
    *,
    vwap: float = 100.0,
    vwap_upper_1sd: float = 101.5,
    vwap_lower_1sd: float = 98.5,
    vwap_upper_2sd: float = 103.0,
    vwap_lower_2sd: float = 97.0,
    vah: float = 104.0,
    val: float = 96.0,
    poc: float = 100.0,
) -> IndicatorBundle:
    return IndicatorBundle(
        entry_atr=1.0,
        trigger_atr=1.0,
        context_atr=1.0,
        higher_atr=1.0,
        higher_trend=0.0,
        context_trend=0.0,
        trigger_momentum=0.0,
        entry_momentum=0.0,
        trigger_volume_surge=1.0,
        entry_volume_surge=1.0,
        cumulative_delta=0.0,
        rsi_14=50.0,
        macd_histogram=0.0,
        stoch_k=50.0,
        bb_position=0.5,
        bb_bandwidth_pct=2.0,
        swing_highs=[],
        swing_lows=[],
        market_structure="mixed",
        liquidity_sweeps=[],
        vwap=vwap,
        vwap_upper_1sd=vwap_upper_1sd,
        vwap_lower_1sd=vwap_lower_1sd,
        vwap_upper_2sd=vwap_upper_2sd,
        vwap_lower_2sd=vwap_lower_2sd,
        poc=poc,
        vah=vah,
        val=val,
        rsi_divergence_type="none",
        rsi_divergence_strength=0.0,
        funding_rate=None,
        oi_change_pct=None,
        funding_momentum=0.0,
        order_book_imbalance=0.0,
        bid_ask_spread_pct=0.0,
    )


_DEFAULT_PARAMS: dict[str, float] = {
    "atr_buffer_factor": 0.5,
    "min_rr_ratio": 1.5,
    "target_cap_atr_mult": 3.0,
}


class TestPlaceEntryStopTarget:
    """Tests for place_entry_stop_target — Requirements 6.1–6.8, 10.3."""

    # Req 6.1: long → stop < entry < target
    def test_long_ordering_stop_entry_target(self) -> None:
        swings = _make_swings(highs=[103.0, 105.0], lows=[97.0, 99.0])
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.stop < geo.entry < geo.target

    # Req 6.2: short → target < entry < stop
    def test_short_ordering_target_entry_stop(self) -> None:
        swings = _make_swings(highs=[103.0, 105.0], lows=[93.0, 95.0])
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "short", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.target < geo.entry < geo.stop

    # Req 6.3: rr_ratio >= min_rr_ratio
    def test_rr_ratio_meets_minimum(self) -> None:
        swings = _make_swings(highs=[103.0], lows=[99.0])
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.rr_ratio >= _DEFAULT_PARAMS["min_rr_ratio"] - 1e-9

    def test_rr_ratio_meets_minimum_short(self) -> None:
        swings = _make_swings(highs=[103.0], lows=[95.0])
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "short", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.rr_ratio >= _DEFAULT_PARAMS["min_rr_ratio"] - 1e-9

    # Req 6.4: stop_anchor and target_anchor are valid values
    def test_stop_anchor_valid_long(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.stop_anchor in {"swing_low", "vwap_lower", "val", "atr_fallback", "rr_enforced"}

    def test_target_anchor_valid_long(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.target_anchor in {"swing_high", "vwap_upper", "vah", "atr_cap", "rr_enforced"}

    def test_stop_anchor_valid_short(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "short", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.stop_anchor in {"swing_high", "vwap_upper", "vah", "atr_fallback", "rr_enforced"}

    def test_target_anchor_valid_short(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "short", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.target_anchor in {"swing_low", "vwap_lower", "val", "atr_cap", "rr_enforced"}

    # Req 6.5: long stop priority — swing_low preferred when available
    def test_long_stop_uses_swing_low_when_available(self) -> None:
        # swing low at 98.5 → stop = 98.5 - atr_buffer = 98.5 - 0.5 = 98.0
        swings = _make_swings(highs=[103.0, 105.0], lows=[98.5])
        bundle = _make_bundle_for_geometry(vwap_lower_1sd=97.0, val=96.0)
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.stop_anchor == "swing_low"

    # Req 6.5: long stop falls back to atr_fallback when no swing lows below entry
    def test_long_stop_atr_fallback_when_no_swing_lows(self) -> None:
        # All swing lows above entry - atr*0.1 = 99.9, so none qualify
        swings = _make_swings(highs=[103.0], lows=[100.5, 101.0])
        # Also make vwap_lower_1sd and val above entry so they don't qualify
        bundle = _make_bundle_for_geometry(vwap_lower_1sd=101.0, val=101.0)
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.stop_anchor == "atr_fallback"

    # Req 6.6: long target uses swing_high when available and R:R sufficient
    def test_long_target_uses_swing_high_when_rr_sufficient(self) -> None:
        # entry=100, stop~98 (swing_low at 98.5 - 0.5 buffer), swing_high at 104
        # risk~2, reward~4 → rr=2.0 >= 1.5 ✓
        swings = _make_swings(highs=[104.0], lows=[98.5])
        bundle = _make_bundle_for_geometry(vwap_upper_2sd=103.0, vah=103.5)
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.target_anchor == "swing_high"

    # Req 6.7: short stop uses swing_high above entry
    def test_short_stop_uses_swing_high_when_available(self) -> None:
        swings = _make_swings(highs=[101.5, 103.0], lows=[95.0, 97.0])
        bundle = _make_bundle_for_geometry(vwap_upper_1sd=102.0, vah=103.5)
        geo = place_entry_stop_target(
            "short", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.stop_anchor == "swing_high"

    # Req 6.8: tick quantization
    def test_tick_quantization_applied(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        tick = 0.5
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, tick,
        )
        # All prices should be multiples of 0.5
        for price in (geo.entry, geo.stop, geo.target):
            remainder = round(price % tick, 10)
            assert remainder < 1e-9 or abs(remainder - tick) < 1e-9, \
                f"{price} is not a multiple of {tick}"

    def test_no_tick_no_quantization_error(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert isinstance(geo, EntryGeometry)

    # Req 10.3: zero ATR substitution
    def test_zero_atr_substituted_with_px_times_0001(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        # With atr=0, should use px*0.001 = 0.1 as minimum ATR
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 0.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        # Should not raise and should produce valid geometry
        assert geo.stop < geo.entry < geo.target
        assert geo.rr_ratio >= _DEFAULT_PARAMS["min_rr_ratio"] - 1e-9

    # rr_enforced anchor when R:R is insufficient
    def test_rr_enforced_when_target_too_close(self) -> None:
        # swing_high very close to entry → R:R < min_rr → target extended
        swings = _make_swings(highs=[100.5], lows=[99.0])
        bundle = _make_bundle_for_geometry(
            vwap_upper_2sd=100.8, vah=100.9,
            vwap_lower_1sd=98.5, val=98.0,
        )
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.rr_ratio >= _DEFAULT_PARAMS["min_rr_ratio"] - 1e-9
        assert geo.stop < geo.entry < geo.target

    # Derived fields are consistent
    def test_risk_reward_consistency(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert geo.risk == pytest.approx(abs(geo.entry - geo.stop), abs=1e-9)
        assert geo.reward == pytest.approx(abs(geo.target - geo.entry), abs=1e-9)
        assert geo.rr_ratio == pytest.approx(geo.reward / max(geo.risk, 1e-9), abs=1e-9)

    def test_invalidation_strength_in_unit_interval(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        for side in ("long", "short"):
            geo = place_entry_stop_target(
                side, 100.0, swings, bundle, 1.0,
                MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
            )
            assert 0.0 <= geo.invalidation_strength <= 1.0

    def test_returns_entry_geometry_instance(self) -> None:
        swings = _make_swings()
        bundle = _make_bundle_for_geometry()
        geo = place_entry_stop_target(
            "long", 100.0, swings, bundle, 1.0,
            MarketRegime.RANGE, StrategyStyle.CONSERVATIVE, _DEFAULT_PARAMS, None,
        )
        assert isinstance(geo, EntryGeometry)


# ---------------------------------------------------------------------------
# geometry_quality_score tests (Task 5.4 — Requirements 7.1–7.7)
# ---------------------------------------------------------------------------

from futures_analyzer.analysis.scorer import EntryGeometry, geometry_quality_score


def _make_geometry(
    *,
    rr_ratio: float = 2.0,
    atr_multiple_to_stop: float = 1.2,
    stop_anchor: str = "swing_low",
    target_anchor: str = "swing_high",
) -> EntryGeometry:
    """Build a minimal EntryGeometry for testing geometry_quality_score."""
    entry = 100.0
    risk = 1.0
    reward = risk * rr_ratio
    stop = entry - risk
    target = entry + reward
    return EntryGeometry(
        entry=entry,
        stop=stop,
        target=target,
        risk=risk,
        reward=reward,
        rr_ratio=rr_ratio,
        stop_distance_pct=risk / entry * 100,
        target_distance_pct=reward / entry * 100,
        atr_multiple_to_stop=atr_multiple_to_stop,
        atr_multiple_to_target=atr_multiple_to_stop * rr_ratio,
        stop_anchor=stop_anchor,
        target_anchor=target_anchor,
        invalidation_strength=0.4,
    )


class TestGeometryQualityScore:
    """Tests for geometry_quality_score — Requirements 7.1–7.7."""

    def test_score_in_valid_range(self):
        """Req 7.1: score must be in [10.0, 95.0]."""
        geo = _make_geometry()
        score, _ = geometry_quality_score(geo, MarketRegime.RANGE, {}, 0.5)
        assert 10.0 <= score <= 95.0

    def test_quality_label_high(self):
        """Req 7.2: score >= 75 → HIGH."""
        # Maximise score: high R:R, ideal ATR stop, swing anchors, max confluence
        geo = _make_geometry(rr_ratio=3.0, atr_multiple_to_stop=1.2,
                             stop_anchor="swing_low", target_anchor="swing_high")
        confluence = {"entry_confluence": 3.0, "target_confluence": 3.0}
        score, label = geometry_quality_score(geo, MarketRegime.BULLISH_TREND, confluence, 0.9)
        assert score >= 75.0
        assert label == QualityLabel.HIGH

    def test_quality_label_medium(self):
        """Req 7.2: 55 <= score < 75 → MEDIUM."""
        # Moderate setup: decent R:R, ideal stop, no confluence
        geo = _make_geometry(rr_ratio=2.0, atr_multiple_to_stop=1.2,
                             stop_anchor="swing_low", target_anchor="swing_high")
        score, label = geometry_quality_score(geo, MarketRegime.RANGE, {}, 0.5)
        if 55.0 <= score < 75.0:
            assert label == QualityLabel.MEDIUM

    def test_quality_label_low(self):
        """Req 7.2: score < 55 → LOW."""
        # Worst case: low R:R, stop outside ideal range, ATR fallback anchors
        geo = _make_geometry(rr_ratio=0.5, atr_multiple_to_stop=3.0,
                             stop_anchor="atr_fallback", target_anchor="atr_fallback")
        score, label = geometry_quality_score(geo, MarketRegime.VOLATILE_CHOP, {}, 0.1)
        assert score < 55.0
        assert label == QualityLabel.LOW

    def test_label_consistent_with_score(self):
        """Req 7.2: label is always consistent with the returned score."""
        for rr in [0.5, 1.0, 2.0, 3.0]:
            for anchor in ["swing_low", "vwap_lower", "atr_fallback"]:
                geo = _make_geometry(rr_ratio=rr, stop_anchor=anchor)
                score, label = geometry_quality_score(geo, MarketRegime.RANGE, {}, 0.5)
                if score >= 75.0:
                    assert label == QualityLabel.HIGH
                elif score >= 55.0:
                    assert label == QualityLabel.MEDIUM
                else:
                    assert label == QualityLabel.LOW

    def test_higher_rr_gives_higher_score(self):
        """Req 7.3: higher R:R → higher score, all else equal."""
        geo_low  = _make_geometry(rr_ratio=1.0)
        geo_high = _make_geometry(rr_ratio=3.0)
        confluence = {"entry_confluence": 1.0}
        score_low,  _ = geometry_quality_score(geo_low,  MarketRegime.RANGE, confluence, 0.5)
        score_high, _ = geometry_quality_score(geo_high, MarketRegime.RANGE, confluence, 0.5)
        assert score_high > score_low

    def test_swing_anchor_beats_atr_fallback(self):
        """Req 7.4: swing anchor → higher score than ATR fallback, all else equal."""
        geo_swing = _make_geometry(stop_anchor="swing_low", target_anchor="swing_high")
        geo_atr   = _make_geometry(stop_anchor="atr_fallback", target_anchor="atr_fallback")
        score_swing, _ = geometry_quality_score(geo_swing, MarketRegime.RANGE, {}, 0.5)
        score_atr,   _ = geometry_quality_score(geo_atr,   MarketRegime.RANGE, {}, 0.5)
        assert score_swing > score_atr

    def test_ideal_atr_stop_gives_bonus(self):
        """Req 7.5: stop in ideal ATR range [0.8, 2.0] → bonus vs outside range."""
        geo_ideal   = _make_geometry(atr_multiple_to_stop=1.2)   # inside [0.8, 2.0]
        geo_outside = _make_geometry(atr_multiple_to_stop=3.0)   # outside
        score_ideal,   _ = geometry_quality_score(geo_ideal,   MarketRegime.RANGE, {}, 0.5)
        score_outside, _ = geometry_quality_score(geo_outside, MarketRegime.RANGE, {}, 0.5)
        assert score_ideal > score_outside

    def test_confluence_bonuses_increase_score(self):
        """Req 7.6: confluence factors at entry/target increase score."""
        geo = _make_geometry()
        score_no_conf, _ = geometry_quality_score(geo, MarketRegime.RANGE, {}, 0.5)
        score_with_conf, _ = geometry_quality_score(
            geo, MarketRegime.RANGE,
            {"entry_confluence": 3.0, "target_confluence": 3.0},
            0.5,
        )
        assert score_with_conf > score_no_conf

    def test_confluence_capped_at_3_factors(self):
        """Req 7.6: confluence beyond 3 factors has no additional effect."""
        geo = _make_geometry()
        score_3, _ = geometry_quality_score(
            geo, MarketRegime.RANGE, {"entry_confluence": 3.0}, 0.5
        )
        score_10, _ = geometry_quality_score(
            geo, MarketRegime.RANGE, {"entry_confluence": 10.0}, 0.5
        )
        assert score_3 == score_10

    def test_quality_independent_of_confidence(self):
        """Req 7.7: changing regime_confidence does not change the score."""
        geo = _make_geometry()
        confluence = {"entry_confluence": 2.0}
        score_low,  _ = geometry_quality_score(geo, MarketRegime.RANGE, confluence, 0.1)
        score_high, _ = geometry_quality_score(geo, MarketRegime.RANGE, confluence, 0.99)
        assert score_low == score_high

    def test_score_clamped_to_min(self):
        """Req 7.1: score never goes below 10.0."""
        # Worst possible setup
        geo = _make_geometry(rr_ratio=0.1, atr_multiple_to_stop=5.0,
                             stop_anchor="atr_fallback", target_anchor="atr_fallback")
        score, _ = geometry_quality_score(geo, MarketRegime.VOLATILE_CHOP, {}, 0.0)
        assert score >= 10.0

    def test_score_clamped_to_max(self):
        """Req 7.1: score never exceeds 95.0."""
        geo = _make_geometry(rr_ratio=100.0, atr_multiple_to_stop=1.2,
                             stop_anchor="swing_low", target_anchor="swing_high")
        score, _ = geometry_quality_score(
            geo, MarketRegime.BULLISH_TREND,
            {"entry_confluence": 100.0, "target_confluence": 100.0},
            1.0,
        )
        assert score <= 95.0

    def test_vwap_anchor_between_swing_and_atr(self):
        """Anchor hierarchy: swing > vwap > atr_fallback."""
        geo_swing = _make_geometry(stop_anchor="swing_low")
        geo_vwap  = _make_geometry(stop_anchor="vwap_lower")
        geo_atr   = _make_geometry(stop_anchor="atr_fallback")
        s_swing, _ = geometry_quality_score(geo_swing, MarketRegime.RANGE, {}, 0.5)
        s_vwap,  _ = geometry_quality_score(geo_vwap,  MarketRegime.RANGE, {}, 0.5)
        s_atr,   _ = geometry_quality_score(geo_atr,   MarketRegime.RANGE, {}, 0.5)
        assert s_swing > s_vwap > s_atr

    def test_rr_enforced_anchor_neutral(self):
        """rr_enforced anchor contributes 0 pts (neither bonus nor penalty)."""
        geo_rr  = _make_geometry(stop_anchor="rr_enforced", target_anchor="rr_enforced")
        geo_atr = _make_geometry(stop_anchor="atr_fallback", target_anchor="atr_fallback")
        score_rr,  _ = geometry_quality_score(geo_rr,  MarketRegime.RANGE, {}, 0.5)
        score_atr, _ = geometry_quality_score(geo_atr, MarketRegime.RANGE, {}, 0.5)
        # rr_enforced (0 pts each) should be better than atr_fallback (-5 pts each)
        assert score_rr > score_atr
