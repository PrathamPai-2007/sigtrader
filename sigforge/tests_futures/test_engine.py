from datetime import UTC, datetime, timedelta

from futures_analyzer.analysis import MarketMode, SetupAnalyzer, StrategyStyle, build_timeframe_plan
from futures_analyzer.analysis.models import Candle, MarketMeta, MarketRegime, QualityLabel
from futures_analyzer.analysis.scorer import _leverage_suggestion, _structure_biases, _volume_surge_ratio


def _candles(start: datetime, n: int, base: float, step: float, volume_start: float = 10.0) -> list[Candle]:
    out: list[Candle] = []
    for i in range(n):
        o = base + (i * step)
        c = o + (step * 0.6)
        hi = max(o, c) + 0.4
        lo = min(o, c) - 0.4
        t0 = start + timedelta(minutes=i * 5)
        out.append(
            Candle(
                open_time=t0,
                close_time=t0 + timedelta(minutes=5),
                open=o,
                high=hi,
                low=lo,
                close=c,
                volume=volume_start + i,
            )
        )
    return out


def _breakout_candles(start: datetime, n: int = 80, base: float = 100.0) -> list[Candle]:
    candles = _candles(start, n - 6, base, 0.0, volume_start=25.0)
    price = candles[-1].close if candles else base
    for i in range(6):
        open_ = price
        close = price + 1.4 + (i * 0.1)
        high = close + 0.3
        low = open_ - 0.2
        t0 = start + timedelta(minutes=(n - 6 + i) * 5)
        candles.append(
            Candle(
                open_time=t0,
                close_time=t0 + timedelta(minutes=5),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=200.0 + i * 40.0,
            )
        )
        price = close
    return candles


def test_engine_returns_enriched_primary_and_secondary_setup() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _candles(start, 80, 100.0, 0.3)
    context = _candles(start, 80, 98.0, 0.2)
    analyzer = SetupAnalyzer(risk_reward=2.0, style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY)
    market = MarketMeta(
        symbol="BTCUSDT",
        mark_price=trigger[-1].close,
        tick_size=0.1,
        step_size=0.001,
        funding_rate=-0.0002,
        open_interest=1_000_000.0,
        open_interest_change_pct=1.8,
    )
    result = analyzer.analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(),
    )
    primary = result.primary_setup
    assert primary.side in ("long", "short")
    assert result.secondary_context.side in ("long", "short")
    assert result.secondary_context.side != primary.side
    assert primary.quality_label in {QualityLabel.HIGH, QualityLabel.MEDIUM, QualityLabel.LOW}
    assert primary.quality_score >= 0.0
    assert primary.leverage_suggestion.endswith("x")
    assert primary.risk_reward_ratio > 0.0
    assert primary.stop_distance_pct > 0.0
    assert primary.target_distance_pct > 0.0
    assert primary.atr_multiple_to_stop > 0.0
    assert primary.atr_multiple_to_target > 0.0
    assert 0.0 <= primary.invalidation_strength <= 1.0
    assert result.market_regime.value in {"bullish_trend", "bearish_trend", "range", "volatile_chop"}
    assert result.regime_confidence >= 0.0
    impacts = [item.impact for item in primary.top_positive_contributors]
    assert impacts == sorted(impacts, reverse=True)
    negative_impacts = [item.impact for item in primary.top_negative_contributors]
    assert negative_impacts == sorted(negative_impacts, reverse=True)
    assert "risk_reward_ratio" in primary.score_components
    assert "invalidation_strength" in primary.score_components


def test_engine_fallback_rr_still_produces_metrics() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _candles(start, 80, 100.0, 0.0001)
    context = _candles(start, 80, 100.0, 0.0001)
    market = MarketMeta(
        symbol="ETHUSDT",
        mark_price=100.01,
        tick_size=0.01,
        step_size=0.001,
        funding_rate=0.0,
        open_interest=100_000.0,
        open_interest_change_pct=0.0,
    )
    result = SetupAnalyzer(risk_reward=2.0, style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY).analyze(
        symbol="ETHUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(),
    )
    assert result.primary_setup.stop_loss != result.primary_setup.entry_price
    assert result.primary_setup.target_price != result.primary_setup.entry_price
    # Structure targets are used as-is; absolute RR floor is 0.8 (scorer gate)
    assert result.primary_setup.risk_reward_ratio >= 0.0


def test_engine_funding_and_oi_bias_can_tilt_direction() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _candles(start, 80, 100.0, -0.01)
    context = _candles(start, 80, 100.0, -0.01)
    market = MarketMeta(
        symbol="XRPUSDT",
        mark_price=100.0,
        tick_size=0.01,
        step_size=0.1,
        funding_rate=0.0005,
        funding_rate_history=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        open_interest=200_000.0,
        open_interest_change_pct=15.0,
    )
    result = SetupAnalyzer(risk_reward=2.0, style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY).analyze(
        symbol="XRPUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(),
    )
    assert result.primary_setup.side == "short"
    assert result.primary_setup.is_tradable is False
    assert result.primary_setup.quality_label in {QualityLabel.LOW, QualityLabel.MEDIUM}



def test_engine_conservative_caps_target_more_than_aggressive() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _candles(start, 80, 100.0, 0.0)
    trigger[-1] = trigger[-1].model_copy(update={"high": 140.0, "close": 100.0, "open": 100.0, "low": 99.5})
    context = _candles(start, 80, 100.0, 0.0)
    market = MarketMeta(
        symbol="BTCUSDT",
        mark_price=100.0,
        tick_size=0.1,
        step_size=0.001,
        funding_rate=-0.0002,
        open_interest=200_000.0,
        open_interest_change_pct=2.0,
    )
    out_cons = SetupAnalyzer(risk_reward=2.0, style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY).analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY),
    )
    out_aggr = SetupAnalyzer(risk_reward=2.0, style=StrategyStyle.AGGRESSIVE, market_mode=MarketMode.INTRADAY).analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(style=StrategyStyle.AGGRESSIVE, market_mode=MarketMode.INTRADAY),
    )
    assert out_cons.primary_setup.side == out_aggr.primary_setup.side
    if out_cons.primary_setup.side == "long":
        assert out_cons.primary_setup.target_price <= out_aggr.primary_setup.target_price
    else:
        assert out_cons.primary_setup.target_price >= out_aggr.primary_setup.target_price



def test_engine_long_term_allows_wider_target_cap_than_intraday() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _candles(start, 80, 100.0, 0.0)
    trigger[-1] = trigger[-1].model_copy(update={"high": 145.0, "close": 100.0, "open": 100.0, "low": 99.5})
    context = _candles(start, 80, 100.0, 0.0)
    market = MarketMeta(
        symbol="BTCUSDT",
        mark_price=100.0,
        tick_size=0.1,
        step_size=0.001,
        funding_rate=-0.0002,
        open_interest=200_000.0,
        open_interest_change_pct=2.0,
    )
    out_intraday = SetupAnalyzer(
        risk_reward=2.0,
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
    ).analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY),
    )
    out_long_term = SetupAnalyzer(
        risk_reward=2.0,
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.LONG_TERM,
    ).analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.LONG_TERM),
    )
    assert out_intraday.primary_setup.side == out_long_term.primary_setup.side
    if out_intraday.primary_setup.side == "long":
        assert out_long_term.primary_setup.target_price >= out_intraday.primary_setup.target_price
    else:
        assert out_long_term.primary_setup.target_price <= out_intraday.primary_setup.target_price


def test_engine_detects_range_or_chop_regime_for_flat_market() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _candles(start, 80, 100.0, 0.0001)
    context = _candles(start, 80, 100.0, 0.0001)
    market = MarketMeta(
        symbol="BTCUSDT",
        mark_price=100.0,
        tick_size=0.1,
        step_size=0.001,
        funding_rate=0.0,
        open_interest=100_000.0,
        open_interest_change_pct=0.0,
    )
    result = SetupAnalyzer(risk_reward=2.0, style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY).analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(),
    )
    assert result.market_regime.value in {"range", "volatile_chop"}


def test_engine_handles_zero_volatility_candles() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _candles(start, 80, 100.0, 0.0)
    context = _candles(start, 80, 100.0, 0.0)
    trigger = [candle.model_copy(update={"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}) for candle in trigger]
    context = [candle.model_copy(update={"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}) for candle in context]
    market = MarketMeta(
        symbol="BTCUSDT",
        mark_price=100.0,
        tick_size=0.1,
        step_size=0.001,
        funding_rate=0.0,
        open_interest=100_000.0,
        open_interest_change_pct=0.0,
    )
    result = SetupAnalyzer(risk_reward=2.0, style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY).analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(),
    )
    assert result.primary_setup.entry_price == 100.0
    assert result.primary_setup.stop_loss != result.primary_setup.entry_price
    assert result.primary_setup.target_price != result.primary_setup.entry_price
    assert result.primary_setup.risk_reward_ratio >= 1.0


def test_engine_preserves_small_tick_precision() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _candles(start, 80, 0.000123, 0.0000002)
    context = _candles(start, 80, 0.000122, 0.00000015)
    tick_size = 0.00000001
    market = MarketMeta(
        symbol="DOGEUSDT",
        mark_price=trigger[-1].close,
        tick_size=tick_size,
        step_size=1.0,
        funding_rate=0.0,
        open_interest=50_000.0,
        open_interest_change_pct=0.0,
    )
    result = SetupAnalyzer(risk_reward=2.0, style=StrategyStyle.CONSERVATIVE, market_mode=MarketMode.INTRADAY).analyze(
        symbol="DOGEUSDT",
        trigger_candles=trigger,
        context_candles=context,
        market=market,
        timeframe_plan=build_timeframe_plan(),
    )
    for price in (
        result.primary_setup.entry_price,
        result.primary_setup.stop_loss,
        result.primary_setup.target_price,
    ):
        units = round(price / tick_size)
        assert abs((units * tick_size) - price) < 1e-12


def test_engine_leverage_ranges_follow_quality_caps() -> None:
    assert _leverage_suggestion(
        stop_distance_pct=0.6,
        quality_label=QualityLabel.LOW,
        confidence=0.9,
        regime=MarketRegime.BULLISH_TREND,
    ) == "2x"
    assert _leverage_suggestion(
        stop_distance_pct=0.6,
        quality_label=QualityLabel.MEDIUM,
        confidence=0.9,
        regime=MarketRegime.BULLISH_TREND,
    ) == "4x"
    assert _leverage_suggestion(
        stop_distance_pct=0.6,
        quality_label=QualityLabel.HIGH,
        confidence=0.9,
        regime=MarketRegime.BULLISH_TREND,
    ) == "6x"
    assert _leverage_suggestion(
        stop_distance_pct=0.6,
        quality_label=QualityLabel.HIGH,
        confidence=0.9,
        regime=MarketRegime.VOLATILE_CHOP,
    ) == "4x"
    assert _leverage_suggestion(
        stop_distance_pct=0.6,
        quality_label=QualityLabel.HIGH,
        confidence=0.3,
        regime=MarketRegime.BULLISH_TREND,
    ) == "1x"


def test_structure_biases_flip_toward_better_entry_location() -> None:
    near_support_long, near_support_short = _structure_biases(px=101.0, support=100.0, resistance=110.0)
    near_resistance_long, near_resistance_short = _structure_biases(px=109.0, support=100.0, resistance=110.0)
    assert near_support_long > near_support_short
    assert near_resistance_short > near_resistance_long


def test_volume_surge_uses_prior_candles_as_baseline() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = _candles(start, 4, 100.0, 0.1, volume_start=10.0)
    candles[:3] = [candle.model_copy(update={"volume": 10.0}) for candle in candles[:3]]
    candles[-1] = candles[-1].model_copy(update={"volume": 50.0})
    assert round(_volume_surge_ratio(candles, window=3), 4) == 5.0


def test_engine_breakout_analysis_returns_valid_new_regime_and_setup_fields() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    trigger = _breakout_candles(start)
    context = _breakout_candles(start, base=99.0)
    higher = _breakout_candles(start, base=98.0)
    market = MarketMeta(
        symbol="BTCUSDT",
        mark_price=trigger[-1].close,
        tick_size=0.1,
        step_size=0.001,
        funding_rate=-0.0001,
        open_interest=900_000.0,
        open_interest_change_pct=2.5,
    )

    result = SetupAnalyzer(
        risk_reward=2.0,
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
    ).analyze(
        symbol="BTCUSDT",
        trigger_candles=trigger,
        context_candles=context,
        higher_candles=higher,
        market=market,
        timeframe_plan=build_timeframe_plan(),
    )

    assert result.market_regime in set(MarketRegime)
    assert result.primary_setup.stop_anchor in {
        "swing_low", "swing_high", "vwap_lower", "vwap_upper", "val", "vah", "atr_fallback",
        "swing_low_sweep", "swing_high_sweep"
    }
    assert result.primary_setup.target_anchor in {
        "swing_high", "swing_low", "vwap_upper", "vwap_lower", "vah", "val", "atr_cap", "rr_enforced",
        "swing_high_sweep", "swing_low_sweep"
    }
    assert result.primary_setup.regime_state == result.market_regime.value
    assert result.primary_setup.signal_strengths
    assert all(0.0 <= value <= 1.0 for value in result.primary_setup.signal_strengths.values())


def test_engine_stronger_signal_environment_increases_confidence() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    strong_trigger = _candles(start, 80, 100.0, 0.45, volume_start=100.0)
    strong_context = _candles(start, 80, 98.0, 0.35, volume_start=120.0)
    flat_trigger = _candles(start, 80, 100.0, 0.0001, volume_start=100.0)
    flat_context = _candles(start, 80, 100.0, 0.0001, volume_start=100.0)

    analyzer = SetupAnalyzer(
        risk_reward=2.0,
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
    )

    strong_result = analyzer.analyze(
        symbol="BTCUSDT",
        trigger_candles=strong_trigger,
        context_candles=strong_context,
        higher_candles=strong_context,
        market=MarketMeta(
            symbol="BTCUSDT",
            mark_price=strong_trigger[-1].close,
            tick_size=0.1,
            step_size=0.001,
            funding_rate=-0.0002,
            open_interest=1_000_000.0,
            open_interest_change_pct=2.0,
        ),
        timeframe_plan=build_timeframe_plan(),
    )
    flat_result = analyzer.analyze(
        symbol="BTCUSDT",
        trigger_candles=flat_trigger,
        context_candles=flat_context,
        higher_candles=flat_context,
        market=MarketMeta(
            symbol="BTCUSDT",
            mark_price=flat_trigger[-1].close,
            tick_size=0.1,
            step_size=0.001,
            funding_rate=0.0,
            open_interest=100_000.0,
            open_interest_change_pct=0.0,
        ),
        timeframe_plan=build_timeframe_plan(),
    )

    assert strong_result.primary_setup.confidence >= flat_result.primary_setup.confidence
