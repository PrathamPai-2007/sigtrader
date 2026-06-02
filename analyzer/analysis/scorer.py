from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean

from analyzer.analysis.models import (
    AnalysisResult,
    Candle,
    ContributorDetail,
    ContributorDirection,
    EnhancedMetrics,
    EntryGeometry,
    EvidenceVector,
    IndicatorBundle,
    MarketMode,
    MarketMeta,
    MarketRegime,
    NormalizedSignals,
    QualityLabel,
    StrategyStyle,
    SwingPoints,
    TimeframePlan,
    TradeSetup,
)
from analyzer.analysis.indicators import (
    LiquiditySweep,
    sigmoid,
    gaussian_peak,
    rsi,
    macd,
    stochastic,
    bollinger_bands,
    rsi_divergence,
    volume_profile_strength,
    adx,
    volume_profile,
    compute_vwap_bands,
    compute_market_structure,
    compute_cumulative_delta,
    compute_adx_slope,
    detect_liquidity_sweeps,
    _swing_pivots,
)
from analyzer.analysis.regime import (
    classify_regime as _regime_classify,
    classify_regime_consensus,
)
from analyzer.config import AppConfig, load_app_config, StrategyConfig
from analyzer.analysis.scoring.utils import _clamp, _quantize, _quality_label, _quality_score_cap_from_confidence
from analyzer.analysis.geometry import (
    find_swing_points,
    select_best_stop,
    select_best_target,
    place_entry_stop_target,
    geometry_quality_score,
)
from analyzer.analysis.normalization import normalize_distension, normalize_signals, _funding_momentum, _oi_funding_biases
from analyzer.analysis.evidence import compute_graded_evidence, _regime_weight_profile, _regime_alignment, _regime_penalty
from analyzer.analysis.confidence import logistic_confidence, logistic_confidence_from_config
from analyzer.analysis.leverage import _leverage_suggestion, DrawdownAdjuster
from analyzer.analysis.enhanced_metrics import _calculate_enhanced_metrics
from analyzer.analysis.indicators_compute import (
    _compute_atr, _ema_value, compute_all_indicators,
    _atr, _structure, _structure_biases, _momentum, _trend_strength,
    _volume_surge_ratio, _buy_sell_pressure, _range_span,
    _volume_divergence_penalties, _confirmation_penalty, _classify_regime,
)
from analyzer.analysis.reversal_confluence import (
    _detect_early_reversal_signals, _round_number_proximity, _score_confluence,
)


def build_timeframe_plan(
    config: AppConfig | None = None,
    *,
    style: StrategyStyle = StrategyStyle.CONSERVATIVE,
    market_mode: MarketMode = MarketMode.INTRADAY,
    preset: str | None = None,
) -> TimeframePlan:
    cfg = config or load_app_config()
    timeframe = cfg.timeframe_for(market_mode)
    return TimeframePlan(
        profile_name=timeframe.profile_name,
        style=style,
        market_mode=market_mode,
        entry_timeframe=timeframe.entry_timeframe,
        context_timeframe=timeframe.context_timeframe,
        trigger_timeframe=timeframe.trigger_timeframe,
        higher_timeframe=timeframe.higher_timeframe,
        lookback_bars=timeframe.lookback_bars,
    )


@dataclass
class _Contribution:
    key: str
    label: str
    value: float
    impact: float
    direction: ContributorDirection
    summary: str


@dataclass
class _EvidenceSnapshot:
    agreement: int
    total: int
    summary: str
    raw_checks: dict[str, bool]


@dataclass
class _SideMetrics:
    side: str
    score: float
    confidence: float
    quality_label: QualityLabel
    quality_score: float
    leverage_suggestion: str
    entry: float
    stop: float
    target: float
    rationale: str
    top_positive_contributors: list[ContributorDetail]
    top_negative_contributors: list[ContributorDetail]
    components: dict[str, float]
    structure_points: dict[str, float]
    risk_reward_ratio: float
    stop_distance_pct: float
    target_distance_pct: float
    atr_multiple_to_stop: float
    atr_multiple_to_target: float
    invalidation_strength: float
    is_tradable: bool
    tradable_reasons: list[str]
    evidence_agreement: int
    evidence_total: int
    deliberation_summary: str
    stop_anchor: str = "atr_fallback"
    target_anchor: str = "atr_cap"
    signal_strengths: dict[str, float] = field(default_factory=dict)
    evidence_weighted_sum: float = 0.0
    logistic_input: float = 0.0
    entry_type: str = "market"


def _contributor_catalog(side: str, regime: MarketRegime) -> dict[str, tuple[str, str]]:
    side_label = "bullish" if side == "long" else "bearish"
    regime_side = "trend" if regime in {MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND} else regime.value.replace("_", " ")
    return {
        "higher_trend": ("Macro Trend", f"The higher timeframe trend is supporting the {side_label} case."),
        "context_trend": ("Context Trend", f"The context timeframe trend is helping this {side_label} setup."),
        "trigger_momentum": ("Trigger Momentum", f"The trigger timeframe momentum is supporting the {side_label} case."),
        "entry_momentum": ("Entry Momentum", f"The entry timeframe momentum is aligned with the {side_label} direction."),
        "momentum": ("Trigger Momentum", f"The trigger timeframe momentum is supporting the {side_label} case."),
        "trend": ("Context Trend", f"The context timeframe trend is helping this {side_label} setup."),
        "entry_confirmation": ("Entry Timing", "The lower timeframe timing is aligned with the setup."),
        "structure_position": ("Structure Position", "Price is at a favorable location within the local structure."),
        "structure": ("Structure", "Nearby market structure gives the setup room to work."),
        "volume_surge": ("Volume Surge", "Participation is expanding in the setup direction."),
        "buy_pressure": ("Order Flow Pressure", "Order flow is showing strong directional commitment."),
        "buy_sell_pressure": ("Order Flow Pressure", "Candle pressure is aligned with the selected side."),
        "oi_funding_bias": ("Funding / OI", "Funding and open-interest context are leaning in this direction."),
        "funding_momentum": ("Funding Momentum", "Funding rate trend is supporting this side — positioning is shifting in this direction."),
        "rsi_alignment": ("RSI Alignment", "RSI is in a favorable zone for momentum continuation."),
        "macd_alignment": ("MACD Alignment", "MACD histogram confirms the setup direction."),
        "bb_alignment": ("BB Position", "Bollinger Band position offers a good entry window."),
        "vwap_alignment": ("VWAP Alignment", "Price relative to VWAP supports the setup."),
        "market_structure_align": ("Structure Align", "Market structure (HH/HL or LH/LL) is aligned with the side."),
        "cumulative_delta_align": ("Delta Align", "Cumulative delta confirms the directional move."),
        "volume_poc_proximity": ("Volume POC", "Price is near a high-volume node — strong support/resistance."),
        "regime_alignment": ("Regime Fit", f"The current {regime_side} regime supports this setup."),
        "volume_divergence_penalty": (
            "Volume Divergence",
            "Momentum is not fully confirmed by pressure or participation.",
        ),
        "confirmation_penalty": (
            "Confirmation Gap",
            "The timeframes are not agreeing cleanly enough yet.",
        ),
        "target_ambition_penalty": (
            "Target Stretch",
            "The target is ambitious relative to current ATR.",
        ),
        "regime_penalty": (
            "Regime Friction",
            "The current regime reduces setup quality for this side.",
        ),
        "timeframe_alignment": (
            "Timeframe Alignment",
            "All timeframes agree on direction — high-conviction setup.",
        ),
        "reversal_signal_count": (
            "Reversal Warning",
            "Multiple signals suggest the move may be exhausting against this side."),
        "entry_confluence": (
            "Entry Confluence",
            "Multiple factors align at the entry level — stronger support/resistance."),
        "target_confluence": (
            "Target Confluence",
            "Multiple factors align at the target level — more likely to be reached."),
    }


def _to_contributor_details(items: list[_Contribution]) -> list[ContributorDetail]:
    return [
        ContributorDetail(
            key=item.key,
            label=item.label,
            value=round(item.value, 6),
            impact=round(item.impact, 6),
            direction=item.direction,
            summary=item.summary,
        )
        for item in items
    ]


def _timeframe_alignment_score(
    side: str,
    *,
    higher_trend: float,
    context_trend: float,
    trigger_momentum: float,
    entry_momentum: float,
) -> float:
    """Return an alignment score in [-1, 1].



    Positive = all timeframes agree with `side`.

    Negative = timeframes conflict with `side`.

    Each of the four signals is normalised to [-1, 1] before averaging.

    """
    direction = 1.0 if side == "long" else -1.0
    signals = [
        higher_trend * direction,
        context_trend * direction,
        trigger_momentum * direction,
        entry_momentum * direction,
    ]
    normalised = [_clamp(s * 20.0, -1.0, 1.0) for s in signals]
    return sum(normalised) / len(normalised)


def _apply_reversal_penalty(
    setup: _SideMetrics,
    signals: dict[str, bool],
    regime: MarketRegime,
    strategy: StrategyConfig,
) -> None:
    """Apply confidence/quality penalty when reversal signals fire.



    1 signal  ??? mild confidence reduction, no block.

    2 signals ??? meaningful reduction.

    3+        ??? strong reduction + trade blocked.

    Volatile chop: even 1 signal blocks.

    """
    fired = [k for k, v in signals.items() if v]
    count = len(fired)

    if count == 0:
        setup.components["reversal_signal_count"] = 0.0
        return

    s = strategy

    if count == 1:
        setup.confidence = _clamp(setup.confidence * s.reversal_1_confidence_mult, 0.0, 1.0)
    elif count == 2:
        setup.confidence = _clamp(setup.confidence * s.reversal_2_confidence_mult, 0.0, 1.0)
        setup.quality_score = _clamp(setup.quality_score - s.reversal_2_quality_deduction, 10.0, 95.0)
    else:
        setup.confidence = _clamp(setup.confidence * s.reversal_3_confidence_mult, 0.0, 1.0)
        setup.quality_score = _clamp(setup.quality_score - s.reversal_3_quality_deduction, 10.0, 95.0)
        setup.tradable_reasons.append(
            f"early reversal: {count} signals fired ({', '.join(fired)})"
        )

    if regime == MarketRegime.VOLATILE_CHOP and count >= 1:
        if not any("early reversal" in r for r in setup.tradable_reasons):
            setup.tradable_reasons.append(
                f"early reversal in volatile chop: {', '.join(fired)}"
            )

    setup.components["reversal_signal_count"] = float(count)
    setup.is_tradable = not setup.tradable_reasons


def _apply_confluence_boost(
    setup: _SideMetrics,
    confluence: dict[str, float],
    strategy: StrategyConfig,
) -> None:
    """Boost quality_score when multiple factors align at entry/target levels.



    Entry confluence (safer stop): up to +10 pts.

    Target confluence (more reachable): up to +5 pts.

    """
    entry_c = confluence["entry_confluence"]
    target_c = confluence["target_confluence"]

    s = strategy

    if entry_c >= 3:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_entry_3_boost, 10.0, 95.0)
    elif entry_c == 2:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_entry_2_boost, 10.0, 95.0)
    elif entry_c == 1:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_entry_1_boost, 10.0, 95.0)

    if target_c >= 2:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_target_2_boost, 10.0, 95.0)
    elif target_c == 1:
        setup.quality_score = _clamp(setup.quality_score + s.confluence_target_1_boost, 10.0, 95.0)

    setup.components["entry_confluence"] = entry_c
    setup.components["target_confluence"] = target_c
    setup.quality_label = _quality_label(setup.quality_score)


class SetupAnalyzer:
    def __init__(
        self,
        *,
        risk_reward: float = 2.0,
        style: StrategyStyle = StrategyStyle.CONSERVATIVE,
        market_mode: MarketMode = MarketMode.INTRADAY,
        filter_overrides: dict[str, float] | None = None,
        config: AppConfig | None = None,
        preset: str | None = None,
    ) -> None:
        self.risk_reward = max(risk_reward, 0.5)
        self.style = style
        self.market_mode = market_mode
        self._filter_overrides: dict[str, float] = filter_overrides or {}
        # Resolve config once at construction ??? avoids repeated imports and
        # lru_cache lookups inside hot paths (_mode_params / _trade_filter_params).
        self._preset_name: str | None = preset
        self._config = config or load_app_config()

    def get_execution_params(self, side: str) -> dict:
        """Return side-specific execution parameters from the active preset.

        Delegates to PresetConfig.execution_overrides for the given side.
        Returns an empty dict when no preset is active or no overrides are set.
        """
        if self._preset_name is None:
            return {}
        try:
            preset = self._config.get_preset(self._preset_name)
            overrides = preset.execution_overrides
            side_cfg = overrides.long if side == "long" else overrides.short
            result: dict = {}
            if side_cfg.tp_rr is not None:
                result["tp_rr"] = side_cfg.tp_rr
            if side_cfg.min_rr_ratio is not None:
                result["min_rr_ratio"] = side_cfg.min_rr_ratio
            if side_cfg.min_quality is not None:
                result["min_quality"] = side_cfg.min_quality
            return result
        except Exception:
            return {}
    def _mode_params(self) -> dict[str, float]:
        tuning = self._config.style_tuning(self.style)
        mode_settings = self._config.market_mode_settings(self.market_mode)
        return {
            "fallback_risk_reward": tuning.fallback_risk_reward,
            "target_cap_atr_mult": (
                mode_settings.target_cap_atr_mult
                if mode_settings.target_cap_atr_mult is not None
                else tuning.target_cap_atr_mult
            ),
            "ambition_penalty_start_atr": tuning.ambition_penalty_start_atr,
            "ambition_penalty_slope": tuning.ambition_penalty_slope,
            "atr_buffer_factor": tuning.atr_buffer_factor,
        }

    def _trade_filter_params(self) -> dict[str, float]:
        tuning = self._config.style_tuning(self.style)
        mode_settings = self._config.market_mode_settings(self.market_mode)

        # Priority: market_mode < style < filter_overrides (runtime)
        # market_mode contributes only fields it explicitly sets (non-None).
        # style (tuning) always has values for every field and overrides market_mode.
        # filter_overrides are runtime caller overrides and win above all.
        mode_params: dict[str, float] = {
            k: v for k, v in {
                "min_confidence": mode_settings.min_confidence,
                "max_stop_distance_pct": mode_settings.max_stop_distance_pct,
                "min_evidence_agreement": mode_settings.min_evidence_agreement,
                "min_evidence_edge": mode_settings.min_evidence_edge,
            }.items() if v is not None
        }

        style_params: dict[str, float] = {
            "min_confidence": tuning.min_confidence,
            "max_confidence": tuning.max_confidence,
            "min_quality": tuning.min_quality,
            "min_rr_ratio": tuning.min_rr_ratio,
            "max_stop_distance_pct": tuning.max_stop_distance_pct,
            "min_evidence_agreement": tuning.min_evidence_agreement,
            "min_evidence_edge": tuning.min_evidence_edge,
        }

        # style_params last ??? style always overrides market_mode
        params = {**mode_params, **style_params, **self._filter_overrides}
        return params

    def _trade_filter_reasons(
        self,
        *,
        confidence: float,
        quality_score: float,
        risk_reward_ratio: float,
        stop_distance_pct: float,
        regime: MarketRegime,
    ) -> list[str]:
        params = self._trade_filter_params()
        reasons: list[str] = []
        if confidence < params["min_confidence"]:
            reasons.append(f"confidence {confidence:.2f} is below {params['min_confidence']:.2f}")
        if confidence > params.get("max_confidence", 1.0):
            reasons.append(f"confidence {confidence:.2f} is above {params['max_confidence']:.2f}")
        if quality_score < params["min_quality"]:
            reasons.append(f"quality {quality_score:.1f} is below {params['min_quality']:.1f}")
        if risk_reward_ratio < params["min_rr_ratio"]:
            reasons.append(f"R:R {risk_reward_ratio:.2f} is below {params['min_rr_ratio']:.2f}")
        if stop_distance_pct > params["max_stop_distance_pct"]:
            reasons.append(
                f"stop distance {stop_distance_pct:.2f}% is above {params['max_stop_distance_pct']:.2f}%"
            )
        return reasons

    def _collect_evidence(
        self,
        side: str,
        *,
        higher_trend: float,
        context_trend: float,
        trigger_momentum: float,
        entry_momentum: float,
        trigger_pressure: float,
        entry_pressure: float,
        trigger_volume_surge: float,
        entry_volume_surge: float,
        regime: MarketRegime,
    ) -> _EvidenceSnapshot:
        direction = 1.0 if side == "long" else -1.0
        _strategy = self._config.strategy
        checks = {
            "macro": higher_trend * direction > 0,
            "context": context_trend * direction > 0,
            "trigger": trigger_momentum * direction > 0,
            "entry": entry_momentum * direction > 0,
            "pressure": ((trigger_pressure + entry_pressure) / 2.0) * direction > _strategy.pressure_threshold,
            "volume": max(trigger_volume_surge, entry_volume_surge) >= _strategy.volume_surge_threshold,
            "regime": _regime_alignment(regime, side) > 0.3,
        }
        agreement = sum(1 for ok in checks.values() if ok)
        total = len(checks)
        aligned = ", ".join(label for label, ok in checks.items() if ok) or "none"
        blocked = ", ".join(label for label, ok in checks.items() if not ok) or "none"
        summary = f"{agreement}/{total} aligned | supporting: {aligned} | missing: {blocked}"
        return _EvidenceSnapshot(agreement=agreement, total=total, summary=summary, raw_checks=checks)

    def _apply_deliberation(
        self,
        setup: _SideMetrics,
        evidence: _EvidenceSnapshot,
        opposing_evidence: _EvidenceSnapshot,
        regime: MarketRegime,
    ) -> None:
        params = self._trade_filter_params()
        s = self._config.strategy
        d = s.deliberation
        setup.evidence_agreement = evidence.agreement
        setup.evidence_total = evidence.total
        setup.deliberation_summary = evidence.summary

        # EVIDENCE FILTER (regime-aware) — Phase 7: soft penalties instead of hard rejects
        # Legacy binary evidence checks now reduce confidence/quality rather than blocking.
        if regime == MarketRegime.RANGE:
            if evidence.agreement < d.range_agreement_threshold:
                setup.confidence = _clamp(setup.confidence * d.range_confidence_mult, 0.0, 1.0)
                setup.quality_score = _clamp(setup.quality_score - d.range_quality_penalty, 0.0, 100.0)
        else:
            if evidence.agreement < params["min_evidence_agreement"]:
                setup.confidence = _clamp(setup.confidence * d.default_agreement_mult, 0.0, 1.0)
                setup.quality_score = _clamp(setup.quality_score - d.default_quality_penalty, 0.0, 100.0)
        if (evidence.agreement - opposing_evidence.agreement) < params["min_evidence_edge"]:
            setup.confidence = _clamp(setup.confidence * d.edge_mult, 0.0, 1.0)
        if not evidence.raw_checks.get("macro", False) and not evidence.raw_checks.get("context", False):
            setup.confidence = _clamp(setup.confidence * d.macro_context_mult, 0.0, 1.0)
            setup.quality_score = _clamp(setup.quality_score - d.macro_context_quality_penalty, 0.0, 100.0)
        if not evidence.raw_checks.get("trigger", False) and not evidence.raw_checks.get("entry", False):
            setup.confidence = _clamp(setup.confidence * d.trigger_entry_mult, 0.0, 1.0)

        if hasattr(s, "allowed_regimes") and regime.value not in s.allowed_regimes:
            setup.tradable_reasons.append(f"regime {regime.value} is globally disabled via StrategyConfig")
        if hasattr(s, "enable_longs") and setup.side == "long" and not s.enable_longs:
            setup.tradable_reasons.append("long setups are globally disabled via StrategyConfig")
        if hasattr(s, "enable_shorts") and setup.side == "short" and not s.enable_shorts:
            setup.tradable_reasons.append("short setups are globally disabled via StrategyConfig")

        if regime == MarketRegime.VOLATILE_CHOP:
            # Cap quality in volatile chop (structural, not a rejection)
            setup.quality_score = min(setup.quality_score, d.chop_quality_cap)
            setup.quality_label = _quality_label(setup.quality_score)
            # Soft penalty for insufficient evidence in chop
            chop_min = max(params["min_evidence_agreement"], d.chop_agreement_threshold)
            if evidence.agreement < chop_min:
                setup.confidence = _clamp(setup.confidence * d.chop_confidence_mult, 0.0, 1.0)
                setup.quality_score = _clamp(setup.quality_score - d.chop_quality_penalty, 0.0, 100.0)
        if regime == MarketRegime.RANGE:
            # Cap quality in range (structural, not a rejection)
            setup.quality_score = min(setup.quality_score, d.range_quality_cap)
            setup.quality_label = _quality_label(setup.quality_score)
            # Soft penalty for insufficient evidence in range
            range_min = d.range_agreement_threshold
            if evidence.agreement < range_min:
                setup.confidence = _clamp(setup.confidence * d.range_confidence_mult, 0.0, 1.0)
                setup.quality_score = _clamp(setup.quality_score - d.range_quality_penalty, 0.0, 100.0)
            # Soft penalty for low quality in range (was hard reject at threshold)
            if setup.quality_score < d.range_low_quality_threshold:
                setup.quality_score = _clamp(setup.quality_score - d.range_low_quality_penalty, 0.0, 100.0)

        setup.is_tradable = not setup.tradable_reasons
        setup.leverage_suggestion = _leverage_suggestion(
            stop_distance_pct=setup.stop_distance_pct,
            quality_label=setup.quality_label,
            confidence=setup.confidence,
            regime=regime,
            risk_reward_ratio=setup.risk_reward_ratio,
            is_tradable=setup.is_tradable,
        )
    # ── Side-specific post-processors ────────────────────────────────────────
    def _analyze_short_sniper(self, setup: "_SideMetrics") -> None:
        """Post-process a short setup with sniper logic.

        Keeps the existing exhaustion penalty (already applied in _build_side).
        Marks entry_type as 'market' — shorts enter at market on confirmation.
        This method is intentionally minimal; all short-specific logic lives in
        _build_side and the soft-penalty block.
        """
        setup.entry_type = "market"
    def _analyze_long_baiter(
        self,
        setup: "_SideMetrics",
        bundle: "IndicatorBundle",
        atr: float,
        px: float,
        trigger_candles: "list[Candle]",
    ) -> None:
        """Post-process a long setup with pullback-baiter logic.

        Applied after _build_side. Enforces:
        - Candle colour gate: green candle = rejection, red candle = bonus
        - FOMO gate: RSI or BB overextended = penalty
        - Near Value check: within configurable ATR dist of VWAP, VAH, or EMA20.
          If NOT near value: apply penalty (never zeros confidence).
        - entry_type = 'limit'
        """
        b = self._config.strategy.baiter
        # Candle colour gate — HARD REQUIREMENT: must be a red candle (pullback entry only)
        if trigger_candles:
            last = trigger_candles[-1]
            if last.close >= last.open:  # green candle — reject immediately
                setup.is_tradable = False
                if "green candle — not a pullback entry" not in setup.tradable_reasons:
                    setup.tradable_reasons.append("green candle — not a pullback entry")
                return
            else:  # red candle — price pulling back, add small bonus
                setup.confidence = _clamp(setup.confidence + b.red_candle_bonus, 0.0, 1.0)
        # FOMO gate (stricter than the inline check in _build_side)
        if bundle.rsi_14 > b.rsi_fomo_threshold or bundle.bb_position > b.bb_fomo_threshold:
            setup.confidence = _clamp(setup.confidence - b.fomo_confidence_penalty, 0.0, 1.0)
        # Near Value check — soft penalty when price is extended from all value anchors.
        _vwap_dist = abs(px - bundle.vwap)
        _vah_dist = abs(px - bundle.vah)
        _ema20_dist = abs(px - bundle.ema_20) if bundle.ema_20 > 0 else 0.0
        _near_value = (
            _vwap_dist < atr * b.near_value_atr_dist
            or _vah_dist < atr * b.near_value_atr_dist
            or _ema20_dist < atr * b.ema20_near_dist
        )
        if not _near_value:
            setup.confidence = _clamp(setup.confidence - b.out_of_value_penalty, 0.0, 1.0)
            if "price outside value area" not in setup.tradable_reasons:
                setup.tradable_reasons.append(
                    f"price not near value (> {b.near_value_atr_dist} ATR from VWAP/VAH and > {b.ema20_near_dist} ATR from EMA20) — confidence penalised"
                )
        # Mark as limit entry
        setup.entry_type = "limit"
        # Hard confidence gate — lock the door on low-conviction setups
        if setup.confidence < b.min_confidence_threshold:
            setup.is_tradable = False
            if f"confidence below {b.min_confidence_threshold} threshold" not in setup.tradable_reasons:
                setup.tradable_reasons.append(f"confidence below {b.min_confidence_threshold} threshold")

    def analyze(
        self,
        *,
        symbol: str,
        trigger_candles: list[Candle],
        context_candles: list[Candle],
        market: MarketMeta,
        timeframe_plan: TimeframePlan,
        entry_candles: list[Candle] | None = None,
        higher_candles: list[Candle] | None = None,
        long_log=None,
    ) -> AnalysisResult:
        entry_candles = entry_candles or trigger_candles
        higher_candles = higher_candles or context_candles
        if min(len(entry_candles), len(trigger_candles), len(context_candles), len(higher_candles)) < 30:
            raise ValueError("Insufficient candle history: need at least 30 bars per timeframe")

        # ── Step 1: Compute All Indicators Once ──────────────────────────────
        bundle = compute_all_indicators(
            entry_candles,
            trigger_candles,
            context_candles,
            higher_candles,
            market,
            config=self._config,
        )
        atr = bundle.trigger_atr
        px = market.mark_price
        # Scale structure window to timeframe
        _tf = timeframe_plan.trigger_timeframe
        _tf_mins = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}.get(_tf, 15)
        _structure_window = 30 if _tf_mins <= 5 else (40 if _tf_mins <= 30 else 60)
        support, resistance = _structure(trigger_candles, window=_structure_window)
        # ── Step 2: Market Regime Classification ─────────────────────────────
        regime_result = classify_regime_consensus(
            context_candles=context_candles,
            higher_candles=higher_candles,
            trigger_candles=trigger_candles,
            trigger_atr=atr,
            px=px,
        )
        regime = regime_result.regime
        # ── Step 3: Build Setups (Long & Short) ──────────────────────────────
        # Legacy penalties derived from bundle for the build_side call
        vp = volume_profile_strength(trigger_candles, px)
        # Use bundle-derived values for legacy paths to ensure consistency
        long_div_penalty, short_div_penalty = _volume_divergence_penalties(
            bundle.trigger_momentum,
            bundle.cumulative_delta,
            bundle.trigger_volume_surge,
        )
        long_setup = self._build_side(
            side="long",
            px=px,
            support=support,
            resistance=resistance,
            atr=atr,
            divergence_penalty=long_div_penalty,
            confirmation_penalty=0.0, # Handled inside build_side from bundle
            regime=regime,
            regime_penalty=_regime_penalty(regime, "long", self._config),
            tick=market.tick_size,
            volume_profile=vp,
            alignment_score=0.0,      # Handled inside build_side from bundle
            market_meta=market,
            bundle=bundle,
            trigger_candles=trigger_candles,
            _weight_overrides={"volume_poc_proximity": 1.5, "market_structure_align": 1.5},
        )
        short_setup = self._build_side(
            side="short",
            px=px,
            support=support,
            resistance=resistance,
            atr=atr,
            divergence_penalty=short_div_penalty,
            confirmation_penalty=0.0,
            regime=regime,
            regime_penalty=_regime_penalty(regime, "short", self._config),
            tick=market.tick_size,
            volume_profile=vp,
            alignment_score=0.0,
            market_meta=market,
            bundle=bundle,
            trigger_candles=trigger_candles,
        )

        # ── Step 4: Evidence & Deliberation ──────────────────────────────────
        # Evidence Snapshot (derived from bundle metrics for the deliberation step)
        direction_long = 1.0
        direction_short = -1.0
        long_evidence = self._collect_evidence(
            "long",
            higher_trend=bundle.higher_trend,
            context_trend=bundle.context_trend,
            trigger_momentum=bundle.trigger_momentum,
            entry_momentum=bundle.entry_momentum,
            trigger_pressure=bundle.cumulative_delta,
            entry_pressure=bundle.cumulative_delta,
            trigger_volume_surge=bundle.trigger_volume_surge,
            entry_volume_surge=bundle.entry_volume_surge,
            regime=regime,
        )
        short_evidence = self._collect_evidence(
            "short",
            higher_trend=bundle.higher_trend,
            context_trend=bundle.context_trend,
            trigger_momentum=bundle.trigger_momentum,
            entry_momentum=bundle.entry_momentum,
            trigger_pressure=bundle.cumulative_delta,
            entry_pressure=bundle.cumulative_delta,
            trigger_volume_surge=bundle.trigger_volume_surge,
            entry_volume_surge=bundle.entry_volume_surge,
            regime=regime,
        )

        self._apply_deliberation(long_setup, long_evidence, short_evidence, regime)
        self._apply_deliberation(short_setup, short_evidence, long_evidence, regime)

        # ── Step 5: Specialized Analysis Gates ────────────────────────────────
        self._analyze_short_sniper(short_setup)
        self._analyze_long_baiter(long_setup, bundle, atr, px, trigger_candles)

        # ── Enhanced LONG entry filters stage (Enhanced Safety) ───────────────
        if self._config.strategy.long_entry_filters.enable_enhanced_filters:
            from analyzer.analysis.long_entry_filters import (
                apply_long_entry_filters,
                adjust_long_confidence_threshold,
            )

            filter_result = apply_long_entry_filters(
                higher_candles=higher_candles,
                trigger_candles=trigger_candles,
                market_structure=bundle.market_structure,
                volume_surge=bundle.trigger_volume_surge,
                swing_highs=bundle.swing_highs,
                current_price=px,
                regime=regime,
                trigger_momentum=bundle.trigger_momentum,
                config=self._config,
            )

            # Apply confidence penalties and adjustments to long_setup
            long_setup.confidence = adjust_long_confidence_threshold(
                base_confidence=long_setup.confidence,
                regime=regime,
                filter_result=filter_result,
                config=self._config,
            )

            # If filters reject the long entry, mark as untradable and attach reasons
            if not filter_result.allow_long:
                long_setup.is_tradable = False
                reason_msg = f"Enhanced Safety Reject: {filter_result.overall_reason}"
                if reason_msg not in long_setup.tradable_reasons:
                    long_setup.tradable_reasons.append(reason_msg)

        # ── Step 6: Final Selection & Telemetry ───────────────────────────────
        tradable_setups = [setup for setup in (long_setup, short_setup) if setup.is_tradable]
        if tradable_setups:
            primary = max(
                tradable_setups,
                key=lambda item: (
                    item.evidence_agreement, 
                    item.score, 
                    item.quality_score, 
                    item.confidence,
                    item.components.get("oi_funding_bias", 0.0)
                ),
            )
        else:
            primary = max(
                [long_setup, short_setup],
                key=lambda item: (
                    item.evidence_agreement, 
                    item.score, 
                    item.quality_score, 
                    item.confidence,
                    item.components.get("oi_funding_bias", 0.0)
                ),
            )

        secondary = short_setup if primary.side == "long" else long_setup
        warnings: list[str] = []
        # Check for missing data elements and add telemetry warning
        if (
            market.funding_rate is None 
            or not market.funding_rate_history 
            or market.open_interest_change_pct is None
        ):
            warnings.append(
                "Partial Evidence: Funding rate or Open Interest data is missing from market metadata. "
                "OI/Funding scoring components will default to neutral."
            )
        if not tradable_setups:
            warnings.append("No setup passed the hard trade filters and deliberation checks; treat this symbol as a no-trade.")
        if not primary.is_tradable and primary.tradable_reasons:
            warnings.append("Primary setup was filtered out: " + "; ".join(primary.tradable_reasons))
        if primary.quality_label == QualityLabel.LOW:
            warnings.append("Primary setup quality is low; treat the signal as exploratory.")
        if primary.invalidation_strength < 0.35:
            warnings.append("Primary invalidation level is weak relative to current ATR and structure.")

        # Calculate enhanced metrics (derived from bundle for efficiency)
        enhanced_metrics = _calculate_enhanced_metrics(
            entry_candles=entry_candles,
            trigger_candles=trigger_candles,
            context_candles=context_candles,
            market_meta=market,
        )

        # Apply final boosts and penalties
        self._apply_enhanced_metrics_boost(long_setup, enhanced_metrics, "long")
        self._apply_enhanced_metrics_boost(short_setup, enhanced_metrics, "short")

        long_reversal = _detect_early_reversal_signals(
            "long",
            trigger_candles=trigger_candles,
            entry_candles=entry_candles,
            support=support,
            resistance=resistance,
            rsi_divergence_type=enhanced_metrics.rsi_divergence_type,
            macd_histogram=enhanced_metrics.macd_histogram,
        )
        short_reversal = _detect_early_reversal_signals(
            "short",
            trigger_candles=trigger_candles,
            entry_candles=entry_candles,
            support=support,
            resistance=resistance,
            rsi_divergence_type=enhanced_metrics.rsi_divergence_type,
            macd_histogram=enhanced_metrics.macd_histogram,
        )
        _apply_reversal_penalty(long_setup, long_reversal, regime, self._config.strategy)
        _apply_reversal_penalty(short_setup, short_reversal, regime, self._config.strategy)

        long_confluence = _score_confluence(
            "long",
            entry=long_setup.entry,
            target=long_setup.target,
            support=support,
            resistance=resistance,
            atr=atr,
            volume_profile=vp,
        )
        short_confluence = _score_confluence(
            "short",
            entry=short_setup.entry,
            target=short_setup.target,
            support=support,
            resistance=resistance,
            atr=atr,
            volume_profile=vp,
        )
        _apply_confluence_boost(long_setup, long_confluence, self._config.strategy)
        _apply_confluence_boost(short_setup, short_confluence, self._config.strategy)

        # Final quality label sync
        long_setup.quality_label = _quality_label(long_setup.quality_score)
        short_setup.quality_label = _quality_label(short_setup.quality_score)

        from analyzer.analysis.decay import compute_ttl
        from datetime import UTC, datetime as _dt
        _now = _dt.now(UTC)
        _ttl = compute_ttl(timeframe_plan.trigger_timeframe)
        _valid_until = _now + _ttl

        return AnalysisResult(
            primary_setup=TradeSetup(
                side=primary.side,
                entry_price=primary.entry,
                target_price=primary.target,
                stop_loss=primary.stop,
                leverage_suggestion=primary.leverage_suggestion,
                confidence=primary.confidence,
                quality_label=primary.quality_label,
                quality_score=primary.quality_score,
                rationale=primary.rationale,
                top_positive_contributors=primary.top_positive_contributors,
                top_negative_contributors=primary.top_negative_contributors,
                score_components=primary.components,
                structure_points=primary.structure_points,
                risk_reward_ratio=primary.risk_reward_ratio,
                stop_distance_pct=primary.stop_distance_pct,
                target_distance_pct=primary.target_distance_pct,
                atr_multiple_to_stop=primary.atr_multiple_to_stop,
                atr_multiple_to_target=primary.atr_multiple_to_target,
                invalidation_strength=primary.invalidation_strength,
                is_tradable=primary.is_tradable,
                tradable_reasons=primary.tradable_reasons,
                evidence_agreement=primary.evidence_agreement,
                evidence_total=primary.evidence_total,
                deliberation_summary=primary.deliberation_summary,
                valid_until=_valid_until,
                ttl_seconds=_ttl.total_seconds(),
                stop_anchor=primary.stop_anchor,
                target_anchor=primary.target_anchor,
                regime_state=regime.value,
                signal_strengths=primary.signal_strengths,
                evidence_weighted_sum=primary.evidence_weighted_sum,
                logistic_input=primary.logistic_input,
                entry_type=primary.entry_type,
            ),
            secondary_context=TradeSetup(
                side=secondary.side,
                entry_price=secondary.entry,
                target_price=secondary.target,
                stop_loss=secondary.stop,
                leverage_suggestion=secondary.leverage_suggestion,
                confidence=secondary.confidence,
                quality_label=secondary.quality_label,
                quality_score=secondary.quality_score,
                rationale=secondary.rationale,
                top_positive_contributors=secondary.top_positive_contributors,
                top_negative_contributors=secondary.top_negative_contributors,
                score_components=secondary.components,
                structure_points=secondary.structure_points,
                risk_reward_ratio=secondary.risk_reward_ratio,
                stop_distance_pct=secondary.stop_distance_pct,
                target_distance_pct=secondary.target_distance_pct,
                atr_multiple_to_stop=secondary.atr_multiple_to_stop,
                atr_multiple_to_target=secondary.atr_multiple_to_target,
                invalidation_strength=secondary.invalidation_strength,
                is_tradable=secondary.is_tradable,
                tradable_reasons=secondary.tradable_reasons,
                evidence_agreement=secondary.evidence_agreement,
                evidence_total=secondary.evidence_total,
                deliberation_summary=secondary.deliberation_summary,
                valid_until=_valid_until,
                ttl_seconds=_ttl.total_seconds(),
                stop_anchor=secondary.stop_anchor,
                target_anchor=secondary.target_anchor,
                regime_state=regime.value,
                signal_strengths=secondary.signal_strengths,
                evidence_weighted_sum=secondary.evidence_weighted_sum,
                logistic_input=secondary.logistic_input,
                entry_type=secondary.entry_type,
            ),
            timeframe_plan=timeframe_plan,
            market_snapshot_meta=market.model_copy(update={"symbol": symbol}),
            market_regime=regime,
            regime_confidence=regime_result.confidence,
            enhanced_metrics=enhanced_metrics,
            warnings=warnings,
        )


    def _apply_enhanced_metrics_boost(
        self,
        setup: _SideMetrics,
        enhanced_metrics: EnhancedMetrics,
        side: str,
    ) -> None:
        """Apply enhanced metrics adjustments to setup scoring."""
        s = self._config.strategy

        # 1. RSI-based confidence adjustment ??? directionally aware
        rsi_val = enhanced_metrics.rsi_14
        if side == "long":
            if 40 < rsi_val < 65:  # healthy momentum zone for longs
                setup.confidence *= s.rsi_neutral_confidence_mult
            elif rsi_val >= 75:  # overbought ??? bad long entry
                setup.confidence *= s.rsi_extreme_confidence_mult
            elif rsi_val <= 25:  # oversold ??? good long entry
                setup.confidence *= s.rsi_mild_extreme_confidence_mult
            elif rsi_val >= 65:  # mildly overbought
                setup.confidence *= s.rsi_mild_extreme_confidence_mult
        else:  # short
            if 35 < rsi_val < 60:  # healthy momentum zone for shorts
                setup.confidence *= s.rsi_neutral_confidence_mult
            elif rsi_val <= 25:  # oversold ??? bad short entry
                setup.confidence *= s.rsi_extreme_confidence_mult
            elif rsi_val >= 75:  # overbought ??? good short entry
                setup.confidence *= s.rsi_mild_extreme_confidence_mult
            elif rsi_val <= 35:  # mildly oversold
                setup.confidence *= s.rsi_mild_extreme_confidence_mult

        # 2. MACD momentum confirmation
        if enhanced_metrics.macd_histogram > 0 and side == "long":
            setup.confidence += s.macd_confirm_confidence_delta
        elif enhanced_metrics.macd_histogram < 0 and side == "short":
            setup.confidence += s.macd_confirm_confidence_delta

        # 3. Bollinger Band position for entry timing
        bb_pos = enhanced_metrics.bollinger_position
        if s.bb_mid_low < bb_pos < s.bb_mid_high:
            setup.quality_score += s.bb_mid_quality_boost
        elif bb_pos < s.bb_extreme_low or bb_pos > s.bb_extreme_high:
            setup.quality_score -= s.bb_extreme_quality_penalty

        # 4. Order book imbalance for entry strength
        if abs(enhanced_metrics.order_book_imbalance) > s.ob_imbalance_threshold:
            setup.confidence += s.ob_imbalance_confidence_delta

        # 5. Volatility-based stop adjustment
        vol_rank = enhanced_metrics.volatility_rank
        if vol_rank > s.vol_rank_high_threshold:
            setup.stop_distance_pct *= s.vol_rank_high_stop_mult
        elif vol_rank < s.vol_rank_low_threshold:
            setup.stop_distance_pct *= s.vol_rank_low_stop_mult

        # Clamp to valid ranges
        setup.confidence = max(0.0, min(1.0, setup.confidence))
        setup.quality_score = max(10.0, min(95.0, setup.quality_score))

    def _build_side(
        self,
        *,
        side: str,
        px: float,
        support: float,
        resistance: float,
        atr: float,
        divergence_penalty: float,
        confirmation_penalty: float,
        regime: MarketRegime,
        regime_penalty: float,
        tick: float | None,
        volume_profile: dict | None = None,
        alignment_score: float = 0.0,
        # New signal pipeline inputs
        entry_candles: list[Candle] | None = None,
        trigger_candles: list[Candle] | None = None,
        context_candles: list[Candle] | None = None,
        higher_candles: list[Candle] | None = None,
        market_meta: MarketMeta | None = None,
        bundle: IndicatorBundle | None = None,
        _weight_overrides: dict[str, float] | None = None,
    ) -> _SideMetrics:
        strategy = self._config.strategy
        weights, penalty_multiplier, confidence_ceiling = _regime_weight_profile(regime, side, self._config)
        mode_params = self._mode_params()
        atr_buffer = max(atr * mode_params["atr_buffer_factor"], px * strategy.min_risk_px_factor)

        # ?????? Indicator Collection ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        _bundle = bundle
        _use_new_pipeline = (
            _bundle is not None
            or (
                entry_candles is not None
                and trigger_candles is not None
                and context_candles is not None
                and higher_candles is not None
                and market_meta is not None
            )
        )

        if _use_new_pipeline and _bundle is None:
            _bundle = compute_all_indicators(
                entry_candles,  # type: ignore[arg-type]
                trigger_candles,  # type: ignore[arg-type]
                context_candles,  # type: ignore[arg-type]
                higher_candles,  # type: ignore[arg-type]
                market_meta,  # type: ignore[arg-type]
                indicator_params=strategy.indicator_params,
                signal_transforms=strategy.signal_transforms,
                config=self._config,
            )

        # ?????? Signal Normalization & Evidence ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        _signals: NormalizedSignals | None = None
        _evidence: EvidenceVector | None = None
        _new_confidence: float | None = None
        _geometry: EntryGeometry | None = None
        _logistic_input = 0.0

        if _use_new_pipeline and _bundle is not None and market_meta is not None:
            _signals = normalize_signals(_bundle, market_meta, side, regime, config=self._config, signal_transforms=strategy.signal_transforms)
            _cfg_weights, _penalty_multiplier, _confidence_ceiling = _regime_weight_profile(regime, side, self._config)

            _NORMALIZED_SIGNAL_FIELDS = {
                "higher_trend", "context_trend", "trigger_momentum", "entry_momentum",
                "volume_surge", "buy_pressure", "oi_funding_bias", "funding_momentum",
                "structure_position", "rsi_alignment", "macd_alignment", "bb_alignment",
                "vwap_alignment", "market_structure_align", "cumulative_delta_align",
                "volume_poc_proximity",
            }

            _cfg_new_keys = {k: v for k, v in _cfg_weights.items() if k in _NORMALIZED_SIGNAL_FIELDS}
            _cfg_new_sum = sum(_cfg_new_keys.values())

            if abs(_cfg_new_sum - 1.0) < 0.1 and len(_cfg_new_keys) >= 12:
                _signal_weights = _cfg_new_keys
            else:
                _signal_weights = dict(strategy.default_signal_weights) or {
                    k: 1.0 / len(_NORMALIZED_SIGNAL_FIELDS)
                    for k in _NORMALIZED_SIGNAL_FIELDS
                }

            if _weight_overrides:
                for _k, _mult in _weight_overrides.items():
                    if _k in _signal_weights:
                        _signal_weights[_k] = _signal_weights[_k] * _mult
                _w_total = sum(_signal_weights.values()) or 1.0
                _signal_weights = {k: v / _w_total for k, v in _signal_weights.items()}

            _evidence = compute_graded_evidence(_signals, regime, side, _signal_weights, regime_gate_thresholds=getattr(strategy, "regime_gate_thresholds", None), config=self._config)

            # Exhaustion penalty
            import dataclasses as _dc
            _signal_values = [getattr(_signals, f.name) for f in _dc.fields(_signals)]
            _overheated_count = sum(1 for v in _signal_values if v > 0.85)

            # Pull exhaustion settings from config
            _ex = strategy.exhaustion_settings
            if side == "short":
                _ex_penalty = _ex.short_penalty if _overheated_count >= _ex.short_count_limit else 0.0
            else:
                _ex_penalty = _ex.long_penalty if _overheated_count >= _ex.long_count_limit else 0.0

            _exhaustion_adjusted_sum = _evidence.weighted_sum - _ex_penalty
            _new_confidence = logistic_confidence_from_config(_exhaustion_adjusted_sum, regime, side, config=self._config)
            _new_confidence = min(_new_confidence, _confidence_ceiling)

            _strategy_logistic = getattr(strategy, "logistic_params", {}) or {}
            _logistic_params = (
                _strategy_logistic.get(f"{regime.value}_{side}")
                or _strategy_logistic.get(regime.value)
                or _strategy_logistic.get("default")
            )
            if isinstance(_logistic_params, dict):
                _steepness = _logistic_params.get("steepness", 6.0)
            elif _logistic_params is not None:
                _steepness = getattr(_logistic_params, "steepness", 6.0)
            else:
                _steepness = 6.0
            _logistic_input = _steepness * _evidence.weighted_sum

            # Recompute confirmation and alignment from bundle if available
            if _bundle is not None:
                confirmation_penalty = _confirmation_penalty(
                    side,
                    higher_trend=_bundle.higher_trend,
                    context_trend=_bundle.context_trend,
                    trigger_momentum=_bundle.trigger_momentum,
                    entry_momentum=_bundle.entry_momentum,
                    trigger_pressure=_bundle.cumulative_delta,
                    entry_pressure=_bundle.cumulative_delta,
                    trigger_volume_surge=_bundle.trigger_volume_surge,
                    entry_volume_surge=_bundle.entry_volume_surge,
                )
                alignment_score = _timeframe_alignment_score(
                    side,
                    higher_trend=_bundle.higher_trend,
                    context_trend=_bundle.context_trend,
                    trigger_momentum=_bundle.trigger_momentum,
                    entry_momentum=_bundle.entry_momentum,
                )

        # ?????? Geometry ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        if _use_new_pipeline and _bundle is not None and trigger_candles is not None:
            _swings = find_swing_points(trigger_candles)
            _symbol = market_meta.symbol if market_meta is not None else ""
            _geometry = place_entry_stop_target(
                side,
                px,
                _swings,
                _bundle,
                atr,
                regime,
                self.style,
                {**mode_params, "min_rr_ratio": self._trade_filter_params().get("min_rr_ratio", 1.5)},
                tick,
                _symbol,
            )
            entry = _geometry.entry
            structure_stop = _geometry.stop
            target = _geometry.target
            risk = _geometry.risk
            reward = _geometry.reward
            structure_reference = support if side == "long" else resistance
            rationale = f"{side.capitalize()} setup identified via multi-timeframe confluence and regime-specific geometry."
        else:
            # Minimal legacy geometry
            entry = px
            if side == "long":
                structure_stop = support - atr_buffer
                risk = max(entry - structure_stop, px * strategy.min_risk_px_factor)
                structure_target = resistance
                fallback_target = entry + (risk * self.risk_reward)
                raw_target = structure_target if structure_target > entry + (risk * 0.8) else fallback_target
                max_target_distance = max((atr * mode_params["target_cap_atr_mult"]), px * 0.002)
                target = min(raw_target, entry + max_target_distance)
                structure_reference = support
            else:
                structure_stop = resistance + atr_buffer
                risk = max(structure_stop - entry, px * strategy.min_risk_px_factor)
                structure_target = support
                fallback_target = entry - (risk * self.risk_reward)
                raw_target = structure_target if structure_target < entry - (risk * 0.8) else fallback_target
                max_target_distance = max((atr * mode_params["target_cap_atr_mult"]), px * 0.002)
                target = max(raw_target, entry - max_target_distance)
                structure_reference = resistance
            reward = abs(target - entry)
            rationale = f"{side.capitalize()} setup (legacy geometry)."

        # Volume profile stop adjustment (legacy only)
        if not _use_new_pipeline and volume_profile is not None:
            if side == "long":
                val = volume_profile.get("val", 0.0)
                if val > support and abs(px - val) < atr * strategy.volume_profile_atr_proximity:
                    structure_stop = val - atr_buffer
                    structure_reference = val
                    risk = max(entry - structure_stop, px * strategy.min_risk_px_factor)
            else:
                vah = volume_profile.get("vah", 0.0)
                if vah > 0 and vah < resistance and abs(px - vah) < atr * strategy.volume_profile_atr_proximity:
                    structure_stop = vah + atr_buffer
                    structure_reference = vah
                    risk = max(structure_stop - entry, px * strategy.min_risk_px_factor)

        reward = abs(target - entry)
        target_distance_atr = reward / max(atr, 1e-9)
        ambition_excess = max(target_distance_atr - mode_params["ambition_penalty_start_atr"], 0.0)
        ambition_penalty = ambition_excess * mode_params["ambition_penalty_slope"]

        # ?????? Telemetry / Contributors ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        catalog = _contributor_catalog(side, regime)
        positive_contribs: list[_Contribution] = []
        weighted_positive_total = 0.0
        components: dict[str, float] = {}

        if _use_new_pipeline and _signals is not None:
            import dataclasses as _dc
            signal_dict = {f.name: getattr(_signals, f.name) for f in _dc.fields(_signals)}
            for key, value in signal_dict.items():
                weight = _signal_weights.get(key, 0.0)
                if weight == 0:
                    continue
                impact = value * weight
                weighted_positive_total += impact
                components[key] = impact
                if key in catalog:
                    label, summary = catalog[key]
                    positive_contribs.append(
                        _Contribution(
                            key=key,
                            label=label,
                            value=value,
                            impact=impact,
                            direction=ContributorDirection.POSITIVE,
                            summary=summary,
                        )
                    )
        else:
            # Derive raw_positive from bundle if available for legacy path consistency
            raw_positive = {}
            if _bundle is not None:
                direction = 1.0 if side == "long" else -1.0
                raw_positive = {
                    "higher_trend": max(_bundle.higher_trend * direction, 0.0),
                    "momentum": max(_bundle.trigger_momentum * direction, 0.0),
                    "trend": max(_bundle.context_trend * direction, 0.0),
                    "entry_confirmation": max((_bundle.entry_momentum * direction + max(_bundle.cumulative_delta * direction, 0.0)) / 2.0, 0.0),
                    "structure": 0.5, # placeholder for structure bias if not provided
                    "volume_surge": max(_bundle.trigger_volume_surge - 1.0, 0.0),
                    "buy_sell_pressure": max(_bundle.cumulative_delta * direction, 0.0),
                    "regime_alignment": _regime_alignment(regime, side, self._config),
                }

            if volume_profile is not None:
                if volume_profile.get("near_poc"):
                    raw_positive["volume_poc_proximity"] = strategy.volume_poc_near_score
                elif side == "long" and volume_profile.get("near_val"):
                    raw_positive["volume_poc_proximity"] = strategy.volume_poc_side_score
                elif side == "short" and volume_profile.get("near_vah"):
                    raw_positive["volume_poc_proximity"] = strategy.volume_poc_side_score
                else:
                    raw_positive["volume_poc_proximity"] = 0.0

            for key, value in raw_positive.items():
                impact = value * weights.get(key, 1.0)
                weighted_positive_total += impact
                components[key] = impact
                if key in catalog:
                    label, summary = catalog[key]
                    positive_contribs.append(
                        _Contribution(
                            key=key,
                            label=label,
                            value=value,
                            impact=impact,
                            direction=ContributorDirection.POSITIVE,
                            summary=summary,
                        )
                    )


        negative_raw = {
            "volume_divergence_penalty": divergence_penalty,
            "confirmation_penalty": confirmation_penalty,
            "target_ambition_penalty": ambition_penalty,
            "regime_penalty": regime_penalty,
        }
        negative_contribs: list[_Contribution] = []
        weighted_negative_total = 0.0
        for key, value in negative_raw.items():
            if key == "volume_divergence_penalty":
                impact = value * penalty_multiplier
            elif key == "confirmation_penalty":
                impact = value * strategy.confirmation_penalty_weight
            elif key == "target_ambition_penalty":
                impact = value * strategy.target_ambition_penalty_weight
            else:
                impact = value * strategy.regime_penalty_weight
            weighted_negative_total += impact
            components[key] = impact
            label, summary = catalog[key]
            negative_contribs.append(
                _Contribution(
                    key=key,
                    label=label,
                    value=value,
                    impact=impact,
                    direction=ContributorDirection.NEGATIVE,
                    summary=summary,
                )
            )

        score = max(weighted_positive_total - weighted_negative_total, 0.0)

        # ?????? Confidence computation ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # When the new signal pipeline was used, confidence comes from the
        # logistic sigmoid over the evidence weighted sum (already capped by
        # confidence_ceiling above). Otherwise fall back to the legacy formula.
        if _new_confidence is not None:
            confidence = _new_confidence
        else:
            base_confidence = score / strategy.score_confidence_divisor
            if regime == MarketRegime.VOLATILE_CHOP:
                if _signals is not None:
                    _src = {
                        "higher_trend": _signals.higher_trend,
                        "trend": _signals.context_trend,
                        "momentum": _signals.trigger_momentum,
                        "entry_confirmation": _signals.entry_momentum,
                    }
                else:
                    _src = raw_positive
                strong_agreement = sum(
                    1
                    for key in ("higher_trend", "trend", "momentum", "entry_confirmation")
                    if _src.get(key, 0.0) > strategy.volatile_chop_strong_signal_threshold
                )
                confidence_ceiling = min(
                    confidence_ceiling + (strong_agreement * strategy.volatile_chop_boost_per_signal),
                    strategy.volatile_chop_confidence_ceiling_cap,
                )
            confidence = _clamp(base_confidence, 0.0, confidence_ceiling)

            # Timeframe alignment multiplier (legacy path only)
            if alignment_score >= strategy.alignment_strong_threshold:
                confidence = _clamp(confidence * strategy.alignment_strong_confidence_mult, 0.0, confidence_ceiling)
            elif alignment_score >= strategy.alignment_neutral_high:
                pass  # mostly agree ??? no change
            elif alignment_score >= strategy.alignment_neutral_low:
                confidence = _clamp(confidence * strategy.alignment_weak_confidence_mult, 0.0, confidence_ceiling)
            else:
                confidence = _clamp(confidence * strategy.alignment_opposed_confidence_mult, 0.0, confidence_ceiling)

        if _use_new_pipeline and _geometry is not None:
            rr_ratio = _geometry.rr_ratio
            stop_distance_pct = _geometry.stop_distance_pct
            target_distance_pct = _geometry.target_distance_pct
            atr_multiple_to_stop = _geometry.atr_multiple_to_stop
            atr_multiple_to_target = _geometry.atr_multiple_to_target
            invalidation_strength = _geometry.invalidation_strength
        else:
            rr_ratio = reward / max(risk, 1e-9)
            stop_distance_pct = (risk / max(entry, 1e-9)) * 100.0
            target_distance_pct = (reward / max(entry, 1e-9)) * 100.0
            atr_multiple_to_stop = risk / max(atr, 1e-9)
            atr_multiple_to_target = reward / max(atr, 1e-9)
            structure_gap = abs(entry - structure_reference)
            range_context = abs(resistance - support) / max(atr, 1e-9)
            formula_penalty = strategy.invalidation_formula_penalty if structure_gap <= (px * strategy.invalidation_proximity_threshold_pct) else 0.0
            invalidation_strength = _clamp(
                ((structure_gap / max(atr, 1e-9)) * strategy.invalidation_structure_gap_weight)
                + ((atr_buffer / max(atr, 1e-9)) * strategy.invalidation_atr_buffer_weight)
                + (min(range_context, strategy.invalidation_range_context_cap) / strategy.invalidation_range_context_cap * strategy.invalidation_range_context_weight)
                - formula_penalty,
                0.0,
                1.0,
            )

        raw_quality_score = _clamp(
            strategy.quality_base_score
            + min(weighted_positive_total, strategy.quality_positive_cap) * strategy.quality_positive_weight
            + min(rr_ratio, strategy.quality_rr_cap) * strategy.quality_rr_weight
            + invalidation_strength * strategy.quality_invalidation_weight
            - min(weighted_negative_total, strategy.quality_negative_cap) * strategy.quality_negative_weight,
            strategy.quality_min_score,
            strategy.quality_max_score,
        )
        if _use_new_pipeline and _geometry is not None:
            # Build confluence_dict from the score_confluence result (computed later in
            # analyze(), but we need it here ??? use a placeholder that will be overridden
            # by _apply_confluence_boost in analyze()).
            _confluence_dict: dict[str, float] = {"entry_confluence": 0.0, "target_confluence": 0.0}
            geo_quality_score, geo_quality_label = geometry_quality_score(
                _geometry, regime, _confluence_dict, 0.5, config=self._config
            )
            quality_score = geo_quality_score
            quality_label = geo_quality_label
        else:
            quality_score = min(raw_quality_score, _quality_score_cap_from_confidence(confidence))
            quality_label = _quality_label(quality_score)
        # ?????? Primary trade filter ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # Centralised gate evaluated BEFORE _SideMetrics is constructed.
        # Downstream reversal penalties may append further reasons and
        # re-evaluate is_tradable via _apply_reversal_penalty.
        _filter_params = self._trade_filter_params()
        # --- VWAP distension penalty ---
        # Penalise setups where price has chased too far from VWAP.
        # A setup 2.5+ ATRs from VWAP loses 25% of its confidence.
        if _bundle is not None:
            _distension = normalize_distension(px, _bundle.vwap, atr)
            confidence = _clamp(confidence * (1.0 - (_distension * 0.25)), 0.0, 1.0)
        reasons: list[str] = []
        # ── Phase 7: Soft penalties replace early hard rejections ─────────────
        # Checks 1-5 and 7 are now soft: they reduce confidence/quality instead
        # of immediately rejecting. Hard gates at the end enforce absolute floors.
        # --- Confidence soft penalty (regime-aware) ---
        if regime == MarketRegime.RANGE:
            if confidence < 0.55:
                confidence = _clamp(confidence * 0.85, 0.0, 1.0)
        else:
            if confidence < _filter_params["min_confidence"]:
                confidence = _clamp(confidence * 0.85, 0.0, 1.0)
        # --- Quality soft penalty (regime-aware) ---
        if regime == MarketRegime.RANGE:
            if quality_score < 48:
                quality_score = _clamp(quality_score - 5.0, 0.0, 100.0)
        else:
            if quality_score < _filter_params["min_quality"]:
                quality_score = _clamp(quality_score - 5.0, 0.0, 100.0)
        # --- Evidence agreement soft penalty (new pipeline only) ---
        if _evidence is not None:
            _min_ev = _filter_params.get("min_evidence_agreement", 0)
            if _evidence.signal_count_above_threshold < _min_ev:
                confidence = _clamp(confidence * 0.90, 0.0, 1.0)
        # --- Regime filter (hard: disabled regimes are always rejected) ---
        if hasattr(strategy, "allowed_regimes") and regime.value not in strategy.allowed_regimes:
            reasons.append(f"regime {regime.value} disabled")
        # --- Trend alignment soft penalty (new pipeline only) ---
        if _bundle is not None:
            direction = 1 if side == "long" else -1
            macro_ok = _bundle.higher_trend * direction > 0
            context_ok = _bundle.context_trend * direction > 0
            if not (macro_ok or context_ok):
                confidence = _clamp(confidence * 0.90, 0.0, 1.0)
                quality_score = _clamp(quality_score - 3.0, 0.0, 100.0)
        # --- Long-trend short protection (mandatory confidence penalty) ---
        if regime == MarketRegime.BULLISH_TREND and side == "short":
            confidence = _clamp(confidence - 0.15, 0.0, 1.0)
        # --- FOMO penalty: prevent buying overextended longs ---
        # RSI > 70 or BB position > 0.8 signals a "top of wick" entry in a bullish run.
        if regime == MarketRegime.BULLISH_TREND and side == "long" and _bundle is not None:
            if _bundle.rsi_14 > 70 or _bundle.bb_position > 0.8:
                confidence = _clamp(confidence - 0.25, 0.0, 1.0)
        # --- Max confidence cap (hard: never allow overconfident signals) ---
        if _filter_params.get("max_confidence", 1.0) and confidence > _filter_params.get("max_confidence", 1.0):
            reasons.append(f"confidence {confidence:.2f} is above {_filter_params['max_confidence']:.2f}")
        # --- Stop distance hard gate (structural safety) ---
        if stop_distance_pct > _filter_params["max_stop_distance_pct"]:
            reasons.append(
                f"stop distance {stop_distance_pct:.2f}% is above {_filter_params['max_stop_distance_pct']:.2f}%"
            )
        # ── Final absolute safety gates ───────────────────────────────────────
        # These are the ONLY hard rejection floors. All other checks above are
        # soft penalties that reduce scores without blocking the setup entirely.
        if confidence < 0.60:
            reasons.append(f"confidence {confidence:.2f} below absolute floor (0.60)")
        if rr_ratio < 0.8:
            reasons.append(f"R:R {rr_ratio:.2f} below absolute floor (0.8)")
        if quality_score < 25.0:
            reasons.append(f"quality {quality_score:.1f} below absolute floor (25)")

        tradable_reasons = reasons
        is_tradable = len(reasons) == 0
        leverage_suggestion = _leverage_suggestion(
            stop_distance_pct=stop_distance_pct,
            quality_label=quality_label,
            confidence=confidence,
            regime=regime,
            risk_reward_ratio=rr_ratio,
            is_tradable=is_tradable,
        )

        positive_contribs.sort(key=lambda item: item.impact, reverse=True)
        negative_contribs.sort(key=lambda item: item.impact, reverse=True)
        top_positive = _to_contributor_details(positive_contribs[:3])
        top_negative = _to_contributor_details([item for item in negative_contribs if item.impact > 0][:3])

        components["risk_reward_ratio"] = rr_ratio
        components["stop_distance_pct"] = stop_distance_pct
        components["target_distance_pct"] = target_distance_pct
        components["atr_multiple_to_stop"] = atr_multiple_to_stop
        components["atr_multiple_to_target"] = atr_multiple_to_target
        components["invalidation_strength"] = invalidation_strength
        components["score"] = score
        components["timeframe_alignment"] = round(alignment_score, 4)

        # ?????? Evidence agreement from new pipeline ???????????????????????????????????????????????????????????????????????????????????????
        # When the new signal pipeline was used, populate evidence_agreement from
        # EvidenceVector.signal_count_above_threshold and evidence_total = 16
        # (total number of NormalizedSignals fields).
        if _evidence is not None:
            _evidence_agreement = _evidence.signal_count_above_threshold
            _evidence_total = 16  # total NormalizedSignals fields
        else:
            _evidence_agreement = 0
            _evidence_total = 0

        _signal_strengths = {
            key: round(value, 6)
            for key, value in (asdict(_signals).items() if _signals is not None else [])
        }

        # Pressure value for structure_points telemetry
        if _signals is not None:
            _pressure_val = getattr(_signals, "buy_pressure", 0.0)
        elif _bundle is not None:
            _dir = 1.0 if side == "long" else -1.0
            _pressure_val = _bundle.cumulative_delta * _dir
        else:
            _pressure_val = 0.0
        return _SideMetrics(
            side=side,
            score=score,
            confidence=round(confidence, 4),
            quality_label=quality_label,
            quality_score=round(quality_score, 1),
            leverage_suggestion=leverage_suggestion,
            entry=_quantize(entry, tick),
            stop=_quantize(structure_stop, tick),
            target=_quantize(target, tick),
            rationale=rationale,
            top_positive_contributors=top_positive,
            top_negative_contributors=top_negative,
            components={key: round(value, 6) for key, value in components.items()},
            structure_points={
                "support": round(support, 6),
                "resistance": round(resistance, 6),
                "atr": round(atr, 6),
                "pressure": round(_pressure_val, 6),
            },
            risk_reward_ratio=round(rr_ratio, 4),
            stop_distance_pct=round(stop_distance_pct, 4),
            target_distance_pct=round(target_distance_pct, 4),
            atr_multiple_to_stop=round(atr_multiple_to_stop, 4),
            atr_multiple_to_target=round(atr_multiple_to_target, 4),
            invalidation_strength=round(invalidation_strength, 4),
            is_tradable=is_tradable,
            tradable_reasons=tradable_reasons,
            evidence_agreement=_evidence_agreement,
            evidence_total=_evidence_total,
            deliberation_summary="",
            stop_anchor=_geometry.stop_anchor if _geometry is not None else "atr_fallback",
            target_anchor=_geometry.target_anchor if _geometry is not None else "atr_cap",
            signal_strengths=_signal_strengths,
            evidence_weighted_sum=round(_evidence.weighted_sum, 6) if _evidence is not None else 0.0,
            logistic_input=round(_logistic_input, 6),
        )


# ?????? Drawdown Recovery Adjuster ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

# DrawdownAdjuster and _leverage_suggestion live in analyzer.analysis.leverage
# and are re-exported from here for backward compatibility via the import above.
