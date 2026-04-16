"""
Configuration loader.

This module contains NO strategy logic. All numeric constants
live in futures_analyzer.config.json. Pydantic model defaults are only
used for schema-safe bootstrap values when keys are missing from JSON.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from futures_analyzer.analysis.models import MarketMode, StrategyStyle
from futures_analyzer.logging import get_logger

log = get_logger(__name__)


# ── Preset configuration ─────────────────────────────────────────────────────

class PresetConfig(BaseModel):
    """Simplified preset configuration."""
    profile_name: str
    entry_timeframe: str
    trigger_timeframe: str
    context_timeframe: str
    higher_timeframe: str
    lookback_bars: int
    target_cap_atr_mult: float
    min_confidence: float
    max_confidence: float
    max_stop_distance_pct: float
    min_evidence_agreement: int
    min_evidence_edge: int
    candidate_quote_asset: str
    candidate_min_quote_volume: float
    candidate_pool_limit: int
    candidate_kline_interval: str
    candidate_kline_lookback: int
    candidate_return_weight: float
    candidate_range_weight: float
    candidate_consistency_weight: float
    fallback_risk_reward: float
    ambition_penalty_start_atr: float
    ambition_penalty_slope: float
    min_quality: float
    min_rr_ratio: float

    @field_validator("min_confidence", "max_confidence")
    @classmethod
    def _unit_interval(cls, v: float, info: object) -> float:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> "PresetConfig":
        if self.min_confidence > self.max_confidence:
            raise ValueError(
                f"min_confidence ({self.min_confidence}) > max_confidence ({self.max_confidence})"
            )
        return self


# ── Cache ─────────────────────────────────────────────────────────────────────

class CacheConfig(BaseModel):
    market_meta_ttl_seconds: float = 12.0
    realtime_klines_ttl_seconds: float = 20.0
    historical_klines_ttl_seconds: float = 3600.0
    replay_lookback_cap: int = 96


# ── Strategy sub-configs ──────────────────────────────────────────────────────

class LogisticParams(BaseModel):
    steepness: float = 6.0
    midpoint: float = 0.5

    @field_validator("steepness")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("steepness must be > 0")
        return v

    @field_validator("midpoint")
    @classmethod
    def _open_unit(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError("midpoint must be in (0, 1)")
        return v


class SignalTransforms(BaseModel):
    """Sigmoid/tanh scaling factors — all come from config.json."""
    higher_trend_scale: float = 10.0
    context_trend_scale: float = 10.0
    trigger_momentum_scale: float = 15.0
    entry_momentum_scale: float = 15.0
    buy_pressure_scale: float = 5.0
    funding_momentum_scale: float = 5.0
    macd_histogram_scale: float = 100.0
    vwap_dev_scale: float = 20.0
    volume_surge_cap: float = 2.0
    oi_funding_norm: float = 0.5
    poc_proximity_pct_cap: float = 2.0
    trend_tanh_scale: float = 100.0
    momentum_tanh_scale: float = 50.0


class IndicatorParams(BaseModel):
    """Indicator computation periods — all come from config.json."""
    ema_fast_period: int = 20
    ema_slow_period: int = 50
    roc_period: int = 14
    volume_window: int = 20
    atr_period: int = 14
    pivot_n: int = 2


class RegimeClassifierConfig(BaseModel):
    """Regime classifier thresholds and consensus weights — all from config.json."""
    adx_trend_threshold: float = 25.0
    adx_weak_threshold: float = 20.0
    atr_volatile_percentile: float = 85.0
    atr_trend_max_percentile: float = 75.0
    adx_slope_exhaustion: float = -2.0
    adx_slope_breakout: float = 3.0
    consensus_weight_higher: float = 0.50
    consensus_weight_context: float = 0.35
    consensus_weight_trigger: float = 0.15
    adx_conf_base: float = 0.5
    adx_conf_range: float = 0.45
    adx_conf_adx_base: float = 20.0
    adx_conf_adx_range: float = 60.0
    di_bias_margin: float = 1.1


class GeometryQualityConfig(BaseModel):
    """Geometry quality scoring parameters — all from config.json."""
    rr_weight: float = 30.0
    rr_cap: float = 3.0
    stop_atr_ideal_min: float = 0.8
    stop_atr_ideal_max: float = 2.0
    stop_atr_bonus: float = 10.0
    stop_atr_penalty: float = 8.0
    anchor_swing_bonus: float = 12.0
    anchor_vwap_bonus: float = 8.0
    anchor_vp_bonus: float = 6.0
    anchor_atr_penalty: float = 5.0
    entry_confluence_bonus: float = 3.0
    target_confluence_bonus: float = 2.0
    base_score: float = 20.0
    min_score: float = 10.0
    max_score: float = 95.0


class MomentumExpansionConfig(BaseModel):
    """Momentum expansion filter configuration."""
    min_momentum_threshold: float = 0.05  # Temporarily lowered from 0.15 to 0.05
    min_volume_surge: float = 1.1  # Temporarily lowered from 1.3 to 1.1


class DelayedEntryConfig(BaseModel):
    """Delayed entry confirmation configuration."""
    min_confirmation_candles: int = 1
    max_confirmation_candles: int = 3
    min_price_continuation_pct: float = 0.05  # Temporarily lowered from 0.1 to 0.05
    min_momentum_sustain: float = 0.05  # Temporarily lowered from 0.1 to 0.05


class TrendDominanceConfig(BaseModel):
    """Trend dominance filter configuration."""
    min_ema_separation_pct: float = 1.5
    min_adx_threshold: float = 25.0
    min_di_margin: float = 5.0


class PullbackTrapConfig(BaseModel):
    """Pullback trap detection configuration."""
    short_term_momentum_threshold: float = 0.02
    higher_tf_bearish_threshold: float = -0.01
    min_momentum_divergence: float = 0.03


class StructureConfirmationConfig(BaseModel):
    """Structure confirmation requirements configuration."""
    min_volume_expansion: float = 1.5
    higher_high_buffer_pct: float = 0.001
    min_confirmations_required: int = 2


class ConfidenceAdjustmentsConfig(BaseModel):
    """LONG confidence adjustment penalties configuration."""
    base_long_penalty: float = 0.15
    weak_trend_penalty: float = 0.20
    momentum_divergence_penalty: float = 0.15
    no_confirmation_penalty: float = 0.25
    bearish_regime_penalty: float = 0.25
    volatile_chop_penalty: float = 0.20


class LongEntryFiltersConfig(BaseModel):
    """Enhanced LONG entry filters configuration."""
    enable_enhanced_filters: bool = True
    momentum_expansion: MomentumExpansionConfig = Field(default_factory=MomentumExpansionConfig)
    delayed_entry: DelayedEntryConfig = Field(default_factory=DelayedEntryConfig)
    structure_confirmation: StructureConfirmationConfig = Field(default_factory=StructureConfirmationConfig)
    trend_dominance: TrendDominanceConfig = Field(default_factory=TrendDominanceConfig)
    pullback_trap: PullbackTrapConfig = Field(default_factory=PullbackTrapConfig)
    confidence_adjustments: ConfidenceAdjustmentsConfig = Field(default_factory=ConfidenceAdjustmentsConfig)


class ExecutionSideOverride(BaseModel):
    """Execution constraints for one trade side. All fields optional — absent
    means fall back to the shared preset value."""
    min_rr_ratio: float | None = None
    min_quality: float | None = None
    tp_rr: float | None = None


class ExecutionOverridesConfig(BaseModel):
    """Side-specific execution overrides. Absent keys fall back to preset."""
    long: ExecutionSideOverride = Field(default_factory=ExecutionSideOverride)
    short: ExecutionSideOverride = Field(default_factory=ExecutionSideOverride)


# ── Strategy config ───────────────────────────────────────────────────────────

class StrategyConfig(BaseModel):
    """
    All strategy parameters.  Every field has a safe Pydantic default so the
    system can start even if config.json omits a key, but the canonical values
    are always in config.json — not here.
    """
    # ── Filters ──────────────────────────────────────────────────────────────
    pressure_threshold: float = 0.05
    volume_surge_threshold: float = 1.05
    enable_longs: bool = True
    enable_shorts: bool = True
    allowed_regimes: list[str] = Field(
        default_factory=lambda: ["range", "bearish_trend", "bullish_trend", "breakout", "exhaustion"]
    )

    # ── Signal pipeline ───────────────────────────────────────────────────────
    signal_transforms: SignalTransforms = Field(default_factory=SignalTransforms)
    indicator_params: IndicatorParams = Field(default_factory=IndicatorParams)
    default_signal_weights: dict[str, float] = Field(default_factory=dict)
    regime_gate_thresholds: dict[str, float] = Field(default_factory=dict)
    logistic_params: dict[str, LogisticParams] = Field(default_factory=dict)

    # ── Regime ────────────────────────────────────────────────────────────────
    regime_classifier: RegimeClassifierConfig = Field(default_factory=RegimeClassifierConfig)
    regime_weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    regime_penalty_multiplier: dict[str, float] = Field(default_factory=dict)
    regime_confidence_ceiling: dict[str, float] = Field(default_factory=dict)
    regime_alignment: dict[str, float] = Field(default_factory=dict)
    regime_penalty: dict[str, float] = Field(default_factory=dict)

    # ── Geometry / quality ────────────────────────────────────────────────────
    geometry_quality: GeometryQualityConfig = Field(default_factory=GeometryQualityConfig)
    leverage_caps: dict[str, int] = Field(default_factory=dict)
    leverage_floors: dict[str, int] = Field(default_factory=dict)
    
    # ── LONG Entry Filters ────────────────────────────────────────────────────
    long_entry_filters: LongEntryFiltersConfig = Field(default_factory=LongEntryFiltersConfig)

    # ── Side-specific execution overrides ────────────────────────────────────
    execution_overrides: ExecutionOverridesConfig = Field(default_factory=ExecutionOverridesConfig)
    confidence_quality_caps: dict[str, float] = Field(
        default_factory=lambda: {"below_0_45": 54.9, "below_0_70": 74.9, "default": 95.0}
    )

    # ── Scoring weights ───────────────────────────────────────────────────────
    score_confidence_divisor: float = 18.0
    quality_base_score: float = 32.0
    quality_positive_cap: float = 10.0
    quality_positive_weight: float = 4.8
    quality_rr_weight: float = 6.0
    quality_rr_cap: float = 3.0
    quality_invalidation_weight: float = 10.0
    quality_negative_cap: float = 10.0
    quality_negative_weight: float = 5.2
    quality_min_score: float = 10.0
    quality_max_score: float = 95.0

    # ── Confirmation / penalty ────────────────────────────────────────────────
    confirmation_low_cutoff: int = 2
    confirmation_mid_cutoff: int = 4
    confirmation_low_penalty: float = 1.3
    confirmation_mid_penalty: float = 0.55
    dual_trend_penalty: float = 0.35
    dual_momentum_penalty: float = 0.35
    dual_pressure_penalty: float = 0.2
    confirmation_penalty_weight: float = 3.0
    target_ambition_penalty_weight: float = 10.0
    regime_penalty_weight: float = 4.0

    # ── Reversal penalties ────────────────────────────────────────────────────
    reversal_1_confidence_mult: float = 0.90
    reversal_2_confidence_mult: float = 0.75
    reversal_3_confidence_mult: float = 0.60
    reversal_2_quality_deduction: float = 8.0
    reversal_3_quality_deduction: float = 15.0

    # ── Confluence boosts ─────────────────────────────────────────────────────
    confluence_entry_3_boost: float = 10.0
    confluence_entry_2_boost: float = 6.0
    confluence_entry_1_boost: float = 2.0
    confluence_target_2_boost: float = 5.0
    confluence_target_1_boost: float = 2.0

    # ── Enhanced metrics ──────────────────────────────────────────────────────
    rsi_neutral_confidence_mult: float = 1.05
    rsi_extreme_confidence_mult: float = 0.85
    rsi_mild_extreme_confidence_mult: float = 0.90
    macd_confirm_confidence_delta: float = 0.03
    bb_mid_quality_boost: float = 5.0
    bb_extreme_quality_penalty: float = 3.0
    bb_mid_low: float = 0.35
    bb_mid_high: float = 0.65
    bb_extreme_low: float = 0.2
    bb_extreme_high: float = 0.8
    ob_imbalance_threshold: float = 0.3
    ob_imbalance_confidence_delta: float = 0.02
    vol_rank_high_threshold: float = 75.0
    vol_rank_low_threshold: float = 25.0
    vol_rank_high_stop_mult: float = 1.15
    vol_rank_low_stop_mult: float = 0.85

    # ── Timeframe alignment ───────────────────────────────────────────────────
    alignment_strong_threshold: float = 0.6
    alignment_neutral_high: float = 0.2
    alignment_neutral_low: float = -0.2
    alignment_strong_confidence_mult: float = 1.10
    alignment_weak_confidence_mult: float = 0.90
    alignment_opposed_confidence_mult: float = 0.75

    # ── Invalidation strength ─────────────────────────────────────────────────
    invalidation_structure_gap_weight: float = 0.35
    invalidation_atr_buffer_weight: float = 0.45
    invalidation_range_context_weight: float = 0.2
    invalidation_range_context_cap: float = 3.0
    invalidation_formula_penalty: float = 0.2
    invalidation_proximity_threshold_pct: float = 0.0015

    # ── Entry / stop geometry ─────────────────────────────────────────────────
    atr_buffer_factor: float = 0.5
    min_risk_px_factor: float = 0.001
    volume_poc_near_score: float = 0.8
    volume_poc_side_score: float = 0.4
    volume_profile_atr_proximity: float = 1.5

    # ── Leverage suggestion ───────────────────────────────────────────────────
    leverage_stop_wide: float = 3.0
    leverage_stop_mid: float = 2.0
    leverage_stop_narrow: float = 1.2
    leverage_stop_tight: float = 0.8
    leverage_confidence_high: float = 0.8
    leverage_confidence_low: float = 0.6
    leverage_rr_good: float = 2.0
    leverage_rr_poor: float = 1.5
    leverage_volatile_chop_cap: int = 4

    # ── OI / funding ──────────────────────────────────────────────────────────
    funding_rate_scale: float = 10_000.0
    funding_rate_divisor: float = 50.0
    oi_funding_partial_weight: float = 0.6
    funding_momentum_weight: float = 0.5
    funding_momentum_slope_scale: float = 20_000.0

    # ── Volatile chop ─────────────────────────────────────────────────────────
    volatile_chop_boost_per_signal: float = 0.04
    volatile_chop_confidence_ceiling_cap: float = 0.68
    volatile_chop_strong_signal_threshold: float = 0.15

    @model_validator(mode="after")
    def _validate_regime_weights(self) -> "StrategyConfig":
        new_style_keys = {"breakout", "exhaustion", "transition"}
        for key, weights in self.regime_weights.items():
            if key not in new_style_keys:
                continue
            total = sum(weights.values())
            if total > 0.0 and abs(total - 1.0) > 1e-6:
                log.warning(
                    "config.regime_weights_sum",
                    key=key,
                    total=total,
                    detail=f"regime_weights['{key}'] sums to {total:.6f}, expected 1.0",
                )
        return self


# ── Slippage / drawdown / portfolio ──────────────────────────────────────────

class SlippageConfig(BaseModel):
    volatility_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"low": 0.7, "normal": 1.0, "high": 1.4, "extreme": 2.0}
    )
    default_model: str = "moderate"
    min_viable_rr: float = 1.0


class DrawdownConfig(BaseModel):
    lookback: int = 10
    mild_threshold_pct: float = 5.0
    moderate_threshold_pct: float = 15.0
    severe_threshold_pct: float = 30.0
    mild_consecutive_losses: int = 2
    moderate_consecutive_losses: int = 3
    severe_consecutive_losses: int = 5

    @field_validator("lookback", "mild_consecutive_losses", "moderate_consecutive_losses", "severe_consecutive_losses")
    @classmethod
    def _positive_int(cls, v: int, info: object) -> int:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if v <= 0:
            raise ValueError(f"{name} must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> "DrawdownConfig":
        if not (0 < self.mild_threshold_pct < self.moderate_threshold_pct < self.severe_threshold_pct):
            raise ValueError("Drawdown thresholds must be strictly increasing: mild < moderate < severe")
        if not (self.mild_consecutive_losses < self.moderate_consecutive_losses < self.severe_consecutive_losses):
            raise ValueError("Consecutive loss thresholds must be strictly increasing")
        return self


class PortfolioConfig(BaseModel):
    max_position_pct: float = 0.20
    max_risk_per_trade_pct: float = 0.02
    max_cluster_risk_pct: float = 0.10
    max_total_risk_pct: float = 0.06
    kelly_fraction: float = 0.25
    min_history_for_kelly: int = 10

    @field_validator("max_position_pct", "max_risk_per_trade_pct", "max_cluster_risk_pct", "max_total_risk_pct", "kelly_fraction")
    @classmethod
    def _unit_interval_pct(cls, v: float, info: object) -> float:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if not (0.0 < v <= 1.0):
            raise ValueError(f"{name} must be in (0, 1], got {v}")
        return v

    @field_validator("min_history_for_kelly")
    @classmethod
    def _positive_history(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"min_history_for_kelly must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _risk_consistency(self) -> "PortfolioConfig":
        if self.max_risk_per_trade_pct > self.max_total_risk_pct:
            raise ValueError(
                f"max_risk_per_trade_pct ({self.max_risk_per_trade_pct}) cannot exceed "
                f"max_total_risk_pct ({self.max_total_risk_pct})"
            )
        return self


# ── Root config ───────────────────────────────────────────────────────────────

class PresetConfig(BaseModel):
    """Simplified preset configuration."""
    profile_name: str
    entry_timeframe: str
    trigger_timeframe: str
    context_timeframe: str
    higher_timeframe: str
    lookback_bars: int
    target_cap_atr_mult: float
    min_confidence: float
    max_confidence: float
    max_stop_distance_pct: float
    min_evidence_agreement: int
    min_evidence_edge: int
    candidate_quote_asset: str
    candidate_min_quote_volume: float
    candidate_pool_limit: int
    candidate_kline_interval: str
    candidate_kline_lookback: int
    candidate_return_weight: float
    candidate_range_weight: float
    candidate_consistency_weight: float
    fallback_risk_reward: float
    ambition_penalty_start_atr: float
    ambition_penalty_slope: float
    min_quality: float
    min_rr_ratio: float


class TimeframeConfig(BaseModel):
    profile_name: str = "auto_core"
    entry_timeframe: str = "5m"
    trigger_timeframe: str = "15m"
    context_timeframe: str = "1h"
    higher_timeframe: str = "4h"
    lookback_bars: int = 600


class MarketModeTuning(BaseModel):
    target_cap_atr_mult: float | None = None
    min_confidence: float | None = None
    max_stop_distance_pct: float | None = None
    min_evidence_agreement: int | None = None
    min_evidence_edge: int | None = None
    candidate_quote_asset: str = "USDT"
    candidate_min_quote_volume: float = 25_000_000.0
    candidate_pool_limit: int = 40
    candidate_kline_interval: str = "1d"
    candidate_kline_lookback: int = 14
    candidate_return_weight: float = 0.5
    candidate_range_weight: float = 0.3
    candidate_consistency_weight: float = 0.2

    @field_validator("min_confidence")
    @classmethod
    def _unit_interval_opt(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {v}")
        return v

    @field_validator("max_stop_distance_pct")
    @classmethod
    def _positive_pct_opt(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"max_stop_distance_pct must be positive, got {v}")
        return v

    @field_validator("candidate_return_weight", "candidate_range_weight", "candidate_consistency_weight")
    @classmethod
    def _unit_weight(cls, v: float, info: object) -> float:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "MarketModeTuning":
        total = (
            self.candidate_return_weight
            + self.candidate_range_weight
            + self.candidate_consistency_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"candidate weights must sum to 1.0, got {total:.6f}")
        return self


class StyleTuning(BaseModel):
    fallback_risk_reward: float
    target_cap_atr_mult: float
    ambition_penalty_start_atr: float
    ambition_penalty_slope: float
    min_confidence: float
    max_confidence: float = 1.0
    min_quality: float
    min_rr_ratio: float
    max_stop_distance_pct: float
    min_evidence_agreement: int
    min_evidence_edge: int
    atr_buffer_factor: float = 0.3

    @field_validator("min_confidence", "max_confidence")
    @classmethod
    def _unit_interval(cls, v: float, info: object) -> float:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> "StyleTuning":
        if self.min_confidence > self.max_confidence:
            raise ValueError(
                f"min_confidence ({self.min_confidence}) > max_confidence ({self.max_confidence})"
            )
        if self.min_rr_ratio > self.fallback_risk_reward:
            log.warning(
                "config.style_tuning_inconsistency",
                detail=(
                    f"min_rr_ratio ({self.min_rr_ratio}) > fallback_risk_reward "
                    f"({self.fallback_risk_reward}); may filter all setups"
                ),
            )
        return self

    @field_validator("min_quality")
    @classmethod
    def _quality_range(cls, v: float) -> float:
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"min_quality must be in [0, 100], got {v}")
        return v

    @field_validator("min_rr_ratio", "fallback_risk_reward")
    @classmethod
    def _positive_ratio(cls, v: float, info: object) -> float:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if v <= 0:
            raise ValueError(f"{name} must be positive, got {v}")
        return v

    @field_validator("max_stop_distance_pct")
    @classmethod
    def _positive_pct(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_stop_distance_pct must be positive, got {v}")
        return v

    @field_validator("min_evidence_agreement", "min_evidence_edge")
    @classmethod
    def _non_negative_int(cls, v: int, info: object) -> int:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if v < 0:
            raise ValueError(f"{name} must be >= 0, got {v}")
        return v


class AppConfig(BaseModel):
    presets: dict[str, PresetConfig]
    market_modes: dict[str, TimeframeConfig] = Field(default_factory=dict)
    market_mode_tuning: dict[str, MarketModeTuning] = Field(default_factory=dict)
    styles: dict[str, StyleTuning] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    strategy: StrategyConfig
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    drawdown: DrawdownConfig = Field(default_factory=DrawdownConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    find_fallback_top: int = 3

    def get_preset(self, preset_name: str) -> PresetConfig:
        """Get preset configuration by name."""
        if preset_name not in self.presets:
            available = ", ".join(self.presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        return self.presets[preset_name]

    def timeframe_for(self, market_mode: "MarketMode") -> TimeframeConfig:
        return self.market_modes[market_mode.value]

    def style_tuning(self, style: "StrategyStyle") -> StyleTuning:
        return self.styles[style.value]

    def market_mode_settings(self, market_mode: "MarketMode") -> MarketModeTuning:
        return self.market_mode_tuning[market_mode.value]


# ── Loader ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path.cwd() / "futures_analyzer.config.json"


def _normalize_payload(payload: dict) -> dict:
    """
    Minimal normalization: handle legacy key renames only.
    No default injection — config.json is the single source of truth.
    """
    normalized = dict(payload)

    # Legacy rename: "modes" → "styles"
    if "styles" not in normalized and "modes" in normalized:
        normalized["styles"] = normalized.pop("modes")

    # Legacy rename: "timeframe" → "market_modes"
    if "market_modes" not in normalized and "timeframe" in normalized:
        log.warning("config.legacy_key", detail="'timeframe' key renamed to 'market_modes'")
        normalized["market_modes"] = {"intraday": normalized.pop("timeframe")}

    return normalized


def ensure_default_config(path: Path = DEFAULT_CONFIG_PATH) -> Path:
    """Write a minimal valid config.json if none exists."""
    if not path.exists():
        from futures_analyzer._config_template import MINIMAL_CONFIG
        path.write_text(json.dumps(MINIMAL_CONFIG, indent=2) + "\n", encoding="utf-8")
        log.warning("config.created_default", path=str(path))
    return path


@lru_cache(maxsize=1)
def load_app_config(path: Path | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    ensure_default_config(config_path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    payload = _normalize_payload(raw)
    log.info(
        "config.loaded",
        path=str(config_path.resolve()),
        ts=datetime.now().isoformat(timespec="seconds"),
    )
    return AppConfig.model_validate(payload)


def refresh_app_config() -> None:
    """Clear the config cache so the next call re-reads disk."""
    if hasattr(load_app_config, "cache_clear"):
        load_app_config.cache_clear()
