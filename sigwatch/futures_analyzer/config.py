from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from futures_analyzer.analysis.models import MarketMode, StrategyStyle


class TimeframeConfig(BaseModel):
    profile_name: str = "auto_core"
    entry_timeframe: str = "5m"
    trigger_timeframe: str = "15m"
    context_timeframe: str = "1h"
    higher_timeframe: str = "4h"
    lookback_bars: int = 600


class StyleTuning(BaseModel):
    fallback_risk_reward: float
    target_cap_atr_mult: float
    ambition_penalty_start_atr: float
    ambition_penalty_slope: float
    min_confidence: float
    min_quality: float
    min_rr_ratio: float
    max_stop_distance_pct: float
    min_evidence_agreement: int
    min_evidence_edge: int


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


class CacheConfig(BaseModel):
    market_meta_ttl_seconds: float = 12.0
    realtime_klines_ttl_seconds: float = 20.0
    historical_klines_ttl_seconds: float = 3600.0
    replay_lookback_cap: int = 96  # max trigger bars to walk back in chart replay


class StrategyConfig(BaseModel):
    pressure_threshold: float = 0.05
    volume_surge_threshold: float = 1.05
    regime_weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    regime_penalty_multiplier: dict[str, float] = Field(default_factory=dict)
    regime_confidence_ceiling: dict[str, float] = Field(default_factory=dict)
    regime_alignment: dict[str, float] = Field(default_factory=dict)
    regime_penalty: dict[str, float] = Field(default_factory=dict)
    leverage_caps: dict[str, int] = Field(default_factory=dict)
    leverage_floors: dict[str, int] = Field(default_factory=dict)
    confidence_quality_caps: dict[str, float] = Field(
        default_factory=lambda: {"below_0_45": 54.9, "below_0_70": 74.9, "default": 95.0}
    )
    score_confidence_divisor: float = 18.0
    quality_base_score: float = 32.0
    quality_positive_cap: float = 10.0
    quality_positive_weight: float = 4.8
    quality_rr_weight: float = 6.0
    quality_invalidation_weight: float = 10.0
    quality_negative_cap: float = 10.0
    quality_negative_weight: float = 5.2
    quality_min_score: float = 10.0
    quality_max_score: float = 95.0
    confirmation_low_cutoff: int = 2
    confirmation_mid_cutoff: int = 4
    confirmation_low_penalty: float = 1.3
    confirmation_mid_penalty: float = 0.55
    dual_trend_penalty: float = 0.35
    dual_momentum_penalty: float = 0.35
    dual_pressure_penalty: float = 0.2
    volatile_chop_boost_per_signal: float = 0.04
    volatile_chop_confidence_ceiling_cap: float = 0.82
    volatile_chop_strong_signal_threshold: float = 0.15
    target_ambition_penalty_weight: float = 10.0
    confirmation_penalty_weight: float = 3.0
    regime_penalty_weight: float = 4.0


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


class PortfolioConfig(BaseModel):
    max_position_pct: float = 0.20        # max single position as fraction of capital
    max_risk_per_trade_pct: float = 0.02  # max dollar risk per trade as fraction of capital
    max_cluster_risk_pct: float = 0.10    # max combined risk for a correlated cluster
    max_total_risk_pct: float = 0.06      # max aggregate portfolio risk
    kelly_fraction: float = 0.25          # quarter-Kelly scaling
    min_history_for_kelly: int = 10       # fall back to equal-weight below this


class AppConfig(BaseModel):
    market_modes: dict[str, TimeframeConfig]
    market_mode_tuning: dict[str, MarketModeTuning]
    cache: CacheConfig = Field(default_factory=CacheConfig)
    strategy: StrategyConfig
    styles: dict[str, StyleTuning]
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    drawdown: DrawdownConfig = Field(default_factory=DrawdownConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    # When `find` yields zero tradable setups, show this many near-miss
    # candidates with a warning instead of an empty result.
    # Set to 0 to disable the fallback entirely.
    find_fallback_top: int = 3

    def timeframe_for(self, market_mode: MarketMode) -> TimeframeConfig:
        return self.market_modes[market_mode.value]

    def style_tuning(self, style: StrategyStyle) -> StyleTuning:
        return self.styles[style.value]

    def market_mode_settings(self, market_mode: MarketMode) -> MarketModeTuning:
        return self.market_mode_tuning[market_mode.value]


DEFAULT_CONFIG_PATH = Path.cwd() / "futures_analyzer.config.json"


def _default_payload() -> dict:
    return {
        "market_modes": {
            "intraday": {
                "profile_name": "intraday_core",
                "entry_timeframe": "5m",
                "trigger_timeframe": "15m",
                "context_timeframe": "1h",
                "higher_timeframe": "4h",
                "lookback_bars": 600,
            },
            "long_term": {
                "profile_name": "long_term_swing",
                "entry_timeframe": "1h",
                "trigger_timeframe": "4h",
                "context_timeframe": "1d",
                "higher_timeframe": "1w",
                "lookback_bars": 900,
            },
        },
        "market_mode_tuning": {
            "intraday": {
                "target_cap_atr_mult": 1.9,
                "min_confidence": 0.44,
                "max_stop_distance_pct": 2.8,
                "min_evidence_agreement": 5,
                "min_evidence_edge": 1,
                "candidate_quote_asset": "USDT",
                "candidate_min_quote_volume": 25000000.0,
                "candidate_pool_limit": 40,
                "candidate_kline_interval": "1h",
                "candidate_kline_lookback": 48,
                "candidate_return_weight": 0.45,
                "candidate_range_weight": 0.4,
                "candidate_consistency_weight": 0.15,
            },
            "long_term": {
                "target_cap_atr_mult": 4.8,
                "min_confidence": 0.5,
                "max_stop_distance_pct": 8.0,
                "min_evidence_agreement": 4,
                "min_evidence_edge": 1,
                "candidate_quote_asset": "USDT",
                "candidate_min_quote_volume": 40000000.0,
                "candidate_pool_limit": 30,
                "candidate_kline_interval": "1d",
                "candidate_kline_lookback": 14,
                "candidate_return_weight": 0.55,
                "candidate_range_weight": 0.15,
                "candidate_consistency_weight": 0.3,
            },
        },
        "cache": {
            "market_meta_ttl_seconds": 12.0,
            "realtime_klines_ttl_seconds": 20.0,
            "historical_klines_ttl_seconds": 3600.0,
            "replay_lookback_cap": 96,
        },
        "strategy": {
            "pressure_threshold": 0.05,
            "volume_surge_threshold": 1.05,
            "regime_weights": {
                "default": {
                    "higher_trend": 6.0,
                    "momentum": 4.0,
                    "trend": 5.0,
                    "entry_confirmation": 3.0,
                    "structure": 1.5,
                    "volume_surge": 1.5,
                    "buy_sell_pressure": 2.3,
                    "oi_funding_bias": 1.5,
                    "regime_alignment": 1.8,
                    "volume_poc_proximity": 2.5,
                },
                "bullish_trend_long": {"higher_trend": 7.5, "momentum": 4.8, "trend": 6.5, "entry_confirmation": 3.3},
                "bullish_trend_short": {"higher_trend": 3.2, "momentum": 2.5, "trend": 3.6, "structure": 1.0, "oi_funding_bias": 1.2},
                "bearish_trend_short": {"higher_trend": 7.5, "momentum": 4.8, "trend": 6.5, "entry_confirmation": 3.3},
                "bearish_trend_long": {"higher_trend": 3.2, "momentum": 2.5, "trend": 3.6, "structure": 1.0, "oi_funding_bias": 1.2},
                "range": {
                    "higher_trend": 3.0,
                    "momentum": 2.6,
                    "trend": 3.0,
                    "entry_confirmation": 2.2,
                    "structure": 2.2,
                    "volume_surge": 1.1,
                    "buy_sell_pressure": 1.8,
                    "oi_funding_bias": 1.0,
                },
                "volatile_chop": {
                    "higher_trend": 2.8,
                    "momentum": 2.8,
                    "trend": 2.8,
                    "entry_confirmation": 2.0,
                    "structure": 1.2,
                    "volume_surge": 1.4,
                    "buy_sell_pressure": 2.1,
                    "oi_funding_bias": 1.0,
                },
            },
            "regime_penalty_multiplier": {
                "default": 2.0,
                "volatile_chop": 2.8,
            },
            "regime_confidence_ceiling": {
                "default": 1.0,
                "bullish_trend_short": 0.92,
                "bearish_trend_long": 0.92,
                "range": 0.84,
                "volatile_chop": 0.68,
            },
            "regime_alignment": {
                "range": 0.25,
                "bullish_trend_long": 0.9,
                "bearish_trend_short": 0.9,
            },
            "regime_penalty": {
                "volatile_chop": 0.6,
                "bullish_trend_short": 0.4,
                "bearish_trend_long": 0.4,
            },
            "leverage_caps": {"low": 2, "medium": 4, "high": 6},
            "leverage_floors": {"low": 2, "medium": 2, "high": 3},
        },
        "styles": {
            "conservative": {
                "fallback_risk_reward": 1.2,
                "target_cap_atr_mult": 1.75,
                "ambition_penalty_start_atr": 1.25,
                "ambition_penalty_slope": 0.12,
                "min_confidence": 0.45,
                "min_quality": 65.0,
                "min_rr_ratio": 1.0,
                "max_stop_distance_pct": 2.5,
                "min_evidence_agreement": 5,
                "min_evidence_edge": 1,
            },
            "aggressive": {
                "fallback_risk_reward": 1.8,
                "target_cap_atr_mult": 3.5,
                "ambition_penalty_start_atr": 2.5,
                "ambition_penalty_slope": 0.06,
                "min_confidence": 0.4,
                "min_quality": 60.0,
                "min_rr_ratio": 1.5,
                "max_stop_distance_pct": 4.0,
                "min_evidence_agreement": 4,
                "min_evidence_edge": 1,
            },
        },
        "slippage": {
            "volatility_multipliers": {
                "low": 0.7,
                "normal": 1.0,
                "high": 1.4,
                "extreme": 2.0,
            },
            "default_model": "moderate",
            "min_viable_rr": 1.0,
        },
    }


def ensure_default_config(path: Path = DEFAULT_CONFIG_PATH) -> Path:
    if not path.exists():
        path.write_text(json.dumps(_default_payload(), indent=2) + "\n", encoding="utf-8")
    return path


def _normalize_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if "styles" not in normalized and "modes" in normalized:
        normalized["styles"] = normalized.pop("modes")
    if "market_modes" not in normalized:
        timeframe = normalized.pop("timeframe", None)
        intraday = timeframe or TimeframeConfig().model_dump(mode="json")
        normalized["market_modes"] = {
            "intraday": intraday,
            "long_term": {
                "profile_name": "long_term_swing",
                "entry_timeframe": "1h",
                "trigger_timeframe": "4h",
                "context_timeframe": "1d",
                "higher_timeframe": "1w",
                "lookback_bars": max(int(intraday.get("lookback_bars", 600)), 900),
            },
        }
    if "market_mode_tuning" not in normalized:
        normalized["market_mode_tuning"] = {
            "intraday": {
                "target_cap_atr_mult": 1.9,
                "min_confidence": 0.44,
                "max_stop_distance_pct": 2.8,
                "min_evidence_agreement": 5,
                "min_evidence_edge": 1,
                "candidate_quote_asset": "USDT",
                "candidate_min_quote_volume": 25000000.0,
                "candidate_pool_limit": 40,
                "candidate_kline_interval": "1h",
                "candidate_kline_lookback": 48,
                "candidate_return_weight": 0.45,
                "candidate_range_weight": 0.4,
                "candidate_consistency_weight": 0.15,
            },
            "long_term": {
                "target_cap_atr_mult": 4.8,
                "min_confidence": 0.5,
                "max_stop_distance_pct": 8.0,
                "min_evidence_agreement": 4,
                "min_evidence_edge": 1,
                "candidate_quote_asset": "USDT",
                "candidate_min_quote_volume": 40000000.0,
                "candidate_pool_limit": 30,
                "candidate_kline_interval": "1d",
                "candidate_kline_lookback": 14,
                "candidate_return_weight": 0.55,
                "candidate_range_weight": 0.15,
                "candidate_consistency_weight": 0.3,
            },
        }
    return normalized


@lru_cache(maxsize=1)
def load_app_config(path: Path | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    ensure_default_config(config_path)
    payload = _normalize_payload(json.loads(config_path.read_text(encoding="utf-8")))
    return AppConfig.model_validate(payload)


def refresh_app_config() -> None:
    """Refresh by clearing the global config cache."""
    load_app_config.cache_clear()
