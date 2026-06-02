"""Configuration package for the analyzer.

All public symbols from ``loader`` (formerly ``config.py``) and ``presets``
(formerly ``config_presets.py``) are re-exported here so that existing
import paths keep working unchanged:

    from analyzer.config import AppConfig, load_app_config   # still works
    from analyzer.config import get_preset                   # also works
"""
from analyzer.config.loader import (
    AppConfig,
    BaiterSettings,
    CacheConfig,
    ConfidenceAdjustmentsConfig,
    DEFAULT_CONFIG_PATH,
    DelayedEntryConfig,
    DeliberationSettings,
    DrawdownConfig,
    ExecutionOverridesConfig,
    ExecutionSideOverride,
    ExhaustionSettings,
    GeometryQualityConfig,
    IndicatorParams,
    LogisticParams,
    LongEntryFiltersConfig,
    MarketModeTuning,
    MomentumExpansionConfig,
    PortfolioConfig,
    PresetConfig,
    PullbackTrapConfig,
    RegimeClassifierConfig,
    SignalTransforms,
    SlippageConfig,
    StrategyConfig,
    StructureConfirmationConfig,
    StyleTuning,
    TimeframeConfig,
    TrendDominanceConfig,
    ensure_default_config,
    load_app_config,
    refresh_app_config,
)
from analyzer.config.presets import (
    PresetProfile,
    StrategyPreset,
    PRESETS,
    get_preset,
    list_presets,
)

__all__ = [
    # loader
    "AppConfig", "BaiterSettings", "CacheConfig", "ConfidenceAdjustmentsConfig",
    "DEFAULT_CONFIG_PATH", "DelayedEntryConfig", "DeliberationSettings", "DrawdownConfig",
    "ExecutionOverridesConfig", "ExecutionSideOverride", "ExhaustionSettings",
    "GeometryQualityConfig", "IndicatorParams", "LogisticParams", "LongEntryFiltersConfig",
    "MarketModeTuning", "MomentumExpansionConfig", "PortfolioConfig", "PresetConfig",
    "PullbackTrapConfig", "RegimeClassifierConfig", "SignalTransforms", "SlippageConfig",
    "StrategyConfig", "StructureConfirmationConfig", "StyleTuning", "TimeframeConfig",
    "TrendDominanceConfig", "ensure_default_config", "load_app_config", "refresh_app_config",
    # presets
    "PresetProfile", "StrategyPreset", "PRESETS", "get_preset", "list_presets",
]
