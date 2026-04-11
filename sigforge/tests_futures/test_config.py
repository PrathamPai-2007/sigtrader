from pathlib import Path

from futures_analyzer.analysis import build_timeframe_plan
from futures_analyzer.analysis.models import MarketMode, QualityLabel, StrategyStyle, TradeSetup
from futures_analyzer.config import (
    AppConfig,
    CacheConfig,
    MarketModeTuning,
    StrategyConfig,
    StyleTuning,
    TimeframeConfig,
    load_app_config,
    refresh_app_config,
)


def test_default_repo_config_file_loads() -> None:
    config = load_app_config()
    path = Path("futures_analyzer.config.json")
    assert path.exists()
    assert config.market_modes["intraday"].lookback_bars == 600
    assert config.market_modes["intraday"].entry_timeframe == "5m"
    assert config.market_modes["long_term"].entry_timeframe == "1h"
    assert config.market_mode_tuning["intraday"].min_confidence == 0.38
    assert config.market_mode_tuning["long_term"].candidate_kline_interval == "1d"
    assert "conservative" in config.styles


def test_build_timeframe_plan_uses_loaded_config(monkeypatch) -> None:
    custom = AppConfig(
        market_modes={
            "intraday": TimeframeConfig(
                profile_name="custom_intraday",
                entry_timeframe="3m",
                trigger_timeframe="30m",
                context_timeframe="2h",
                higher_timeframe="1d",
                lookback_bars=420,
            ),
            "long_term": TimeframeConfig(
                profile_name="custom_long_term",
                entry_timeframe="1h",
                trigger_timeframe="4h",
                context_timeframe="1d",
                higher_timeframe="1w",
                lookback_bars=960,
            ),
        },
        market_mode_tuning={
            "intraday": MarketModeTuning(
                target_cap_atr_mult=1.8,
                min_confidence=0.42,
                max_stop_distance_pct=2.6,
                min_evidence_agreement=5,
                min_evidence_edge=1,
            ),
            "long_term": MarketModeTuning(
                target_cap_atr_mult=4.6,
                min_confidence=0.5,
                max_stop_distance_pct=7.5,
                min_evidence_agreement=4,
                min_evidence_edge=1,
            ),
        },
        cache=CacheConfig(),
        strategy=StrategyConfig(),
        styles={
            "conservative": StyleTuning(
                fallback_risk_reward=0.7,
                target_cap_atr_mult=1.5,
                ambition_penalty_start_atr=1.1,
                ambition_penalty_slope=0.1,
                min_confidence=0.4,
                min_quality=60.0,
                min_rr_ratio=1.2,
                max_stop_distance_pct=3.0,
                min_evidence_agreement=4,
                min_evidence_edge=1,
            ),
            "aggressive": StyleTuning(
                fallback_risk_reward=1.1,
                target_cap_atr_mult=3.0,
                ambition_penalty_start_atr=2.0,
                ambition_penalty_slope=0.05,
                min_confidence=0.35,
                min_quality=55.0,
                min_rr_ratio=1.1,
                max_stop_distance_pct=4.5,
                min_evidence_agreement=3,
                min_evidence_edge=1,
            ),
        },
    )
    monkeypatch.setattr("futures_analyzer.config.load_app_config", lambda: custom)
    plan = build_timeframe_plan(style=StrategyStyle.AGGRESSIVE, market_mode=MarketMode.INTRADAY)
    assert plan.profile_name == "custom_intraday"
    assert plan.entry_timeframe == "3m"
    assert plan.trigger_timeframe == "30m"
    assert plan.context_timeframe == "2h"
    assert plan.higher_timeframe == "1d"
    assert plan.lookback_bars == 420
    assert plan.style == StrategyStyle.AGGRESSIVE
    assert plan.market_mode == MarketMode.INTRADAY


def test_repo_config_populates_new_strategy_sections() -> None:
    config_path = Path(__file__).resolve().parents[1] / "futures_analyzer.config.json"
    refresh_app_config()
    config = load_app_config(config_path)

    assert "default" in config.strategy.logistic_params
    assert "breakout" in config.strategy.regime_weights
    assert "exhaustion" in config.strategy.regime_weights
    assert "transition" in config.strategy.regime_weights
    assert abs(sum(config.strategy.regime_weights["breakout"].values()) - 1.0) <= 1e-6
    assert abs(sum(config.strategy.regime_weights["exhaustion"].values()) - 1.0) <= 1e-6
    assert abs(sum(config.strategy.regime_weights["transition"].values()) - 1.0) <= 1e-6
    assert config.strategy.geometry_quality.rr_weight == 30.0
    assert config.strategy.geometry_quality.max_score == 95.0


def test_trade_setup_deserializes_without_new_revamp_fields() -> None:
    setup = TradeSetup.model_validate(
        {
            "side": "long",
            "entry_price": 100.0,
            "target_price": 104.0,
            "stop_loss": 98.0,
            "confidence": 0.64,
            "quality_label": QualityLabel.MEDIUM,
            "quality_score": 61.0,
            "rationale": "legacy payload",
        }
    )

    assert setup.stop_anchor == "atr_fallback"
    assert setup.target_anchor == "atr_cap"
    assert setup.regime_state == ""
    assert setup.signal_strengths == {}
    assert setup.evidence_weighted_sum == 0.0
    assert setup.logistic_input == 0.0
