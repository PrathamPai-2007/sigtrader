
import dataclasses

from futures_analyzer.analysis.models import (
    ContributorDetail,
    ContributorDirection,
    EvidenceVector,
    MarketRegime,
    NormalizedSignals,
    QualityLabel,
)
from futures_analyzer.config import AppConfig, load_app_config, StrategyConfig
from futures_analyzer.analysis.scoring.utils import _clamp, _quality_label, _quality_score_cap_from_confidence

# TYPE_CHECKING imports to avoid circular dependency at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from futures_analyzer.analysis.long_pipeline_log import LongPipelineLog


def compute_graded_evidence(
    signals: NormalizedSignals,
    regime: MarketRegime,
    side: str,
    weights: dict[str, float],
    *,
    debug: bool = False,
    regime_gate_thresholds: dict | None = None,
    config: AppConfig | None = None,
) -> EvidenceVector:
    """Compute a weighted evidence strength score from normalized signals.

    Replaces the binary 7-check system with a continuous weighted sum.
    """
    # Build signal dict from NormalizedSignals fields
    signal_dict: dict[str, float] = {
        f.name: getattr(signals, f.name)
        for f in dataclasses.fields(signals)
    }

    # Weighted sum: dot product of regime weights and directional signal values.
    # Convert each normalized signal from [0, 1] to directional edge in [-1, 1]:
    #   1.0 -> +1 (bullish dominance), 0.5 -> 0 (neutral), 0.0 -> -1 (bearish dominance)
    raw_sum = 0.0
    positive_contribution = 0.0
    negative_contribution = 0.0
    for name, value in signal_dict.items():
        weight = weights.get(name, 0.0)
        if weight == 0:
            continue

        # Convert [0,1] → [-1,1]
        directional_value = (value - 0.5) * 2.0

        contribution = weight * directional_value
        raw_sum += contribution
        if contribution >= 0.0:
            positive_contribution += contribution
        else:
            negative_contribution += -contribution

    weighted_sum = raw_sum

    # Count signals above 0.5 threshold
    signal_count_above_threshold = sum(1 for v in signal_dict.values() if v > 0.5)

    # Top-3 strongest and bottom-3 weakest signals
    sorted_desc = sorted(signal_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_asc = sorted(signal_dict.items(), key=lambda kv: kv[1])
    strongest_signals = sorted_desc[:3]
    weakest_signals = sorted_asc[:3]

    # Regime gate check — thresholds from config.strategy.regime_gate_thresholds
    if regime_gate_thresholds is None:
        try:
            cfg = config or load_app_config()
            thresholds = getattr(cfg.strategy, "regime_gate_thresholds", {}) or {}
        except Exception:
            thresholds = {}
    else:
        thresholds = regime_gate_thresholds

    def _thr(key: str, default: float) -> float:
        return thresholds.get(key, default)

    if regime in (MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND):
        regime_gate_passed = signals.higher_trend > _thr("trend", 0.4)
    elif regime == MarketRegime.BREAKOUT:
        regime_gate_passed = signals.volume_surge > _thr("breakout", 0.3)
    elif regime == MarketRegime.EXHAUSTION:
        regime_gate_passed = signals.rsi_alignment > _thr("exhaustion", 0.3)
    elif regime == MarketRegime.RANGE:
        regime_gate_passed = signals.structure_position > _thr("range", 0.3)
    elif regime == MarketRegime.VOLATILE_CHOP:
        regime_gate_passed = signals.higher_trend > _thr("volatile_chop", 0.5)
    elif regime == MarketRegime.TRANSITION:
        regime_gate_passed = signals.higher_trend > _thr("transition", 0.35)
    else:
        regime_gate_passed = True

    debug_payload: dict[str, float] | None = None
    if debug:
        total = positive_contribution + negative_contribution
        signal_balance = (positive_contribution - negative_contribution) / max(total, 1e-12)
        debug_payload = {
            "raw_score": weighted_sum,
            "positive_contribution": positive_contribution,
            "negative_contribution": negative_contribution,
            "signal_balance": signal_balance,
        }

    return EvidenceVector(
        weighted_sum=weighted_sum,
        signal_count_above_threshold=signal_count_above_threshold,
        strongest_signals=strongest_signals,
        weakest_signals=weakest_signals,
        regime_gate_passed=regime_gate_passed,
        debug=debug_payload,
    )


def _regime_weight_profile(regime: MarketRegime, side: str) -> tuple[dict[str, float], float, float]:
    """Return regime-specific weights, penalty multiplier, and confidence ceiling."""
    from futures_analyzer.config import load_app_config
    strategy = load_app_config().strategy
    regime_key = regime.value
    side_key = f"{regime_key}_{side}"
    weights = dict(strategy.regime_weights.get("default", {}))
    if regime in {MarketRegime.BULLISH_TREND, MarketRegime.BEARISH_TREND}:
        weights.update(strategy.regime_weights.get(side_key, {}))
    else:
        weights.update(strategy.regime_weights.get(regime_key, {}))
    penalty_multiplier = strategy.regime_penalty_multiplier.get(regime_key, strategy.regime_penalty_multiplier["default"])
    confidence_ceiling = strategy.regime_confidence_ceiling.get(side_key, strategy.regime_confidence_ceiling.get(regime_key, strategy.regime_confidence_ceiling["default"]))
    return weights, penalty_multiplier, confidence_ceiling


def _regime_alignment(regime: MarketRegime, side: str) -> float:
    """Return alignment score between regime and side."""
    from futures_analyzer.config import load_app_config
    strategy = load_app_config().strategy
    return strategy.regime_alignment.get(f"{regime.value}_{side}", strategy.regime_alignment.get(regime.value, 0.0))


def _regime_penalty(regime: MarketRegime, side: str) -> float:
    """Return penalty score for regime/side combination."""
    from futures_analyzer.config import load_app_config
    strategy = load_app_config().strategy
    return strategy.regime_penalty.get(f"{regime.value}_{side}", strategy.regime_penalty.get(regime.value, 0.0))
