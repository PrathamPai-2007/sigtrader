from __future__ import annotations

from futures_analyzer.analysis.models import MarketRegime
from futures_analyzer.analysis.indicators import sigmoid
from futures_analyzer.analysis.scoring.utils import _clamp
from futures_analyzer.config import AppConfig, load_app_config, StrategyConfig


def logistic_confidence(
    evidence_weighted_sum: float,
    regime: MarketRegime,
    side: str,
    strategy: StrategyConfig,
) -> float:
    """Map evidence weighted sum to confidence via logistic sigmoid.

    Uses regime/side-specific steepness from config.logistic_params.
    Falls back to the "default" entry in config.logistic_params, then to
    a minimal safe steepness value.

    Midpoint shifting is intentionally not applied: the directional scoring
    model is already centered around 0.0.
    """
    key = f"{regime.value}_{side}"
    logistic_params = getattr(strategy, "logistic_params", {}) or {}
    params = (
        logistic_params.get(key)
        or logistic_params.get(regime.value)
        or logistic_params.get("default")
    )
    if params is None:
        steepness = 6.0  # Minimal safe fallback
    else:
        if isinstance(params, dict):
            steepness = params.get("steepness", 6.0)
        else:
            steepness = getattr(params, "steepness", 6.0)

    x = steepness * evidence_weighted_sum
    return _clamp(sigmoid(x), 0.0, 1.0)


def logistic_confidence_from_config(
    evidence_weighted_sum: float,
    regime: MarketRegime,
    side: str,
    config: AppConfig | None = None,
) -> float:
    """Convenience wrapper — requires config param."""
    cfg = config or load_app_config()
    return logistic_confidence(evidence_weighted_sum, regime, side, cfg.strategy)
