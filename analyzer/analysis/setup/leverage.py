"""Leverage suggestion and drawdown-adjustment logic."""
from __future__ import annotations

from analyzer.analysis.models import MarketRegime, QualityLabel, TradeSetup
from analyzer.analysis.scoring.utils import _clamp, _quality_label
from analyzer.config import load_app_config


def _leverage_suggestion(
    *,
    stop_distance_pct: float,
    quality_label: QualityLabel,
    confidence: float,
    regime: MarketRegime,
    risk_reward_ratio: float = 2.0,
    is_tradable: bool = True,
) -> str:
    if not is_tradable or confidence < 0.45:
        return "1x"

    strategy = load_app_config().strategy
    cap_map = {
        QualityLabel.LOW: strategy.leverage_caps["low"],
        QualityLabel.MEDIUM: strategy.leverage_caps["medium"],
        QualityLabel.HIGH: strategy.leverage_caps["high"],
    }
    floor_map = {
        QualityLabel.LOW: strategy.leverage_floors["low"],
        QualityLabel.MEDIUM: strategy.leverage_floors["medium"],
        QualityLabel.HIGH: strategy.leverage_floors["high"],
    }

    cap = cap_map[quality_label]
    floor = floor_map[quality_label]

    if stop_distance_pct >= strategy.leverage_stop_wide:
        base = floor
    elif stop_distance_pct >= strategy.leverage_stop_mid:
        base = floor + 1
    elif stop_distance_pct >= strategy.leverage_stop_narrow:
        base = floor + 2
    elif stop_distance_pct >= strategy.leverage_stop_tight:
        base = floor + 3
    else:
        base = cap

    if confidence >= strategy.leverage_confidence_high:
        base += 1
    elif confidence < strategy.leverage_confidence_low:
        base -= 1

    if risk_reward_ratio >= strategy.leverage_rr_good:
        base += 1
    elif risk_reward_ratio < strategy.leverage_rr_poor:
        base -= 1

    if regime == MarketRegime.VOLATILE_CHOP:
        cap = min(cap, strategy.leverage_volatile_chop_cap)
        floor = min(floor, cap)
        base -= 2
    elif regime == MarketRegime.RANGE:
        base -= 1

    base = int(_clamp(base, floor, cap))
    return f"{base}x"


class DrawdownAdjuster:
    """Applies leverage and quality penalties when recent predictions are in drawdown.

    Severity thresholds and step sizes are driven by AppConfig.drawdown so they
    can be tuned without touching code.
    """

    # Leverage step ladder used to reduce suggestions (e.g. "5x" -> "3x")
    _LEVERAGE_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]

    @classmethod
    def apply(
        cls,
        setup: "TradeSetup",
        drawdown_state: "DrawdownState",
    ) -> "TradeSetup":
        """Return a copy of *setup* with leverage and quality adjusted for drawdown.

        No-op when severity is "none".
        """
        from history.models import DrawdownState  # local to avoid circular

        severity = drawdown_state.severity
        if severity == "none":
            return setup

        quality_penalty, confidence_bump, leverage_steps_down = {
            "mild":     (5.0,  0.05, 1),
            "moderate": (10.0, 0.10, 2),
            "severe":   (15.0, 0.15, None),  # None = floor at 1x
        }[severity]

        new_quality = _clamp(setup.quality_score - quality_penalty, 10.0, 95.0)
        new_quality_label = _quality_label(new_quality)
        new_confidence = _clamp(setup.confidence - confidence_bump, 0.0, 1.0)

        # Reduce leverage
        current_lev = int(setup.leverage_suggestion.rstrip("x") or 1)
        if leverage_steps_down is None:
            new_lev = 1
        else:
            try:
                idx = cls._LEVERAGE_STEPS.index(current_lev)
            except ValueError:
                idx = min(
                    range(len(cls._LEVERAGE_STEPS)),
                    key=lambda i: abs(cls._LEVERAGE_STEPS[i] - current_lev),
                )
            new_lev = cls._LEVERAGE_STEPS[max(0, idx - leverage_steps_down)]

        warning = (
            f"drawdown guard ({severity}): "
            f"{drawdown_state.current_drawdown_pct:.1f}% drawdown, "
            f"{drawdown_state.consecutive_losses} consecutive loss(es) — "
            f"leverage reduced to {new_lev}x, quality -{quality_penalty:.0f} pts"
        )

        updated_reasons = list(setup.tradable_reasons)
        if setup.is_tradable:
            updated_reasons.append(warning)

        return setup.model_copy(update={
            "quality_score": new_quality,
            "quality_label": new_quality_label,
            "confidence": new_confidence,
            "leverage_suggestion": f"{new_lev}x",
            "tradable_reasons": updated_reasons,
        })
