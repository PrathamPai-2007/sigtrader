"""Signal decay — TTL computation and freshness-adjusted confidence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from futures_analyzer.analysis.models import TradeSetup

# Maps trigger timeframe string to bar duration in seconds
_TF_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
}


def compute_ttl(trigger_timeframe: str, multiplier: float = 3.0) -> timedelta:
    """Return TTL duration based on trigger timeframe bar size × multiplier.

    Examples (default multiplier=3):
        15m  → 45 min
        1h   → 3 hours
        4h   → 12 hours
        1d   → 3 days
    """
    bar_seconds = _TF_SECONDS.get(trigger_timeframe, 900)
    return timedelta(seconds=bar_seconds * multiplier)


def apply_decay(setup: TradeSetup, as_of: datetime) -> TradeSetup:
    """Return a copy of *setup* with freshness and effective confidence applied.

    If ``valid_until`` is None the setup is returned unchanged — decay is
    opt-in and only activates once the scorer attaches a TTL.

    Freshness is linear: 1.0 at creation, 0.0 at ``valid_until``, clamped to
    [0.0, 1.0].  Effective confidence = original_confidence × freshness.
    """
    if setup.valid_until is None:
        return setup

    now = as_of
    deadline = setup.valid_until.replace(tzinfo=UTC) if setup.valid_until.tzinfo is None else setup.valid_until
    now = now.replace(tzinfo=UTC) if now.tzinfo is None else now

    remaining = (deadline - now).total_seconds()

    ttl_seconds = getattr(setup, "ttl_seconds", None)
    if ttl_seconds and ttl_seconds > 0:
        freshness = max(0.0, min(1.0, remaining / ttl_seconds))
    else:
        # Fallback: treat remaining > 0 as fresh, <= 0 as stale
        freshness = 1.0 if remaining > 0 else 0.0

    effective_confidence = round(setup.confidence * freshness, 6)
    is_stale = freshness <= 0.0

    return setup.model_copy(update={
        "freshness_at_retrieval": round(freshness, 4),
        "is_stale": is_stale,
        "confidence": effective_confidence,
    })
