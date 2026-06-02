from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class EvaluationOutcome(str, Enum):
    UNRESOLVED = "unresolved"
    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    NEITHER = "neither"
    AMBIGUOUS_SAME_BAR = "ambiguous_same_bar"


class HistorySnapshot(BaseModel):
    id: int
    prediction_id: str
    symbol: str
    as_of: datetime
    command: str
    style: str = "conservative"
    market_mode: str = "intraday"
    profile_name: str = "auto_core"
    entry_timeframe: str = "5m"
    trigger_timeframe: str = "15m"
    context_timeframe: str = "1h"
    higher_timeframe: str = "4h"
    side: str
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    quality_label: str
    quality_score: float
    regime: str
    regime_confidence: float
    risk_reward_ratio: float
    stop_distance_pct: float
    target_distance_pct: float
    atr_multiple_to_stop: float
    atr_multiple_to_target: float
    invalidation_strength: float
    top_positive_contributors_json: str
    top_negative_contributors_json: str
    score_components_json: str
    analysis_json: str
    evaluation_status: str = EvaluationOutcome.UNRESOLVED.value
    outcome: str | None = None
    resolved_at: datetime | None = None
    max_favorable_excursion_pct: float | None = None
    max_adverse_excursion_pct: float | None = None
    pnl_at_24h_close_pct: float | None = None
    is_profitable_at_24h_close: bool | None = None
    # Enhanced metrics (None for legacy rows)
    rsi_14: float | None = None
    macd_histogram: float | None = None
    bb_position: float | None = None
    order_book_imbalance: float | None = None
    volatility_regime: str | None = None
    adjusted_rr_ratio: float | None = None
    total_slippage_pct: float | None = None


class SnapshotEvaluation(BaseModel):
    evaluation_status: str = EvaluationOutcome.UNRESOLVED.value
    outcome: str
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    pnl_at_24h_close_pct: float
    is_profitable_at_24h_close: bool


class StatsBucket(BaseModel):
    bucket: str
    sample_count: int
    target_hit_rate: float
    stop_hit_rate: float
    profitable_at_24h_rate: float
    average_24h_pnl: float
    average_mfe: float
    average_mae: float


class HistoryStatsReport(BaseModel):
    overall_feedback: StatsBucket | None = None
    confidence_buckets: list[StatsBucket] = Field(default_factory=list)
    quality_buckets: list[StatsBucket] = Field(default_factory=list)
    regime_buckets: list[StatsBucket] = Field(default_factory=list)


class HistoryCompareBy(str, Enum):
    SYMBOL = "symbol"
    STYLE = "style"
    MODE = "mode"
    CONFIDENCE = "confidence"


class HistoryCompareReport(BaseModel):
    compare_by: HistoryCompareBy
    overall_feedback: StatsBucket | None = None
    buckets: list[StatsBucket] = Field(default_factory=list)


class EnhancedMetricsFilter(BaseModel):
    min_rsi: float | None = None
    max_rsi: float | None = None
    regime: str | None = None
    min_ob_imbalance: float | None = None
    volatility_regime: str | None = None


class WindowEvaluationBucket(BaseModel):
    window: str
    sample_count: int
    win_rate: float
    avg_pnl_pct: float
    avg_mfe_pct: float


# ── Drawdown ──────────────────────────────────────────────────────────────────

@dataclass
class DrawdownState:
    """Rolling drawdown state computed from recent resolved predictions."""
    lookback: int                  # number of resolved trades examined
    cumulative_pnl_pct: float      # sum of pnl_at_24h_close_pct over lookback
    max_drawdown_pct: float        # peak-to-trough on the rolling PnL curve
    current_drawdown_pct: float    # distance from last peak to current equity
    consecutive_losses: int        # unbroken loss streak at the tail
    severity: str                  # "none" | "mild" | "moderate" | "severe"
    sample_count: int              # actual resolved trades found (may be < lookback)
