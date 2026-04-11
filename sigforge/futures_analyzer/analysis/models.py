from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class StrategyStyle(str, Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


class MarketMode(str, Enum):
    INTRADAY = "intraday"
    LONG_TERM = "long_term"


class MarketRegime(str, Enum):
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    RANGE = "range"
    VOLATILE_CHOP = "volatile_chop"
    BREAKOUT = "breakout"
    EXHAUSTION = "exhaustion"
    TRANSITION = "transition"


class QualityLabel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ContributorDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Candle(BaseModel):
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class TimeframePlan(BaseModel):
    profile_name: str = "auto_core"
    style: StrategyStyle = StrategyStyle.CONSERVATIVE
    market_mode: MarketMode = MarketMode.INTRADAY
    entry_timeframe: str = "5m"
    context_timeframe: str
    trigger_timeframe: str
    higher_timeframe: str = "4h"
    lookback_bars: int = 600


class MarketMeta(BaseModel):
    symbol: str
    tick_size: float | None = None
    step_size: float | None = None
    mark_price: float
    funding_rate: float | None = None
    funding_rate_history: list[float] = Field(default_factory=list)
    open_interest: float | None = None
    open_interest_change_pct: float | None = None
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ContributorDetail(BaseModel):
    key: str
    label: str
    value: float
    impact: float
    direction: ContributorDirection
    summary: str


class TradeSetup(BaseModel):
    side: str
    entry_price: float
    target_price: float
    stop_loss: float
    leverage_suggestion: str = "1x"
    confidence: float = Field(ge=0.0, le=1.0)
    quality_label: QualityLabel = QualityLabel.MEDIUM
    quality_score: float = Field(default=50.0, ge=0.0, le=100.0)
    rationale: str
    top_positive_contributors: list[ContributorDetail] = Field(default_factory=list)
    top_negative_contributors: list[ContributorDetail] = Field(default_factory=list)
    score_components: dict[str, float] = Field(default_factory=dict)
    structure_points: dict[str, float] = Field(default_factory=dict)
    risk_reward_ratio: float = 0.0
    stop_distance_pct: float = 0.0
    target_distance_pct: float = 0.0
    atr_multiple_to_stop: float = 0.0
    atr_multiple_to_target: float = 0.0
    invalidation_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    is_tradable: bool = True
    tradable_reasons: list[str] = Field(default_factory=list)
    evidence_agreement: int = 0
    evidence_total: int = 0
    deliberation_summary: str = ""
    # Signal decay / TTL
    valid_until: datetime | None = None
    ttl_seconds: float | None = None
    freshness_at_retrieval: float | None = None
    is_stale: bool = False
    # New fields (all optional with defaults for backward compat)
    stop_anchor: str = "atr_fallback"
    target_anchor: str = "atr_cap"
    regime_state: str = ""
    signal_strengths: dict[str, float] = Field(default_factory=dict)
    evidence_weighted_sum: float = 0.0
    logistic_input: float = 0.0


class EnhancedMetrics(BaseModel):
    """Enhanced market metrics from advanced data sources."""
    rsi_14: float = Field(default=50.0, ge=0.0, le=100.0)
    macd_value: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    stochastic_k: float = Field(default=50.0, ge=0.0, le=100.0)
    stochastic_d: float = Field(default=50.0, ge=0.0, le=100.0)
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_bandwidth_pct: float = 0.0
    bollinger_position: float = Field(default=0.5, ge=0.0, le=1.0)
    rsi_divergence: bool = False
    rsi_divergence_type: str = "none"
    bid_ask_spread_pct: float = 0.0
    bid_ask_ratio: float = 1.0
    order_book_imbalance: float = Field(default=0.0, ge=-1.0, le=1.0)
    volatility_rank: float = Field(default=50.0, ge=0.0, le=100.0)
    volatility_regime: str = "normal"
    vwap: float = 0.0
    vwap_deviation_pct: float = 0.0
    liquidity_score: float = Field(default=50.0, ge=0.0, le=100.0)
    slippage_estimate_pct: float = 0.0


class AnalysisResult(BaseModel):
    prediction_id: str | None = None
    primary_setup: TradeSetup
    secondary_context: TradeSetup
    timeframe_plan: TimeframePlan
    market_snapshot_meta: MarketMeta
    market_regime: MarketRegime = MarketRegime.RANGE
    regime_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    chart_replay_last_tradable_at: datetime | None = None
    enhanced_metrics: EnhancedMetrics = Field(default_factory=EnhancedMetrics)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = "Analytical guidance only. Not financial advice and not guaranteed."
