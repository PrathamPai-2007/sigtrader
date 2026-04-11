from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlippageReport:
    symbol: str
    side: str
    order_size_usd: float
    raw_entry: float
    raw_target: float
    raw_stop: float
    raw_rr: float
    entry_slippage_pct: float
    target_slippage_pct: float
    stop_slippage_pct: float
    total_slippage_pct: float
    adj_entry: float
    adj_target: float
    adj_stop: float
    adj_rr: float
    rr_degradation: float       # raw_rr - adj_rr
    is_still_viable: bool       # adj_rr >= min_viable_rr
    model_used: str             # conservative | moderate | optimistic
    volatility_regime: str
    liquidity_score: float
    spread_pct: float


@dataclass
class CorrelationPair:
    symbol_a: str
    symbol_b: str
    correlation: float          # Pearson, -1 to 1
    window_bars: int
    interval: str


@dataclass
class CorrelationReport:
    symbols: list[str]
    interval: str
    window_bars: int
    pairs: list[CorrelationPair] = field(default_factory=list)
    diversification_score: float = 0.0   # 0-100; higher = less correlated
    cluster_groups: list[list[str]] = field(default_factory=list)
    hedge_pairs: list[CorrelationPair] = field(default_factory=list)
