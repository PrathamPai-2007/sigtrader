from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev

from futures_analyzer.analysis.models import MarketMode, StrategyStyle


@dataclass
class BacktestConfig:
    symbol: str
    start: datetime
    end: datetime
    style: StrategyStyle = StrategyStyle.CONSERVATIVE
    market_mode: MarketMode = MarketMode.INTRADAY
    risk_reward: float = 2.0
    preset: str | None = None
    order_size_usd: float | None = None   # if set, slippage is applied to each trade
    daily_volume_usd: float | None = None  # estimated 24h quote volume; derived from candles if None


@dataclass
class BacktestTrade:
    bar_time: datetime
    side: str
    entry: float
    target: float
    stop: float
    confidence: float
    quality_score: float
    quality_label: str
    regime: str
    outcome: str = "unresolved"   # target_hit | stop_hit | neither | ambiguous_same_bar
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestReport:
    config: BacktestConfig
    trades: list[BacktestTrade] = field(default_factory=list)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    stop_anchor_counts: dict[str, int] = field(default_factory=dict)
    target_anchor_counts: dict[str, int] = field(default_factory=dict)
    bars_skipped_insufficient_history: int = 0
    bars_skipped_analysis_error: int = 0

    # ── aggregated ──────────────────────────────────────────────────────────
    total_trades: int = 0
    target_hit_rate: float = 0.0
    stop_hit_rate: float = 0.0
    win_rate: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    expectancy: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_approx: float = 0.0

    def compute_aggregates(self) -> None:
        """Populate all aggregate fields from self.trades."""
        resolved = [t for t in self.trades if t.outcome != "unresolved"]
        self.total_trades = len(resolved)
        if not resolved:
            return

        target_hits = sum(1 for t in resolved if t.outcome == "target_hit")
        stop_hits = sum(1 for t in resolved if t.outcome == "stop_hit")
        wins = sum(1 for t in resolved if t.pnl_pct > 0)

        self.target_hit_rate = target_hits / self.total_trades
        self.stop_hit_rate = stop_hits / self.total_trades
        self.win_rate = wins / self.total_trades
        self.avg_pnl_pct = mean(t.pnl_pct for t in resolved)
        self.avg_mfe_pct = mean(t.mfe_pct for t in resolved)
        self.avg_mae_pct = mean(t.mae_pct for t in resolved)

        win_pnls = [t.pnl_pct for t in resolved if t.pnl_pct > 0]
        loss_pnls = [t.pnl_pct for t in resolved if t.pnl_pct <= 0]
        avg_win = mean(win_pnls) if win_pnls else 0.0
        avg_loss = mean(loss_pnls) if loss_pnls else 0.0
        self.expectancy = (self.win_rate * avg_win) + ((1 - self.win_rate) * avg_loss)

        # Max drawdown on cumulative PnL curve
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in resolved:
            cumulative += t.pnl_pct
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        self.max_drawdown_pct = max_dd

        # Simplified Sharpe (mean / stdev of per-trade PnL)
        pnls = [t.pnl_pct for t in resolved]
        if len(pnls) >= 2:
            try:
                sd = stdev(pnls)
                self.sharpe_approx = (self.avg_pnl_pct / sd) if sd > 0 else 0.0
            except Exception:
                self.sharpe_approx = 0.0
