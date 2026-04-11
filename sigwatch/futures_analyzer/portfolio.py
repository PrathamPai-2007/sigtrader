"""Portfolio-level risk management for scan / find results.

Given a list of ranked AnalysisResult objects and a total capital amount,
PortfolioRiskManager allocates position sizes that respect:

  1. Per-symbol position cap  (max_position_pct of capital)
  2. Per-trade dollar-risk cap (max_risk_per_trade_pct of capital)
  3. Correlated-cluster risk cap (max_cluster_risk_pct of capital)
  4. Total portfolio risk cap  (max_total_risk_pct of capital)

Kelly-fraction sizing is used when enough history exists; otherwise equal-weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from futures_analyzer.analysis.models import AnalysisResult
from futures_analyzer.config import PortfolioConfig
from futures_analyzer.logging import get_logger

log = get_logger(__name__)


@dataclass
class PositionAllocation:
    symbol: str
    side: str
    notional_usd: float           # recommended position size in USD
    size_pct_of_capital: float    # notional / total_capital
    risk_usd: float               # notional * stop_distance_pct / 100
    risk_pct_of_capital: float    # risk_usd / total_capital
    cluster_id: str | None        # correlation cluster label (e.g. "A")
    capped: bool                  # True if size was reduced by a portfolio rule
    cap_reason: str | None        # human-readable reason for the cap


@dataclass
class PortfolioRiskReport:
    total_capital: float
    total_notional_usd: float
    total_risk_usd: float
    total_risk_pct: float          # total_risk_usd / total_capital
    allocations: list[PositionAllocation] = field(default_factory=list)
    cluster_warnings: list[str] = field(default_factory=list)
    breached_rules: list[str] = field(default_factory=list)


class PortfolioRiskManager:
    """Allocates position sizes across a ranked list of setups."""

    def __init__(self, config: PortfolioConfig) -> None:
        self._cfg = config

    # ── Public API ────────────────────────────────────────────────────────────

    def allocate(
        self,
        ranked_results: list[tuple[int, AnalysisResult]],
        total_capital: float,
        *,
        kelly_inputs: dict[str, tuple[float, float, float]] | None = None,
        correlation_clusters: list[list[str]] | None = None,
    ) -> PortfolioRiskReport:
        """Compute position allocations for *ranked_results*.

        Args:
            ranked_results: Output of _rank_tradable_results — list of (rank, result).
            total_capital: Total capital in USD.
            kelly_inputs: Optional per-symbol (win_rate, avg_win_pct, avg_loss_pct).
                          Falls back to equal-weight when absent or insufficient.
            correlation_clusters: Optional list of symbol groups from CorrelationAnalyzer.
        """
        if not ranked_results or total_capital <= 0:
            return PortfolioRiskReport(
                total_capital=total_capital,
                total_notional_usd=0.0,
                total_risk_usd=0.0,
                total_risk_pct=0.0,
            )

        cfg = self._cfg
        n = len(ranked_results)
        ki = kelly_inputs or {}

        # Build cluster lookup: symbol → cluster label
        cluster_map = self._build_cluster_map(correlation_clusters or [])

        # Step 1 — base notional via Kelly or equal-weight
        allocations: list[PositionAllocation] = []
        for _rank, result in ranked_results:
            sym = result.market_snapshot_meta.symbol
            stop_pct = max(result.primary_setup.stop_distance_pct, 0.01)

            base_notional = self._kelly_notional(
                total_capital=total_capital,
                symbol=sym,
                kelly_inputs=ki,
                n_setups=n,
                min_history=cfg.min_history_for_kelly,
                kelly_fraction=cfg.kelly_fraction,
            )

            allocations.append(PositionAllocation(
                symbol=sym,
                side=result.primary_setup.side,
                notional_usd=base_notional,
                size_pct_of_capital=base_notional / total_capital,
                risk_usd=base_notional * stop_pct / 100.0,
                risk_pct_of_capital=(base_notional * stop_pct / 100.0) / total_capital,
                cluster_id=cluster_map.get(sym),
                capped=False,
                cap_reason=None,
            ))

        # Step 2 — per-symbol position cap
        for alloc in allocations:
            max_notional = total_capital * cfg.max_position_pct
            if alloc.notional_usd > max_notional:
                alloc.notional_usd = max_notional
                alloc.capped = True
                alloc.cap_reason = f"position cap ({cfg.max_position_pct:.0%} of capital)"

        # Step 3 — per-trade dollar-risk cap
        for alloc in allocations:
            result = next(r for _, r in ranked_results if r.market_snapshot_meta.symbol == alloc.symbol)
            stop_pct = max(result.primary_setup.stop_distance_pct, 0.01)
            max_risk_usd = total_capital * cfg.max_risk_per_trade_pct
            implied_notional = (max_risk_usd / stop_pct) * 100.0
            if alloc.notional_usd > implied_notional:
                alloc.notional_usd = implied_notional
                alloc.capped = True
                alloc.cap_reason = f"risk cap ({cfg.max_risk_per_trade_pct:.1%} of capital per trade)"

        # Step 4 — cluster risk cap
        cluster_warnings: list[str] = []
        if correlation_clusters:
            for cluster in correlation_clusters:
                cluster_allocs = [a for a in allocations if a.symbol in cluster]
                if not cluster_allocs:
                    continue
                cluster_risk = sum(
                    a.notional_usd * self._stop_pct_for(a.symbol, ranked_results) / 100.0
                    for a in cluster_allocs
                )
                cluster_risk_pct = cluster_risk / total_capital
                max_cluster_risk = total_capital * cfg.max_cluster_risk_pct
                if cluster_risk > max_cluster_risk:
                    scale = max_cluster_risk / cluster_risk
                    label = self._cluster_label(cluster)
                    for a in cluster_allocs:
                        a.notional_usd *= scale
                        a.capped = True
                        a.cap_reason = f"cluster risk cap — cluster {label} scaled to {cfg.max_cluster_risk_pct:.0%}"
                    cluster_warnings.append(
                        f"Cluster {label} ({', '.join(cluster)}): combined risk "
                        f"{cluster_risk_pct:.2%} exceeded {cfg.max_cluster_risk_pct:.0%} cap — scaled down."
                    )

        # Step 5 — total portfolio risk cap
        breached: list[str] = []
        total_risk = sum(
            a.notional_usd * self._stop_pct_for(a.symbol, ranked_results) / 100.0
            for a in allocations
        )
        if total_risk > total_capital * cfg.max_total_risk_pct:
            scale = (total_capital * cfg.max_total_risk_pct) / total_risk
            for a in allocations:
                a.notional_usd *= scale
                a.capped = True
                a.cap_reason = a.cap_reason or f"total portfolio risk cap ({cfg.max_total_risk_pct:.0%})"
            breached.append(
                f"Total portfolio risk exceeded {cfg.max_total_risk_pct:.0%} cap — all positions scaled down."
            )

        # Recompute derived fields after all caps
        for alloc in allocations:
            stop_pct = self._stop_pct_for(alloc.symbol, ranked_results)
            alloc.size_pct_of_capital = alloc.notional_usd / total_capital
            alloc.risk_usd = alloc.notional_usd * stop_pct / 100.0
            alloc.risk_pct_of_capital = alloc.risk_usd / total_capital

        total_notional = sum(a.notional_usd for a in allocations)
        total_risk_final = sum(a.risk_usd for a in allocations)

        log.info(
            "portfolio.allocated",
            n_positions=len(allocations),
            total_notional=round(total_notional, 2),
            total_risk_pct=round(total_risk_final / total_capital * 100, 2),
        )

        return PortfolioRiskReport(
            total_capital=total_capital,
            total_notional_usd=round(total_notional, 2),
            total_risk_usd=round(total_risk_final, 2),
            total_risk_pct=round(total_risk_final / total_capital, 4),
            allocations=allocations,
            cluster_warnings=cluster_warnings,
            breached_rules=breached,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _kelly_notional(
        self,
        *,
        total_capital: float,
        symbol: str,
        kelly_inputs: dict[str, tuple[float, float, float]],
        n_setups: int,
        min_history: int,
        kelly_fraction: float,
    ) -> float:
        """Return base notional using Kelly fraction or equal-weight fallback."""
        equal_weight = total_capital / n_setups
        if symbol not in kelly_inputs:
            return equal_weight

        win_rate, avg_win, avg_loss = kelly_inputs[symbol]
        # Need at least min_history data points implied by non-default values
        if avg_win <= 0 or avg_loss <= 0:
            return equal_weight

        # Full Kelly: f = (p * b - q) / b  where b = avg_win/avg_loss
        b = avg_win / avg_loss
        q = 1.0 - win_rate
        full_kelly = (win_rate * b - q) / b if b > 0 else 0.0
        full_kelly = max(0.0, full_kelly)

        return total_capital * full_kelly * kelly_fraction

    @staticmethod
    def _stop_pct_for(symbol: str, ranked_results: list[tuple[int, AnalysisResult]]) -> float:
        for _, r in ranked_results:
            if r.market_snapshot_meta.symbol == symbol:
                return max(r.primary_setup.stop_distance_pct, 0.01)
        return 1.0

    @staticmethod
    def _build_cluster_map(clusters: list[list[str]]) -> dict[str, str]:
        label_map: dict[str, str] = {}
        for i, group in enumerate(clusters):
            label = chr(ord("A") + i) if i < 26 else str(i + 1)
            for sym in group:
                label_map[sym] = label
        return label_map

    @staticmethod
    def _cluster_label(cluster: list[str]) -> str:
        return cluster[0][:3] if cluster else "?"
