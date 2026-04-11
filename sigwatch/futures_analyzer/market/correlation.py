from __future__ import annotations

import asyncio
import math
from datetime import datetime

from futures_analyzer.analysis.concurrency import ParallelAnalyzer
from futures_analyzer.analysis.models import Candle
from futures_analyzer.market.models import CorrelationPair, CorrelationReport


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pure-Python Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)
    return num / den if den > 0 else 0.0


def _log_returns(closes: list[float]) -> list[float]:
    result: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]
        if prev > 0 and curr > 0:
            result.append(math.log(curr / prev))
        else:
            result.append(0.0)
    return result


def _align_series(
    series_map: dict[str, list[tuple[datetime, float]]],
) -> dict[str, list[float]]:
    """Inner-join all symbols on shared timestamps, return aligned close lists."""
    if not series_map:
        return {}
    # Find timestamps present in every symbol
    common: set[datetime] | None = None
    for ts_closes in series_map.values():
        ts_set = {ts for ts, _ in ts_closes}
        common = ts_set if common is None else common & ts_set
    if not common:
        return {}
    sorted_ts = sorted(common)
    aligned: dict[str, list[float]] = {}
    for symbol, ts_closes in series_map.items():
        lookup = {ts: close for ts, close in ts_closes}
        aligned[symbol] = [lookup[ts] for ts in sorted_ts]
    return aligned


def _cluster(
    symbols: list[str],
    pairs: list[CorrelationPair],
    threshold: float,
) -> list[list[str]]:
    """Union-find clustering of symbols with correlation >= threshold."""
    parent = {s: s for s in symbols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for pair in pairs:
        if pair.correlation >= threshold:
            union(pair.symbol_a, pair.symbol_b)

    groups: dict[str, list[str]] = {}
    for s in symbols:
        root = find(s)
        groups.setdefault(root, []).append(s)

    return [g for g in groups.values() if len(g) >= 2]


class CorrelationAnalyzer:
    """Computes pairwise return correlations across a set of symbols.

    Uses pure-Python Pearson correlation on log returns — no numpy/pandas.
    Candles are fetched in parallel via ParallelAnalyzer.
    """

    def __init__(
        self,
        *,
        window_bars: int = 100,
        interval: str = "1h",
        high_corr_threshold: float = 0.75,
        hedge_threshold: float = -0.6,
    ) -> None:
        self.window_bars = window_bars
        self.interval = interval
        self.high_corr_threshold = high_corr_threshold
        self.hedge_threshold = hedge_threshold

    async def analyze(
        self,
        symbols: list[str],
        *,
        provider=None,
    ) -> CorrelationReport:
        """Fetch candles for all symbols and compute the full correlation matrix.

        Args:
            symbols: List of Binance futures symbols.
            provider: Optional BinanceFuturesProvider (created internally if None).

        Returns:
            CorrelationReport with pairs, clusters, hedge pairs, and
            diversification score.
        """
        from futures_analyzer.providers import BinanceFuturesProvider

        owned = provider is None
        provider = provider or BinanceFuturesProvider()

        try:
            series_map = await self._fetch_series(symbols, provider)
        finally:
            if owned:
                await provider.aclose()

        aligned = _align_series(series_map)
        if len(aligned) < 2:
            return CorrelationReport(
                symbols=symbols,
                interval=self.interval,
                window_bars=self.window_bars,
            )

        # Compute log returns per symbol
        returns_map = {sym: _log_returns(closes) for sym, closes in aligned.items()}

        # Pairwise Pearson
        sym_list = sorted(returns_map)
        pairs: list[CorrelationPair] = []
        for i, a in enumerate(sym_list):
            for b in sym_list[i + 1:]:
                ra, rb = returns_map[a], returns_map[b]
                # Align lengths (should already match but be safe)
                n = min(len(ra), len(rb))
                corr = _pearson(ra[:n], rb[:n])
                pairs.append(CorrelationPair(
                    symbol_a=a,
                    symbol_b=b,
                    correlation=round(corr, 4),
                    window_bars=self.window_bars,
                    interval=self.interval,
                ))

        # Sort by absolute correlation descending
        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)

        # Diversification score
        if pairs:
            avg_abs = sum(abs(p.correlation) for p in pairs) / len(pairs)
            div_score = round((1.0 - avg_abs) * 100.0, 2)
        else:
            div_score = 100.0

        clusters = _cluster(sym_list, pairs, self.high_corr_threshold)
        hedge_pairs = [p for p in pairs if p.correlation <= self.hedge_threshold]

        return CorrelationReport(
            symbols=sym_list,
            interval=self.interval,
            window_bars=self.window_bars,
            pairs=pairs,
            diversification_score=div_score,
            cluster_groups=clusters,
            hedge_pairs=hedge_pairs,
        )

    async def _fetch_series(
        self,
        symbols: list[str],
        provider,
    ) -> dict[str, list[tuple[datetime, float]]]:
        """Fetch close-price series for all symbols in parallel."""
        pa = ParallelAnalyzer(max_concurrent=8)

        async def fetch_one(symbol: str):
            candles = await provider.fetch_klines(
                symbol=symbol,
                interval=self.interval,
                limit=self.window_bars + 1,
                min_required_candles=10,
            )
            return [(c.open_time, c.close) for c in candles]

        results = await pa.analyze_batch(symbols, fetch_one, fail_fast=False)
        series_map: dict[str, list[tuple[datetime, float]]] = {}
        for symbol, ts_closes, error in results:
            if error is None and ts_closes:
                series_map[symbol] = ts_closes
        return series_map

    def warn_concentrated_setups(
        self,
        report: CorrelationReport,
        ranked_symbols: list[str],
        *,
        threshold: float | None = None,
    ) -> list[str]:
        """Return warning strings for highly-correlated pairs in ranked_symbols.

        Used by scan/find to annotate output when top setups are concentrated.
        """
        threshold = threshold if threshold is not None else self.high_corr_threshold
        sym_set = set(ranked_symbols)
        warnings: list[str] = []
        for pair in report.pairs:
            if pair.symbol_a not in sym_set or pair.symbol_b not in sym_set:
                continue
            if pair.correlation >= threshold:
                warnings.append(
                    f"  Correlation warning: {pair.symbol_a} and {pair.symbol_b} "
                    f"are {pair.correlation:.2f} correlated — concentrated risk."
                )
            elif pair.correlation <= self.hedge_threshold:
                warnings.append(
                    f"  Hedge opportunity: {pair.symbol_a} and {pair.symbol_b} "
                    f"are {pair.correlation:.2f} correlated — natural hedge."
                )
        return warnings
