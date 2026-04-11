from __future__ import annotations

import asyncio

from futures_analyzer.analysis.models import MarketMeta, TradeSetup
from futures_analyzer.history.evaluation import SlippageCalculator, SlippageModel
from futures_analyzer.market.models import SlippageReport


# Volatility regime → slippage multiplier
_DEFAULT_VOL_MULTIPLIERS: dict[str, float] = {
    "low": 0.7,
    "normal": 1.0,
    "high": 1.4,
    "extreme": 2.0,
}


def _raw_rr(setup: TradeSetup) -> float:
    entry = setup.entry_price
    target = setup.target_price
    stop = setup.stop_loss
    if setup.side == "long":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    return reward / risk if risk > 0 else 0.0


class SlippageAdvisor:
    """Estimates realistic per-trade slippage using live order book data and
    volatility regime scaling.

    Wraps the existing SlippageCalculator (history/evaluation.py) and
    EnhancedDataProvider (providers/microstructure.py) with position-size
    awareness so the R:R impact is concrete rather than a flat percentage.
    """

    def __init__(
        self,
        *,
        model: SlippageModel = SlippageModel.MODERATE,
        min_viable_rr: float = 1.0,
    ) -> None:
        self.model = model
        self.min_viable_rr = min_viable_rr

    def _vol_multiplier(self, regime: str) -> float:
        try:
            from futures_analyzer.config import load_app_config
            cfg = load_app_config()
            multipliers = getattr(cfg, "slippage", None)
            if multipliers is not None:
                return multipliers.volatility_multipliers.get(regime, 1.0)
        except Exception:
            pass
        return _DEFAULT_VOL_MULTIPLIERS.get(regime, 1.0)

    async def estimate(
        self,
        setup: TradeSetup,
        market: MarketMeta,
        *,
        order_size_usd: float = 1000.0,
        daily_volume_usd: float | None = None,
        candles: list | None = None,
        provider=None,
    ) -> SlippageReport:
        """Estimate slippage for a trade setup.

        Args:
            setup: The TradeSetup to evaluate.
            market: MarketMeta for the symbol.
            order_size_usd: Position size in USD.
            daily_volume_usd: 24h volume; used to compute order_size_pct.
                              Falls back to a conservative 10M if not provided.
            candles: Recent candles for volatility calculation (optional).
            provider: BinanceFuturesProvider instance (optional; created if None).

        Returns:
            SlippageReport with adjusted prices and R:R impact.
        """
        from futures_analyzer.providers.microstructure import EnhancedDataProvider

        owned_provider = provider is None
        enhanced = EnhancedDataProvider()
        try:
            # ── order book ──────────────────────────────────────────────────
            try:
                ob = await enhanced.fetch_order_book_snapshot(market.symbol)
                spread_pct = ob.spread_pct
                book_depth_usd = (ob.bid_volume + ob.ask_volume) * market.mark_price
                liquidity_score = 50.0  # default; refined below
            except Exception:
                spread_pct = 0.05
                book_depth_usd = max(daily_volume_usd or 10_000_000.0, 1.0)
                liquidity_score = 50.0

            # ── volatility regime ────────────────────────────────────────────
            vol_regime = "normal"
            if candles:
                try:
                    vol_metrics = await enhanced.fetch_volatility_metrics(market.symbol, candles)
                    vol_regime = vol_metrics.volatility_regime
                    # Refine liquidity from order book + daily volume
                    if daily_volume_usd and daily_volume_usd > 0:
                        liq = await enhanced.fetch_liquidity_metrics(
                            market.symbol, ob, daily_volume_usd
                        )
                        liquidity_score = liq.liquidity_score
                        spread_pct = liq.bid_ask_spread_pct
                except Exception:
                    pass
        finally:
            await enhanced.aclose()

        # ── base slippage ────────────────────────────────────────────────────
        vol_mult = self._vol_multiplier(vol_regime)
        effective_daily_vol = daily_volume_usd or 10_000_000.0
        order_size_pct = (order_size_usd / effective_daily_vol) * 100.0

        # Scale by how large the order is relative to visible book depth
        depth_scale = max(1.0, order_size_usd / max(book_depth_usd, 1.0))

        base_slip = SlippageCalculator.calculate_slippage(
            order_size_pct=order_size_pct * depth_scale,
            bid_ask_spread_pct=spread_pct,
            liquidity_score=liquidity_score,
            model=self.model,
        ) * vol_mult

        # Side asymmetry: exits (target/stop) typically have slightly less
        # market-impact than entries because they're limit-friendly
        entry_slip = base_slip
        exit_slip = base_slip * 0.6

        exec_metrics = SlippageCalculator.adjust_prices(
            setup.entry_price,
            setup.target_price,
            setup.stop_loss,
            entry_slip,
            exit_slip,
            exit_slip,
            setup.side,
        )

        raw_rr = _raw_rr(setup)
        adj_rr = exec_metrics.adjusted_rr_ratio

        return SlippageReport(
            symbol=market.symbol,
            side=setup.side,
            order_size_usd=order_size_usd,
            raw_entry=setup.entry_price,
            raw_target=setup.target_price,
            raw_stop=setup.stop_loss,
            raw_rr=round(raw_rr, 4),
            entry_slippage_pct=round(entry_slip, 6),
            target_slippage_pct=round(exit_slip, 6),
            stop_slippage_pct=round(exit_slip, 6),
            total_slippage_pct=round(exec_metrics.total_slippage_pct, 6),
            adj_entry=round(exec_metrics.adjusted_entry_price, 8),
            adj_target=round(exec_metrics.adjusted_target_price, 8),
            adj_stop=round(exec_metrics.adjusted_stop_price, 8),
            adj_rr=round(adj_rr, 4),
            rr_degradation=round(raw_rr - adj_rr, 4),
            is_still_viable=adj_rr >= self.min_viable_rr,
            model_used=self.model.value,
            volatility_regime=vol_regime,
            liquidity_score=round(liquidity_score, 2),
            spread_pct=round(spread_pct, 6),
        )

    async def estimate_from_params(
        self,
        *,
        symbol: str,
        side: str,
        entry: float,
        target: float,
        stop: float,
        order_size_usd: float = 1000.0,
        daily_volume_usd: float | None = None,
    ) -> SlippageReport:
        """Convenience wrapper that builds a minimal TradeSetup + MarketMeta
        from raw price parameters — useful for the standalone CLI command."""
        from futures_analyzer.analysis.models import (
            TradeSetup as TS,
            MarketMeta as MM,
            QualityLabel,
        )
        from datetime import UTC, datetime

        if side == "long":
            risk = max(entry - stop, 1e-9)
            reward = max(target - entry, 0.0)
        else:
            risk = max(stop - entry, 1e-9)
            reward = max(entry - target, 0.0)
        rr = reward / risk

        setup = TS(
            side=side,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            confidence=0.5,
            quality_label=QualityLabel.MEDIUM,
            rationale="manual",
            risk_reward_ratio=rr,
            stop_distance_pct=abs(entry - stop) / entry * 100,
            target_distance_pct=abs(target - entry) / entry * 100,
        )
        meta = MM(symbol=symbol, mark_price=entry, as_of=datetime.now(UTC))
        return await self.estimate(setup, meta, order_size_usd=order_size_usd, daily_volume_usd=daily_volume_usd)
