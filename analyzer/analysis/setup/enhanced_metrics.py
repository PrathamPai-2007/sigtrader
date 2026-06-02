"""Calculate enhanced technical indicators and market microstructure metrics."""
from __future__ import annotations

from analyzer.analysis.models import Candle, EnhancedMetrics, MarketMeta
from analyzer.analysis.indicators import rsi, macd, stochastic, bollinger_bands, rsi_divergence


def _calculate_enhanced_metrics(
    entry_candles: list[Candle],
    trigger_candles: list[Candle],
    context_candles: list[Candle],
    market_meta: MarketMeta,
    order_book_data: dict[str, float] | None = None,
    volatility_data: dict[str, float] | None = None,
    vwap_data: dict[str, float] | None = None,
) -> EnhancedMetrics:
    """Calculate enhanced technical indicators and market microstructure metrics."""
    # Advanced technical indicators on entry timeframe
    rsi_14 = rsi(entry_candles, period=14)
    macd_result = macd(entry_candles, fast=12, slow=26, signal=9)
    stoch_result = stochastic(entry_candles, period=14, smooth_k=3, smooth_d=3)
    bb_result = bollinger_bands(entry_candles, period=20, std_dev=2.0)
    div_detected, div_type, _div_strength = rsi_divergence(entry_candles, period=14, lookback=30)

    # Order book metrics
    bid_ask_spread = order_book_data.get("spread_pct", 0.0) if order_book_data else 0.0
    bid_ask_ratio = order_book_data.get("bid_ask_ratio", 1.0) if order_book_data else 1.0
    ob_imbalance = order_book_data.get("imbalance", 0.0) if order_book_data else 0.0

    # Volatility metrics
    vol_rank = volatility_data.get("volatility_rank", 50.0) if volatility_data else 50.0
    vol_regime = volatility_data.get("volatility_regime", "normal") if volatility_data else "normal"

    # VWAP metrics
    vwap = vwap_data.get("vwap", entry_candles[-1].close if entry_candles else 0.0) if vwap_data else (entry_candles[-1].close if entry_candles else 0.0)
    vwap_dev = vwap_data.get("vwap_deviation_pct", 0.0) if vwap_data else 0.0

    liquidity_score = 50.0
    slippage_est = bid_ask_spread * 1.5

    return EnhancedMetrics(
        rsi_14=rsi_14,
        macd_value=macd_result.value,
        macd_signal=macd_result.signal or 0.0,
        macd_histogram=macd_result.histogram or 0.0,
        stochastic_k=stoch_result.value,
        stochastic_d=stoch_result.signal or 0.0,
        bollinger_upper=bb_result["upper"],
        bollinger_middle=bb_result["middle"],
        bollinger_lower=bb_result["lower"],
        bollinger_bandwidth_pct=bb_result["bandwidth"],
        bollinger_position=bb_result["position"],
        rsi_divergence=div_detected,
        rsi_divergence_type=div_type,
        bid_ask_spread_pct=bid_ask_spread,
        bid_ask_ratio=bid_ask_ratio,
        order_book_imbalance=ob_imbalance,
        volatility_rank=vol_rank,
        volatility_regime=vol_regime,
        vwap=vwap,
        vwap_deviation_pct=vwap_dev,
        liquidity_score=liquidity_score,
        slippage_estimate_pct=slippage_est,
    )
