"""Enhanced data sources for improved market analysis."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any

import httpx

from futures_analyzer.config import load_app_config


@dataclass
class OrderBookSnapshot:
    """Order book depth snapshot."""
    symbol: str
    timestamp: float
    bid_volume: float
    ask_volume: float
    bid_ask_ratio: float
    spread_pct: float
    top_bid: float
    top_ask: float
    imbalance: float  # -1 to 1, negative = more sell pressure


@dataclass
class VolatilityMetrics:
    """Volatility analysis metrics."""
    current_volatility: float
    volatility_rank: float  # 0-100 percentile
    volatility_trend: float  # -1 to 1, positive = increasing
    volatility_regime: str  # "low", "normal", "high", "extreme"


@dataclass
class LiquidityMetrics:
    """Liquidity analysis metrics."""
    bid_ask_spread_pct: float
    order_book_depth_20: float  # Total volume at top 20 levels
    liquidity_score: float  # 0-100
    slippage_estimate_pct: float  # Estimated slippage for 1% of daily volume


@dataclass
class MarketMicrostructure:
    """Market microstructure indicators."""
    order_book: OrderBookSnapshot
    volatility: VolatilityMetrics
    liquidity: LiquidityMetrics
    vwap: float  # Volume-weighted average price
    vwap_deviation_pct: float  # Current price deviation from VWAP


class EnhancedDataProvider:
    """Provides enhanced market data beyond basic OHLCV."""
    
    def __init__(self, timeout: float = 20.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)
        self._orderbook_cache: dict[str, tuple[float, OrderBookSnapshot]] = {}
        self._volatility_cache: dict[str, tuple[float, VolatilityMetrics]] = {}
        self._vwap_cache: dict[tuple[str, str], tuple[float, float]] = {}
    
    async def aclose(self) -> None:
        await self._client.aclose()
    
    @staticmethod
    def _cache_hit(cache_entry: tuple[float, Any] | None, ttl_seconds: float) -> Any | None:
        """Check if cache entry is still valid."""
        if cache_entry is None:
            return None
        ts, value = cache_entry
        if (time.monotonic() - ts) > ttl_seconds:
            return None
        return copy.deepcopy(value)
    
    async def fetch_order_book_snapshot(
        self,
        symbol: str,
        *,
        limit: int = 20,
        cache_ttl: float = 5.0,
    ) -> OrderBookSnapshot:
        """Fetch order book depth snapshot from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            limit: Number of levels to fetch (5, 10, 20, 50, 100, 500, 1000)
            cache_ttl: Cache time-to-live in seconds
        
        Returns:
            OrderBookSnapshot with bid/ask analysis
        """
        cached = self._cache_hit(self._orderbook_cache.get(symbol), cache_ttl)
        if cached is not None:
            return cached
        
        try:
            url = f"https://fapi.binance.com/fapi/v1/depth"
            params = {"symbol": symbol, "limit": limit}
            resp = await self._client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            if not bids or not asks:
                raise ValueError(f"Invalid order book data for {symbol}")
            
            # Calculate metrics
            bid_volume = sum(float(b[1]) for b in bids)
            ask_volume = sum(float(a[1]) for a in asks)
            bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
            
            top_bid = float(bids[0][0])
            top_ask = float(asks[0][0])
            spread_pct = ((top_ask - top_bid) / top_bid * 100) if top_bid > 0 else 0.0
            
            # Imbalance: -1 (all sell) to 1 (all buy)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.0
            
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=time.time(),
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                bid_ask_ratio=bid_ask_ratio,
                spread_pct=spread_pct,
                top_bid=top_bid,
                top_ask=top_ask,
                imbalance=imbalance,
            )
            
            self._orderbook_cache[symbol] = (time.monotonic(), snapshot)
            return copy.deepcopy(snapshot)
        
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch order book for {symbol}: {exc}") from exc
    
    async def fetch_volatility_metrics(
        self,
        symbol: str,
        candles: list[Any],
        *,
        cache_ttl: float = 60.0,
    ) -> VolatilityMetrics:
        """Calculate volatility metrics from candles.
        
        Args:
            symbol: Trading pair
            candles: List of Candle objects
            cache_ttl: Cache time-to-live in seconds
        
        Returns:
            VolatilityMetrics with current and historical volatility
        """
        cached = self._cache_hit(self._volatility_cache.get(symbol), cache_ttl)
        if cached is not None:
            return cached
        
        if len(candles) < 20:
            metrics = VolatilityMetrics(
                current_volatility=0.0,
                volatility_rank=50.0,
                volatility_trend=0.0,
                volatility_regime="normal",
            )
            self._volatility_cache[symbol] = (time.monotonic(), metrics)
            return metrics
        
        # Calculate returns
        closes = [c.close for c in candles]
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        
        # Current volatility (last 20 periods)
        recent_returns = returns[-20:]
        current_vol = (sum(r ** 2 for r in recent_returns) / len(recent_returns)) ** 0.5
        
        # Historical volatility (all periods)
        historical_vol = (sum(r ** 2 for r in returns) / len(returns)) ** 0.5
        
        # Volatility rank (percentile of current vs historical)
        vol_rank = (current_vol / historical_vol * 100) if historical_vol > 0 else 50.0
        vol_rank = min(100.0, max(0.0, vol_rank))
        
        # Volatility trend
        vol_trend = (current_vol - historical_vol) / historical_vol if historical_vol > 0 else 0.0
        
        # Regime classification
        if vol_rank > 75:
            regime = "extreme"
        elif vol_rank > 60:
            regime = "high"
        elif vol_rank > 40:
            regime = "normal"
        else:
            regime = "low"
        
        metrics = VolatilityMetrics(
            current_volatility=current_vol,
            volatility_rank=vol_rank,
            volatility_trend=vol_trend,
            volatility_regime=regime,
        )
        
        self._volatility_cache[symbol] = (time.monotonic(), metrics)
        return metrics
    
    async def fetch_liquidity_metrics(
        self,
        symbol: str,
        order_book: OrderBookSnapshot,
        daily_volume: float,
    ) -> LiquidityMetrics:
        """Calculate liquidity metrics.
        
        Args:
            symbol: Trading pair
            order_book: OrderBookSnapshot
            daily_volume: 24h trading volume in quote asset
        
        Returns:
            LiquidityMetrics with spread and depth analysis
        """
        # Bid-ask spread
        spread_pct = order_book.spread_pct
        
        # Order book depth (sum of bid and ask volumes at top 20 levels)
        depth_20 = order_book.bid_volume + order_book.ask_volume
        
        # Liquidity score (0-100)
        # Based on spread and depth
        spread_score = max(0.0, 100.0 - (spread_pct * 1000))  # Penalize wide spreads
        depth_score = min(100.0, (depth_20 / daily_volume * 100)) if daily_volume > 0 else 0.0
        liquidity_score = (spread_score * 0.4 + depth_score * 0.6)
        
        # Slippage estimate for 1% of daily volume
        slippage_volume = daily_volume * 0.01
        slippage_pct = (slippage_volume / depth_20 * spread_pct) if depth_20 > 0 else spread_pct
        
        return LiquidityMetrics(
            bid_ask_spread_pct=spread_pct,
            order_book_depth_20=depth_20,
            liquidity_score=liquidity_score,
            slippage_estimate_pct=slippage_pct,
        )
    
    async def calculate_vwap(
        self,
        symbol: str,
        candles: list[Any],
        *,
        cache_ttl: float = 60.0,
    ) -> float:
        """Calculate Volume-Weighted Average Price.
        
        Args:
            symbol: Trading pair
            candles: List of Candle objects
            cache_ttl: Cache time-to-live in seconds
        
        Returns:
            VWAP value
        """
        cache_key = (symbol, "vwap")
        cached = self._cache_hit(self._vwap_cache.get(cache_key), cache_ttl)
        if cached is not None:
            return cached
        
        if not candles:
            return 0.0
        
        typical_prices = [(c.high + c.low + c.close) / 3.0 for c in candles]
        volumes = [c.volume for c in candles]
        
        cumulative_tp_volume = sum(tp * vol for tp, vol in zip(typical_prices, volumes))
        cumulative_volume = sum(volumes)
        
        vwap = cumulative_tp_volume / cumulative_volume if cumulative_volume > 0 else candles[-1].close
        
        self._vwap_cache[cache_key] = (time.monotonic(), vwap)
        return vwap
    
    async def fetch_market_microstructure(
        self,
        symbol: str,
        candles: list[Any],
        daily_volume: float,
        *,
        orderbook_limit: int = 20,
    ) -> MarketMicrostructure:
        """Fetch comprehensive market microstructure data.
        
        Args:
            symbol: Trading pair
            candles: List of Candle objects
            daily_volume: 24h trading volume
            orderbook_limit: Order book depth limit
        
        Returns:
            MarketMicrostructure with all metrics
        """
        # Fetch in parallel
        order_book, volatility, vwap = await asyncio.gather(
            self.fetch_order_book_snapshot(symbol, limit=orderbook_limit),
            self.fetch_volatility_metrics(symbol, candles),
            self.calculate_vwap(symbol, candles),
        )
        
        liquidity = await self.fetch_liquidity_metrics(symbol, order_book, daily_volume)
        
        current_price = candles[-1].close if candles else 0.0
        vwap_deviation = ((current_price - vwap) / vwap * 100) if vwap > 0 else 0.0
        
        return MarketMicrostructure(
            order_book=order_book,
            volatility=volatility,
            liquidity=liquidity,
            vwap=vwap,
            vwap_deviation_pct=vwap_deviation,
        )
