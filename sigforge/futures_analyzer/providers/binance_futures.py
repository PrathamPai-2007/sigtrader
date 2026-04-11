from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime, timedelta
import math
import time
from typing import Any

import httpx

from futures_analyzer.analysis.models import Candle, MarketMeta, MarketMode
from futures_analyzer.analysis.validation import CandleValidator, DataValidationError


class BinanceFuturesProvider:
    base_url = "https://fapi.binance.com"

    # Maximum number of entries in each per-symbol cache dict.
    _CACHE_MAX_MARKET_META = 200
    _CACHE_MAX_CONTEXT = 500
    _CACHE_MAX_KLINES = 300

    # Binance Futures allows 1200 weight/min. We cap concurrent requests to
    # avoid bursting into the limit during large scan/find operations.
    _MAX_CONCURRENT_REQUESTS = 10

    def __init__(self, timeout: float = 20.0) -> None:
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._exchange_info_cache: dict[str, Any] | None = None
        self._market_meta_cache: dict[str, tuple[float, MarketMeta]] = {}
        self._historical_market_context_cache: dict[
            tuple[str, int, str], tuple[float, tuple[float | None, float | None, float | None]]
        ] = {}
        self._kline_cache: dict[tuple[str, str, int, int | None, int | None], tuple[float, list[Candle]]] = {}
        self._ticker_24h_cache: tuple[float, list[dict[str, Any]]] | None = None
        # Semaphore to cap concurrent outbound requests and stay within
        # Binance's 1200 weight/min rate limit during large scans.
        self._request_semaphore = asyncio.Semaphore(self._MAX_CONCURRENT_REQUESTS)
        # Lazy import to avoid circular dependency
        from futures_analyzer.providers.microstructure import EnhancedDataProvider
        self._enhanced_provider = EnhancedDataProvider(timeout=timeout)

    @staticmethod
    def _evict_oldest(cache: dict, max_size: int) -> None:
        """Remove the oldest half of entries when cache exceeds max_size.
        
        Thread-safe: Creates a snapshot of keys before deletion to avoid
        'dictionary changed size during iteration' errors.
        """
        if len(cache) >= max_size:
            # Create snapshot of cache items to avoid modification during iteration
            cache_snapshot = list(cache.items())
            sorted_items = sorted(cache_snapshot, key=lambda item: item[1][0])
            keys_to_delete = [k for k, _ in sorted_items[: max_size // 2]]
            for k in keys_to_delete:
                cache.pop(k, None)  # Use pop with default to handle concurrent deletions

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._enhanced_provider.aclose()

    @staticmethod
    def _cache_hit(cache_entry, ttl_seconds: float):
        if cache_entry is None:
            return None
        ts, value = cache_entry
        if (time.monotonic() - ts) > ttl_seconds:
            return None
        return copy.deepcopy(value)

    async def _get_json_with_retry(self, path: str, *, params: dict[str, Any] | None = None, attempts: int = 4) -> Any:
        delay = 0.5
        last_error: Exception | None = None
        for i in range(attempts):
            try:
                resp = await self._client.get(path, params=params)
                # Handle rate limiting explicitly with longer backoff
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", delay * 4))
                    wait = min(retry_after, 60.0)
                    if i < attempts - 1:
                        await asyncio.sleep(wait)
                        delay = min(delay * 2, 16.0)
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if i >= attempts - 1:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
            except Exception as exc:
                last_error = exc
                if i >= attempts - 1:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 4.0)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Unexpected request failure")

    async def fetch_exchange_info(self) -> dict[str, Any]:
        if self._exchange_info_cache is None:
            info = await self._get_json_with_retry("/fapi/v1/exchangeInfo")
            if not isinstance(info, dict):
                raise ValueError("Invalid exchange info payload")
            if not isinstance(info.get("symbols"), list):
                raise ValueError("Exchange info payload is missing the symbols list")
            self._exchange_info_cache = info
        return self._exchange_info_cache

    async def fetch_24h_tickers(self) -> list[dict[str, Any]]:
        from futures_analyzer.config import load_app_config
        cached_tickers = self._cache_hit(
            self._ticker_24h_cache,
            load_app_config().cache.realtime_klines_ttl_seconds,
        )
        if cached_tickers is not None:
            return cached_tickers
        data = await self._get_json_with_retry("/fapi/v1/ticker/24hr")
        if not isinstance(data, list):
            raise ValueError("Invalid 24h ticker payload")
        tickers = [copy.deepcopy(row) for row in data if isinstance(row, dict)]
        self._ticker_24h_cache = (time.monotonic(), tickers)
        return copy.deepcopy(tickers)

    @staticmethod
    def _coerce_float(value: Any, *, field_name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {field_name} in Binance payload: {value!r}") from exc

    async def fetch_market_meta(self, symbol: str) -> MarketMeta:
        from futures_analyzer.config import load_app_config
        cached_meta = self._cache_hit(
            self._market_meta_cache.get(symbol),
            load_app_config().cache.market_meta_ttl_seconds,
        )
        if cached_meta is not None:
            return cached_meta
        info = await self.fetch_exchange_info()
        symbol_row = None
        for row in info.get("symbols", []):
            if row.get("symbol") == symbol:
                symbol_row = row
                break
        if symbol_row is None:
            raise ValueError(f"Symbol not found in Binance futures exchange info: {symbol}")
        tick_size = None
        step_size = None
        for filter_row in symbol_row.get("filters", []):
            if filter_row.get("filterType") == "PRICE_FILTER":
                try:
                    tick_size = float(filter_row.get("tickSize"))
                except (TypeError, ValueError):
                    tick_size = None
            if filter_row.get("filterType") == "LOT_SIZE":
                try:
                    step_size = float(filter_row.get("stepSize"))
                except (TypeError, ValueError):
                    step_size = None

        premium_result, funding_result, oi_result, oi_hist_result = await asyncio.gather(
            self._get_json_with_retry("/fapi/v1/premiumIndex", params={"symbol": symbol}),
            self._get_json_with_retry("/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 8}),
            self._get_json_with_retry("/fapi/v1/openInterest", params={"symbol": symbol}),
            self._get_json_with_retry(
                "/futures/data/openInterestHist",
                params={"symbol": symbol, "period": "5m", "limit": 2},
            ),
            return_exceptions=True,
        )

        if isinstance(premium_result, Exception) or not isinstance(premium_result, dict):
            raise ValueError("Invalid premium index payload")
        mark_price = self._coerce_float(premium_result.get("markPrice"), field_name="mark price")

        funding_rate = None
        funding_rate_history: list[float] = []
        if not isinstance(funding_result, Exception) and isinstance(funding_result, list) and funding_result:
            try:
                for row in funding_result:
                    funding_rate_history.append(float(row.get("fundingRate")))
                funding_rate = funding_rate_history[-1] if funding_rate_history else None
            except (TypeError, ValueError):
                funding_rate = None
                funding_rate_history = []

        open_interest = None
        if not isinstance(oi_result, Exception) and isinstance(oi_result, dict):
            try:
                open_interest = float(oi_result.get("openInterest"))
            except (TypeError, ValueError):
                open_interest = None

        oi_change_pct = None
        if not isinstance(oi_hist_result, Exception) and isinstance(oi_hist_result, list) and len(oi_hist_result) >= 2:
            try:
                prev = float(oi_hist_result[-2].get("sumOpenInterest"))
                curr = float(oi_hist_result[-1].get("sumOpenInterest"))
                if prev > 0:
                    oi_change_pct = ((curr / prev) - 1.0) * 100.0
            except (TypeError, ValueError):
                oi_change_pct = None

        meta = MarketMeta(
            symbol=symbol,
            tick_size=tick_size,
            step_size=step_size,
            mark_price=mark_price,
            funding_rate=funding_rate,
            funding_rate_history=funding_rate_history,
            open_interest=open_interest,
            open_interest_change_pct=oi_change_pct,
            as_of=datetime.now(UTC),
        )
        self._evict_oldest(self._market_meta_cache, self._CACHE_MAX_MARKET_META)
        self._market_meta_cache[symbol] = (time.monotonic(), meta)
        return meta

    async def fetch_historical_market_context(
        self,
        *,
        symbol: str,
        as_of: datetime,
        interval: str = "5m",
    ) -> tuple[float | None, float | None, float | None]:
        from futures_analyzer.config import load_app_config
        context_time = as_of.astimezone(UTC)
        rounded_key = int(context_time.timestamp() // 60)
        cache_key = (symbol, rounded_key, interval)
        cached = self._cache_hit(
            self._historical_market_context_cache.get(cache_key),
            load_app_config().cache.historical_klines_ttl_seconds,
        )
        if cached is not None:
            return cached

        funding_rate = None
        open_interest = None
        oi_change_pct = None
        end_ms = int(context_time.timestamp() * 1000)
        start_ms = int((context_time - timedelta(days=7)).timestamp() * 1000)
        try:
            funding_rows = await self._get_json_with_retry(
                "/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1, "endTime": end_ms, "startTime": start_ms},
            )
            if isinstance(funding_rows, list) and funding_rows:
                funding_rate = float(funding_rows[-1].get("fundingRate"))
        except Exception:
            funding_rate = None

        oi_period = self._open_interest_hist_period(interval)
        try:
            oi_hist = await self._get_json_with_retry(
                "/futures/data/openInterestHist",
                params={
                    "symbol": symbol,
                    "period": oi_period,
                    "limit": 2,
                    "endTime": end_ms,
                },
            )
            if isinstance(oi_hist, list) and oi_hist:
                open_interest = float(oi_hist[-1].get("sumOpenInterest"))
                if len(oi_hist) >= 2:
                    prev = float(oi_hist[-2].get("sumOpenInterest"))
                    curr = float(oi_hist[-1].get("sumOpenInterest"))
                    if prev > 0:
                        oi_change_pct = ((curr / prev) - 1.0) * 100.0
        except Exception:
            open_interest = None
            oi_change_pct = None

        context = (funding_rate, open_interest, oi_change_pct)
        self._evict_oldest(self._historical_market_context_cache, self._CACHE_MAX_CONTEXT)
        self._historical_market_context_cache[cache_key] = (time.monotonic(), context)
        return context

    @staticmethod
    def _open_interest_hist_period(interval: str) -> str:
        supported = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
        if interval in supported:
            return interval
        if interval == "3m":
            return "5m"
        if interval == "8h":
            return "6h"
        if interval in {"3d", "1w"}:
            return "1d"
        return "5m"

    async def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 300,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_required_candles: int = 30,
    ) -> list[Candle]:
        # Validate symbol
        symbol = symbol.strip().upper()
        is_valid, error = CandleValidator.validate_symbol(symbol)
        if not is_valid:
            raise DataValidationError(f"Invalid symbol: {error}")
        
        # Validate interval
        if interval not in {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"}:
            raise DataValidationError(f"Invalid interval: {interval}")
        
        from futures_analyzer.config import load_app_config
        capped_limit = max(1, min(limit, 1500))
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": capped_limit}
        if start_time is not None:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)
        start_ms = params.get("startTime")
        end_ms = params.get("endTime")
        cache_key = (symbol, interval, capped_limit, start_ms if isinstance(start_ms, int) else None, end_ms if isinstance(end_ms, int) else None)
        config = load_app_config().cache
        ttl = (
            config.historical_klines_ttl_seconds
            if start_time is not None or end_time is not None
            else config.realtime_klines_ttl_seconds
        )
        cached_klines = self._cache_hit(self._kline_cache.get(cache_key), ttl)
        if cached_klines is not None:
            return cached_klines
        data = await self._get_json_with_retry("/fapi/v1/klines", params=params)
        if not isinstance(data, list):
            return []
        candles: list[Candle] = []
        for row in data:
            if not isinstance(row, list) or len(row) < 7:
                continue
            try:
                candles.append(
                    Candle(
                        open_time=datetime.fromtimestamp(float(row[0]) / 1000, tz=UTC),
                        close_time=datetime.fromtimestamp(float(row[6]) / 1000, tz=UTC),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                )
            except (TypeError, ValueError, OSError):
                continue
        
        # Validate candle data
        is_valid, warnings = CandleValidator.validate_candles(
            candles,
            min_required=min_required_candles,
            symbol=symbol,
        )
        
        if not is_valid:
            raise DataValidationError(f"Invalid candle data for {symbol}: {warnings[0]}")
        
        if warnings:
            from futures_analyzer.logging import get_logger as _get_logger
            _log = _get_logger(__name__)
            for w in warnings:
                _log.warning("candle_validation.warning", symbol=symbol, detail=w)
        
        self._evict_oldest(self._kline_cache, self._CACHE_MAX_KLINES)
        self._kline_cache[cache_key] = (time.monotonic(), candles)
        return candles

    async def fetch_intraday_candidates(
        self,
        *,
        limit: int = 20,
        quote_asset: str = "USDT",
        min_quote_volume: float = 25_000_000.0,
    ) -> list[str]:
        capped_limit = max(1, min(limit, 100))
        exchange_info = await self.fetch_exchange_info()
        ticker_rows = await self.fetch_24h_tickers()

        eligible_symbols = {
            row.get("symbol")
            for row in exchange_info.get("symbols", [])
            if row.get("symbol")
            and row.get("status") == "TRADING"
            and row.get("contractType") == "PERPETUAL"
            and row.get("quoteAsset") == quote_asset
        }

        ranked: list[tuple[float, str]] = []
        for row in ticker_rows:
            symbol = row.get("symbol")
            if not isinstance(symbol, str) or symbol not in eligible_symbols:
                continue
            try:
                quote_volume = self._coerce_float(row.get("quoteVolume"), field_name="quote volume")
                last_price = self._coerce_float(row.get("lastPrice"), field_name="last price")
                high_price = self._coerce_float(row.get("highPrice"), field_name="high price")
                low_price = self._coerce_float(row.get("lowPrice"), field_name="low price")
                price_change_pct = abs(self._coerce_float(row.get("priceChangePercent"), field_name="price change percent"))
            except ValueError:
                continue
            if quote_volume < min_quote_volume or last_price <= 0:
                continue
            range_pct = max(((high_price - low_price) / last_price) * 100.0, 0.0)
            liquidity_multiplier = max(math.log10(max(quote_volume, 1.0)) - 6.0, 0.5)
            opportunity_score = ((price_change_pct * 0.55) + (range_pct * 0.45)) * liquidity_multiplier
            ranked.append((opportunity_score, symbol))

        ranked.sort(reverse=True)
        return [symbol for _, symbol in ranked[:capped_limit]]

    async def fetch_long_term_candidates(
        self,
        *,
        limit: int = 20,
        quote_asset: str = "USDT",
        min_quote_volume: float = 40_000_000.0,
        interval: str = "1d",
        lookback: int = 14,
        candidate_pool_limit: int = 30,
    ) -> list[str]:
        from futures_analyzer.config import load_app_config
        capped_limit = max(1, min(limit, 100))
        exchange_info = await self.fetch_exchange_info()
        ticker_rows = await self.fetch_24h_tickers()

        eligible_symbols = {
            row.get("symbol")
            for row in exchange_info.get("symbols", [])
            if row.get("symbol")
            and row.get("status") == "TRADING"
            and row.get("contractType") == "PERPETUAL"
            and row.get("quoteAsset") == quote_asset
        }

        liquid_rows: list[tuple[float, str]] = []
        for row in ticker_rows:
            symbol = row.get("symbol")
            if not isinstance(symbol, str) or symbol not in eligible_symbols:
                continue
            try:
                quote_volume = self._coerce_float(row.get("quoteVolume"), field_name="quote volume")
            except ValueError:
                continue
            if quote_volume < min_quote_volume:
                continue
            liquid_rows.append((quote_volume, symbol))

        liquid_rows.sort(reverse=True)
        shortlisted_symbols = [symbol for _, symbol in liquid_rows[: max(candidate_pool_limit, capped_limit)]]
        settings = load_app_config().market_mode_settings(MarketMode.LONG_TERM)
        ranked: list[tuple[float, str]] = []
        for symbol in shortlisted_symbols:
            candles = await self.fetch_klines(symbol=symbol, interval=interval, limit=lookback, min_required_candles=1)
            if len(candles) < 3:
                continue
            first_close = candles[0].close
            last_close = candles[-1].close
            if first_close <= 0 or last_close <= 0:
                continue
            closes = [candle.close for candle in candles]
            up_bars = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
            consistency = up_bars / max(len(closes) - 1, 1)
            net_return_pct = abs(((last_close / first_close) - 1.0) * 100.0)
            high = max(candle.high for candle in candles)
            low = min(candle.low for candle in candles)
            range_pct = max(((high - low) / last_close) * 100.0, 0.0)
            volume_proxy = next((volume for volume, name in liquid_rows if name == symbol), min_quote_volume)
            liquidity_multiplier = max(math.log10(max(volume_proxy, 1.0)) - 6.0, 0.5)
            opportunity_score = (
                (net_return_pct * settings.candidate_return_weight)
                + (range_pct * settings.candidate_range_weight)
                + (consistency * 100.0 * settings.candidate_consistency_weight)
            ) * liquidity_multiplier
            ranked.append((opportunity_score, symbol))

        ranked.sort(reverse=True)
        return [symbol for _, symbol in ranked[:capped_limit]]

    async def fetch_candidate_symbols(self, *, market_mode: MarketMode, limit: int = 20) -> list[str]:
        from futures_analyzer.config import load_app_config
        settings = load_app_config().market_mode_settings(market_mode)
        if market_mode == MarketMode.LONG_TERM:
            return await self.fetch_long_term_candidates(
                limit=limit,
                quote_asset=settings.candidate_quote_asset,
                min_quote_volume=settings.candidate_min_quote_volume,
                interval=settings.candidate_kline_interval,
                lookback=settings.candidate_kline_lookback,
                candidate_pool_limit=settings.candidate_pool_limit,
            )
        return await self.fetch_intraday_candidates(
            limit=limit,
            quote_asset=settings.candidate_quote_asset,
            min_quote_volume=settings.candidate_min_quote_volume,
        )
