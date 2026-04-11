import asyncio
from datetime import UTC, datetime

import httpx
import pytest

from futures_analyzer.analysis.models import Candle, MarketMode
from futures_analyzer.providers.binance_futures import BinanceFuturesProvider



def test_provider_parses_exchange_filters_and_klines() -> None:
    provider = BinanceFuturesProvider()
    seen_params: dict[str, object] = {}

    async def fake_get_json_with_retry(path: str, *, params=None, attempts=4):
        if path.endswith("exchangeInfo"):
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        ],
                    }
                ]
            }
        if path.endswith("premiumIndex"):
            return {"markPrice": "65000.5"}
        if path.endswith("fundingRate"):
            return [{"fundingRate": "0.0001"}]
        if path.endswith("openInterest"):
            return {"openInterest": "12345.67"}
        if path.endswith("openInterestHist"):
            return [
                {"sumOpenInterest": "1000.0"},
                {"sumOpenInterest": "1020.0"},
            ]
        if path.endswith("klines"):
            seen_params.update(params or {})
            return [
                [1700000000000, "100", "101", "99", "100.5", "123", 1700000300000],
                [1700000300000, "100.5", "102", "100", "101.5", "111", 1700000600000],
            ]
        return {}

    provider._get_json_with_retry = fake_get_json_with_retry  # type: ignore[method-assign]

    async def run() -> None:
        meta = await provider.fetch_market_meta("BTCUSDT")
        assert meta.tick_size == 0.1
        assert meta.step_size == 0.001
        assert meta.mark_price == 65000.5
        assert meta.funding_rate == 0.0001
        assert meta.open_interest == 12345.67
        assert meta.open_interest_change_pct == pytest.approx(2.0, rel=1e-9)
        candles = await provider.fetch_klines(
            symbol="BTCUSDT",
            interval="5m",
            limit=2,
            start_time=datetime(2026, 1, 1, tzinfo=UTC),
            end_time=datetime(2026, 1, 2, tzinfo=UTC),
            min_required_candles=2,
        )
        assert len(candles) == 2
        assert candles[0].high == 101.0
        assert candles[1].close == 101.5
        assert seen_params["interval"] == "5m"
        assert "startTime" in seen_params
        assert "endTime" in seen_params
        await provider.aclose()

    asyncio.run(run())


def test_provider_skips_malformed_kline_rows_and_rejects_bad_mark_price() -> None:
    provider = BinanceFuturesProvider()

    async def fake_get_json_with_retry(path: str, *, params=None, attempts=4):
        if path.endswith("exchangeInfo"):
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        ],
                    }
                ]
            }
        if path.endswith("premiumIndex"):
            return {"markPrice": "not-a-number"}
        if path.endswith("klines"):
            return [
                [1700000000000, "100", "101", "99", "100.5", "123", 1700000300000],
                [1700000300000, "bad-open", "102", "100", "101.5", "111", 1700000600000],
                ["bad-time", "100.5", "102", "100", "101.5", "111", 1700000600000],
                [1700000600000, "101.5", "103", "101", "102.5", "150", 1700000900000],
            ]
        return {}

    provider._get_json_with_retry = fake_get_json_with_retry  # type: ignore[method-assign]

    async def run() -> None:
        with pytest.raises(ValueError, match="mark price"):
            await provider.fetch_market_meta("BTCUSDT")
        candles = await provider.fetch_klines(symbol="BTCUSDT", interval="5m", limit=4, min_required_candles=2)
        assert len(candles) == 2
        assert candles[0].close == 100.5
        assert candles[1].close == 102.5
        await provider.aclose()

    asyncio.run(run())


def test_provider_retry_exhaustion_surfaces_last_error(monkeypatch) -> None:
    provider = BinanceFuturesProvider()
    calls: list[str] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(f"sleep:{delay}")

    async def fake_get(path: str, params=None):
        calls.append(path)
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("futures_analyzer.providers.binance_futures.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(provider._client, "get", fake_get)

    async def run() -> None:
        with pytest.raises(httpx.ReadTimeout):
            await provider._get_json_with_retry("/fapi/v1/klines", attempts=3)
        assert calls.count("/fapi/v1/klines") == 3
        await provider.aclose()

    asyncio.run(run())


def test_provider_caches_recent_market_meta_and_klines() -> None:
    provider = BinanceFuturesProvider()
    calls = {"premium": 0, "klines": 0}

    async def fake_get_json_with_retry(path: str, *, params=None, attempts=4):
        if path.endswith("exchangeInfo"):
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        ],
                    }
                ]
            }
        if path.endswith("premiumIndex"):
            calls["premium"] += 1
            return {"markPrice": "65000.5"}
        if path.endswith("fundingRate"):
            return [{"fundingRate": "0.0001"}]
        if path.endswith("openInterest"):
            return {"openInterest": "12345.67"}
        if path.endswith("openInterestHist"):
            return [
                {"sumOpenInterest": "1000.0"},
                {"sumOpenInterest": "1020.0"},
            ]
        if path.endswith("klines"):
            calls["klines"] += 1
            return [
                [1700000000000, "100", "101", "99", "100.5", "123", 1700000300000],
                [1700000300000, "100.5", "102", "100", "101.5", "111", 1700000600000],
            ]
        return {}

    provider._get_json_with_retry = fake_get_json_with_retry  # type: ignore[method-assign]

    async def run() -> None:
        await provider.fetch_market_meta("BTCUSDT")
        await provider.fetch_market_meta("BTCUSDT")
        await provider.fetch_klines(symbol="BTCUSDT", interval="5m", limit=2, min_required_candles=2)
        await provider.fetch_klines(symbol="BTCUSDT", interval="5m", limit=2, min_required_candles=2)
        assert calls["premium"] == 1
        assert calls["klines"] == 1
        await provider.aclose()

    asyncio.run(run())


def test_provider_fetches_historical_market_context() -> None:
    provider = BinanceFuturesProvider()
    seen_params: dict[str, dict[str, object]] = {}

    async def fake_get_json_with_retry(path: str, *, params=None, attempts=4):
        seen_params[path] = dict(params or {})
        if path.endswith("fundingRate"):
            return [{"fundingRate": "0.0003"}]
        if path.endswith("openInterestHist"):
            return [
                {"sumOpenInterest": "1000.0"},
                {"sumOpenInterest": "1030.0"},
            ]
        return {}

    provider._get_json_with_retry = fake_get_json_with_retry  # type: ignore[method-assign]

    async def run() -> None:
        as_of = datetime(2026, 1, 2, tzinfo=UTC)
        funding_rate, open_interest, oi_change_pct = await provider.fetch_historical_market_context(
            symbol="BTCUSDT",
            as_of=as_of,
            interval="15m",
        )
        assert funding_rate == 0.0003
        assert open_interest == 1030.0
        assert oi_change_pct == pytest.approx(3.0, rel=1e-9)
        assert seen_params["/fapi/v1/fundingRate"]["endTime"] == int(as_of.timestamp() * 1000)
        assert seen_params["/futures/data/openInterestHist"]["period"] == "15m"
        await provider.aclose()

    asyncio.run(run())


def test_provider_ranks_intraday_candidates_from_liquid_movers() -> None:
    provider = BinanceFuturesProvider()

    async def fake_get_json_with_retry(path: str, *, params=None, attempts=4):
        if path.endswith("exchangeInfo"):
            return {
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                    {"symbol": "ETHUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                    {"symbol": "XRPUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                    {"symbol": "BNBBUSD", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "BUSD"},
                ]
            }
        if path.endswith("ticker/24hr"):
            return [
                {
                    "symbol": "BTCUSDT",
                    "quoteVolume": "1800000000",
                    "lastPrice": "68000",
                    "highPrice": "69500",
                    "lowPrice": "66000",
                    "priceChangePercent": "3.8",
                },
                {
                    "symbol": "ETHUSDT",
                    "quoteVolume": "950000000",
                    "lastPrice": "3500",
                    "highPrice": "3680",
                    "lowPrice": "3370",
                    "priceChangePercent": "4.6",
                },
                {
                    "symbol": "XRPUSDT",
                    "quoteVolume": "3000000",
                    "lastPrice": "0.60",
                    "highPrice": "0.64",
                    "lowPrice": "0.57",
                    "priceChangePercent": "5.0",
                },
                {
                    "symbol": "BNBBUSD",
                    "quoteVolume": "450000000",
                    "lastPrice": "620",
                    "highPrice": "640",
                    "lowPrice": "600",
                    "priceChangePercent": "2.0",
                },
            ]
        return {}

    provider._get_json_with_retry = fake_get_json_with_retry  # type: ignore[method-assign]

    async def run() -> None:
        symbols = await provider.fetch_intraday_candidates(limit=3)
        assert symbols == ["ETHUSDT", "BTCUSDT"]
        await provider.aclose()

    asyncio.run(run())


def test_provider_ranks_long_term_candidates_from_trend_quality() -> None:
    provider = BinanceFuturesProvider()

    async def fake_get_json_with_retry(path: str, *, params=None, attempts=4):
        if path.endswith("exchangeInfo"):
            return {
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                    {"symbol": "ETHUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                    {"symbol": "SOLUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT"},
                ]
            }
        if path.endswith("ticker/24hr"):
            return [
                {"symbol": "BTCUSDT", "quoteVolume": "1800000000"},
                {"symbol": "ETHUSDT", "quoteVolume": "1100000000"},
                {"symbol": "SOLUSDT", "quoteVolume": "600000000"},
            ]
        return {}

    async def fake_fetch_klines(*, symbol: str, interval: str, limit: int = 300, start_time=None, end_time=None, min_required_candles: int = 30):
        base = datetime(2026, 1, 1, tzinfo=UTC)
        if symbol == "ETHUSDT":
            closes = [100, 104, 108, 111, 114]
        elif symbol == "BTCUSDT":
            closes = [100, 102, 103, 104, 106]
        else:
            closes = [100, 98, 102, 97, 103]
        candles = []
        for idx, close in enumerate(closes):
            candles.append(
                Candle(
                    open_time=base,
                    close_time=base,
                    open=float(closes[max(idx - 1, 0)]),
                    high=float(close + 1),
                    low=float(close - 1),
                    close=float(close),
                    volume=1000.0 + idx,
                )
            )
        return candles

    provider._get_json_with_retry = fake_get_json_with_retry  # type: ignore[method-assign]
    provider.fetch_klines = fake_fetch_klines  # type: ignore[method-assign]

    async def run() -> None:
        symbols = await provider.fetch_candidate_symbols(market_mode=MarketMode.LONG_TERM, limit=3)
        assert symbols[:2] == ["ETHUSDT", "BTCUSDT"]
        await provider.aclose()

    asyncio.run(run())
