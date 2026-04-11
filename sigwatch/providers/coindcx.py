"""CoinDCX data provider — implements the same interface as BinanceFuturesProvider."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import httpx

# Import shared models from sigforge
from futures_analyzer.analysis.models import Candle, MarketMeta, MarketMode


_BASE_URL = "https://api.coindcx.com"
_FUTURES_BASE_URL = "https://api.coindcx.com"  # update if CoinDCX exposes a separate futures host

_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1d",
}


class CoinDCXProvider:
    """Fetches market data from CoinDCX and maps it to sigforge's internal models."""

    def __init__(self, *, api_key: str = "", api_secret: str = "", timeout: float = 20.0) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── Public interface (mirrors BinanceFuturesProvider) ─────────────────────

    async def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 500,
        **_kwargs: Any,
    ) -> list[Candle]:
        """Fetch OHLCV candles for a symbol."""
        raise NotImplementedError("CoinDCXProvider.fetch_klines — to be implemented in Phase 1")

    async def fetch_market_meta(self, symbol: str) -> MarketMeta:
        """Fetch current market metadata (price, funding rate, OI, tick size)."""
        raise NotImplementedError("CoinDCXProvider.fetch_market_meta — to be implemented in Phase 1")

    async def fetch_24h_tickers(self) -> list[dict[str, Any]]:
        """Fetch 24h ticker data for all symbols."""
        raise NotImplementedError("CoinDCXProvider.fetch_24h_tickers — to be implemented in Phase 1")

    async def fetch_candidate_symbols(
        self,
        *,
        market_mode: MarketMode,
        limit: int = 20,
    ) -> list[str]:
        """Return the top liquid symbols to scan."""
        raise NotImplementedError("CoinDCXProvider.fetch_candidate_symbols — to be implemented in Phase 1")

    # ── Auth helpers ──────────────────────────────────────────────────────────

    def _sign(self, body: dict) -> str:
        payload = json.dumps(body, separators=(",", ":"))
        return hmac.new(
            self._api_secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, body: dict) -> dict[str, str]:
        return {
            "X-AUTH-APIKEY": self._api_key,
            "X-AUTH-SIGNATURE": self._sign(body),
            "Content-Type": "application/json",
        }
