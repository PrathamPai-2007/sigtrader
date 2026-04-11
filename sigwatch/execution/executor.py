"""Order execution layer for CoinDCX."""
from __future__ import annotations

from providers.coindcx import CoinDCXProvider


class OrderExecutor:
    """Places, cancels, and tracks orders via the CoinDCX REST API."""

    def __init__(self, provider: CoinDCXProvider, *, dry_run: bool = True) -> None:
        self._provider = provider
        self._dry_run = dry_run

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        raise NotImplementedError("Phase 2")

    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        raise NotImplementedError("Phase 2")

    async def place_stop_limit_order(
        self, symbol: str, side: str, quantity: float, stop_price: float, limit_price: float
    ) -> dict:
        raise NotImplementedError("Phase 2")

    async def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError("Phase 2")

    async def get_open_orders(self, symbol: str) -> list[dict]:
        raise NotImplementedError("Phase 2")

    async def get_balance(self, asset: str) -> float:
        raise NotImplementedError("Phase 2")
