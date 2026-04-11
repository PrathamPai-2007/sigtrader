"""Position monitor — watches open positions and triggers exits."""
from __future__ import annotations

from execution.executor import OrderExecutor


class PositionMonitor:
    """Polls or listens via WebSocket for price updates and enforces stop/target exits."""

    def __init__(self, executor: OrderExecutor) -> None:
        self._executor = executor

    async def check_exits(self, open_positions: list[dict]) -> None:
        """Check each open position against its stop and target. Close if hit."""
        raise NotImplementedError("Phase 3")
