"""Main 24/7 bot loop: scan → signal → size → execute."""
from __future__ import annotations

import asyncio
import os

from futures_analyzer.analysis.concurrency import ParallelAnalyzer
from futures_analyzer.config import load_app_config
from futures_analyzer.history.service import HistoryService

from bot.config import BotConfig
from execution.executor import OrderExecutor
from execution.monitor import PositionMonitor
from providers.coindcx import CoinDCXProvider


async def run_bot(config: BotConfig | None = None) -> None:
    cfg = config or BotConfig.load()

    provider = CoinDCXProvider(
        api_key=os.environ.get("COINDCX_API_KEY", ""),
        api_secret=os.environ.get("COINDCX_API_SECRET", ""),
    )
    executor = OrderExecutor(provider, dry_run=cfg.dry_run)
    monitor = PositionMonitor(executor)
    history = HistoryService()  # noqa: F841 — wired in Phase 2

    print(f"sigwatch starting — dry_run={cfg.dry_run}, symbols={cfg.symbols}")

    try:
        while True:
            # Phase 1: wire CoinDCXProvider.fetch_klines and run SetupAnalyzer
            # Phase 2: filter by cfg.min_quality / cfg.min_confidence, execute entries
            # Phase 3: enforce stops/targets via monitor
            await monitor.check_exits([])
            await asyncio.sleep(cfg.scan_interval_seconds)
    finally:
        await provider.aclose()
