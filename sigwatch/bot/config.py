"""Bot configuration loader."""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BotConfig:
    symbols: list[str]
    scan_interval_seconds: int
    capital_usdt: float
    dry_run: bool
    max_daily_loss_pct: float
    max_open_positions: int
    min_quality: float
    min_confidence: float

    @classmethod
    def load(cls, path: Path | None = None) -> "BotConfig":
        config_path = path or Path(__file__).parent.parent / "bot_config.json"
        with open(config_path) as f:
            data = json.load(f)
        return cls(**data)
