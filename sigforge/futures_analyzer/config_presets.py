"""Pre-configured strategy presets to reduce configuration complexity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from futures_analyzer.analysis.models import MarketMode, StrategyStyle


class StrategyPreset(str, Enum):
    """Pre-configured strategy presets."""
    SCALPER = "scalper"  # High frequency, tight stops, small targets
    POSITION_TRADER = "position_trader"  # Long-term, macro trends


@dataclass
class PresetConfig:
    """Configuration preset with sensible defaults."""
    name: str
    description: str
    entry_timeframe: str
    trigger_timeframe: str
    context_timeframe: str
    higher_timeframe: str
    lookback_bars: int
    min_confidence: float
    min_quality: float
    min_rr_ratio: float
    max_stop_distance_pct: float
    min_evidence_agreement: int
    min_evidence_edge: int
    target_cap_atr_mult: float
    leverage_cap: int
    leverage_floor: int


# Pre-configured presets
PRESETS: dict[StrategyPreset, PresetConfig] = {
    StrategyPreset.SCALPER: PresetConfig(
        name="Scalper",
        description="High-frequency trend-following scalper — trades with momentum, avoids range chop",
        entry_timeframe="1m",
        trigger_timeframe="5m",
        context_timeframe="15m",
        higher_timeframe="1h",
        lookback_bars=300,
        min_confidence=0.44,
        min_quality=28.0,
        min_rr_ratio=1.5,
        max_stop_distance_pct=2.5,
        min_evidence_agreement=4,   # forces trend alignment — range rarely hits 4
        min_evidence_edge=1,        # must be clearly better than the opposing side
        target_cap_atr_mult=2.5,
        leverage_cap=5,
        leverage_floor=2,
    ),
    StrategyPreset.POSITION_TRADER: PresetConfig(
        name="Position Trader",
        description="Long-term trading following macro trends with conservative approach",
        entry_timeframe="5m",
        trigger_timeframe="15m",
        context_timeframe="1h",
        higher_timeframe="4h",
        lookback_bars=600,
        min_confidence=0.56,
        min_quality=45.0,
        min_rr_ratio=1.25,
        max_stop_distance_pct=2.0,
        min_evidence_agreement=2,
        min_evidence_edge=1,
        target_cap_atr_mult=1.3,
        leverage_cap=2,
        leverage_floor=2,
    ),
}


def get_preset(preset: StrategyPreset | str) -> PresetConfig:
    """Get a preset configuration by name or enum.
    
    Args:
        preset: StrategyPreset enum or string name
    
    Returns:
        PresetConfig with all settings
    
    Raises:
        ValueError: If preset not found
    """
    if isinstance(preset, str):
        try:
            preset = StrategyPreset(preset)
        except ValueError:
            available = ", ".join(p.value for p in StrategyPreset)
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    if preset not in PRESETS:
        raise ValueError(f"Preset {preset} not configured")
    
    return PRESETS[preset]


def list_presets() -> list[dict[str, Any]]:
    """List all available presets with descriptions.
    
    Returns:
        List of preset info dicts
    """
    return [
        {
            "name": preset.value,
            "display_name": config.name,
            "description": config.description,
        }
        for preset, config in PRESETS.items()
    ]
