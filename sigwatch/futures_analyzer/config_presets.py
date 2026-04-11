"""Pre-configured strategy presets to reduce configuration complexity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from futures_analyzer.analysis.models import MarketMode, StrategyStyle


class StrategyPreset(str, Enum):
    """Pre-configured strategy presets."""
    SCALPER = "scalper"  # High frequency, tight stops, small targets
    DAY_TRADER = "day_trader"  # Intraday, medium timeframes
    SWING_TRADER = "swing_trader"  # Multi-day, larger moves
    POSITION_TRADER = "position_trader"  # Long-term, macro trends
    CONSERVATIVE = "conservative"  # Low risk, high quality only
    AGGRESSIVE = "aggressive"  # Higher risk, more opportunities


@dataclass
class PresetConfig:
    """Configuration preset with sensible defaults."""
    name: str
    description: str
    style: StrategyStyle
    market_mode: MarketMode
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
        description="High-frequency trading with tight stops and small targets",
        style=StrategyStyle.AGGRESSIVE,
        market_mode=MarketMode.INTRADAY,
        entry_timeframe="1m",
        trigger_timeframe="5m",
        context_timeframe="15m",
        higher_timeframe="1h",
        lookback_bars=300,
        min_confidence=0.55,
        min_quality=45.0,
        min_rr_ratio=1.2,
        max_stop_distance_pct=0.5,
        min_evidence_agreement=2,
        min_evidence_edge=1,
        target_cap_atr_mult=1.5,
        leverage_cap=5,
        leverage_floor=2,
    ),
    StrategyPreset.DAY_TRADER: PresetConfig(
        name="Day Trader",
        description="Intraday trading with medium timeframes",
        style=StrategyStyle.AGGRESSIVE,
        market_mode=MarketMode.INTRADAY,
        entry_timeframe="5m",
        trigger_timeframe="15m",
        context_timeframe="1h",
        higher_timeframe="4h",
        lookback_bars=600,
        min_confidence=0.60,
        min_quality=50.0,
        min_rr_ratio=1.5,
        max_stop_distance_pct=1.0,
        min_evidence_agreement=3,
        min_evidence_edge=1,
        target_cap_atr_mult=2.0,
        leverage_cap=6,
        leverage_floor=2,
    ),
    StrategyPreset.SWING_TRADER: PresetConfig(
        name="Swing Trader",
        description="Multi-day trading with larger moves",
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        entry_timeframe="1h",
        trigger_timeframe="4h",
        context_timeframe="1d",
        higher_timeframe="1w",
        lookback_bars=600,
        min_confidence=0.65,
        min_quality=55.0,
        min_rr_ratio=2.0,
        max_stop_distance_pct=2.0,
        min_evidence_agreement=4,
        min_evidence_edge=2,
        target_cap_atr_mult=3.0,
        leverage_cap=4,
        leverage_floor=1,
    ),
    StrategyPreset.POSITION_TRADER: PresetConfig(
        name="Position Trader",
        description="Long-term trading following macro trends",
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.LONG_TERM,
        entry_timeframe="1h",
        trigger_timeframe="4h",
        context_timeframe="1d",
        higher_timeframe="1w",
        lookback_bars=900,
        min_confidence=0.70,
        min_quality=60.0,
        min_rr_ratio=2.5,
        max_stop_distance_pct=3.0,
        min_evidence_agreement=5,
        min_evidence_edge=2,
        target_cap_atr_mult=4.0,
        leverage_cap=3,
        leverage_floor=1,
    ),
    StrategyPreset.CONSERVATIVE: PresetConfig(
        name="Conservative",
        description="Low-risk strategy with strict filters",
        style=StrategyStyle.CONSERVATIVE,
        market_mode=MarketMode.INTRADAY,
        entry_timeframe="5m",
        trigger_timeframe="15m",
        context_timeframe="1h",
        higher_timeframe="4h",
        lookback_bars=600,
        min_confidence=0.75,
        min_quality=65.0,
        min_rr_ratio=2.5,
        max_stop_distance_pct=1.5,
        min_evidence_agreement=5,
        min_evidence_edge=2,
        target_cap_atr_mult=2.5,
        leverage_cap=2,
        leverage_floor=1,
    ),
    StrategyPreset.AGGRESSIVE: PresetConfig(
        name="Aggressive",
        description="High-risk strategy with relaxed filters",
        style=StrategyStyle.AGGRESSIVE,
        market_mode=MarketMode.INTRADAY,
        entry_timeframe="5m",
        trigger_timeframe="15m",
        context_timeframe="1h",
        higher_timeframe="4h",
        lookback_bars=600,
        min_confidence=0.50,
        min_quality=40.0,
        min_rr_ratio=1.2,
        max_stop_distance_pct=2.0,
        min_evidence_agreement=2,
        min_evidence_edge=0,
        target_cap_atr_mult=3.0,
        leverage_cap=6,
        leverage_floor=3,
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
            "style": config.style.value,
            "market_mode": config.market_mode.value,
        }
        for preset, config in PRESETS.items()
    ]
