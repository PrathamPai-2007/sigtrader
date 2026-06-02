# Backward-compatibility shim — canonical location is analyzer.config.presets
from analyzer.config.presets import *  # noqa: F401, F403
from analyzer.config.presets import PresetProfile, PresetConfig, StrategyPreset, PRESETS, get_preset, list_presets
