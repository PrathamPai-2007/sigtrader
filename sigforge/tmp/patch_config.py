"""Patch config.py: insert TimeframeConfig, MarketModeTuning, StyleTuning before AppConfig,
and add timeframe_for / style_tuning / market_mode_settings methods to AppConfig."""

with open("futures_analyzer/config.py", encoding="utf-8") as f:
    content = f.read()

# ── 1. Insert the 3 missing classes before AppConfig ──────────────────────────
MARKER = "    min_rr_ratio: float\n\n\nclass AppConfig(BaseModel):"

NEW_CLASSES = """\
    min_rr_ratio: float


class TimeframeConfig(BaseModel):
    profile_name: str = "auto_core"
    entry_timeframe: str = "5m"
    trigger_timeframe: str = "15m"
    context_timeframe: str = "1h"
    higher_timeframe: str = "4h"
    lookback_bars: int = 600


class MarketModeTuning(BaseModel):
    target_cap_atr_mult: float | None = None
    min_confidence: float | None = None
    max_stop_distance_pct: float | None = None
    min_evidence_agreement: int | None = None
    min_evidence_edge: int | None = None
    candidate_quote_asset: str = "USDT"
    candidate_min_quote_volume: float = 25_000_000.0
    candidate_pool_limit: int = 40
    candidate_kline_interval: str = "1d"
    candidate_kline_lookback: int = 14
    candidate_return_weight: float = 0.5
    candidate_range_weight: float = 0.3
    candidate_consistency_weight: float = 0.2

    @field_validator("min_confidence")
    @classmethod
    def _unit_interval_opt(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {v}")
        return v

    @field_validator("max_stop_distance_pct")
    @classmethod
    def _positive_pct_opt(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"max_stop_distance_pct must be positive, got {v}")
        return v

    @field_validator("candidate_return_weight", "candidate_range_weight", "candidate_consistency_weight")
    @classmethod
    def _unit_weight(cls, v: float, info: object) -> float:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "MarketModeTuning":
        total = (
            self.candidate_return_weight
            + self.candidate_range_weight
            + self.candidate_consistency_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"candidate weights must sum to 1.0, got {total:.6f}")
        return self


class StyleTuning(BaseModel):
    fallback_risk_reward: float
    target_cap_atr_mult: float
    ambition_penalty_start_atr: float
    ambition_penalty_slope: float
    min_confidence: float
    max_confidence: float = 1.0
    min_quality: float
    min_rr_ratio: float
    max_stop_distance_pct: float
    min_evidence_agreement: int
    min_evidence_edge: int
    atr_buffer_factor: float = 0.3

    @field_validator("min_confidence", "max_confidence")
    @classmethod
    def _unit_interval(cls, v: float, info: object) -> float:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_consistency(self) -> "StyleTuning":
        if self.min_confidence > self.max_confidence:
            raise ValueError(
                f"min_confidence ({self.min_confidence}) > max_confidence ({self.max_confidence})"
            )
        if self.min_rr_ratio > self.fallback_risk_reward:
            log.warning(
                "config.style_tuning_inconsistency",
                detail=(
                    f"min_rr_ratio ({self.min_rr_ratio}) > fallback_risk_reward "
                    f"({self.fallback_risk_reward}); may filter all setups"
                ),
            )
        return self

    @field_validator("min_quality")
    @classmethod
    def _quality_range(cls, v: float) -> float:
        if not (0.0 <= v <= 100.0):
            raise ValueError(f"min_quality must be in [0, 100], got {v}")
        return v

    @field_validator("min_rr_ratio", "fallback_risk_reward")
    @classmethod
    def _positive_ratio(cls, v: float, info: object) -> float:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if v <= 0:
            raise ValueError(f"{name} must be positive, got {v}")
        return v

    @field_validator("max_stop_distance_pct")
    @classmethod
    def _positive_pct(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_stop_distance_pct must be positive, got {v}")
        return v

    @field_validator("min_evidence_agreement", "min_evidence_edge")
    @classmethod
    def _non_negative_int(cls, v: int, info: object) -> int:
        name = info.field_name if hasattr(info, "field_name") else "field"
        if v < 0:
            raise ValueError(f"{name} must be >= 0, got {v}")
        return v


class AppConfig(BaseModel):\
"""

assert MARKER in content, f"MARKER not found. Occurrences of 'min_rr_ratio: float': {content.count('min_rr_ratio: float')}"
content = content.replace(MARKER, NEW_CLASSES, 1)
print("Step 1: 3 classes inserted OK")

# ── 2. Add the 3 missing methods to AppConfig ─────────────────────────────────
# Insert after the existing get_preset method
OLD_METHOD_BLOCK = """\
    def get_preset(self, preset_name: str) -> PresetConfig:
        \"\"\"Get preset configuration by name.\"\"\"
        if preset_name not in self.presets:
            available = ", ".join(self.presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        return self.presets[preset_name]"""

NEW_METHOD_BLOCK = """\
    def get_preset(self, preset_name: str) -> PresetConfig:
        \"\"\"Get preset configuration by name.\"\"\"
        if preset_name not in self.presets:
            available = ", ".join(self.presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        return self.presets[preset_name]

    def timeframe_for(self, market_mode: "MarketMode") -> TimeframeConfig:
        return self.market_modes[market_mode.value]

    def style_tuning(self, style: "StrategyStyle") -> StyleTuning:
        return self.styles[style.value]

    def market_mode_settings(self, market_mode: "MarketMode") -> MarketModeTuning:
        return self.market_mode_tuning[market_mode.value]\
"""

assert OLD_METHOD_BLOCK in content, "get_preset block not found"
content = content.replace(OLD_METHOD_BLOCK, NEW_METHOD_BLOCK, 1)
print("Step 2: 3 methods added to AppConfig OK")

# ── 3. Add market_modes, market_mode_tuning, styles fields to AppConfig ───────
OLD_FIELDS = """\
class AppConfig(BaseModel):
    presets: dict[str, PresetConfig]
    cache: CacheConfig = Field(default_factory=CacheConfig)
    strategy: StrategyConfig"""

NEW_FIELDS = """\
class AppConfig(BaseModel):
    presets: dict[str, PresetConfig]
    market_modes: dict[str, TimeframeConfig] = Field(default_factory=dict)
    market_mode_tuning: dict[str, MarketModeTuning] = Field(default_factory=dict)
    styles: dict[str, StyleTuning] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    strategy: StrategyConfig\
"""

assert OLD_FIELDS in content, "AppConfig fields block not found"
content = content.replace(OLD_FIELDS, NEW_FIELDS, 1)
print("Step 3: AppConfig fields added OK")

with open("futures_analyzer/config.py", "w", encoding="utf-8") as f:
    f.write(content)
print("config.py written successfully")
