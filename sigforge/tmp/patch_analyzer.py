"""Patch scorer.py: store _preset_name and add get_execution_params."""

with open("futures_analyzer/analysis/scorer.py", encoding="utf-8") as f:
    content = f.read()

# ── Step 1: store preset in __init__ ─────────────────────────────────────────
OLD1 = "self._config = config or load_app_config()"
NEW1 = "self._preset_name: str | None = preset\n\n        self._config = config or load_app_config()"

assert content.count(OLD1) == 1, f"Step 1: expected 1 match, got {content.count(OLD1)}"
content = content.replace(OLD1, NEW1, 1)
print("Step 1: _preset_name stored OK")

# ── Step 2: add get_execution_params before _mode_params ─────────────────────
OLD2 = "    def _mode_params(self) -> dict[str, float]:"

NEW2 = """\
    def get_execution_params(self, side: str) -> dict:
        \"\"\"Return side-specific execution parameters from the active preset.

        Delegates to PresetConfig.execution_overrides for the given side.
        Returns an empty dict when no preset is active or no overrides are set.
        \"\"\"
        if self._preset_name is None:
            return {}
        try:
            preset = self._config.get_preset(self._preset_name)
            overrides = preset.execution_overrides
            side_cfg = overrides.long if side == "long" else overrides.short
            result: dict = {}
            if side_cfg.tp_rr is not None:
                result["tp_rr"] = side_cfg.tp_rr
            if side_cfg.min_rr_ratio is not None:
                result["min_rr_ratio"] = side_cfg.min_rr_ratio
            if side_cfg.min_quality is not None:
                result["min_quality"] = side_cfg.min_quality
            return result
        except Exception:
            return {}

    def _mode_params(self) -> dict[str, float]:\
"""

assert content.count(OLD2) == 1, f"Step 2: expected 1 match, got {content.count(OLD2)}"
content = content.replace(OLD2, NEW2, 1)
print("Step 2: get_execution_params added OK")

with open("futures_analyzer/analysis/scorer.py", "w", encoding="utf-8") as f:
    f.write(content)
print("scorer.py written")
