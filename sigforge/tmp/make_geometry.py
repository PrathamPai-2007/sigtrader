"""Script: extract geometry functions from scorer.py and write geometry.py"""
import re

with open("futures_analyzer/analysis/scorer.py", encoding="utf-8") as f:
    content = f.read()

# --- geometry functions (find_swing_points … end of place_entry_stop_target) ---
geo_start = content.find("\ndef find_swing_points(")
geo_end   = content.find("\ndef geometry_quality_score(")
geometry_fns = content[geo_start:geo_end].strip()

# --- geometry_quality_score ---
gqs_start = geo_end
rest = content[gqs_start + 1:]
next_top = re.search(r"\ndef [a-z_]|\nclass [A-Z_]", rest)
gqs_end = gqs_start + 1 + next_top.start() if next_top else gqs_start + 3000
gqs_fn = content[gqs_start:gqs_end].strip()

header = (
    '"""Geometry module — entry/stop/target placement and quality scoring.\n'
    "\n"
    "All functions are pure: inputs are prices, ATR, swing points, and VWAP/VP\n"
    "structure from IndicatorBundle. No dependency on evidence, confidence, or\n"
    "filter layers.\n"
    '"""\n'
    "from __future__ import annotations\n"
    "\n"
    "from futures_analyzer.analysis.models import (\n"
    "    Candle,\n"
    "    EntryGeometry,\n"
    "    IndicatorBundle,\n"
    "    MarketRegime,\n"
    "    QualityLabel,\n"
    "    StrategyStyle,\n"
    "    SwingPoints,\n"
    ")\n"
    "from futures_analyzer.analysis.indicators import _swing_pivots\n"
    "from futures_analyzer.analysis.scoring.utils import _clamp, _quantize, _quality_label\n"
    "from futures_analyzer.config import AppConfig, load_app_config\n"
    "\n"
)

output = header + geometry_fns + "\n\n\n" + gqs_fn + "\n"

with open("futures_analyzer/analysis/geometry.py", "w", encoding="utf-8") as f:
    f.write(output)

print(f"geometry.py written: {len(output.splitlines())} lines")
print(f"  geometry_fns: {len(geometry_fns.splitlines())} lines")
print(f"  gqs_fn: {len(gqs_fn.splitlines())} lines")
