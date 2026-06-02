# Backward-compatibility shim — canonical location is analyzer.analysis.setup.geometry
from analyzer.analysis.setup.geometry import *  # noqa: F401, F403
from analyzer.analysis.setup.geometry import find_swing_points, select_best_stop, select_best_target, place_entry_stop_target, geometry_quality_score, _compute_entry, _build_stop_candidates, _build_target_candidates
