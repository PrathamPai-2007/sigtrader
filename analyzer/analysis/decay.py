# Backward-compatibility shim — canonical location is analyzer.analysis.pipeline.decay
from analyzer.analysis.pipeline.decay import *  # noqa: F401, F403
from analyzer.analysis.pipeline.decay import compute_ttl, apply_decay
