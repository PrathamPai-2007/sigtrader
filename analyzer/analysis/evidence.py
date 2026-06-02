# Backward-compatibility shim — canonical location is analyzer.analysis.pipeline.evidence
from analyzer.analysis.pipeline.evidence import *  # noqa: F401, F403
from analyzer.analysis.pipeline.evidence import compute_graded_evidence, _regime_weight_profile, _regime_alignment, _regime_penalty
