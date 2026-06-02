# Backward-compatibility shim — canonical location is analyzer.analysis.utils.scoring.utils
from analyzer.analysis.utils.scoring.utils import *  # noqa: F401, F403
from analyzer.analysis.utils.scoring.utils import _clamp, _quantize, _quality_label, _quality_score_cap_from_confidence
