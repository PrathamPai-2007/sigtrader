# Backward-compatibility shim — canonical location is analyzer.analysis.utils.indicators_compute
from analyzer.analysis.utils.indicators_compute import *  # noqa: F401, F403
from analyzer.analysis.utils.indicators_compute import compute_all_indicators, _compute_atr, _ema_value, _atr, _structure, _structure_biases, _momentum, _trend_strength, _volume_surge_ratio, _buy_sell_pressure, _range_span, _volume_divergence_penalties, _confirmation_penalty, _classify_regime
