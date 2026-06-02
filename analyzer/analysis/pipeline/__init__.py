"""Signal → confidence pipeline stages.

Pipeline order:
  regime.py        — classify market regime from multi-timeframe candles
  normalization.py — scale raw indicators to [0, 1] signal strengths
  evidence.py      — compute weighted evidence from normalized signals
  confidence.py    — map evidence score → confidence via logistic sigmoid
  decay.py         — apply time-based freshness decay to setup confidence
"""
from analyzer.analysis.pipeline.regime import (
    PerTFRegime,
    RegimeResult,
    classify_regime,
    classify_regime_consensus,
)
from analyzer.analysis.pipeline.normalization import (
    normalize_distension,
    normalize_signals,
    _funding_momentum,
    _oi_funding_biases,
)
from analyzer.analysis.pipeline.evidence import (
    compute_graded_evidence,
    _regime_weight_profile,
    _regime_alignment,
    _regime_penalty,
)
from analyzer.analysis.pipeline.confidence import (
    logistic_confidence,
    logistic_confidence_from_config,
)
from analyzer.analysis.pipeline.decay import (
    compute_ttl,
    apply_decay,
)
