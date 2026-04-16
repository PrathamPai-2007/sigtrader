"""Structured LONG pipeline debug logger.

Tracks how many LONG candidates survive each filter stage per backtest run.
Aggregates counts — does NOT log per-candle to avoid spam.

Usage
-----
    from futures_analyzer.analysis.long_pipeline_log import LongPipelineLog

    log = LongPipelineLog()
    log.generated()
    log.rejected_macro("higher_trend=0.000, context_trend=-0.1")
    log.passed_macro()
    ...
    print(log.summary())          # human-readable
    print(log.as_dict())          # machine-readable

The log instance is meant to be created once per backtest run and passed
into the pipeline.  It is intentionally easy to remove: delete the import
and the two call-sites in runner.py and scorer.py.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class LongPipelineLog:
    """Aggregated LONG candidate counts per pipeline stage."""

    # ── Stage counters ────────────────────────────────────────────────────────
    generated: int = 0
    after_macro: int = 0
    after_regime: int = 0
    after_evidence: int = 0
    after_logistic: int = 0
    after_long_filters: int = 0
    entered: int = 0

    # ── Rejection reason tallies (one bucket per stage) ───────────────────────
    rejected_macro_reasons: Counter[str] = field(default_factory=Counter)
    rejected_regime_reasons: Counter[str] = field(default_factory=Counter)
    rejected_evidence_reasons: Counter[str] = field(default_factory=Counter)
    rejected_logistic_reasons: Counter[str] = field(default_factory=Counter)
    rejected_long_filter_reasons: Counter[str] = field(default_factory=Counter)
    rejected_final_reasons: Counter[str] = field(default_factory=Counter)

    # ── Stage recording helpers ───────────────────────────────────────────────

    def record_generated(self) -> None:
        self.generated += 1

    # --- macro filter ---
    def record_passed_macro(self) -> None:
        self.after_macro += 1

    def record_rejected_macro(self, reason: str) -> None:
        self.rejected_macro_reasons[reason] += 1

    # --- regime filter ---
    def record_passed_regime(self) -> None:
        self.after_regime += 1

    def record_rejected_regime(self, reason: str) -> None:
        self.rejected_regime_reasons[reason] += 1

    # --- evidence filter ---
    def record_passed_evidence(self) -> None:
        self.after_evidence += 1

    def record_rejected_evidence(self, reason: str) -> None:
        self.rejected_evidence_reasons[reason] += 1

    # --- logistic / confidence scoring ---
    def record_passed_logistic(self) -> None:
        self.after_logistic += 1

    def record_rejected_logistic(self, reason: str) -> None:
        self.rejected_logistic_reasons[reason] += 1

    # --- enhanced LONG-specific filters ---
    def record_passed_long_filters(self) -> None:
        self.after_long_filters += 1

    def record_rejected_long_filters(self, reason: str) -> None:
        self.rejected_long_filter_reasons[reason] += 1

    # --- final entry ---
    def record_entered(self) -> None:
        self.entered += 1

    def record_rejected_final(self, reason: str) -> None:
        self.rejected_final_reasons[reason] += 1

    # ── Output helpers ────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        """Machine-readable summary dict."""
        return {
            "LONG_PIPELINE_DEBUG": {
                "generated": self.generated,
                "after_macro": self.after_macro,
                "after_regime": self.after_regime,
                "after_evidence": self.after_evidence,
                "after_logistic": self.after_logistic,
                "after_long_filters": self.after_long_filters,
                "entered": self.entered,
            },
            "rejection_reasons": {
                "macro": dict(self.rejected_macro_reasons.most_common(5)),
                "regime": dict(self.rejected_regime_reasons.most_common(5)),
                "evidence": dict(self.rejected_evidence_reasons.most_common(5)),
                "logistic": dict(self.rejected_logistic_reasons.most_common(5)),
                "long_filters": dict(self.rejected_long_filter_reasons.most_common(5)),
                "final": dict(self.rejected_final_reasons.most_common(5)),
            },
        }

    def summary(self) -> str:
        """Human-readable pipeline funnel."""
        d = self.as_dict()["LONG_PIPELINE_DEBUG"]
        lines = [
            "",
            "╔══════════════════════════════════════════╗",
            "║         LONG PIPELINE DEBUG SUMMARY      ║",
            "╠══════════════════════════════════════════╣",
            f"║  generated          : {d['generated']:>6}             ║",
            f"║  after_macro        : {d['after_macro']:>6}             ║",
            f"║  after_regime       : {d['after_regime']:>6}             ║",
            f"║  after_evidence     : {d['after_evidence']:>6}             ║",
            f"║  after_logistic     : {d['after_logistic']:>6}             ║",
            f"║  after_long_filters : {d['after_long_filters']:>6}             ║",
            f"║  entered            : {d['entered']:>6}             ║",
            "╠══════════════════════════════════════════╣",
        ]

        def _section(title: str, counter: Counter) -> list[str]:
            if not counter:
                return []
            out = [f"║  {title}:"]
            for reason, count in counter.most_common(5):
                short = (reason[:36] + "…") if len(reason) > 37 else reason
                out.append(f"║    [{count:>4}] {short}")
            return out

        lines += _section("Macro rejections", self.rejected_macro_reasons)
        lines += _section("Regime rejections", self.rejected_regime_reasons)
        lines += _section("Evidence rejections", self.rejected_evidence_reasons)
        lines += _section("Logistic rejections", self.rejected_logistic_reasons)
        lines += _section("Long-filter rejections", self.rejected_long_filter_reasons)
        lines += _section("Final rejections", self.rejected_final_reasons)
        lines.append("╚══════════════════════════════════════════╝")
        return "\n".join(lines)
