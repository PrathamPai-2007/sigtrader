"""
Phase 7: Convert early hard rejections to soft penalties.
Preserves final safety gates: confidence < 0.45, rr < 0.8, quality < 25.
"""

with open("futures_analyzer/analysis/scorer.py", encoding="utf-8") as f:
    content = f.read()

# ── Locate the primary filter gate block ─────────────────────────────────────
# The block starts with "reasons: list[str] = []" (inside _build_side, after
# _filter_params is assigned) and ends just before "tradable_reasons = reasons".

OLD_GATE = """\
        reasons: list[str] = []



        # --- Confidence floor (regime-aware) ---

        if regime == MarketRegime.RANGE:

            if confidence < 0.55:

                reasons.append(f"confidence {confidence:.2f} below threshold (range)")

        else:

            if confidence < _filter_params["min_confidence"]:

                reasons.append(f"confidence {confidence:.2f} below threshold")



        # --- Quality floor (regime-aware) ---

        if regime == MarketRegime.RANGE:

            if quality_score < 48:

                reasons.append(f"quality {quality_score:.1f} below threshold (range)")

        else:

            if quality_score < _filter_params["min_quality"]:

                reasons.append(f"quality {quality_score:.1f} below threshold")



        # --- Evidence agreement (new pipeline only) ---

        if _evidence is not None:

            _min_ev = _filter_params.get("min_evidence_agreement", 0)

            if _evidence.signal_count_above_threshold < _min_ev:

                reasons.append(

                    f"evidence agreement {_evidence.signal_count_above_threshold} below {_min_ev}"

                )



        # --- Regime filter ---

        if hasattr(strategy, "allowed_regimes") and regime.value not in strategy.allowed_regimes:

            reasons.append(f"regime {regime.value} disabled")



        # --- Trend alignment (RELAXED ??? macro OR context must agree, new pipeline only) ---

        if _bundle is not None:

            direction = 1 if side == "long" else -1

            macro_ok = _bundle.higher_trend * direction > 0

            context_ok = _bundle.context_trend * direction > 0

            if not (macro_ok or context_ok):

                reasons.append("macro and context trends are not aligned")



        # --- Legacy filters (R:R, stop distance, max confidence) ---

        if _filter_params.get("max_confidence", 1.0) and confidence > _filter_params.get("max_confidence", 1.0):

            reasons.append(f"confidence {confidence:.2f} is above {_filter_params['max_confidence']:.2f}")

        if rr_ratio < _filter_params["min_rr_ratio"]:

            reasons.append(f"R:R {rr_ratio:.2f} is below {_filter_params['min_rr_ratio']:.2f}")

        if stop_distance_pct > _filter_params["max_stop_distance_pct"]:

            reasons.append(

                f"stop distance {stop_distance_pct:.2f}% is above {_filter_params['max_stop_distance_pct']:.2f}%"

            )\
"""

NEW_GATE = """\
        reasons: list[str] = []

        # ── Phase 7: Soft penalties replace early hard rejections ─────────────
        # Checks 1-5 and 7 are now soft: they reduce confidence/quality instead
        # of immediately rejecting. Hard gates at the end enforce absolute floors.

        # --- Confidence soft penalty (regime-aware) ---
        if regime == MarketRegime.RANGE:
            if confidence < 0.55:
                confidence = _clamp(confidence * 0.85, 0.0, 1.0)
        else:
            if confidence < _filter_params["min_confidence"]:
                confidence = _clamp(confidence * 0.85, 0.0, 1.0)

        # --- Quality soft penalty (regime-aware) ---
        if regime == MarketRegime.RANGE:
            if quality_score < 48:
                quality_score = _clamp(quality_score - 5.0, 0.0, 100.0)
        else:
            if quality_score < _filter_params["min_quality"]:
                quality_score = _clamp(quality_score - 5.0, 0.0, 100.0)

        # --- Evidence agreement soft penalty (new pipeline only) ---
        if _evidence is not None:
            _min_ev = _filter_params.get("min_evidence_agreement", 0)
            if _evidence.signal_count_above_threshold < _min_ev:
                confidence = _clamp(confidence * 0.90, 0.0, 1.0)

        # --- Regime filter (hard: disabled regimes are always rejected) ---
        if hasattr(strategy, "allowed_regimes") and regime.value not in strategy.allowed_regimes:
            reasons.append(f"regime {regime.value} disabled")

        # --- Trend alignment soft penalty (new pipeline only) ---
        if _bundle is not None:
            direction = 1 if side == "long" else -1
            macro_ok = _bundle.higher_trend * direction > 0
            context_ok = _bundle.context_trend * direction > 0
            if not (macro_ok or context_ok):
                confidence = _clamp(confidence * 0.90, 0.0, 1.0)
                quality_score = _clamp(quality_score - 3.0, 0.0, 100.0)

        # --- Max confidence cap (hard: never allow overconfident signals) ---
        if _filter_params.get("max_confidence", 1.0) and confidence > _filter_params.get("max_confidence", 1.0):
            reasons.append(f"confidence {confidence:.2f} is above {_filter_params['max_confidence']:.2f}")

        # --- Stop distance hard gate (structural safety) ---
        if stop_distance_pct > _filter_params["max_stop_distance_pct"]:
            reasons.append(
                f"stop distance {stop_distance_pct:.2f}% is above {_filter_params['max_stop_distance_pct']:.2f}%"
            )

        # ── Final absolute safety gates ───────────────────────────────────────
        # These are the ONLY hard rejection floors. All other checks above are
        # soft penalties that reduce scores without blocking the setup entirely.
        if confidence < 0.45:
            reasons.append(f"confidence {confidence:.2f} below absolute floor (0.45)")
        if rr_ratio < 0.8:
            reasons.append(f"R:R {rr_ratio:.2f} below absolute floor (0.8)")
        if quality_score < 25.0:
            reasons.append(f"quality {quality_score:.1f} below absolute floor (25)")\
"""

assert OLD_GATE in content, "ERROR: old gate block not found — check for encoding differences"
content = content.replace(OLD_GATE, NEW_GATE, 1)
print("Filter gate replaced OK")

with open("futures_analyzer/analysis/scorer.py", "w", encoding="utf-8") as f:
    f.write(content)
print("scorer.py written")
