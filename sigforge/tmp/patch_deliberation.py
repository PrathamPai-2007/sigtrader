"""
Phase 7 Part 2: Convert _apply_deliberation hard rejections to soft penalties.
Preserves: regime disabled, enable_longs/shorts, absolute quality/confidence floors.
"""

with open("futures_analyzer/analysis/scorer.py", encoding="utf-8") as f:
    content = f.read()

OLD_DELIB = """\
        # EVIDENCE FILTER (regime-aware)

        if regime == MarketRegime.RANGE:

            if evidence.agreement < 2:

                setup.tradable_reasons.append(

                    f"evidence agreement {evidence.agreement}/{evidence.total} is below 2 (range)"

                )

        else:

            if evidence.agreement < params["min_evidence_agreement"]:

                setup.tradable_reasons.append(

                    f"evidence agreement {evidence.agreement}/{evidence.total} is below {params['min_evidence_agreement']}"

                )

        if (evidence.agreement - opposing_evidence.agreement) < params["min_evidence_edge"]:

            setup.tradable_reasons.append(

                f"evidence edge over opposing side is only {evidence.agreement - opposing_evidence.agreement}"

            )

        if not evidence.raw_checks.get("macro", False) and not evidence.raw_checks.get("context", False):

            setup.tradable_reasons.append("macro and context trends are not aligned for this side")

        if not evidence.raw_checks.get("trigger", False) and not evidence.raw_checks.get("entry", False):

            setup.tradable_reasons.append("trigger and entry momentum are both unsupportive")

        if evidence.agreement < params["min_evidence_agreement"]:

            setup.quality_score = min(setup.quality_score, _quality_score_cap_from_confidence(0.0))

            setup.quality_label = _quality_label(setup.quality_score)\
"""

NEW_DELIB = """\
        # EVIDENCE FILTER (regime-aware) — Phase 7: soft penalties instead of hard rejects
        # Legacy binary evidence checks now reduce confidence/quality rather than blocking.

        if regime == MarketRegime.RANGE:
            if evidence.agreement < 2:
                setup.confidence = _clamp(setup.confidence * 0.90, 0.0, 1.0)
                setup.quality_score = _clamp(setup.quality_score - 3.0, 0.0, 100.0)
        else:
            if evidence.agreement < params["min_evidence_agreement"]:
                setup.confidence = _clamp(setup.confidence * 0.90, 0.0, 1.0)
                setup.quality_score = _clamp(setup.quality_score - 3.0, 0.0, 100.0)

        if (evidence.agreement - opposing_evidence.agreement) < params["min_evidence_edge"]:
            setup.confidence = _clamp(setup.confidence * 0.92, 0.0, 1.0)

        if not evidence.raw_checks.get("macro", False) and not evidence.raw_checks.get("context", False):
            setup.confidence = _clamp(setup.confidence * 0.90, 0.0, 1.0)
            setup.quality_score = _clamp(setup.quality_score - 3.0, 0.0, 100.0)

        if not evidence.raw_checks.get("trigger", False) and not evidence.raw_checks.get("entry", False):
            setup.confidence = _clamp(setup.confidence * 0.92, 0.0, 1.0)\
"""

assert OLD_DELIB in content, "ERROR: old deliberation block not found"
content = content.replace(OLD_DELIB, NEW_DELIB, 1)
print("Step 1: evidence filter softened OK")

# ── Also soften the regime-specific quality caps and evidence checks ──────────
OLD_CHOP = """\
        if regime == MarketRegime.VOLATILE_CHOP:

            setup.quality_score = min(setup.quality_score, 50.0)

            setup.quality_label = _quality_label(setup.quality_score)

            

            chop_min = max(params["min_evidence_agreement"], 3)

            if evidence.agreement < chop_min:

                setup.tradable_reasons.append(

                    f"volatile chop requires {chop_min}+ evidence signals, got {evidence.agreement}"

                )



        if regime == MarketRegime.RANGE:

            setup.quality_score = min(setup.quality_score, 52.0)

            setup.quality_label = _quality_label(setup.quality_score)



            # Range needs at least 2 signals

            range_min = 2

            if evidence.agreement < range_min:

                setup.tradable_reasons.append(

                    f"range regime requires {range_min}+ evidence signals, got {evidence.agreement}"

                )



            # RANGE: enforce only relaxed threshold (no overrides here)

            if setup.quality_score < 45:

                setup.tradable_reasons.append(

                    f"quality {setup.quality_score:.1f} below threshold (range)"

                )\
"""

NEW_CHOP = """\
        if regime == MarketRegime.VOLATILE_CHOP:
            # Cap quality in volatile chop (structural, not a rejection)
            setup.quality_score = min(setup.quality_score, 50.0)
            setup.quality_label = _quality_label(setup.quality_score)
            # Soft penalty for insufficient evidence in chop
            chop_min = max(params["min_evidence_agreement"], 3)
            if evidence.agreement < chop_min:
                setup.confidence = _clamp(setup.confidence * 0.88, 0.0, 1.0)
                setup.quality_score = _clamp(setup.quality_score - 5.0, 0.0, 100.0)

        if regime == MarketRegime.RANGE:
            # Cap quality in range (structural, not a rejection)
            setup.quality_score = min(setup.quality_score, 52.0)
            setup.quality_label = _quality_label(setup.quality_score)
            # Soft penalty for insufficient evidence in range
            range_min = 2
            if evidence.agreement < range_min:
                setup.confidence = _clamp(setup.confidence * 0.90, 0.0, 1.0)
                setup.quality_score = _clamp(setup.quality_score - 3.0, 0.0, 100.0)
            # Soft penalty for low quality in range (was hard reject at 45)
            if setup.quality_score < 45:
                setup.quality_score = _clamp(setup.quality_score - 3.0, 0.0, 100.0)\
"""

assert OLD_CHOP in content, "ERROR: old chop/range block not found"
content = content.replace(OLD_CHOP, NEW_CHOP, 1)
print("Step 2: chop/range regime blocks softened OK")

with open("futures_analyzer/analysis/scorer.py", "w", encoding="utf-8") as f:
    f.write(content)
print("scorer.py written")
