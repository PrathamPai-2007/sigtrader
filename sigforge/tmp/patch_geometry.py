"""
Phase 8: Execution Layer Isolation.
- rr_enforced only fires when no structure target was selected (atr_cap fallback)
- SHORT: cap target_distance <= stop_distance * 1.2
- LONG: cap target_distance <= stop_distance * 1.0 (1:1 minimum, no forced extension)
- Stops unchanged.
"""

with open("futures_analyzer/analysis/geometry.py", encoding="utf-8") as f:
    content = f.read()

# ── Step 1: Replace select_best_target to track whether a structure anchor won ─
# The current function returns (price, label). We need to know if the winner
# was a structure anchor so we can skip rr_enforced for it.
# Solution: change the R:R enforcement block — check target_anchor instead.

OLD_RR_ENFORCE = """\
    # Enforce minimum R:R

    risk = abs(entry - stop)

    reward = abs(target - entry)

    if reward / max(risk, 1e-9) < min_rr:

        if side == "long":

            target = entry + risk * min_rr

        else:

            target = entry - risk * min_rr

        target_anchor = "rr_enforced"\
"""

NEW_RR_ENFORCE = """\
    # Enforce minimum R:R — only when no structure anchor was selected.
    # If a swing, vwap, or volume-profile target was chosen, trust it directly.
    # rr_enforced is used ONLY as a last resort when the atr_cap fallback fires.
    risk = abs(entry - stop)
    reward = abs(target - entry)
    _structure_anchors = {"swing_high_sweep", "swing_high", "swing_low_sweep",
                          "swing_low", "vwap_upper", "vwap_lower", "vah", "val"}
    _has_structure_target = target_anchor in _structure_anchors
    if not _has_structure_target and reward / max(risk, 1e-9) < min_rr:
        if side == "long":
            target = entry + risk * min_rr
        else:
            target = entry - risk * min_rr
        target_anchor = "rr_enforced"

    # Phase 8: Soft TP cap — prevent unrealistic targets.
    # SHORT: cap target distance to 1.2× stop distance (realistic downside).
    # LONG:  cap target distance to 1.0× stop distance (1:1 minimum, no forced extension).
    # Only applies when a structure target exists; atr_cap and rr_enforced are
    # already bounded by target_cap_atr_mult and min_rr respectively.
    risk = abs(entry - stop)
    if _has_structure_target:
        if side == "short":
            _max_reward = risk * 1.2
            if abs(target - entry) > _max_reward:
                target = entry - _max_reward
                target_anchor = "tp_capped"
        else:  # long
            _max_reward = risk * 1.0
            if abs(target - entry) > _max_reward:
                target = entry + _max_reward
                target_anchor = "tp_capped"\
"""

assert OLD_RR_ENFORCE in content, "ERROR: rr_enforce block not found"
content = content.replace(OLD_RR_ENFORCE, NEW_RR_ENFORCE, 1)
print("Step 1+4: rr_enforced demoted to fallback, TP cap added OK")

# ── Step 2: Update docstring to reflect new behaviour ────────────────────────
OLD_DOC = """\
    \"\"\"Place entry, stop, and target using swing pivots and VWAP anchors.



    Long stop priority: (1) nearest swing low below entry minus ATR buffer,

    (2) VWAP lower 1SD, (3) VAL, (4) ATR fallback.

    Long target priority: (1) nearest swing high above entry, (2) VWAP upper 2SD,

    (3) VAH, (4) ATR cap.

    Symmetric mirror logic applies for short setups.

    Enforces minimum R:R by extending target if needed.

    Quantizes entry/stop/target to tick size.

    Substitutes max(atr, px * 0.001) when ATR is zero.

    \"\"\"\
"""

NEW_DOC = """\
    \"\"\"Place entry, stop, and target using swing pivots and VWAP anchors.

    Long stop priority:   (1) swing low sweep, (2) VWAP lower 1SD, (3) VAL, (4) ATR fallback.
    Long target priority: (1) swing high sweep, (2) VWAP upper 2SD, (3) VAH, (4) ATR cap.
    Short stop priority:  (1) swing high, (2) VWAP upper 1SD, (3) VAH, (4) ATR fallback.
    Short target priority:(1) swing low, (2) VWAP lower 2SD, (3) VAL, (4) ATR cap.

    Structure targets (swing/vwap/vp) are used directly — rr_enforced fires ONLY
    when the atr_cap fallback is selected and the resulting R:R is below min_rr.

    Phase 8 TP cap: SHORT target capped at stop_distance * 1.2; LONG at * 1.0.
    Stops are never modified by this function.
    Quantizes entry/stop/target to tick size.
    Substitutes max(atr, px * 0.001) when ATR is zero.
    \"\"\"\
"""

assert OLD_DOC in content, "ERROR: docstring not found"
content = content.replace(OLD_DOC, NEW_DOC, 1)
print("Step 2: docstring updated OK")

with open("futures_analyzer/analysis/geometry.py", "w", encoding="utf-8") as f:
    f.write(content)
print("geometry.py written")
