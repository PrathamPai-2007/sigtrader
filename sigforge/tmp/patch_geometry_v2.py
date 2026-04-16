"""
Phase 8 v2: Corrected geometry patch.
- rr_enforced skipped only when structure target already meets min_rr
- SHORT TP cap at 2.5x risk (prevents extreme targets, respects min_rr=1.5)
- LONG TP cap deferred (future use)
- tp_capped anchor added to valid set
"""

with open("futures_analyzer/analysis/geometry.py", encoding="utf-8") as f:
    content = f.read()

# ── Replace the Phase 8 v1 block with corrected logic ────────────────────────
OLD_BLOCK = """\
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

NEW_BLOCK = """\
    # Enforce minimum R:R.
    # Phase 8: rr_enforced is demoted to fallback — it is skipped when a
    # structure anchor (swing/vwap/vp) was selected AND already meets min_rr.
    # This prevents overriding realistic structure targets with distant forced ones.
    risk = abs(entry - stop)
    reward = abs(target - entry)
    _structure_anchors = {"swing_high_sweep", "swing_high", "swing_low_sweep",
                          "swing_low", "vwap_upper", "vwap_lower", "vah", "val"}
    _has_structure_target = target_anchor in _structure_anchors
    _rr_met = reward / max(risk, 1e-9) >= min_rr
    if not (_has_structure_target and _rr_met) and reward / max(risk, 1e-9) < min_rr:
        # Only extend to rr_enforced when:
        # - no structure target was selected, OR
        # - structure target exists but doesn't meet min_rr
        if side == "long":
            target = entry + risk * min_rr
        else:
            target = entry - risk * min_rr
        target_anchor = "rr_enforced"

    # Phase 8: Soft TP cap for SHORT — prevents unrealistic distant targets.
    # Cap at 2.5× risk so extreme swing targets are trimmed while still
    # satisfying min_rr (typically 1.5). LONG cap deferred (future use).
    risk = abs(entry - stop)
    if _has_structure_target and side == "short":
        _max_reward = risk * 2.5
        if abs(target - entry) > _max_reward:
            target = entry - _max_reward
            target_anchor = "tp_capped"\
"""

assert OLD_BLOCK in content, "ERROR: v1 block not found"
content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)
print("Phase 8 v2: corrected rr_enforced logic and SHORT TP cap OK")

# ── Update docstring ──────────────────────────────────────────────────────────
OLD_DOC = """\
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

NEW_DOC = """\
    \"\"\"Place entry, stop, and target using swing pivots and VWAP anchors.

    Long stop priority:   (1) swing low sweep, (2) VWAP lower 1SD, (3) VAL, (4) ATR fallback.
    Long target priority: (1) swing high sweep, (2) VWAP upper 2SD, (3) VAH, (4) ATR cap.
    Short stop priority:  (1) swing high, (2) VWAP upper 1SD, (3) VAH, (4) ATR fallback.
    Short target priority:(1) swing low, (2) VWAP lower 2SD, (3) VAL, (4) ATR cap.

    Phase 8 — rr_enforced demotion:
      rr_enforced fires only when a structure target was selected but doesn't
      meet min_rr, OR when no structure target was available (atr_cap fallback).
      Structure targets that already meet min_rr are used as-is.

    Phase 8 — SHORT TP cap:
      When a structure target is selected for SHORT, it is capped at 2.5× risk
      to prevent unrealistic distant targets. LONG cap deferred.

    Stops are never modified by this function.
    Quantizes entry/stop/target to tick size.
    Substitutes max(atr, px * 0.001) when ATR is zero.
    \"\"\"\
"""

assert OLD_DOC in content, "ERROR: docstring not found"
content = content.replace(OLD_DOC, NEW_DOC, 1)
print("Docstring updated OK")

with open("futures_analyzer/analysis/geometry.py", "w", encoding="utf-8") as f:
    f.write(content)
print("geometry.py written")
