# Scorer.py Decomposition Plan (Revised & Safe)

This document outlines a **safe, dependency-aware refactor plan** to decompose `scorer.py` (~3000 lines) into modular components **without changing behavior**.

---

# 🎯 Goals

* Improve readability and maintainability
* Eliminate duplicated logic
* Enable future long/short pipeline separation
* Preserve 100% backward compatibility
* Avoid breaking existing backtests

---

# ⚠️ Core Principles

1. **NO logic changes during decomposition**
2. **Move code, don’t rewrite it**
3. **Preserve function signatures exactly**
4. **Use re-exports to maintain API**
5. **Test after every phase**

---

# 🧱 Correct Dependency Order

Modules must be extracted in this order to avoid circular imports:

```
utils
  ↓
indicators
  ↓
normalization
  ↓
evidence
  ↓
confidence
  ↓
geometry
  ↓
quality
  ↓
filters
  ↓
setup_analyzer
  ↓
scorer (re-export layer)
```

---

# 🔥 Phase 0 — Deduplication (MANDATORY)

Before splitting anything:

### Remove duplicate logic:

* `_compute_atr` vs `_atr`
* Any duplicate EMA / clamp / quantize logic
* Ensure only ONE implementation exists per function

### Ensure:

* `_clamp`, `_quantize`, `_quality_label` are single-source
* No shadowed or redundant helpers

---

# 📦 Phase 1 — utils.py

Extract pure helper functions:

* `_clamp`
* `_quantize`
* `_quality_label`
* `_quality_score_cap_from_confidence`

✅ No dependencies
✅ Safest first step

---

# 📊 Phase 2 — indicators.py

Move all indicator computation logic:

* `compute_all_indicators`
* `_compute_atr`
* `_ema_value`
* `_funding_momentum`
* `_oi_funding_biases`
* VWAP, RSI, MACD helpers

Depends on:

* utils
* config

---

# 📉 Phase 3 — normalization.py

Move signal normalization:

* `normalize_signals`

Depends on:

* indicators
* utils
* config

---

# 🧠 Phase 4 — evidence.py

Move evidence computation:

* `compute_graded_evidence`
* `_regime_weight_profile`
* `_regime_alignment`
* `_regime_penalty`

Depends on:

* normalization
* config

---

# 📈 Phase 5 — confidence.py

Move logistic mapping:

* `logistic_confidence`
* `logistic_confidence_from_config`

Depends on:

* evidence
* config

---

# 📐 Phase 6 — geometry.py

Move all entry/stop/target logic:

* `find_swing_points`
* `select_best_stop`
* `select_best_target`
* `place_entry_stop_target`

⚠️ Important:

* Geometry depends on indicators (ATR, VWAP, structure)

Depends on:

* indicators
* utils
* config

---

# ⭐ Phase 7 — quality.py

Move scoring logic:

* `geometry_quality_score`

Depends on:

* geometry
* config

---

# 🚫 Phase 8 — filters.py

Move trade gating logic:

* `_trade_filter_params`
* `_trade_filter_reasons`
* execution thresholds
* RR / quality checks

Depends on:

* geometry
* evidence
* config

---

# 🧩 Phase 9 — setup_analyzer (CRITICAL SPLIT)

Break the large class into multiple files:

```
setup_analyzer/
    ├── analyzer.py          # main class
    ├── builder.py           # _build_setup
    ├── deliberation.py      # _apply_deliberation
    ├── filters.py           # filter logic
```

Goal:

* Each file < 500 lines
* Clear separation of responsibilities

---

# 🔁 Phase 10 — scorer.py (RE-EXPORT LAYER)

Final file should only contain:

```python
from .utils import *
from .indicators import *
from .normalization import *
from .evidence import *
from .confidence import *
from .geometry import *
from .quality import *
from .filters import *
from .setup_analyzer import SetupAnalyzer
```

✅ Ensures backward compatibility
✅ No breaking imports

---

# 🧪 Testing Strategy

After EACH phase:

1. Run:

   ```
   pytest -q
   ```

2. Run a small backtest:

   * Compare trade count
   * Compare PnL
   * Compare rejection reasons

3. Verify:

   * No missing imports
   * No circular dependencies

---

# 🚨 Risks to Avoid

### ❌ Do NOT:

* Modify logic
* Change thresholds
* Touch config behavior
* Refactor AND optimize at the same time

---

### ⚠️ Watch for:

* Circular imports
* Hidden dependencies
* Implicit shared state

---

# 📏 Target File Sizes

| Module           | Target Size          |
| ---------------- | -------------------- |
| utils.py         | ~100 lines           |
| indicators.py    | ~250 lines           |
| normalization.py | ~200 lines           |
| evidence.py      | ~180 lines           |
| confidence.py    | ~100 lines           |
| geometry.py      | ~350 lines           |
| quality.py       | ~150 lines           |
| filters.py       | ~400 lines           |
| setup_analyzer   | ~600–800 lines total |
| scorer.py        | ~50 lines            |

---

# 🧠 Final Outcome

After decomposition:

* You can **debug each stage independently**
* You can **separate long/short cleanly later**
* You can **tune without breaking other parts**
* You gain **true modular trading architecture**

---

# 🚀 Next Step (After This)

Once decomposition is complete:

👉 THEN split:

* long pipeline
* short pipeline

---

# 🧠 Key Insight

> First make the system **clean and observable**
> Then make it **profitable**


Proceed phase-by-phase. Do not rush.
