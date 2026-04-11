# SigForge Refactor Plan (Multi-Phase)

## Overview

This document outlines a structured, multi-phase refactor plan to
transform the trading bot into a clean, fully config-driven, and
reliable system.

------------------------------------------------------------------------

## Phase 1: Establish Single Source of Truth

**Goal:** Eliminate all duplicated and hardcoded configuration.

### Tasks:

-   Remove all hardcoded constants from:
    -   scorer.py
    -   config.py
    -   backtest modules
-   Ensure all weights, thresholds, and transforms exist ONLY in
    `futures_analyzer.config.json`
-   Refactor `config.py` to:
    -   ONLY load and validate config
    -   Provide typed accessors

### Deliverables:

-   Clean config.json
-   No numeric literals in logic (except trivial ones like 0,1)

------------------------------------------------------------------------

## Phase 2: Rebuild Scoring Engine

**Goal:** Make scoring deterministic, interpretable, and tunable.

### Tasks:

-   Refactor scorer.py into pipeline:
    -   raw signals → transforms → weighted sum → normalized score
-   Remove hidden multipliers (e.g. \*10.0)
-   Add transform config:

``` json
"transforms": {
  "higher_trend_scale": 6.0
}
```

-   Ensure score is bounded (0--1)

### Deliverables:

-   Clean scoring function
-   Unit-testable scoring module

------------------------------------------------------------------------

## Phase 3: Fix Regime Logic

**Goal:** Ensure regime filtering is correct and not destructive.

### Tasks:

-   Replace overwriting conditions with composable logic (AND / OR)
-   Remove or disable `volatile_chop`
-   Ensure regime gating aligns with config

### Deliverables:

-   Correct regime filtering
-   No logical overwrites

------------------------------------------------------------------------

## Phase 4: Align Risk & Execution Logic

**Goal:** Ensure actual trade execution matches configured expectations.

### Tasks:

-   Enforce min R:R properly
-   Ensure SL/TP logic matches config
-   Fix MAE \> MFE issue:
    -   tighten stops
    -   enforce targets
-   Validate slippage model integration

### Deliverables:

-   Consistent trade outcomes
-   Improved expectancy alignment

------------------------------------------------------------------------

## Phase 5: Add Observability & Validation

**Goal:** Make system debuggable and trustworthy.

### Tasks:

-   Add score breakdown logging
-   Add explainability:
    -   contribution per signal
-   Add unit tests:
    -   scoring correctness
    -   regime gating
-   Add sanity checks:
    -   confidence monotonicity

### Deliverables:

-   Debug-friendly system
-   Reliable evaluation pipeline

------------------------------------------------------------------------

## Final Outcome

-   Fully config-driven architecture
-   No hidden logic
-   Reliable backtesting
-   Interpretable signals
