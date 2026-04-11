# sigforge

`sigforge` is a CLI for analyzing Binance perpetual futures setups. It does not place
trades. It pulls market data, classifies the current regime, scores long and short
ideas, builds entry/stop/target geometry, and returns `no trade` when the setup
does not clear the configured filters.

## Installation

Requires Python 3.10+. Run from inside the `sigforge/` directory.

```bash
pip install -e .
# dev dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Startup dashboard
sigforge

# Analyse one symbol
sigforge analyse --symbol BTCUSDT
sigforge analyse --symbol BTCUSDT --style aggressive
sigforge analyse --symbol BTCUSDT --mode long_term
sigforge analyse --symbol BTCUSDT --preset swing_trader
sigforge analyse --symbol BTCUSDT --order-size 5000
sigforge analyse --symbol BTCUSDT --json

# Scan a list of symbols
sigforge scan --symbols BTCUSDT,ETHUSDT,SOLUSDT
sigforge scan --symbols BTCUSDT,ETHUSDT,SOLUSDT --top 3 --correlate
sigforge scan --symbols BTCUSDT,ETHUSDT,SOLUSDT --capital 10000

# Auto-find best setups from liquid candidates
sigforge find
sigforge find --universe 30 --top 8 --mode long_term
sigforge find --capital 10000 --correlate

# Backtest
sigforge backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-06-30
sigforge backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31 --folds 5

# Slippage estimate
sigforge slippage --symbol BTCUSDT --side long --entry 65000 --target 67000 --stop 64000 --order-size 5000

# Correlation analysis
sigforge correlate --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT
```

## History Commands

```bash
sigforge history recent
sigforge history recent --symbol BTCUSDT --limit 10

sigforge history stats --days 30
sigforge history stats --symbol BTCUSDT --regime bullish_trend

sigforge history feedback --days 30
sigforge history feedback --symbol BTCUSDT --limit 20

sigforge history compare --by mode --days 30
sigforge history compare --by symbol

sigforge history drawdown
sigforge history drawdown --symbol BTCUSDT --lookback 20

sigforge history backfill-metrics

# Clear all history (prompts for confirmation)
sigforge history clear
```

## Features

- Multi-timeframe analysis across entry, trigger, context, and higher timeframes
- Seven-state regime model: `bullish_trend`, `bearish_trend`, `range`, `volatile_chop`, `breakout`, `exhaustion`, `transition`
- Normalized 16-signal scoring pipeline with logistic confidence mapping
- Geometry engine for entry, stop, target, anchor selection, and quality scoring
- Strategy styles: `conservative` and `aggressive`
- Trading modes: `intraday` and `long_term`
- Named strategy presets via `--preset`
- Single-symbol analysis, multi-symbol scan, and auto candidate discovery with `find`
- Replay-backed "last feasible setup" timestamps from historical charts
- Local history storage with automatic outcome evaluation and drawdown-aware adjustments
- Portfolio risk controls, slippage estimation, and correlation analysis
- Config-driven tuning via `futures_analyzer.config.json`

## Analysis Pipeline

Each setup flows through the same core stages:

- Indicator collection: ATR, ADX, RSI, MACD, Bollinger Bands, VWAP bands, market structure, cumulative delta, liquidity sweeps, volume profile, funding, and OI context
- Regime classification: multi-timeframe consensus over higher, context, and trigger timeframes
- Signal normalization: 16 side-aware signals are mapped into `[0, 1]`
- Evidence grading: normalized signals are weighted by regime profile to produce a bounded evidence score
- Confidence mapping: evidence is converted to confidence through regime-aware logistic parameters
- Geometry selection: stop/target anchors are chosen from swing structure, VWAP bands, volume profile, or ATR fallback
- Quality scoring: R:R, stop distance, anchor quality, and confluence are combined into a `LOW` / `MEDIUM` / `HIGH` setup label

## Output Highlights

`analyse` and related commands now expose richer setup metadata, including:

- `stop_anchor` and `target_anchor`
- `regime_state`
- `signal_strengths`
- `evidence_weighted_sum`
- `logistic_input`
- Existing trade fields such as confidence, quality, leverage suggestion, and rationale remain backward-compatible

## Config

`sigforge` auto-loads `futures_analyzer.config.json` from the working directory.
The config controls timeframe stacks, style thresholds, regime scoring weights,
logistic confidence parameters, geometry-quality weights, cache TTLs, slippage
defaults, and candidate discovery parameters.

Key config keys:

| Key | Default | Description |
|-----|---------|-------------|
| `find_fallback_top` | `3` | Near-miss candidates shown when no tradable setup is found |
| `strategy.logistic_params` | built-in defaults | Regime-aware steepness and midpoint for confidence calibration |
| `strategy.geometry_quality` | built-in defaults | Weights for R:R, ATR stop quality, anchors, and confluence |
| `strategy.regime_weights.breakout` | normalized profile | Breakout scoring weights, summed to `1.0` |
| `strategy.regime_weights.exhaustion` | normalized profile | Exhaustion scoring weights, summed to `1.0` |
| `strategy.regime_weights.transition` | normalized profile | Transition scoring weights, summed to `1.0` |
| `drawdown.lookback` | `10` | Resolved trades examined for drawdown state |
| `portfolio.max_position_pct` | `0.20` | Max single position as fraction of capital |
| `portfolio.max_risk_per_trade_pct` | `0.02` | Max dollar risk per trade |
| `portfolio.kelly_fraction` | `0.25` | Quarter-Kelly scaling factor |

New config sections are backward-compatible. If an older config omits the new
keys, the application falls back to built-in defaults.

### Config loading behaviour

- Config is loaded once per CLI invocation and cached for the duration of the run.
- Every load prints `[CONFIG] Loaded from <path> at <timestamp>` to stdout so you always know which file is active.
- Pass `--refresh-config` to any of `analyse`, `scan`, or `find` to force a reload from disk and print the full resolved config before execution. Useful when you've edited the file mid-session or are debugging unexpected behaviour.

```bash
sigforge analyse --symbol BTCUSDT --refresh-config
sigforge scan --symbols BTCUSDT,ETHUSDT --refresh-config
sigforge find --refresh-config
```

## Logging

```bash
FUTURES_ANALYZER_LOG_LEVEL=INFO sigforge find
FUTURES_ANALYZER_LOG_FILE=/var/log/sigforge.log sigforge find
```

## Tests

```bash
pytest -q
```

## Notes

- Uses live Binance Futures market data
- History stored in `.data/history.db`; override with `FUTURES_ANALYZER_HISTORY_DB`
- `analyse` is the primary spelling; `analyze` is a compatibility alias
- Analytical tooling only — not financial advice

---

## Changelog

### 2026-04-11 (config system refactor)

**Reliability & debuggability**

- **Single config load per CLI run** — `analyse`, `scan`, and `find` now call `load_app_config()` exactly once at the top of each command and pass the result down. No more hidden global state or stale values mid-run.

- **`--refresh-config` on all main commands** — previously only `analyse` supported this flag. It is now available on `scan` and `find` as well. When passed, the cache is cleared, config is re-read from disk, and the full resolved config is printed before execution.

- **Config load tracing** — `load_app_config()` prints `[CONFIG] Loaded from <path> at <timestamp>` on every cache miss, making it immediately obvious which file is active and when it was read.

- **Removed duplicate `DEFAULT_CONFIG_PATH`** — the constant was defined in both `config.py` and `cli.py`. The `cli.py` copy is removed; it now imports from `config.py`.

- **Slippage config no longer silently falls back** — `SlippageAdvisor._vol_multiplier()` previously called `load_app_config()` inside a bare `except Exception: pass` block, masking any config errors. It now uses the module-level defaults directly; the caller is responsible for passing a correctly configured advisor.

- **Backtest slippage config hoisted out of hot loop** — `BacktestRunner._run_in_memory()` previously called `load_app_config()` on every tradable bar. The slippage model and volatility multiplier are now resolved once before the loop starts.

### 2026-04-08 (production hardening)

**Bug fixes**

- **`trigger_mins` used before assignment in backtest** — `_run_in_memory()` referenced
  `trigger_mins` before it was defined, causing a `NameError` on every backtest run.
  Fixed by moving the definition above the call site.

- **`asyncio.gather` swallowed fetch failures silently** — `BacktestRunner.run()` and
  `walk_forward()` used bare `gather()` so a single failed timeframe fetch would raise
  an unrelated `TypeError` on unpacking. Both now use `return_exceptions=True` and
  raise a clear `RuntimeError` naming the failing timeframe.

- **Cache eviction crashed under concurrency** — `BinanceFuturesProvider._evict_oldest()`
  deleted keys while iterating the dict, raising `RuntimeError: dictionary changed size
  during iteration`. Fixed by snapshotting keys first and using `pop(..., None)`.

- **Per-snapshot evaluation failures aborted the whole batch** — a single bad snapshot
  in `evaluate_due_snapshots()` would raise and skip all remaining snapshots. Each
  snapshot is now evaluated in its own try/except; failures are logged and skipped.

- **Silent history-save failures in scan/find** — bare `except Exception: pass` blocks
  in `_scan_async` and `_find_async` now log a warning so save failures are visible.

- **`backtest` CLI date parsing** — `--start`/`--end` used Typer's `formats=` kwarg
  which is not available in all Typer versions. Replaced with explicit
  `datetime.fromisoformat()` parsing with a clear error message on bad input.

**Validation improvements**

- **Interval validation in `fetch_klines`** — unknown interval strings are now rejected
  before hitting the API.

- **Symbol normalisation** — `validate_symbol` and `fetch_klines` now strip whitespace
  and uppercase the symbol before validation.

- **Config consistency warning** — `StyleTuning` now logs a warning at load time when
  `min_rr_ratio` exceeds `fallback_risk_reward`, which would silently filter all setups.

- **Candle validation warnings are now logged** — previously discarded with `pass`.

- **Prediction ID collision is now logged** — `IntegrityError` on duplicate ID emits a
  `db.prediction_id_collision` warning before retrying.

---

### 2026-04-08

**Performance & correctness fixes**

- **SQL filtering in `HistoryRepository.evaluated()`** — added `symbol`, `days`,
  `is_tradable`, and `limit` parameters pushed down to SQL. `recent_drawdown_state()`
  and `kelly_inputs()` now pass their lookback as a `LIMIT` clause instead of
  fetching all rows and slicing in Python. Noticeable speedup with 500+ history rows.

- **Pydantic range validators on config models** — `StyleTuning`, `MarketModeTuning`,
  `DrawdownConfig`, and `PortfolioConfig` now reject nonsensical values at load time
  (e.g. `min_confidence > 1.0`, misordered drawdown thresholds, `max_risk_per_trade`
  exceeding `max_total_risk`).

- **Config fallback warnings** — `_normalize_payload()` now emits a structured
  `config.missing_key` warning for every top-level key that is absent from the
  loaded file and falls back to a built-in default.

- **`SetupAnalyzer` reuse in walk-forward backtest** — `walk_forward()` previously
  created a new `SetupAnalyzer` instance per fold. It now instantiates one shared
  analyzer before the fold loop and passes it through, eliminating redundant
  object allocation.

**Signal quality fix**

- **RSI divergence — pivot-based detection (SQ-1)** — replaced the global-extremes
  approach in `rsi_divergence()` which fired on almost any trending series. The new
  implementation:
  - Identifies confirmed swing lows/highs via an N-bar left/right window comparison
    (`_swing_pivots()`).
  - Compares only the two most recent confirmed pivot pairs, not global extremes.
  - Requires a minimum price move between pivots (default 0.5%) to filter noise.
  - Returns divergence strength (RSI-point distance between the two pivot RSI
    values) as a third return value — signature is now `tuple[bool, str, float]`.
  - Monotonic trends no longer produce false divergence signals.
