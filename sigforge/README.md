# sigforge

> ⚠️ **This project is under active development.** Interfaces, config keys, and output formats may change without notice. Do not use in production trading systems.

`sigforge` is a CLI for analyzing perpetual futures setups. It does not place trades. It pulls live market data, classifies the current market regime, scores long and short ideas through a multi-stage signal pipeline, builds entry/stop/target geometry, and returns `no trade` when the setup does not clear the final safety gates (confidence < 0.60, R:R < 0.8, quality < 25). Sub-threshold signals are penalized rather than immediately rejected, so marginal setups are scored down rather than silently dropped.

## Installation

Requires Python 3.10+.

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
sigforge analyse --symbol BTCUSDT --preset scalper
sigforge analyse --symbol BTCUSDT --preset position_trader
sigforge analyse --symbol BTCUSDT --order-size 5000
sigforge analyse --symbol BTCUSDT --json
sigforge analyse --symbol BTCUSDT --export report.html

# Scan a list of symbols
sigforge scan --symbols BTCUSDT,ETHUSDT,SOLUSDT
sigforge scan --symbols BTCUSDT,ETHUSDT,SOLUSDT --preset scalper --top 3 --correlate
sigforge scan --symbols BTCUSDT,ETHUSDT,SOLUSDT --capital 10000

# Auto-find best setups from liquid candidates
sigforge find
sigforge find --preset position_trader --universe 30 --top 8
sigforge find --capital 10000 --correlate

# Backtest
sigforge backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-06-30
sigforge backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31 --folds 5

# Slippage estimate
sigforge slippage --symbol BTCUSDT --side long --entry 65000 --target 67000 --stop 64000 --order-size 5000

# Correlation analysis
sigforge correlate --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT

# List available strategy presets
sigforge presets
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
- Two streamlined trading presets: `scalper` and `position_trader`
- Single-symbol analysis, multi-symbol scan, and auto candidate discovery with `find`
- Near-miss fallback candidates when no setup clears hard filters
- Replay-backed "last feasible setup" timestamps from historical charts
- Local history storage with automatic outcome evaluation
- Drawdown-aware position guard applied per-symbol before saving
- Portfolio risk controls with Kelly-fraction sizing and cluster-aware allocation
- Slippage estimation with volatility-adjusted models
- Correlation analysis and concentration warnings
- Walk-forward backtesting with configurable fold count
- Simplified preset-based configuration via `futures_analyzer.config.json`
- Config loaded once per CLI run and cached for the duration of the session

## Analysis Pipeline

Each setup flows through the same core stages:

1. **Indicator collection** — ATR, ADX, RSI, MACD, Bollinger Bands, VWAP bands, EMA20, market structure, cumulative delta, liquidity sweeps, volume profile, funding rate, and open interest context
2. **Regime classification** — multi-timeframe consensus over higher, context, and trigger timeframes using ADX slope, ATR percentile, and DI bias
3. **Signal normalization** — 16 side-aware signals mapped into `[0, 1]` via sigmoid and tanh transforms
4. **Evidence grading** — normalized signals weighted by regime profile to produce a bounded evidence score
5. **Confidence mapping** — evidence converted to confidence through regime-aware logistic parameters (steepness + midpoint per regime)
6. **Enhanced metrics adjustments** — RSI zone, MACD confirmation, Bollinger position, order-book imbalance, and volatility rank applied as confidence/quality multipliers
7. **Timeframe alignment scoring** — cross-timeframe trend agreement applied as a confidence multiplier
8. **Reversal signal detection** — early reversal signals penalize confidence and quality when counter-trend pressure is detected
9. **Confluence scoring** — entry and target price confluence with structure levels adds quality boosts
10. **Geometry selection** — entry anchored to `max(EMA20, nearest swing low)` for longs and `min(EMA20, nearest swing high)` for shorts; stop/target anchors chosen from swing structure, VWAP bands, volume profile, or ATR fallback; stop buffers dynamically scaled by relative volatility (`atr / price × 100`) so volatile assets naturally get wider stops; structure targets used directly; `rr_enforced` fires only as a last resort with a 2.2× minimum R/R
11. **Long baiter post-processing** — limit-order long setups penalized for candle colour, FOMO conditions (RSI > 65, BB > 0.75), and distance from value anchors (VWAP, VAH, EMA20); penalties are soft and never zero confidence
12. **Quality scoring** — R:R, stop distance, anchor quality, and confluence combined into a `LOW` / `MEDIUM` / `HIGH` label
13. **Confidence firewall** — setups with confidence < 0.60 are filtered before recording or execution simulation
14. **Drawdown guard** — per-symbol drawdown state applied before results are saved or ranked

## Presets

The system uses two streamlined presets that contain all necessary configuration. Pass `--preset` to any of `analyse`, `scan`, or `find`.

| Preset | Description |
|--------|-------------|
| `scalper` | High-frequency trend-following with tight stops, small targets, and aggressive filters |
| `position_trader` | Conservative approach with relaxed filters and longer timeframes (default) |

```bash
sigforge presets                          # list all with descriptions
sigforge analyse --symbol BTCUSDT --preset scalper
sigforge find --preset position_trader --top 5
```

Each preset contains its complete configuration including:
- Timeframe stack (entry/trigger/context/higher)
- Filter thresholds (confidence, quality, evidence requirements)
- Risk parameters (stop distance, R:R ratios, target caps)
- Candidate selection criteria (volume, pool size, lookback periods)

## Output

`analyse` and related commands expose full setup metadata:

- `confidence`, `quality_score`, `quality_label` (`LOW` / `MEDIUM` / `HIGH`)
- `stop_anchor` and `target_anchor` (swing, vwap, volume_profile, atr, tp_capped)
- `regime_state`
- `signal_strengths` — per-signal normalized values
- `evidence_weighted_sum` and `logistic_input`
- `leverage_suggestion`
- `risk_reward_ratio`, `stop_distance_pct`, `target_distance_pct`
- `chart_replay_last_tradable_at` — timestamp of the most recent bar where the setup was tradable
- `warnings` — active drawdown guard messages and other runtime notices
- `tradable_reasons` — list of filter failures when `is_tradable` is false

All commands support `--json` for machine-readable output and `--export <file.html|.md|.txt>` for saved reports.

## Config

`sigforge` auto-loads `futures_analyzer.config.json` from the working directory. If the file does not exist, a minimal valid config is written automatically.

Config is loaded **once per CLI invocation** and cached for the duration of the run. Every load prints `[CONFIG] Loaded from <path> at <timestamp>` so you always know which file is active.

### Simplified Structure

The configuration has been streamlined to use preset-based organization:

```json
{
  "presets": {
    "position_trader": { /* complete config for position trader */ },
    "scalper": { /* complete config for scalper */ }
  },
  "strategy": { /* shared strategy parameters */ },
  "cache": { /* cache settings */ },
  "slippage": { /* slippage models */ },
  "drawdown": { /* drawdown thresholds */ },
  "portfolio": { /* portfolio risk controls */ }
}
```

### Key config sections

| Section | Description |
|---------|-------------|
| `presets` | Contains complete configurations for `scalper` and `position_trader` |
| `presets.<name>` | Individual preset with timeframes, filters, and candidate selection |
| `strategy` | Signal weights, regime weights, logistic params, geometry quality, long entry filters |
| `cache` | TTLs for market meta, realtime klines, historical klines |
| `slippage` | Default model and volatility multipliers |
| `drawdown` | Lookback window and severity thresholds |
| `portfolio` | Max position size, max risk per trade, Kelly fraction |

Key preset fields:

| Key | Description |
|-----|-------------|
| `entry_timeframe`, `trigger_timeframe`, `context_timeframe`, `higher_timeframe` | Multi-timeframe analysis stack |
| `lookback_bars` | Historical bars to analyze |
| `min_confidence`, `max_confidence` | Confidence thresholds for trade filters |
| `min_quality`, `min_rr_ratio` | Quality and risk:reward requirements |
| `max_stop_distance_pct` | Maximum stop distance as percentage |
| `min_evidence_agreement`, `min_evidence_edge` | Evidence requirements |
| `target_cap_atr_mult` | Target distance cap in ATR multiples |
| `candidate_*` | Symbol selection criteria for `find` command |

All config sections are backward-compatible. Missing keys fall back to built-in defaults.

> **Long entry filters** — `strategy.long_entry_filters.trend_dominance.min_adx_threshold` controls the ADX floor for trend-dominance checks on long setups (default `22.0`). Set `enable_enhanced_filters: true` to activate the full filter suite.

### Refreshing config mid-session

Pass `--refresh-config` to `analyse`, `scan`, or `find` to force a reload from disk and print the full resolved config before execution.

```bash
sigforge analyse --symbol BTCUSDT --refresh-config
sigforge scan --symbols BTCUSDT,ETHUSDT --refresh-config
sigforge find --refresh-config
```

## Backtest

```bash
sigforge backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-06-30
sigforge backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-12-31 --folds 5 --style aggressive
sigforge backtest --symbol ETHUSDT --start 2024-01-01 --end 2024-12-31 --order-size 1000
```

The backtest runner replays historical bars through the same analysis pipeline used live. With `--folds`, it runs walk-forward validation, splitting the date range into equal folds and reporting per-fold metrics. Slippage is applied when `--order-size` is provided.

Reported metrics per run: total trades, win rate, target hit rate, stop hit rate, average PnL%, average MFE/MAE, expectancy, max drawdown, and approximate Sharpe ratio.

## Portfolio Allocation

Pass `--capital` to `scan` or `find` to get a Kelly-fraction position sizing report alongside the ranked setups.

```bash
sigforge scan --symbols BTCUSDT,ETHUSDT,SOLUSDT --capital 10000 --correlate
sigforge find --top 5 --capital 50000 --correlate
```

Allocation respects `portfolio.max_position_pct`, `portfolio.max_risk_per_trade_pct`, `portfolio.max_cluster_risk_pct`, and `portfolio.max_total_risk_pct`. Kelly sizing activates per-symbol once at least `portfolio.min_history_for_kelly` evaluated trades exist in history.

## Drawdown Guard

After each analysis, the system checks the symbol's recent trade history for drawdown severity (`none` / `mild` / `moderate` / `severe`). When severity is not `none`, position size and confidence are adjusted downward before the result is saved or ranked. The active guard state is included in `warnings` on the output.

Thresholds are configured under `drawdown` in `futures_analyzer.config.json`.

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

- Uses live Binance Futures market data — no API key required for public endpoints
- History stored in `.data/history.db`; override with `FUTURES_ANALYZER_HISTORY_DB`
- Replay and history recording only accept setups with confidence ≥ 0.60
- `analyse` is the primary spelling; `analyze` is a compatibility alias
- Analytical tooling only — not financial advice
