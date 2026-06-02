# Sigforge - Futures Setup Analyzer

`sigforge` is a high-performance CLI tool for analyzing perpetual futures setups on Binance. It focuses on multi-timeframe analysis, market regime classification, and rigorous signal scoring to identify high-probability trading opportunities without placing actual trades.

## Project Overview

- **Core Tech:** Python 3.10+, [Pydantic](https://docs.pydantic.dev/) (Data Modeling), [Typer](https://typer.tiangolo.com/) (CLI), [HTTPX](https://www.python-httpx.org/) (Async API calls).
- **Domain:** Quantitative trading analysis for Binance Perpetual Futures.
- **Key Pipeline:** Indicator Collection -> Regime Classification -> Signal Normalization -> Evidence Grading -> Confidence Mapping -> Geometry Selection -> Quality Scoring.

## Architecture

The project is structured as a modular Python package:

- `analyzer/`: Core analysis package.
  - `config.py`: Manages `analyzer.config.json` loading and validation using Pydantic.
  - `reporting.py`: Text and HTML report rendering.
  - `logging.py`: Centralized logging.
  - `analysis/`: Core signal generation and grading rules.
    - `models.py`: Pydantic models for candles, setups, contributors, etc.
    - `regime.py`: Classification logic for 7 market states (e.g., `bullish_trend`, `volatile_chop`).
    - `scorer.py`: Multi-stage signal scoring and confidence mapping.
    - `geometry.py`: Entry, stop, and target placement logic.
    - `indicators.py`: TA-Lib style indicator computations.
- `backtest/`: Engine for replaying historical data through the analysis pipeline.
- `providers/`: Data fetching logic, primarily `binance_futures.py`.
- `history/`: Local SQLite persistence for tracking analysis outcomes.
- `market/`: Models for slippage, correlation, and market structure.
- `portfolio.py`: Risk management, including Kelly-fraction sizing and cluster-aware allocation.
- `cli.py`: Main entry point using Typer. Subcommands include `analyse`, `scan`, `find`, `backtest`, `history`, `slippage`, `correlate`, and `presets`.

## Development Commands

### Environment Setup
```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running the CLI
```bash
# Launch the dashboard
sigforge

# Analyze a specific symbol
sigforge analyse --symbol BTCUSDT

# Find best setups across a liquid universe
sigforge find --top 5

# Backtest a strategy over a date range
sigforge backtest --symbol BTCUSDT --start 2024-01-01 --end 2024-06-30
```

### Testing
```bash
# Run all tests
pytest

# Run tests with short output
pytest -q
```

## Development Conventions

1.  **Type Safety:** Use strict type hints for all function signatures and class members.
2.  **Data Models:** Always use Pydantic `BaseModel` for data structures that require validation or serialization (found in `*.models.py`).
3.  **Configuration:** Do NOT hardcode strategy parameters. Use `analyzer.config.json` and access them via `load_app_config()`.
4.  **Async/Await:** Use `asyncio` for I/O bound tasks (API calls, database access).
5.  **Logging:** Use the centralized logger from `analyzer.logging`.
6.  **Error Handling:** Use `typer.Exit` for clean CLI exits on expected errors.
7.  **Testing:** Add tests for new analysis signals or geometry logic in `tests/`.

## Key Files
- `pyproject.toml`: Build system and dependencies.
- `analyzer.config.json`: Master configuration for all strategy parameters and thresholds.
- `analyzer/analysis/models.py`: The "source of truth" for internal data structures.
- `.data/history.db`: Local SQLite database for analysis history.
