# sigwatch

A 24/7 automated trading bot built on top of the sigforge signal engine.
Connects to CoinDCX, generates signals using the existing `SetupAnalyzer`,
and executes real orders without human intervention.

This is not a CLI tool. It is a persistent async service designed to run
continuously on a server.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Bot Loop (async)                   │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │ CoinDCX      │    │  Signal Engine           │   │
│  │ DataProvider │───▶│  (SetupAnalyzer reused   │   │
│  └──────────────┘    │   from sigforge)         │   │
│                      └────────────┬─────────────┘   │
│                                   │                  │
│                      ┌────────────▼─────────────┐   │
│                      │  Risk Manager            │   │
│                      │  (DrawdownAdjuster +     │   │
│                      │   PortfolioRiskManager)  │   │
│                      └────────────┬─────────────┘   │
│                                   │                  │
│                      ┌────────────▼─────────────┐   │
│                      │  Order Executor          │   │
│                      │  (CoinDCX REST API)      │   │
│                      └────────────┬─────────────┘   │
│                                   │                  │
│                      ┌────────────▼─────────────┐   │
│                      │  Position Monitor        │   │
│                      │  (WebSocket / polling)   │   │
│                      └──────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
sigwatch/
  providers/
    coindcx.py        ← CoinDCX data provider (replaces BinanceFuturesProvider)
  execution/
    executor.py       ← place, cancel, and track orders via CoinDCX REST
    monitor.py        ← watch open positions, trigger stop/target exits
  bot/
    loop.py           ← main 24/7 async loop: scan → signal → size → execute
    config.py         ← bot-specific settings (symbols, capital, scan interval)
  bot_config.json     ← watchlist, interval, risk limits
  run.py              ← entry point: asyncio.run(run_bot())
  requirements.txt
  Dockerfile
  README.md
```

## What's Reused from sigforge

The signal engine, risk management, and position sizing are shared directly:

| Component | Source |
|---|---|
| `SetupAnalyzer` | `sigforge/futures_analyzer/analysis/scorer.py` |
| `DrawdownAdjuster` | `sigforge/futures_analyzer/analysis/scorer.py` |
| `PortfolioRiskManager` | `sigforge/futures_analyzer/portfolio.py` |
| `HistoryService` | `sigforge/futures_analyzer/history/service.py` |
| `ParallelAnalyzer` | `sigforge/futures_analyzer/analysis/concurrency.py` |

The only new pieces are the CoinDCX data provider, the order execution layer,
and the persistent bot loop.

## Implementation Phases

### Phase 1 — Data Layer
- Build `CoinDCXProvider` implementing the same interface as `BinanceFuturesProvider`
- Verify kline data maps correctly to the `Candle` model
- Test signal generation end-to-end with CoinDCX data

### Phase 2 — Paper Trading
- Build `OrderExecutor` with `dry_run=True` mode that logs instead of placing orders
- Run the full loop for 1–2 weeks, track simulated P&L
- Validate signals and sizing behave as expected

### Phase 3 — Live Execution
- Enable real order placement with small capital
- Add position monitor with stop-loss enforcement
- Add daily loss circuit breaker

### Phase 4 — 24/7 Deployment
- Containerize with Docker
- Deploy to VPS with auto-restart
- Set up alerting (Telegram) for trade events and errors

## Risk Controls

Before going live, these must be in place:

- Max daily loss limit — halt trading if drawdown exceeds X% in 24h
- Max open positions — cap concurrent trades
- Duplicate signal guard — don't re-enter a symbol already in a position
- API error circuit breaker — stop trading if N consecutive API calls fail
- Dry-run mode — simulate orders without real execution

## Configuration

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
  "scan_interval_seconds": 300,
  "capital_usdt": 1000,
  "dry_run": true,
  "max_daily_loss_pct": 3.0,
  "max_open_positions": 3
}
```

Store API credentials in environment variables, never in config files:

```bash
export COINDCX_API_KEY=your_key
export COINDCX_API_SECRET=your_secret
```

## Running

```bash
# Dry run (paper trading)
python run.py

# Docker (24/7)
docker run --restart=always --env-file .env sigwatch
```

## Notes

- CoinDCX requires KYC verification for trading
- Automated trading via API is permitted under their ToS (verify current terms)
- API keys must be stored in environment variables, never in code or config files
- Not financial advice
