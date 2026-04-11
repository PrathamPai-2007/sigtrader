# crypto-predictor

A monorepo containing two independently developed projects built around the same
core signal engine for perpetual futures markets.

## Projects

### sigforge — CLI Analysis Tool

A command-line tool for analysing Binance perpetual futures setups. Scans market
data, builds long/short trade ideas with confidence and quality scoring, and returns
`no trade` when the evidence is weak. Does not place orders.

```bash
cd sigforge
pip install -e .
sigforge analyse --symbol BTCUSDT
sigforge find --top 5 --capital 10000
```

See [sigforge/README.md](sigforge/README.md) for full documentation.

---

### sigwatch — Automated Trading Bot

A 24/7 async trading bot that connects to CoinDCX, uses the same signal engine as
sigforge, and executes real orders without human intervention. Not a CLI — it is a
persistent service designed to run on a server.

```bash
cd sigwatch
pip install -e .
export COINDCX_API_KEY=your_key
export COINDCX_API_SECRET=your_secret
python run.py
```

See [sigwatch/README.md](sigwatch/README.md) for architecture, phases, and deployment.

---

## Independence

These two projects are **fully independent**. Each has its own copy of the signal
engine (`futures_analyzer/`), its own `pyproject.toml`, its own config, its own
tests, and its own dependency tree. They are developed simultaneously but do not
share code at runtime — changes in one do not affect the other.

This is intentional. sigforge and sigwatch will diverge over time as each evolves
toward its own purpose: sigforge as a research and analysis tool, sigwatch as a
live execution system. Keeping them independent means each can be installed,
deployed, tested, and versioned on its own without any coupling.

If a meaningful improvement is made to the shared signal engine in one project,
it should be manually ported to the other when appropriate.

---

## Repository Structure

```
crypto-predictor/
  sigforge/          ← CLI analysis tool (Binance Futures)
    futures_analyzer/
    tests_futures/
    pyproject.toml
    futures_analyzer.config.json
    PLAN.md          ← Improvement backlog for sigforge

  sigwatch/          ← Automated trading bot (CoinDCX)
    futures_analyzer/  ← own independent copy of the signal engine
    bot/               ← main loop, config
    execution/         ← order executor, position monitor
    providers/         ← CoinDCX data provider
    tests_futures/
    pyproject.toml
    bot_config.json
```

## Notes

- Not financial advice
- API keys must always be stored in environment variables, never in code or config files
