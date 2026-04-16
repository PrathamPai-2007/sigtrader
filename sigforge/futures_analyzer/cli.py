from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
import json
from pathlib import Path

import httpx
import typer


from futures_analyzer.analysis import (
    MarketMode,
    SetupAnalyzer,
    StrategyStyle,
    build_timeframe_plan,
    find_latest_tradable_chart_timestamp,
)
from futures_analyzer.config import DEFAULT_CONFIG_PATH, load_app_config, refresh_app_config
from futures_analyzer.config_presets import get_preset, list_presets, StrategyPreset, PresetConfig
from futures_analyzer.analysis.concurrency import ParallelAnalyzer, QueryOptimizer
from futures_analyzer.history import HistoryCompareBy, HistoryService
from futures_analyzer.providers import BinanceFuturesProvider
from futures_analyzer.reporting import (
    render_analysis_text,
    render_correlation_text,
    render_feedback_text,
    render_find_text,
    render_history_compare_text,
    render_history_recent_text,
    render_history_stats_text,
    render_scan_text,
    render_slippage_text,
    render_startup_dashboard,
    write_report,
)
from futures_analyzer.backtest import BacktestConfig, BacktestRunner
from futures_analyzer.backtest.reporter import render_backtest_text, render_walk_forward_text
from futures_analyzer.market import SlippageAdvisor, CorrelationAnalyzer
from futures_analyzer.logging import configure_logging, get_logger
from futures_analyzer.analysis.scorer import DrawdownAdjuster
from futures_analyzer.portfolio import PortfolioRiskManager
from futures_analyzer.reporting import render_drawdown_text, render_portfolio_text

configure_logging()
log = get_logger(__name__)

app = typer.Typer(help="Sigforge — futures setup analyzer for Binance perpetual markets.")
history_app = typer.Typer(help="Stored analysis history and performance summaries.")
app.add_typer(history_app, name="history")


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context) -> None:
    """Sigforge — futures setup analyzer for Binance perpetual markets."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        from importlib.metadata import version as _pkg_version
        ver = _pkg_version("sigforge")
    except Exception:
        ver = "0.1.0"
    service = _history_service()
    overview = None
    pending_count = 0
    last_symbol = None
    last_evaluated_ago = None
    try:
        overview = service.feedback_overview(days=14)
        recent = service.recent(limit=1)
        if recent:
            last_symbol = recent[0].symbol
            delta = datetime.now(timezone.utc) - recent[0].as_of
            hours = int(delta.total_seconds() // 3600)
            last_evaluated_ago = f"{hours}h ago" if hours > 0 else "just now"
        pending = service.recent(limit=200)
        pending_count = sum(1 for s in pending if s.evaluation_status == "pending")
    except Exception:
        pass
    typer.echo(render_startup_dashboard(
        version=ver,
        overview=overview,
        pending_count=pending_count,
        last_symbol=last_symbol,
        last_evaluated_ago=last_evaluated_ago,
    ))


def _history_service() -> HistoryService:
    return HistoryService()


def _export_report(path: Path, *, title: str, text_body: str) -> None:
    try:
        exported = write_report(path, title=title, text_body=text_body)
    except (OSError, ValueError) as exc:
        typer.echo(f"Export failed: {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Exported report to {exported.resolve()}")


def _feedback_text_block(
    *,
    symbol: str | None = None,
    limit: int = 10,
    days: int = 14,
    service: HistoryService | None = None,
) -> str:
    service = service or _history_service()
    rows = service.feedback(symbol=symbol, limit=limit, days=days)
    overview = service.feedback_overview(symbol=symbol, days=days)
    return render_feedback_text(rows, overview)


def _resolve_risk_reward(style: StrategyStyle, risk_reward: float | None, preset: str = "position_trader", config: "AppConfig | None" = None) -> float:
    if risk_reward is not None:
        return risk_reward
    cfg = config or load_app_config()
    preset_config = cfg.get_preset(preset)
    return preset_config.fallback_risk_reward


def _resolve_preset(preset: StrategyPreset | None, style: StrategyStyle, market_mode: MarketMode) -> tuple[StrategyStyle, MarketMode, PresetConfig | None]:
    """Resolve preset configuration, returning style, market_mode, and preset_config.
    
    If preset is provided, get its configuration but keep the explicit style/market_mode.
    If no preset, return the provided style/market_mode with None preset_config.
    """
    if preset is None:
        return style, market_mode, None
    
    preset_config = get_preset(preset)
    return style, market_mode, preset_config



async def _analyze_async(
    symbol: str,
    risk_reward: float | None,
    style: StrategyStyle,
    market_mode: MarketMode,
    *,
    preset_config: PresetConfig | None = None,
    preset_name: str = "position_trader",
    provider: BinanceFuturesProvider | None = None,
):
    # Build the timeframe plan first — preset may override timeframes before
    # we fetch klines, so we must resolve it here rather than after the fetch.
    if preset_config is not None:
        from futures_analyzer.analysis.models import TimeframePlan as _TFP
        timeframe_plan = _TFP(
            profile_name=preset_config.name,
            style=style,
            market_mode=market_mode,
            entry_timeframe=preset_config.entry_timeframe,
            trigger_timeframe=preset_config.trigger_timeframe,
            context_timeframe=preset_config.context_timeframe,
            higher_timeframe=preset_config.higher_timeframe,
            lookback_bars=preset_config.lookback_bars,
        )
        filter_overrides: dict[str, float] = {
            "min_confidence": preset_config.min_confidence,
            "min_quality": preset_config.min_quality,
            "min_rr_ratio": preset_config.min_rr_ratio,
            "max_stop_distance_pct": preset_config.max_stop_distance_pct,
            "min_evidence_agreement": float(preset_config.min_evidence_agreement),
            "min_evidence_edge": float(preset_config.min_evidence_edge),
        }
    else:
        config = load_app_config()
        timeframe_plan = build_timeframe_plan(config=config, style=style, market_mode=market_mode, preset=preset_name)
        filter_overrides = {}

    owned_provider = provider is None
    provider = provider or BinanceFuturesProvider()
    try:
        market = await provider.fetch_market_meta(symbol)
        entry, trigger, context, higher = await asyncio.gather(
            provider.fetch_klines(symbol=symbol, interval=timeframe_plan.entry_timeframe, limit=timeframe_plan.lookback_bars),
            provider.fetch_klines(symbol=symbol, interval=timeframe_plan.trigger_timeframe, limit=timeframe_plan.lookback_bars),
            provider.fetch_klines(symbol=symbol, interval=timeframe_plan.context_timeframe, limit=timeframe_plan.lookback_bars),
            provider.fetch_klines(symbol=symbol, interval=timeframe_plan.higher_timeframe, limit=timeframe_plan.lookback_bars),
        )
        analyzer = SetupAnalyzer(
            risk_reward=_resolve_risk_reward(style, risk_reward, preset_name),
            style=style,
            market_mode=market_mode,
            filter_overrides=filter_overrides,
            preset=preset_name,
        )
        result = analyzer.analyze(
            symbol=symbol,
            entry_candles=entry,
            trigger_candles=trigger,
            context_candles=context,
            higher_candles=higher,
            market=market,
            timeframe_plan=timeframe_plan,
        )
        try:
            config = load_app_config()
            replay_timestamp = await find_latest_tradable_chart_timestamp(
                provider=provider,
                symbol=symbol,
                market=market,
                timeframe_plan=timeframe_plan,
                config=config,
                risk_reward=_resolve_risk_reward(style, risk_reward, preset_name),
                style=style,
                market_mode=market_mode,
                preset=preset_name,
            )
            result = result.model_copy(update={"chart_replay_last_tradable_at": replay_timestamp})
        except Exception as exc:
            log.warning("analysis.replay_timestamp_failed", symbol=symbol, error=str(exc))
        return result
    finally:
        if owned_provider:
            await provider.aclose()


async def _analyze_and_save_async(
    symbol: str,
    risk_reward: float | None,
    style: StrategyStyle,
    market_mode: MarketMode,
    service: "HistoryService",
    *,
    preset_config: "PresetConfig | None" = None,
    preset_name: str = "position_trader",
):
    result = await _analyze_async(
        symbol=symbol,
        risk_reward=risk_reward,
        style=style,
        market_mode=market_mode,
        preset_config=preset_config,
        preset_name=preset_name,
    )
    # Apply drawdown guard before saving
    drawdown_state = service.recent_drawdown_state(symbol=symbol)
    if drawdown_state.severity != "none":
        adjusted_primary = DrawdownAdjuster.apply(result.primary_setup, drawdown_state)
        result = result.model_copy(update={
            "primary_setup": adjusted_primary,
            "warnings": result.warnings + [
                f"Drawdown guard active ({drawdown_state.severity}): "
                f"{drawdown_state.current_drawdown_pct:.1f}% drawdown, "
                f"{drawdown_state.consecutive_losses} consecutive loss(es)."
            ],
        })
    try:
        await service.record_results([result], command="analyse")
    except Exception as exc:
        log.warning("history.save_skipped", symbol=result.market_snapshot_meta.symbol, error=str(exc))
    return result


async def _scan_async(
    symbols: list[str],
    risk_reward: float | None,
    style: StrategyStyle,
    market_mode: MarketMode,
    preset_config: PresetConfig | None = None,
    service: HistoryService | None = None,
    preset_name: str = "position_trader",
):
    symbols = QueryOptimizer.deduplicate_symbols(symbols)
    provider = BinanceFuturesProvider()
    try:
        analyzer = ParallelAnalyzer(max_concurrent=5)

        async def analyze_symbol(symbol: str):
            return await _analyze_async(
                symbol=symbol, risk_reward=risk_reward, style=style,
                market_mode=market_mode, preset_config=preset_config, provider=provider,
                preset_name=preset_name,
            )

        results = await analyzer.analyze_batch(symbols, analyze_symbol, fail_fast=False)
        if service is not None:
            successes = [r for _, r, err in results if err is None]
            if successes:
                try:
                    await service.record_results(successes, command="scan")
                except Exception as exc:
                    log.warning("history.scan_save_failed", error=str(exc))
        return [(item, result, error) for item, result, error in results]
    finally:
        await provider.aclose()


async def _find_async(
    top: int,
    universe: int,
    risk_reward: float | None,
    style: StrategyStyle,
    market_mode: MarketMode,
    preset_config: PresetConfig | None = None,
    service: HistoryService | None = None,
    preset_name: str = "position_trader",
):
    provider = BinanceFuturesProvider()
    try:
        candidate_symbols = await provider.fetch_candidate_symbols(
            market_mode=market_mode, limit=max(top, universe), preset=preset_name,
        )
        if not candidate_symbols:
            return [], []
        analyzer = ParallelAnalyzer(max_concurrent=5)

        async def analyze_symbol(symbol: str):
            return await _analyze_async(
                symbol=symbol, risk_reward=risk_reward, style=style,
                market_mode=market_mode, preset_config=preset_config, provider=provider,
                preset_name=preset_name,
            )

        results = await analyzer.analyze_batch(candidate_symbols, analyze_symbol, fail_fast=False)
        if service is not None:
            successes = [r for _, r, err in results if err is None]
            if successes:
                try:
                    await service.record_results(successes, command="find")
                except Exception as exc:
                    log.warning("history.find_save_failed", error=str(exc))
        return candidate_symbols, [(item, result, error) for item, result, error in results]
    finally:
        await provider.aclose()


def _latest_chart_replay_result(results: list) -> object | None:
    candidates = [r for r in results if getattr(r, "chart_replay_last_tradable_at", None) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.chart_replay_last_tradable_at)


def _chart_replay_summary(result) -> dict | None:
    if result is None or getattr(result, "chart_replay_last_tradable_at", None) is None:
        return None
    return {
        "symbol": result.market_snapshot_meta.symbol,
        "as_of": result.chart_replay_last_tradable_at.isoformat(),
        "style": result.timeframe_plan.style.value,
        "market_mode": result.timeframe_plan.market_mode.value,
    }


def _handle_error(exc: Exception, symbol: str | None = None) -> None:
    prefix = f"{symbol}: " if symbol else ""
    if isinstance(exc, ValueError):
        typer.echo(f"{prefix}{exc}", err=True)
    elif isinstance(exc, httpx.HTTPError):
        typer.echo(f"{prefix}Failed to fetch Binance Futures data: {exc}", err=True)
    else:
        typer.echo(f"{prefix}Unexpected analysis error: {exc}", err=True)
    raise typer.Exit(code=1)


def _rank_tradable_results(successes: list[tuple[str, object]], *, top: int) -> tuple[list[tuple[int, object]], int]:
    tradable = [(sym, r) for sym, r in successes if r.primary_setup.is_tradable]
    filtered_out = len(successes) - len(tradable)
    ranked = sorted(
        tradable,
        key=lambda item: (
            item[1].primary_setup.quality_score,
            item[1].primary_setup.confidence,
            item[1].primary_setup.target_distance_pct,
            item[1].primary_setup.evidence_agreement,
            item[1].primary_setup.risk_reward_ratio,
        ),
        reverse=True,
    )[:top]
    return [(rank, result) for rank, (_, result) in enumerate(ranked, start=1)], filtered_out


def _rank_near_tradable_results(successes: list[tuple[str, object]], *, top: int) -> list[tuple[int, object]]:
    """Return the best non-tradable setups, sorted by quality then confidence.

    Used as a fallback when no setup passes the hard trade filters, so the
    user still gets actionable near-miss candidates with a warning.
    """
    non_tradable = [(sym, r) for sym, r in successes if not r.primary_setup.is_tradable]
    ranked = sorted(
        non_tradable,
        key=lambda item: (
            item[1].primary_setup.quality_score,
            item[1].primary_setup.confidence,
        ),
        reverse=True,
    )[:top]
    return [(rank, result) for rank, (_, result) in enumerate(ranked, start=1)]


def _emit_progress_message(message: str, *, enabled: bool) -> None:
    if enabled:
        typer.echo(message)


def _run_correlation_warnings(
    symbols: list[str],
    ranked_results: list[tuple[int, object]],
) -> list[str]:
    """Fetch correlation and return warning strings. Returns [] on any failure."""
    try:
        ranked_syms = [r.market_snapshot_meta.symbol for _, r in ranked_results]
        ca = CorrelationAnalyzer()
        report = asyncio.run(ca.analyze(symbols))
        return ca.warn_concentrated_setups(report, ranked_syms)
    except Exception:
        return []


def _apply_drawdown_to_batch(
    successes: list[tuple[str, object]],
    service: "HistoryService",
) -> list[tuple[str, object]]:
    """Apply per-symbol drawdown guard to a batch of analysis results."""
    adjusted = []
    for sym, result in successes:
        drawdown_state = service.recent_drawdown_state(symbol=sym)
        if drawdown_state.severity != "none":
            adjusted_primary = DrawdownAdjuster.apply(result.primary_setup, drawdown_state)
            result = result.model_copy(update={
                "primary_setup": adjusted_primary,
                "warnings": result.warnings + [
                    f"Drawdown guard active ({drawdown_state.severity}): "
                    f"{drawdown_state.current_drawdown_pct:.1f}% drawdown, "
                    f"{drawdown_state.consecutive_losses} consecutive loss(es)."
                ],
            })
        adjusted.append((sym, result))
    return adjusted


def _build_portfolio_report(
    ranked_results: list[tuple[int, object]],
    capital: float,
    service: "HistoryService",
    correlation_clusters: list[list[str]] | None = None,
    config: "AppConfig | None" = None,
) -> object:
    """Build a PortfolioRiskReport for the given ranked results and capital."""
    cfg = (config or load_app_config()).portfolio
    manager = PortfolioRiskManager(cfg)
    ki: dict[str, tuple[float, float, float]] = {}
    for _, result in ranked_results:
        sym = result.market_snapshot_meta.symbol
        if len(service.repository.evaluated(symbol=sym)) >= cfg.min_history_for_kelly:
            ki[sym] = service.kelly_inputs(symbol=sym)
    return manager.allocate(
        ranked_results,
        capital,
        kelly_inputs=ki,
        correlation_clusters=correlation_clusters,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command("presets")
def presets() -> None:
    """List available strategy presets."""
    for preset in list_presets():
        typer.echo(f"  {preset['name']:<20} - {preset['display_name']:<20}")
        typer.echo(f"    {preset['description']}\n")


@app.command("analyse")
@app.command("analyze", hidden=True)
def analyse(
    symbol: str = typer.Option(..., "--symbol", help="Binance futures symbol, e.g. BTCUSDT"),
    risk_reward: float | None = typer.Option(None, "--risk-reward", min=0.5, help="Fallback risk:reward."),
    preset: StrategyPreset = typer.Option(StrategyPreset.POSITION_TRADER, "--preset", help="Trading strategy preset."),
    refresh_config: bool = typer.Option(False, "--refresh-config", help="Reload config.json from disk (clears cache)."),
    order_size: float | None = typer.Option(None, "--order-size", min=1.0, help="Position size in USD for slippage analysis."),
    as_json: bool = typer.Option(False, "--json"),
    export: Path | None = typer.Option(None, "--export", help="Write a .html, .md, or .txt report."),
) -> None:
    """Analyse a single symbol."""
    if refresh_config:
        cfg_path = DEFAULT_CONFIG_PATH.resolve()
        print(f"Refreshing config from {cfg_path}...")
        refresh_app_config()
    config = load_app_config()
    if refresh_config:
        print("=== FINAL CONFIG ===")
        print(config.model_dump_json(indent=2))
    
    # Get preset configuration
    preset_config = get_preset(preset)
    # Use default style and market_mode since presets no longer specify them  
    style = StrategyStyle.CONSERVATIVE
    market_mode = MarketMode.INTRADAY
    
    service = _history_service()
    _emit_progress_message(
        "Fetching market data and replaying recent charts to find the latest feasible setup. This can take a moment.",
        enabled=not as_json,
    )
    try:
        result = asyncio.run(_analyze_and_save_async(
            symbol=symbol.upper(), risk_reward=risk_reward, style=style,
            market_mode=market_mode, service=service, preset_config=preset_config,
            preset_name=preset.value,
        ))
    except Exception as exc:
        _handle_error(exc, symbol=symbol.upper())

    slip_report = None
    if order_size is not None:
        try:
            from futures_analyzer.history.evaluation import SlippageModel
            model = SlippageModel(config.slippage.default_model)
            advisor = SlippageAdvisor(model=model, min_viable_rr=config.slippage.min_viable_rr)
            slip_report = asyncio.run(advisor.estimate(
                result.primary_setup, result.market_snapshot_meta, order_size_usd=order_size,
            ))
        except Exception:
            pass

    text_report = render_analysis_text(result)
    feedback_report = _feedback_text_block(symbol=symbol.upper(), service=service, limit=5)
    slip_text = ("\n\n" + render_slippage_text(slip_report)) if slip_report else ""

    if export is not None:
        _export_report(export, title=f"Analysis Report - {result.market_snapshot_meta.symbol}",
                       text_body=f"{text_report}{slip_text}\n\n{feedback_report}")
    if as_json:
        payload: dict = {
            "result": result.model_dump(mode="json"),
            "latest_feasible_chart_replay": _chart_replay_summary(result),
            "feedback": feedback_report,
        }
        if slip_report:
            from dataclasses import asdict
            payload["slippage"] = asdict(slip_report)
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(text_report)
        if slip_report:
            typer.echo("")
            typer.echo(render_slippage_text(slip_report))
        typer.echo("")
        typer.echo(feedback_report)


@app.command("scan")
def scan(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated Binance futures symbols."),
    risk_reward: float | None = typer.Option(None, "--risk-reward", min=0.5),
    preset: StrategyPreset = typer.Option(StrategyPreset.POSITION_TRADER, "--preset", help="Trading strategy preset."),
    top: int = typer.Option(5, "--top", min=1),
    correlate: bool = typer.Option(False, "--correlate", help="Append correlation warnings for top-ranked setups."),
    capital: float | None = typer.Option(None, "--capital", min=1.0, help="Total capital in USD for portfolio allocation."),
    refresh_config: bool = typer.Option(False, "--refresh-config", help="Reload config.json from disk (clears cache)."),
    as_json: bool = typer.Option(False, "--json"),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Scan a list of symbols and rank tradable setups."""
    if refresh_config:
        refresh_app_config()
    config = load_app_config()
    if refresh_config:
        print("=== FINAL CONFIG ===")
        print(config.model_dump_json(indent=2))
    
    # Get preset configuration
    preset_config = get_preset(preset)
    # Use default style and market_mode since presets no longer specify them
    style = StrategyStyle.CONSERVATIVE
    market_mode = MarketMode.INTRADAY
    
    service = _history_service()
    parsed_symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not parsed_symbols:
        raise typer.BadParameter("Please provide at least one symbol via --symbols.")
    _emit_progress_message(
        "Fetching market data and replaying recent charts for the requested symbols. This can take a moment.",
        enabled=not as_json,
    )
    scan_results = asyncio.run(_scan_async(
        parsed_symbols, risk_reward=risk_reward, style=style,
        market_mode=market_mode, preset_config=preset_config, service=service,
        preset_name=preset.value,
    ))
    successes: list[tuple[str, object]] = []
    failures: list[dict[str, str]] = []
    for sym, result, error in scan_results:
        if error is not None:
            failures.append({"symbol": sym, "error": str(error)})
        else:
            successes.append((sym, result))

    # Apply per-symbol drawdown guard
    successes = _apply_drawdown_to_batch(successes, service)

    ranked_results, filtered_out = _rank_tradable_results(successes, top=top)
    latest_replay = _latest_chart_replay_result([r for _, r in successes])
    text_report = render_scan_text(ranked_results, failures,
        latest_replay_symbol=latest_replay.market_snapshot_meta.symbol if latest_replay else None,
        latest_replay_at=latest_replay.chart_replay_last_tradable_at if latest_replay else None,
        filtered_out=filtered_out,
    )

    corr_warnings: list[str] = []
    corr_clusters: list[list[str]] | None = None
    if correlate and len(parsed_symbols) >= 2:
        corr_warnings = _run_correlation_warnings(parsed_symbols, ranked_results)
        try:
            ca = CorrelationAnalyzer()
            corr_report = asyncio.run(ca.analyze(parsed_symbols))
            corr_clusters = corr_report.cluster_groups
        except Exception:
            pass

    portfolio_report = None
    if capital is not None and ranked_results:
        try:
            portfolio_report = _build_portfolio_report(ranked_results, capital, service, corr_clusters, config)
        except Exception as exc:
            log.warning("portfolio.allocation_failed", error=str(exc))

    full_body = text_report
    if corr_warnings:
        full_body += "\n\nCorrelation Warnings\n" + "\n".join(corr_warnings)
    if portfolio_report is not None:
        full_body += "\n\n" + render_portfolio_text(portfolio_report, capital=capital)

    if export is not None:
        _export_report(export, title="Scan Report", text_body=full_body)

    if as_json:
        payload = []
        for rank, result in ranked_results:
            row = result.model_dump(mode="json")
            row["rank"] = rank
            payload.append(row)
        json_out: dict = {
            "results": payload, "errors": failures, "filtered_out": filtered_out,
            "latest_feasible_chart_replay": _chart_replay_summary(latest_replay),
            "feedback": _feedback_text_block(limit=20, service=service),
            "correlation_warnings": corr_warnings,
        }
        if portfolio_report is not None:
            from dataclasses import asdict
            json_out["portfolio"] = asdict(portfolio_report)
        typer.echo(json.dumps(json_out, indent=2))
        return

    typer.echo(text_report)
    if corr_warnings:
        typer.echo("")
        typer.echo("Correlation Warnings")
        for w in corr_warnings:
            typer.echo(w)
    if portfolio_report is not None:
        typer.echo("")
        typer.echo(render_portfolio_text(portfolio_report, capital=capital))
    typer.echo("")
    typer.echo(_feedback_text_block(limit=20, service=service))


@app.command("find")
def find(
    top: int = typer.Option(5, "--top", min=1),
    universe: int = typer.Option(20, "--universe", min=1),
    risk_reward: float | None = typer.Option(None, "--risk-reward", min=0.5),
    preset: StrategyPreset = typer.Option(StrategyPreset.POSITION_TRADER, "--preset", help="Trading strategy preset."),
    correlate: bool = typer.Option(False, "--correlate", help="Append correlation warnings for top-ranked setups."),
    capital: float | None = typer.Option(None, "--capital", min=1.0, help="Total capital in USD for portfolio allocation."),
    refresh_config: bool = typer.Option(False, "--refresh-config", help="Reload config.json from disk (clears cache)."),
    as_json: bool = typer.Option(False, "--json"),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Auto-find best setups from liquid candidates."""
    if refresh_config:
        refresh_app_config()
    config = load_app_config()
    if refresh_config:
        print("=== FINAL CONFIG ===")
        print(config.model_dump_json(indent=2))
    
    # Get preset configuration
    preset_config = get_preset(preset)
    # Use default style and market_mode since presets no longer specify them
    style = StrategyStyle.CONSERVATIVE
    market_mode = MarketMode.INTRADAY
    
    service = _history_service()
    _emit_progress_message(
        "Finding liquid candidates, fetching market data, and replaying recent charts under the current rules. This can take a moment.",
        enabled=not as_json,
    )
    try:
        candidate_symbols, find_results = asyncio.run(_find_async(
            top=top, universe=universe, risk_reward=risk_reward, style=style,
            market_mode=market_mode, preset_config=preset_config, service=service,
            preset_name=preset.value,
        ))
    except Exception as exc:
        _handle_error(exc)

    successes: list[tuple[str, object]] = []
    failures: list[dict[str, str]] = []
    for sym, result, error in find_results:
        if error is not None:
            failures.append({"symbol": sym, "error": str(error)})
        else:
            successes.append((sym, result))

    # Apply per-symbol drawdown guard
    successes = _apply_drawdown_to_batch(successes, service)

    ranked_results, filtered_out = _rank_tradable_results(successes, top=top)
    latest_replay = _latest_chart_replay_result([r for _, r in successes])

    # Fallback: when no setup passes hard filters, surface near-miss candidates.
    fallback_results: list[tuple[int, object]] = []
    fallback_top = config.find_fallback_top
    if not ranked_results and successes and fallback_top > 0:
        fallback_results = _rank_near_tradable_results(successes, top=fallback_top)

    text_report = render_find_text(
        ranked_results, failures,
        candidate_count=len(candidate_symbols),
        market_mode=market_mode.value,
        latest_replay_symbol=latest_replay.market_snapshot_meta.symbol if latest_replay else None,
        latest_replay_at=latest_replay.chart_replay_last_tradable_at if latest_replay else None,
        filtered_out=filtered_out,
        fallback_results=fallback_results,
    )

    corr_warnings: list[str] = []
    corr_clusters: list[list[str]] | None = None
    if correlate and len(successes) >= 2:
        all_syms = [sym for sym, _ in successes]
        corr_warnings = _run_correlation_warnings(all_syms, ranked_results)
        try:
            ca = CorrelationAnalyzer()
            corr_report = asyncio.run(ca.analyze(all_syms))
            corr_clusters = corr_report.cluster_groups
        except Exception:
            pass

    portfolio_report = None
    if capital is not None and ranked_results:
        try:
            portfolio_report = _build_portfolio_report(ranked_results, capital, service, corr_clusters, config)
        except Exception as exc:
            log.warning("portfolio.allocation_failed", error=str(exc))

    full_body = text_report
    if corr_warnings:
        full_body += "\n\nCorrelation Warnings\n" + "\n".join(corr_warnings)
    if portfolio_report is not None:
        full_body += "\n\n" + render_portfolio_text(portfolio_report, capital=capital)

    if export is not None:
        _export_report(export, title="Find Report", text_body=full_body)

    if as_json:
        payload = []
        for rank, result in ranked_results:
            row = result.model_dump(mode="json")
            row["rank"] = rank
            payload.append(row)
        fallback_payload = []
        for rank, result in fallback_results:
            row = result.model_dump(mode="json")
            row["rank"] = rank
            fallback_payload.append(row)
        json_out: dict = {
            "candidates": candidate_symbols, "results": payload, "errors": failures,
            "filtered_out": filtered_out,
            "fallback_results": fallback_payload,
            "latest_feasible_chart_replay": _chart_replay_summary(latest_replay),
            "feedback": _feedback_text_block(limit=20, service=service),
            "correlation_warnings": corr_warnings,
        }
        if portfolio_report is not None:
            from dataclasses import asdict
            json_out["portfolio"] = asdict(portfolio_report)
        typer.echo(json.dumps(json_out, indent=2))
        return

    typer.echo(text_report)
    if corr_warnings:
        typer.echo("")
        typer.echo("Correlation Warnings")
        for w in corr_warnings:
            typer.echo(w)
    if portfolio_report is not None:
        typer.echo("")
        typer.echo(render_portfolio_text(portfolio_report, capital=capital))
    typer.echo("")
    typer.echo(_feedback_text_block(limit=20, service=service))


@app.command("backtest")
def backtest(
    symbol: str = typer.Option(..., "--symbol"),
    start: str = typer.Option(..., "--start", help="Start date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"),
    end: str = typer.Option(..., "--end", help="End date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"),
    style: StrategyStyle = typer.Option(StrategyStyle.CONSERVATIVE, "--style"),
    market_mode: MarketMode = typer.Option(MarketMode.INTRADAY, "--mode"),
    preset: StrategyPreset | None = typer.Option(None, "--preset"),
    risk_reward: float = typer.Option(2.0, "--risk-reward", min=0.5),
    folds: int | None = typer.Option(None, "--folds", min=2, help="Enable walk-forward testing with N folds."),
    verbose: bool = typer.Option(False, "--verbose", help="Print individual trade list."),
    order_size: float | None = typer.Option(None, "--order-size", min=1.0, help="Apply slippage for this USD position size."),
    as_json: bool = typer.Option(False, "--json"),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Backtest the strategy on historical data for a symbol and date range."""
    style, market_mode, _ = _resolve_preset(preset, style, market_mode)
    
    # Parse dates
    try:
        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
        start_utc = start_dt.replace(tzinfo=timezone.utc) if start_dt.tzinfo is None else start_dt
    except ValueError:
        typer.echo(f"Invalid --start format: {start}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS", err=True)
        raise typer.Exit(code=1)
    
    try:
        end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
        end_utc = end_dt.replace(tzinfo=timezone.utc) if end_dt.tzinfo is None else end_dt
    except ValueError:
        typer.echo(f"Invalid --end format: {end}. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS", err=True)
        raise typer.Exit(code=1)
    if end_utc <= start_utc:
        typer.echo("--end must be after --start", err=True)
        raise typer.Exit(code=1)

    config = BacktestConfig(
        symbol=symbol.upper(), start=start_utc, end=end_utc,
        style=style, market_mode=market_mode, risk_reward=risk_reward,
        preset=preset.value if preset else None, order_size_usd=order_size,
    )
    runner = BacktestRunner(config)
    _emit_progress_message(
        f"Running backtest for {symbol.upper()} from {start_utc.date()} to {end_utc.date()}. "
        "Fetching historical data — this may take a moment.",
        enabled=not as_json,
    )
    try:
        if folds is not None:
            reports = asyncio.run(runner.walk_forward(folds=folds, progress=not as_json))
            text_report = render_walk_forward_text(reports)
            if export is not None:
                _export_report(export, title=f"Walk-Forward Backtest — {symbol.upper()}", text_body=text_report)
            if as_json:
                typer.echo(json.dumps([{
                    "fold": i + 1, "start": r.config.start.isoformat(), "end": r.config.end.isoformat(),
                    "total_trades": r.total_trades, "target_hit_rate": r.target_hit_rate,
                    "stop_hit_rate": r.stop_hit_rate, "win_rate": r.win_rate,
                    "avg_pnl_pct": r.avg_pnl_pct, "expectancy": r.expectancy,
                    "max_drawdown_pct": r.max_drawdown_pct, "sharpe_approx": r.sharpe_approx,
                } for i, r in enumerate(reports)], indent=2))
            else:
                typer.echo(text_report)
        else:
            report = asyncio.run(runner.run(progress=not as_json))
            text_report = render_backtest_text(report, verbose=verbose)
            if export is not None:
                _export_report(export, title=f"Backtest Report — {symbol.upper()}", text_body=text_report)
            if as_json:
                typer.echo(json.dumps({
                    "config": {
                        "symbol": config.symbol, "start": config.start.isoformat(),
                        "end": config.end.isoformat(), "style": config.style.value,
                        "market_mode": config.market_mode.value, "risk_reward": config.risk_reward,
                        "preset": config.preset,
                    },
                    "total_trades": report.total_trades, "target_hit_rate": report.target_hit_rate,
                    "stop_hit_rate": report.stop_hit_rate, "win_rate": report.win_rate,
                    "avg_pnl_pct": report.avg_pnl_pct, "avg_mfe_pct": report.avg_mfe_pct,
                    "avg_mae_pct": report.avg_mae_pct, "expectancy": report.expectancy,
                    "max_drawdown_pct": report.max_drawdown_pct, "sharpe_approx": report.sharpe_approx,
                    "trades": [{
                        "bar_time": t.bar_time.isoformat(), "side": t.side,
                        "entry": t.entry, "target": t.target, "stop": t.stop,
                        "confidence": t.confidence, "quality_score": t.quality_score,
                        "quality_label": t.quality_label, "regime": t.regime,
                        "outcome": t.outcome, "pnl_pct": t.pnl_pct,
                        "mfe_pct": t.mfe_pct, "mae_pct": t.mae_pct,
                    } for t in report.trades],
                }, indent=2))
            else:
                typer.echo(text_report)
    except Exception as exc:
        _handle_error(exc)


@app.command("slippage")
def slippage_cmd(
    symbol: str = typer.Option(..., "--symbol"),
    side: str = typer.Option(..., "--side", help="long or short"),
    entry: float = typer.Option(..., "--entry"),
    target: float = typer.Option(..., "--target"),
    stop: float = typer.Option(..., "--stop"),
    order_size: float = typer.Option(1000.0, "--order-size", min=1.0),
    model: str = typer.Option("moderate", "--model", help="conservative, moderate, or optimistic"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Estimate slippage impact for a trade at given prices."""
    from futures_analyzer.history.evaluation import SlippageModel
    from dataclasses import asdict
    side = side.lower()
    if side not in ("long", "short"):
        typer.echo("--side must be 'long' or 'short'", err=True)
        raise typer.Exit(code=1)
    try:
        slip_model = SlippageModel(model)
    except ValueError:
        typer.echo("--model must be one of: conservative, moderate, optimistic", err=True)
        raise typer.Exit(code=1)
    cfg = load_app_config()
    advisor = SlippageAdvisor(model=slip_model, min_viable_rr=cfg.slippage.min_viable_rr)
    try:
        report = asyncio.run(advisor.estimate_from_params(
            symbol=symbol.upper(), side=side, entry=entry,
            target=target, stop=stop, order_size_usd=order_size,
        ))
    except Exception as exc:
        _handle_error(exc)
    if as_json:
        typer.echo(json.dumps(asdict(report), indent=2))
    else:
        typer.echo(render_slippage_text(report))


@app.command("correlate")
def correlate_cmd(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated Binance futures symbols."),
    interval: str = typer.Option("1h", "--interval"),
    window: int = typer.Option(100, "--window", min=10),
    top: int = typer.Option(10, "--top", min=1),
    as_json: bool = typer.Option(False, "--json"),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Compute pairwise return correlations across a set of symbols."""
    from dataclasses import asdict
    parsed = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(parsed) < 2:
        typer.echo("Please provide at least 2 symbols via --symbols.", err=True)
        raise typer.Exit(code=1)
    _emit_progress_message(
        f"Fetching {window} bars of {interval} candles for {len(parsed)} symbols and computing correlations...",
        enabled=not as_json,
    )
    analyzer = CorrelationAnalyzer(window_bars=window, interval=interval)
    try:
        report = asyncio.run(analyzer.analyze(parsed))
    except Exception as exc:
        _handle_error(exc)
    text_report = render_correlation_text(report, top=top)
    if export is not None:
        _export_report(export, title="Correlation Report", text_body=text_report)
    if as_json:
        typer.echo(json.dumps({
            "symbols": report.symbols, "interval": report.interval,
            "window_bars": report.window_bars,
            "diversification_score": report.diversification_score,
            "pairs": [asdict(p) for p in report.pairs],
            "cluster_groups": report.cluster_groups,
            "hedge_pairs": [asdict(p) for p in report.hedge_pairs],
        }, indent=2))
    else:
        typer.echo(text_report)


# ── History subcommands ───────────────────────────────────────────────────────

@history_app.command("recent")
def history_recent(
    symbol: str | None = typer.Option(None, "--symbol"),
    limit: int = typer.Option(10, "--limit", min=1),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Show recent saved predictions."""
    rows = _history_service().recent(symbol=symbol.upper() if symbol else None, limit=limit)
    text_report = render_history_recent_text(rows)
    if export is not None:
        _export_report(export, title="Recent History Report", text_body=text_report)
    typer.echo(text_report)


@history_app.command("stats")
def history_stats(
    symbol: str | None = typer.Option(None, "--symbol"),
    days: int | None = typer.Option(None, "--days", min=1),
    min_rsi: float | None = typer.Option(None, "--min-rsi"),
    max_rsi: float | None = typer.Option(None, "--max-rsi"),
    regime: str | None = typer.Option(None, "--regime"),
    min_ob_imbalance: float | None = typer.Option(None, "--min-ob-imbalance"),
    volatility_regime: str | None = typer.Option(None, "--volatility-regime"),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Show performance stats, optionally filtered by enhanced metrics."""
    service = _history_service()
    sym = symbol.upper() if symbol else None
    has_filter = any(v is not None for v in [min_rsi, max_rsi, regime, min_ob_imbalance, volatility_regime])
    if has_filter:
        report = service.stats_with_filter(
            symbol=sym, days=days, min_rsi=min_rsi, max_rsi=max_rsi,
            regime=regime, min_ob_imbalance=min_ob_imbalance, volatility_regime=volatility_regime,
        )
    else:
        report = service.stats(symbol=sym, days=days)
    text_report = render_history_stats_text(report)
    if export is not None:
        _export_report(export, title="History Stats Report", text_body=text_report)
    typer.echo(text_report)


@history_app.command("feedback")
def history_feedback(
    symbol: str | None = typer.Option(None, "--symbol"),
    limit: int = typer.Option(10, "--limit", min=1),
    days: int | None = typer.Option(None, "--days", min=1),
    min_rsi: float | None = typer.Option(None, "--min-rsi"),
    max_rsi: float | None = typer.Option(None, "--max-rsi"),
    regime: str | None = typer.Option(None, "--regime"),
    min_ob_imbalance: float | None = typer.Option(None, "--min-ob-imbalance"),
    volatility_regime: str | None = typer.Option(None, "--volatility-regime"),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Show resolved prediction feedback."""
    service = _history_service()
    sym = symbol.upper() if symbol else None
    has_filter = any(v is not None for v in [min_rsi, max_rsi, regime, min_ob_imbalance, volatility_regime])
    if has_filter:
        rows = service.repository.evaluated_with_filter(
            symbol=sym, days=days, min_rsi=min_rsi, max_rsi=max_rsi,
            regime=regime, min_ob_imbalance=min_ob_imbalance, volatility_regime=volatility_regime,
        )[:limit]
    else:
        rows = service.feedback(symbol=sym, limit=limit, days=days)
    overview = service.feedback_overview(symbol=sym, days=days)
    text_report = render_feedback_text(rows, overview)
    if export is not None:
        _export_report(export, title="Prediction Feedback Report", text_body=text_report)
    typer.echo(text_report)


@history_app.command("compare")
def history_compare(
    by: HistoryCompareBy = typer.Option(..., "--by"),
    symbol: str | None = typer.Option(None, "--symbol"),
    days: int | None = typer.Option(None, "--days", min=1),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Compare evaluated results by symbol, style, mode, or confidence."""
    report = _history_service().compare(
        compare_by=by, symbol=symbol.upper() if symbol else None, days=days,
    )
    text_report = render_history_compare_text(report)
    if export is not None:
        _export_report(export, title=f"History Compare Report - {by.value}", text_body=text_report)
    typer.echo(text_report)


@history_app.command("windows")
def history_windows(
    symbol: str | None = typer.Option(None, "--symbol"),
    days: int | None = typer.Option(None, "--days", min=1),
    export: Path | None = typer.Option(None, "--export"),
) -> None:
    """Show per-window (4h/24h/7d/30d) performance breakdown."""
    service = _history_service()
    buckets = service.window_stats(symbol=symbol.upper() if symbol else None, days=days)
    if not buckets:
        typer.echo("No window evaluation data found. Run analyse/scan to generate predictions and wait for evaluation.")
        return
    lines = ["Window Performance Breakdown",
             f"  {'WINDOW':<8} {'N':>5}  {'WIN%':>7}  {'AVG PNL%':>10}  {'AVG MFE%':>10}"]
    for b in buckets:
        lines.append(f"  {b.window:<8} {b.sample_count:>5}  {b.win_rate:>7.1%}  {b.avg_pnl_pct:>+10.3f}  {b.avg_mfe_pct:>10.3f}")
    text_report = "\n".join(lines)
    if export is not None:
        _export_report(export, title="Window Performance Report", text_body=text_report)
    typer.echo(text_report)


@history_app.command("backfill-metrics")
def history_backfill_metrics(
    limit: int = typer.Option(500, "--limit", min=1),
) -> None:
    """Backfill enhanced metrics for existing snapshots that are missing them."""
    filled = _history_service().backfill_enhanced_metrics(limit=limit)
    typer.echo(f"Backfilled enhanced metrics for {filled} snapshot(s).")


@history_app.command("drawdown")
def history_drawdown(
    symbol: str | None = typer.Option(None, "--symbol", help="Filter to a specific symbol."),
    lookback: int = typer.Option(10, "--lookback", min=1, help="Number of recent resolved trades to examine."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show rolling drawdown state from recent resolved predictions."""
    state = _history_service().recent_drawdown_state(symbol=symbol, lookback=lookback)
    if as_json:
        from dataclasses import asdict
        typer.echo(json.dumps(asdict(state), indent=2))
    else:
        typer.echo(render_drawdown_text(state))


@history_app.command("clear")
def history_clear() -> None:
    """Delete all stored history records."""
    confirm = typer.prompt("This will permanently delete all history. Continue? [Y/n]", default="n")
    if confirm.strip().lower() != "y":
        typer.echo("Aborted.")
        raise typer.Exit()
    deleted = _history_service().clear_all()
    typer.echo(f"Cleared {deleted} record(s) from history.")
