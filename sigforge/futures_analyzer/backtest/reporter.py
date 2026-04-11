from __future__ import annotations

from collections import defaultdict

from futures_analyzer.backtest.models import BacktestReport, BacktestTrade


def _bucket(trades: list[BacktestTrade], key_fn) -> list[tuple[str, BacktestReport]]:
    """Group trades by key_fn and return mini-reports per bucket."""
    from futures_analyzer.backtest.models import BacktestReport
    grouped: dict[str, list[BacktestTrade]] = defaultdict(list)
    for t in trades:
        grouped[key_fn(t)].append(t)
    result = []
    for bucket_key in sorted(grouped):
        mini = BacktestReport(config=None, trades=grouped[bucket_key])  # type: ignore[arg-type]
        mini.compute_aggregates()
        result.append((bucket_key, mini))
    return result


def _confidence_bucket(trade: BacktestTrade) -> str:
    if trade.confidence < 0.45:
        return "0.00-0.44"
    if trade.confidence < 0.70:
        return "0.45-0.69"
    return "0.70-1.00"


def _stats_line(label: str, report: BacktestReport) -> str:
    return (
        f"  {label:<20} n={report.total_trades:>4} | "
        f"target {report.target_hit_rate:.1%} | stop {report.stop_hit_rate:.1%} | "
        f"win {report.win_rate:.1%} | avg pnl {report.avg_pnl_pct:+.3f}% | "
        f"mfe {report.avg_mfe_pct:.3f}% | mae {report.avg_mae_pct:.3f}% | "
        f"expect {report.expectancy:+.3f}%"
    )


def render_backtest_text(report: BacktestReport, *, verbose: bool = False) -> str:
    cfg = report.config
    lines: list[str] = [
        "Backtest Report",
        f"  Symbol     : {cfg.symbol}",
        f"  Range      : {cfg.start.date()} → {cfg.end.date()}",
        f"  Style      : {cfg.style.value}",
        f"  Mode       : {cfg.market_mode.value}",
        f"  Risk/Reward: {cfg.risk_reward}",
        f"  Preset     : {cfg.preset or 'none'}",
        "",
        "Aggregate Results",
        f"  Total trades   : {report.total_trades}",
        f"  Target hit rate: {report.target_hit_rate:.2%}",
        f"  Stop hit rate  : {report.stop_hit_rate:.2%}",
        f"  Win rate       : {report.win_rate:.2%}",
        f"  Avg PnL        : {report.avg_pnl_pct:+.3f}%",
        f"  Avg MFE        : {report.avg_mfe_pct:.3f}%",
        f"  Avg MAE        : {report.avg_mae_pct:.3f}%",
        f"  Expectancy     : {report.expectancy:+.3f}%",
        f"  Max drawdown   : {report.max_drawdown_pct:.3f}%",
        f"  Sharpe (approx): {report.sharpe_approx:.3f}",
        f"  Bars skipped   : history={report.bars_skipped_insufficient_history}, analysis_errors={report.bars_skipped_analysis_error}",
    ]

    if report.rejection_reasons:
        lines += ["", "Top Rejection Reasons"]
        ranked = sorted(report.rejection_reasons.items(), key=lambda item: (-item[1], item[0]))
        for reason, count in ranked[:10]:
            lines.append(f"  {count:>4} | {reason}")

    if report.stop_anchor_counts:
        lines += ["", "Stop Anchors"]
        ranked = sorted(report.stop_anchor_counts.items(), key=lambda item: (-item[1], item[0]))
        for anchor, count in ranked:
            lines.append(f"  {count:>4} | {anchor}")

    if report.target_anchor_counts:
        lines += ["", "Target Anchors"]
        ranked = sorted(report.target_anchor_counts.items(), key=lambda item: (-item[1], item[0]))
        for anchor, count in ranked:
            lines.append(f"  {count:>4} | {anchor}")

    if report.trades:
        lines += ["", "By Regime"]
        for label, mini in _bucket(report.trades, lambda t: t.regime):
            lines.append(_stats_line(label, mini))

        lines += ["", "By Confidence"]
        for label, mini in _bucket(report.trades, _confidence_bucket):
            lines.append(_stats_line(label, mini))

        lines += ["", "By Side"]
        for label, mini in _bucket(report.trades, lambda t: t.side):
            lines.append(_stats_line(label, mini))

    if verbose and report.trades:
        lines += ["", "Trade List",
                  "  TIME                     SIDE  QUAL   REGIME           CONF  OUTCOME              PNL%    MFE%    MAE%"]
        for t in report.trades:
            lines.append(
                f"  {t.bar_time.isoformat():<24} {t.side.upper():<5} {t.quality_label:<6} "
                f"{t.regime:<16} {t.confidence:.2f}  {t.outcome:<20} "
                f"{t.pnl_pct:>+7.3f} {t.mfe_pct:>7.3f} {t.mae_pct:>7.3f}"
            )

    return "\n".join(lines)


def render_walk_forward_text(reports: list[BacktestReport]) -> str:
    lines = ["Walk-Forward Results", f"  Folds: {len(reports)}", ""]
    for i, report in enumerate(reports, start=1):
        cfg = report.config
        lines.append(
            f"  Fold {i}: {cfg.start.date()} → {cfg.end.date()} | "
            f"n={report.total_trades} | win {report.win_rate:.1%} | "
            f"avg pnl {report.avg_pnl_pct:+.3f}% | expect {report.expectancy:+.3f}% | "
            f"max dd {report.max_drawdown_pct:.3f}%"
        )

    # Combined summary across all folds
    all_trades = [t for r in reports for t in r.trades]
    if all_trades:
        from futures_analyzer.backtest.models import BacktestReport as BR
        combined = BR(config=reports[0].config, trades=all_trades)
        combined.compute_aggregates()
        lines += [
            "",
            "Combined (all folds)",
            f"  n={combined.total_trades} | win {combined.win_rate:.1%} | "
            f"avg pnl {combined.avg_pnl_pct:+.3f}% | expect {combined.expectancy:+.3f}% | "
            f"max dd {combined.max_drawdown_pct:.3f}% | sharpe {combined.sharpe_approx:.3f}",
        ]

    return "\n".join(lines)
