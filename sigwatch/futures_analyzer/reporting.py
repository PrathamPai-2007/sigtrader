from __future__ import annotations

from html import escape
from pathlib import Path

from futures_analyzer.analysis.models import AnalysisResult, ContributorDetail
from futures_analyzer.history.models import HistoryCompareReport, HistorySnapshot, HistoryStatsReport, StatsBucket


def _format_contributors(prefix: str, contributors: list[ContributorDetail]) -> list[str]:
    if not contributors:
        return [f"{prefix} none"]
    return [f"{prefix} {item.label}: {item.summary} (impact {item.impact:.3f})" for item in contributors]


def _last_feasible_line(
    *,
    label: str,
    timestamp: object,
    symbol: str | None = None,
) -> str:
    if timestamp is None:
        return f"{label}: none found in replay window."
    if symbol:
        return f"{label}: {symbol} at {timestamp.isoformat()}"
    return f"{label}: {timestamp.isoformat()}"


def render_startup_dashboard(
    *,
    version: str,
    overview: "StatsBucket | None",
    pending_count: int,
    last_symbol: str | None,
    last_evaluated_ago: str | None,
) -> str:
    lines = [
        "",
        "  crypto-predictor  " + version,
        "  Binance Futures Setup Analyzer",
        "",
        "  Recent performance (last 14 days)",
        "  " + "─" * 34,
    ]
    if overview is not None:
        lines += [
            f"  Evaluated   : {overview.sample_count} setups",
            f"  Win rate    : {overview.target_hit_rate:.1%}",
            f"  Pending     : {pending_count} awaiting outcome",
            f"  Last symbol : {last_symbol or 'n/a'}" + (f"  ({last_evaluated_ago})" if last_evaluated_ago else ""),
        ]
    else:
        lines.append("  No history yet — run  analyse  to get started.")
    lines += [
        "",
        "  Available commands",
        "  " + "─" * 34,
        "  analyse   Analyse a single symbol",
        "  scan      Scan a list of symbols",
        "  find      Auto-find best setups",
        "  presets   List strategy presets",
        "  history   View stored history",
        "",
        "  Run  crypto-predictor <command> --help  for details.",
        "",
    ]
    return "\n".join(lines)


def render_analysis_text(result: AnalysisResult) -> str:
    primary = result.primary_setup
    secondary = result.secondary_context
    meta = result.market_snapshot_meta
    lines = [
        f"Prediction ID: {result.prediction_id or 'pending'}",
        f"Symbol: {meta.symbol}",
        f"As Of (UTC): {meta.as_of.isoformat()}",
        f"Market Regime: {result.market_regime.value} ({result.regime_confidence:.2f})",
        _last_feasible_line(
            label="Last feasible setup from chart replay for this symbol under current rules",
            timestamp=result.chart_replay_last_tradable_at,
        ),
        "",
        "Primary Setup",
        f"  Side: {primary.side.upper()}",
        f"  Entry: {primary.entry_price}",
        f"  Target: {primary.target_price}",
        f"  Stop Loss: {primary.stop_loss}",
        f"  Potential Leverage: {primary.leverage_suggestion}",
        f"  Confidence: {primary.confidence:.2f}",
        f"  Quality: {primary.quality_label.value.upper()} ({primary.quality_score:.1f}/100)",
        f"  Tradable: {'YES' if primary.is_tradable else 'NO'}",
        f"  Evidence Agreement: {primary.evidence_agreement}/{primary.evidence_total}",
        f"  R:R Ratio: {primary.risk_reward_ratio:.2f}",
        f"  Rationale: {primary.rationale}",
        f"  Deliberation: {primary.deliberation_summary or 'Awaiting full agreement review.'}",
        "  Top Positives:",
    ]
    if primary.tradable_reasons:
        lines.append("  Trade Filters:")
        for reason in primary.tradable_reasons:
            lines.append(f"    - {reason}")
    lines.extend(_format_contributors("    +", primary.top_positive_contributors))
    lines.append("  Top Negatives:")
    lines.extend(_format_contributors("    -", primary.top_negative_contributors))
    lines.extend(
        [
            "",
            "Secondary Context",
            f"  Side: {secondary.side.upper()} | Leverage {secondary.leverage_suggestion} | Quality {secondary.quality_label.value.upper()} ({secondary.quality_score:.1f}) | Confidence {secondary.confidence:.2f} | Tradable {'YES' if secondary.is_tradable else 'NO'} | Evidence {secondary.evidence_agreement}/{secondary.evidence_total}",
            f"  Entry {secondary.entry_price} | Target {secondary.target_price} | Stop {secondary.stop_loss}",
            f"  R:R {secondary.risk_reward_ratio:.2f}",
        ]
    )
    if result.warnings:
        lines.append("")
        lines.append("Warnings")
        for warning in result.warnings:
            lines.append(f"  - {warning}")
    lines.extend(["", f"Disclaimer: {result.disclaimer}"])
    return "\n".join(lines)


def render_scan_text(
    ranked: list[tuple[int, AnalysisResult]],
    failures: list[dict[str, str]],
    *,
    title: str = "Scan Results",
    intro_lines: list[str] | None = None,
    latest_replay_symbol: str | None = None,
    latest_replay_at: object = None,
    filtered_out: int = 0,
) -> str:
    lines = [
        title,
    ]
    if intro_lines:
        lines.extend(intro_lines)
    lines.append(
        _last_feasible_line(
            label="Last feasible setup from chart replay under current rules",
            symbol=latest_replay_symbol,
            timestamp=latest_replay_at,
        )
    )
    lines.append("RK  ID        SYMBOL     SIDE  QUAL   REGIME           LEV R:R  ENTRY      TARGET     STOP       STOP% TARGET%")
    for rank, result in ranked:
        primary = result.primary_setup
        lines.append(
            f"{rank:>2}  {(result.prediction_id or 'pending'):<8} {result.market_snapshot_meta.symbol:<10} {primary.side.upper():<5} "
            f"{primary.quality_label.value:<6} {result.market_regime.value:<14} "
            f"{primary.leverage_suggestion:>3} {primary.risk_reward_ratio:>4.2f} "
            f"{primary.entry_price:>10.4f} {primary.target_price:>10.4f} {primary.stop_loss:>10.4f} "
            f"{primary.stop_distance_pct:>6.3f} {primary.target_distance_pct:>7.3f}"
        )
    if not ranked:
        lines.append("No tradable setups met the current filter criteria.")
    if filtered_out:
        lines.append(f"Filtered out {filtered_out} setup(s) that failed the hard trade filters.")
    if failures:
        lines.append("")
        lines.append("Errors")
        for failure in failures:
            lines.append(f"  {failure['symbol']}: {failure['error']}")
    return "\n".join(lines)


def render_find_text(
    ranked: list[tuple[int, AnalysisResult]],
    failures: list[dict[str, str]],
    *,
    candidate_count: int,
    market_mode: str,
    latest_replay_symbol: str | None = None,
    latest_replay_at: object = None,
    filtered_out: int = 0,
    fallback_results: list[tuple[int, AnalysisResult]] | None = None,
) -> str:
    body = render_scan_text(
        ranked,
        failures,
        title="Find Results",
        intro_lines=[f"Analyzed {candidate_count} liquid {market_mode.replace('_', ' ')} candidate symbol(s)."],
        latest_replay_symbol=latest_replay_symbol,
        latest_replay_at=latest_replay_at,
        filtered_out=filtered_out,
    )
    if fallback_results:
        lines = [
            "",
            "⚠  No setup passed the hard trade filters. "
            "Showing near-miss candidates below — review carefully before acting.",
            "",
            "NEAR-MISS CANDIDATES (not tradable under current rules)",
            "RK  ID        SYMBOL     SIDE  QUAL   REGIME           LEV R:R  ENTRY      TARGET     STOP       STOP% TARGET%",
        ]
        for rank, result in fallback_results:
            primary = result.primary_setup
            lines.append(
                f"{rank:>2}  {(result.prediction_id or 'pending'):<8} "
                f"{result.market_snapshot_meta.symbol:<10} {primary.side.upper():<5} "
                f"{primary.quality_label.value:<6} {result.market_regime.value:<14} "
                f"{primary.leverage_suggestion:>3} {primary.risk_reward_ratio:>4.2f} "
                f"{primary.entry_price:>10.4f} {primary.target_price:>10.4f} {primary.stop_loss:>10.4f} "
                f"{primary.stop_distance_pct:>6.3f} {primary.target_distance_pct:>7.3f}"
            )
        body += "\n".join(lines)
    return body


def render_history_recent_text(rows: list[HistorySnapshot]) -> str:
    if not rows:
        return "No saved analysis history found."
    lines = ["Recent History"]
    for row in rows:
        outcome = row.outcome or row.evaluation_status
        lines.append(
            f"  {row.prediction_id} | {row.as_of.isoformat()} | {row.symbol} | {row.side.upper()} | {row.quality_label.upper()} "
            f"({row.quality_score:.1f}) | conf {row.confidence:.2f} | regime {row.regime} | "
            f"style {row.style} | mode {row.market_mode} | cmd {row.command} | "
            f"profile {row.profile_name} | tf {row.entry_timeframe}/{row.trigger_timeframe}/{row.context_timeframe}/{row.higher_timeframe} | outcome {outcome}"
        )
    return "\n".join(lines)


def render_history_stats_text(report: HistoryStatsReport) -> str:
    lines: list[str] = []

    if report.overall_feedback is not None:
        row = report.overall_feedback
        lines.append(
            "Overall Feedback"
        )
        lines.append(
            f"  {row.bucket}: n={row.sample_count} | target {row.target_hit_rate:.2%} | stop {row.stop_hit_rate:.2%} | "
            f"profitable {row.profitable_at_24h_rate:.2%} | avg pnl {row.average_24h_pnl:.3f}% | "
            f"avg mfe {row.average_mfe:.3f}% | avg mae {row.average_mae:.3f}%"
        )
        lines.append("")

    def append_section(title: str, rows: list[StatsBucket]) -> None:
        lines.append(title)
        if not rows:
            lines.append("  No evaluated snapshots yet.")
            return
        for row in rows:
            lines.append(
                f"  {row.bucket}: n={row.sample_count} | target {row.target_hit_rate:.2%} | stop {row.stop_hit_rate:.2%} | "
                f"profitable {row.profitable_at_24h_rate:.2%} | avg pnl {row.average_24h_pnl:.3f}% | "
                f"avg mfe {row.average_mfe:.3f}% | avg mae {row.average_mae:.3f}%"
            )

    append_section("Confidence Buckets", report.confidence_buckets)
    lines.append("")
    append_section("Quality Buckets", report.quality_buckets)
    lines.append("")
    append_section("Regime Buckets", report.regime_buckets)
    return "\n".join(lines)


def render_feedback_text(rows: list[HistorySnapshot], overview: StatsBucket | None) -> str:
    lines: list[str] = ["Prediction Feedback"]
    if overview is not None:
        lines.append(
            f"Summary: n={overview.sample_count} | target {overview.target_hit_rate:.2%} | stop {overview.stop_hit_rate:.2%} | "
            f"profitable {overview.profitable_at_24h_rate:.2%} | avg pnl {overview.average_24h_pnl:.3f}%"
        )
    if not rows:
        lines.append("No resolved predictions yet.")
        return "\n".join(lines)
    for row in rows:
        outcome = row.outcome or row.evaluation_status
        pnl = f"{row.pnl_at_24h_close_pct:.3f}%" if row.pnl_at_24h_close_pct is not None else "n/a"
        lines.append(
            f"  {row.prediction_id} | {row.symbol} | {row.side.upper()} | {row.style} | {row.market_mode} | outcome {outcome} | pnl {pnl} | "
            f"mfe {row.max_favorable_excursion_pct if row.max_favorable_excursion_pct is not None else 'n/a'} | "
            f"mae {row.max_adverse_excursion_pct if row.max_adverse_excursion_pct is not None else 'n/a'}"
        )
    return "\n".join(lines)


def render_history_compare_text(report: HistoryCompareReport) -> str:
    lines: list[str] = [f"History Compare By {report.compare_by.value.upper()}"]
    if report.overall_feedback is not None:
        row = report.overall_feedback
        lines.append(
            f"Overall: n={row.sample_count} | target {row.target_hit_rate:.2%} | stop {row.stop_hit_rate:.2%} | "
            f"profitable {row.profitable_at_24h_rate:.2%} | avg pnl {row.average_24h_pnl:.3f}%"
        )
    if not report.buckets:
        lines.append("No evaluated snapshots yet.")
        return "\n".join(lines)
    lines.append("")
    for row in report.buckets:
        lines.append(
            f"  {row.bucket}: n={row.sample_count} | target {row.target_hit_rate:.2%} | stop {row.stop_hit_rate:.2%} | "
            f"profitable {row.profitable_at_24h_rate:.2%} | avg pnl {row.average_24h_pnl:.3f}% | "
            f"avg mfe {row.average_mfe:.3f}% | avg mae {row.average_mae:.3f}%"
        )
    return "\n".join(lines)


def write_report(path: Path, *, title: str, text_body: str) -> Path:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".pdf":
        raise ValueError("Direct PDF export is not supported. Export to .html and use your browser's Print to PDF.")
    if suffix == ".html":
        html = _html_document(title=title, text_body=text_body)
        path.write_text(html, encoding="utf-8")
        return path
    if suffix in {".md", ".markdown"}:
        path.write_text(_markdown_document(title=title, text_body=text_body), encoding="utf-8")
        return path
    path.write_text(text_body, encoding="utf-8")
    return path


def _markdown_document(*, title: str, text_body: str) -> str:
    return f"# {title}\n\n```text\n{text_body}\n```\n"


def _html_document(*, title: str, text_body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #5c6773;
      --accent: #b45309;
      --border: #ded4c4;
      --shadow: 0 20px 45px rgba(31, 41, 51, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 32px 18px;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180, 83, 9, 0.12), transparent 30%),
        linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
    }}
    .report {{
      max-width: 960px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .hero {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(135deg, rgba(180, 83, 9, 0.12), rgba(255, 255, 255, 0.7));
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.8rem, 3vw, 2.4rem);
      line-height: 1.1;
    }}
    p {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 1rem;
    }}
    pre {{
      margin: 0;
      padding: 28px 32px 32px;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 0.95rem;
      line-height: 1.55;
      color: var(--ink);
      background: transparent;
    }}
    @media print {{
      body {{
        padding: 0;
        background: #fff;
      }}
      .report {{
        border: none;
        box-shadow: none;
      }}
    }}
  </style>
</head>
<body>
  <article class="report">
    <header class="hero">
      <h1>{escape(title)}</h1>
      <p>Readable export from crypto-predictor. Open in a browser and print to PDF if needed.</p>
    </header>
    <pre>{escape(text_body)}</pre>
  </article>
</body>
</html>
"""


def render_slippage_text(report: object) -> str:
    viable = "YES" if report.is_still_viable else "NO — R:R below threshold after slippage"
    lines = [
        "Slippage Analysis",
        f"  Symbol          : {report.symbol}",
        f"  Side            : {report.side.upper()}",
        f"  Order size      : ${report.order_size_usd:,.0f}",
        f"  Model           : {report.model_used}",
        f"  Volatility      : {report.volatility_regime}",
        f"  Spread          : {report.spread_pct:.4f}%",
        f"  Liquidity score : {report.liquidity_score:.1f}/100",
        "",
        "  Price Impact",
        f"  {'':20} {'Raw':>12}  {'Adjusted':>12}  {'Slippage':>10}",
        f"  {'Entry':20} {report.raw_entry:>12.4f}  {report.adj_entry:>12.4f}  {report.entry_slippage_pct:>9.4f}%",
        f"  {'Target':20} {report.raw_target:>12.4f}  {report.adj_target:>12.4f}  {report.target_slippage_pct:>9.4f}%",
        f"  {'Stop':20} {report.raw_stop:>12.4f}  {report.adj_stop:>12.4f}  {report.stop_slippage_pct:>9.4f}%",
        "",
        f"  Raw R:R         : {report.raw_rr:.3f}",
        f"  Adjusted R:R    : {report.adj_rr:.3f}",
        f"  R:R degradation : {report.rr_degradation:.3f}",
        f"  Still viable    : {viable}",
    ]
    return "\n".join(lines)


def render_correlation_text(report: object, *, top: int = 10) -> str:
    lines = [
        "Correlation Analysis",
        f"  Symbols  : {', '.join(report.symbols)}",
        f"  Interval : {report.interval}",
        f"  Window   : {report.window_bars} bars",
        f"  Diversification score: {report.diversification_score:.1f}/100",
        "",
    ]

    if report.pairs:
        lines.append(f"  Top {min(top, len(report.pairs))} pairs by |correlation|")
        lines.append(f"  {'SYMBOL A':<12} {'SYMBOL B':<12} {'CORR':>7}")
        for pair in report.pairs[:top]:
            bar = "█" * int(abs(pair.correlation) * 10)
            sign = "+" if pair.correlation >= 0 else "-"
            lines.append(f"  {pair.symbol_a:<12} {pair.symbol_b:<12} {pair.correlation:>+7.3f}  {sign}{bar}")

    if report.cluster_groups:
        lines += ["", "  Correlated clusters (risk concentration)"]
        for i, group in enumerate(report.cluster_groups, 1):
            lines.append(f"    Group {i}: {', '.join(group)}")

    if report.hedge_pairs:
        lines += ["", "  Hedge pairs (negative correlation)"]
        for pair in report.hedge_pairs:
            lines.append(f"    {pair.symbol_a} / {pair.symbol_b}: {pair.correlation:+.3f}")

    if not report.pairs:
        lines.append("  Not enough aligned data to compute correlations.")

    return "\n".join(lines)


def render_drawdown_text(state: object) -> str:
    """Render a DrawdownState as human-readable text."""
    severity_label = state.severity.upper()
    lines = [
        "Drawdown Recovery Analysis",
        f"  Severity          : {severity_label}",
        f"  Sample trades     : {state.sample_count} (lookback {state.lookback})",
        f"  Cumulative PnL    : {state.cumulative_pnl_pct:+.3f}%",
        f"  Max drawdown      : {state.max_drawdown_pct:.3f}%",
        f"  Current drawdown  : {state.current_drawdown_pct:.3f}%",
        f"  Consecutive losses: {state.consecutive_losses}",
    ]
    if state.severity == "none":
        lines.append("  Status            : No drawdown guard active.")
    else:
        lines.append(
            f"  Status            : Drawdown guard ACTIVE — leverage and quality "
            f"will be reduced on new setups until drawdown recovers."
        )
    return "\n".join(lines)


def render_portfolio_text(report: object, *, capital: float) -> str:
    """Render a PortfolioRiskReport as human-readable text."""
    lines = [
        f"Portfolio Allocation  (capital: ${capital:,.2f})",
        f"  Total notional : ${report.total_notional_usd:,.2f}  ({report.total_notional_usd / capital:.1%})",
        f"  Total risk     : ${report.total_risk_usd:,.2f}  ({report.total_risk_pct:.2%})",
        "",
        f"  {'SYM':<12} {'SIDE':<6} {'NOTIONAL':>12}  {'RISK $':>10}  {'RISK%':>7}  {'CLUSTER':>8}  CAPPED",
    ]
    for alloc in report.allocations:
        cluster = alloc.cluster_id or "—"
        capped = f"yes  ({alloc.cap_reason})" if alloc.capped else "no"
        lines.append(
            f"  {alloc.symbol:<12} {alloc.side.upper():<6} "
            f"${alloc.notional_usd:>11,.2f}  "
            f"${alloc.risk_usd:>9,.2f}  "
            f"{alloc.risk_pct_of_capital:>6.2%}  "
            f"{cluster:>8}  {capped}"
        )
    if report.cluster_warnings:
        lines += ["", "  Cluster Warnings"]
        for w in report.cluster_warnings:
            lines.append(f"    {w}")
    if report.breached_rules:
        lines += ["", "  Rule Breaches"]
        for b in report.breached_rules:
            lines.append(f"    {b}")
    return "\n".join(lines)
