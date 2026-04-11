import asyncio
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path

from typer.testing import CliRunner

from futures_analyzer.analysis.models import (
    AnalysisResult,
    Candle,
    ContributorDetail,
    ContributorDirection,
    MarketMode,
    MarketMeta,
    MarketRegime,
    QualityLabel,
    StrategyStyle,
    TimeframePlan,
    TradeSetup,
)
from futures_analyzer.cli import _analyze_async, app
from futures_analyzer.history.models import HistorySnapshot, HistoryStatsReport, StatsBucket

runner = CliRunner()


def _scratch_path(name: str) -> Path:
    path = Path(name)
    if path.exists():
        path.unlink()
    return path


class _FakeHistoryService:
    def __init__(self) -> None:
        self.saved: list[tuple[list[AnalysisResult], str]] = []

    async def record_results(self, results: list[AnalysisResult], *, command: str) -> list[str]:
        self.saved.append((results, command))
        ids: list[str] = []
        for idx, result in enumerate(results, start=1):
            result.prediction_id = result.prediction_id or f"BTC{idx:05d}"
            ids.append(result.prediction_id)
        return ids

    def recent(self, *, symbol: str | None = None, limit: int = 20):
        now = datetime(2026, 1, 2, tzinfo=UTC)
        return [
            HistorySnapshot(
                id=1,
                prediction_id="BTC00001",
                symbol=symbol or "BTCUSDT",
                as_of=now,
                command="analyse",
                style="conservative",
                market_mode="intraday",
                entry_timeframe="5m",
                trigger_timeframe="15m",
                context_timeframe="1h",
                higher_timeframe="4h",
                side="long",
                entry_price=100.0,
                target_price=103.0,
                stop_loss=98.5,
                confidence=0.72,
                quality_label="high",
                quality_score=82.0,
                regime="bullish_trend",
                regime_confidence=0.8,
                risk_reward_ratio=2.0,
                stop_distance_pct=1.5,
                target_distance_pct=3.0,
                atr_multiple_to_stop=1.2,
                atr_multiple_to_target=2.4,
                invalidation_strength=0.78,
                top_positive_contributors_json="[]",
                top_negative_contributors_json="[]",
                score_components_json="{}",
                analysis_json="{}",
                evaluation_status="resolved",
                outcome="target_hit",
            )
        ]

    def stats(self, *, symbol: str | None = None, days: int | None = None) -> HistoryStatsReport:
        bucket = StatsBucket(
            bucket="high",
            sample_count=3,
            target_hit_rate=0.6667,
            stop_hit_rate=0.0,
            profitable_at_24h_rate=1.0,
            average_24h_pnl=2.4,
            average_mfe=3.5,
            average_mae=0.9,
        )
        return HistoryStatsReport(
            overall_feedback=bucket.model_copy(update={"bucket": "overall"}),
            confidence_buckets=[bucket.model_copy(update={"bucket": "0.70-1.00"})],
            quality_buckets=[bucket],
            regime_buckets=[bucket.model_copy(update={"bucket": "bullish_trend"})],
        )

    def feedback(self, *, symbol: str | None = None, limit: int = 20, days: int | None = None):
        return self.recent(symbol=symbol, limit=limit)

    def feedback_overview(self, *, symbol: str | None = None, days: int | None = None):
        return self.stats(symbol=symbol, days=days).overall_feedback

    def compare(self, *, compare_by, symbol: str | None = None, days: int | None = None):
        bucket = StatsBucket(
            bucket="intraday",
            sample_count=2,
            target_hit_rate=0.5,
            stop_hit_rate=0.0,
            profitable_at_24h_rate=1.0,
            average_24h_pnl=1.8,
            average_mfe=2.5,
            average_mae=0.7,
        )
        from futures_analyzer.history.models import HistoryCompareReport

        return HistoryCompareReport(compare_by=compare_by, overall_feedback=bucket, buckets=[bucket])

    def recent_drawdown_state(self, *, symbol: str | None = None, lookback: int | None = None):
        from futures_analyzer.history.models import DrawdownState
        return DrawdownState(
            lookback=lookback or 10,
            cumulative_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            current_drawdown_pct=0.0,
            consecutive_losses=0,
            severity="none",
            sample_count=0,
        )

    def kelly_inputs(self, *, symbol: str | None = None, lookback: int = 50):
        return (0.5, 1.0, 1.0)
def _contributor(key: str, label: str, impact: float, direction: ContributorDirection) -> ContributorDetail:
    return ContributorDetail(
        key=key,
        label=label,
        value=impact / 2,
        impact=impact,
        direction=direction,
        summary=f"{label} summary",
    )



def _fake_result(symbol: str = "BTCUSDT") -> AnalysisResult:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    setup_long = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=103.0,
        stop_loss=98.5,
        leverage_suggestion="3x",
        confidence=0.72,
        quality_label=QualityLabel.HIGH,
        quality_score=84.0,
        rationale="Synthetic long setup",
        top_positive_contributors=[
            _contributor("trend", "Trend", 2.4, ContributorDirection.POSITIVE),
            _contributor("momentum", "Momentum", 1.8, ContributorDirection.POSITIVE),
        ],
        top_negative_contributors=[
            _contributor("target_ambition_penalty", "Target Stretch", 0.4, ContributorDirection.NEGATIVE)
        ],
        score_components={"trend": 2.4, "risk_reward_ratio": 2.0, "invalidation_strength": 0.78},
        structure_points={"support": 98.5, "resistance": 103.2, "atr": 1.2},
        risk_reward_ratio=2.0,
        stop_distance_pct=1.5,
        target_distance_pct=3.0,
        atr_multiple_to_stop=1.25,
        atr_multiple_to_target=2.5,
        invalidation_strength=0.78,
        evidence_agreement=6,
        evidence_total=7,
        deliberation_summary="6/7 aligned",
    )
    setup_short = TradeSetup(
        side="short",
        entry_price=100.0,
        target_price=97.0,
        stop_loss=101.5,
        leverage_suggestion="2x",
        confidence=0.41,
        quality_label=QualityLabel.MEDIUM,
        quality_score=62.0,
        rationale="Synthetic short context",
        top_positive_contributors=[_contributor("structure", "Structure", 1.1, ContributorDirection.POSITIVE)],
        top_negative_contributors=[],
        score_components={"structure": 1.1},
        structure_points={"support": 98.5, "resistance": 103.2, "atr": 1.2},
        risk_reward_ratio=2.0,
        stop_distance_pct=1.5,
        target_distance_pct=3.0,
        atr_multiple_to_stop=1.25,
        atr_multiple_to_target=2.5,
        invalidation_strength=0.55,
        evidence_agreement=4,
        evidence_total=7,
        deliberation_summary="4/7 aligned",
    )
    return AnalysisResult(
        prediction_id="BTC00001",
        primary_setup=setup_long,
        secondary_context=setup_short,
        timeframe_plan=TimeframePlan(
            profile_name="intraday_core",
            style=StrategyStyle.CONSERVATIVE,
            market_mode=MarketMode.INTRADAY,
            entry_timeframe="5m",
            trigger_timeframe="15m",
            context_timeframe="1h",
            higher_timeframe="4h",
            lookback_bars=600,
        ),
        market_snapshot_meta=MarketMeta(symbol=symbol, tick_size=0.1, step_size=0.001, mark_price=100.2, as_of=now),
        market_regime=MarketRegime.BULLISH_TREND,
        regime_confidence=0.76,
        chart_replay_last_tradable_at=datetime(2025, 12, 31, 23, 45, tzinfo=UTC),
    )



def test_cli_analyze_human(monkeypatch) -> None:
    history = _FakeHistoryService()

    async def fake_analyze_async(symbol: str, risk_reward: float, style: StrategyStyle, market_mode: MarketMode, **kwargs):
        assert symbol == "BTCUSDT"
        assert risk_reward is None
        assert style == StrategyStyle.CONSERVATIVE
        assert market_mode == MarketMode.INTRADAY
        return _fake_result(symbol)

    monkeypatch.setattr("futures_analyzer.cli._analyze_async", fake_analyze_async)
    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: history)
    result = runner.invoke(app, ["analyse", "--symbol", "BTCUSDT"])
    assert result.exit_code == 0
    assert "Fetching market data and replaying recent charts to find the latest feasible setup. This can take a moment." in result.stdout
    assert "Top Positives" in result.stdout
    assert "Top Negatives" in result.stdout
    assert "R:R Ratio: 2.00" in result.stdout
    assert "Potential Leverage: 3x" in result.stdout
    assert "Prediction ID: BTC00001" in result.stdout
    assert "Last feasible setup from chart replay for this symbol under current rules: 2025-12-31T23:45:00+00:00" in result.stdout
    assert "Prediction Feedback" in result.stdout
    assert "ATR To Stop" not in result.stdout
    assert "ATR To Target" not in result.stdout
    assert "Invalidation Strength" not in result.stdout
    assert history.saved and history.saved[0][1] == "analyse"


def test_cli_analyze_export_html(monkeypatch) -> None:
    history = _FakeHistoryService()
    export_path = _scratch_path("analysis_report_test.html")

    async def fake_analyze_async(symbol: str, risk_reward: float, style: StrategyStyle, market_mode: MarketMode, **kwargs):
        return _fake_result(symbol)

    monkeypatch.setattr("futures_analyzer.cli._analyze_async", fake_analyze_async)
    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: history)
    result = runner.invoke(app, ["analyse", "--symbol", "BTCUSDT", "--export", str(export_path)])
    assert result.exit_code == 0
    assert export_path.exists()
    html = export_path.read_text(encoding="utf-8")
    assert "Analysis Report - BTCUSDT" in html
    assert "print to PDF" in html
    assert "Last feasible setup from chart replay for this symbol under current rules: 2025-12-31T23:45:00+00:00" in html
    assert "Potential Leverage: 3x" in html
    assert "Prediction Feedback" in html



def test_cli_analyze_json(monkeypatch) -> None:
    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: _FakeHistoryService())

    async def fake_analyze_async(symbol: str, risk_reward: float, style: StrategyStyle, market_mode: MarketMode, **kwargs):
        assert style == StrategyStyle.AGGRESSIVE
        assert market_mode == MarketMode.LONG_TERM
        return _fake_result(symbol)

    monkeypatch.setattr("futures_analyzer.cli._analyze_async", fake_analyze_async)
    result = runner.invoke(app, ["analyse", "--symbol", "ETHUSDT", "--style", "aggressive", "--mode", "long_term", "--json"])
    assert result.exit_code == 0
    assert '"result"' in result.stdout
    assert '"quality_label": "high"' in result.stdout
    assert '"top_positive_contributors"' in result.stdout
    assert '"risk_reward_ratio": 2.0' in result.stdout
    assert '"invalidation_strength": 0.78' in result.stdout
    assert '"latest_feasible_chart_replay"' in result.stdout
    assert '"feedback"' in result.stdout
    assert "Fetching market data and replaying recent charts" not in result.stdout



def test_cli_scan_human(monkeypatch) -> None:
    history = _FakeHistoryService()

    async def fake_scan_async(symbols: list[str], risk_reward: float, style: StrategyStyle, market_mode: MarketMode, **kwargs):
        results = [_fake_result(symbol) for symbol in symbols]
        await history.record_results(results, command="scan")
        return [(result.market_snapshot_meta.symbol, result, None) for result in results]

    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: history)
    monkeypatch.setattr("futures_analyzer.cli._scan_async", fake_scan_async)
    result = runner.invoke(app, ["scan", "--symbols", "BTCUSDT,ETHUSDT"])
    assert result.exit_code == 0
    assert "Fetching market data and replaying recent charts for the requested symbols. This can take a moment." in result.stdout
    assert "RK  ID" in result.stdout
    assert "BTCUSDT" in result.stdout
    assert "BTC00001" in result.stdout
    assert "LEV" in result.stdout
    assert "R:R" in result.stdout
    assert "100.0000" in result.stdout
    assert "103.0000" in result.stdout
    assert "98.5000" in result.stdout
    assert "Last feasible setup from chart replay under current rules: BTCUSDT at 2025-12-31T23:45:00+00:00" in result.stdout
    assert "+ Trend; Momentum" not in result.stdout
    assert "- Target Stretch" not in result.stdout
    assert "CONF" not in result.stdout
    assert "Prediction Feedback" in result.stdout
    assert history.saved and len(history.saved[0][0]) == 2
    assert len(history.saved) == 1



def test_cli_scan_json(monkeypatch) -> None:
    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: _FakeHistoryService())

    async def fake_scan_async(symbols: list[str], risk_reward: float, style: StrategyStyle, market_mode: MarketMode, **kwargs):
        return [("BTCUSDT", _fake_result("BTCUSDT"), None), ("ETHUSDT", None, RuntimeError("bad symbol"))]

    monkeypatch.setattr("futures_analyzer.cli._scan_async", fake_scan_async)
    result = runner.invoke(app, ["scan", "--symbols", "BTCUSDT,ETHUSDT", "--json"])
    assert result.exit_code == 0
    assert '"rank": 1' in result.stdout
    assert '"errors"' in result.stdout
    assert '"symbol": "ETHUSDT"' in result.stdout
    assert '"latest_feasible_chart_replay"' in result.stdout
    assert '"feedback"' in result.stdout
    assert "Fetching market data and replaying recent charts" not in result.stdout



def test_cli_find_human(monkeypatch) -> None:
    history = _FakeHistoryService()

    async def fake_find_async(
        top: int,
        universe: int,
        risk_reward: float | None,
        style: StrategyStyle,
        market_mode: MarketMode,
        **kwargs,
    ):
        assert top == 3
        assert universe == 8
        assert risk_reward is None
        assert style == StrategyStyle.AGGRESSIVE
        assert market_mode == MarketMode.INTRADAY
        result = _fake_result("BTCUSDT")
        await history.record_results([result], command="find")
        return ["BTCUSDT", "ETHUSDT"], [("BTCUSDT", result, None), ("ETHUSDT", None, RuntimeError("bad symbol"))]

    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: history)
    monkeypatch.setattr("futures_analyzer.cli._find_async", fake_find_async)
    result = runner.invoke(app, ["find", "--top", "3", "--universe", "8"])
    assert result.exit_code == 0
    assert "Finding liquid candidates, fetching market data, and replaying recent charts under the current rules. This can take a moment." in result.stdout
    assert "Find Results" in result.stdout
    assert "Analyzed 2 liquid intraday candidate symbol(s)." in result.stdout
    assert "Last feasible setup from chart replay under current rules: BTCUSDT at 2025-12-31T23:45:00+00:00" in result.stdout
    assert "BTCUSDT" in result.stdout
    assert "Prediction Feedback" in result.stdout
    assert history.saved and history.saved[0][1] == "find"
    assert len(history.saved) == 1


def test_cli_find_json(monkeypatch) -> None:
    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: _FakeHistoryService())

    async def fake_find_async(
        top: int,
        universe: int,
        risk_reward: float | None,
        style: StrategyStyle,
        market_mode: MarketMode,
        **kwargs,
    ):
        return ["BTCUSDT", "ETHUSDT"], [("BTCUSDT", _fake_result("BTCUSDT"), None), ("ETHUSDT", None, RuntimeError("bad symbol"))]

    monkeypatch.setattr("futures_analyzer.cli._find_async", fake_find_async)
    result = runner.invoke(app, ["find", "--json"])
    assert result.exit_code == 0
    assert '"candidates"' in result.stdout
    assert '"BTCUSDT"' in result.stdout
    assert '"rank": 1' in result.stdout
    assert '"errors"' in result.stdout
    assert '"latest_feasible_chart_replay"' in result.stdout
    assert '"feedback"' in result.stdout
    assert "Finding liquid candidates, fetching market data" not in result.stdout


def test_structured_logger_accepts_keyword_fields() -> None:
    from futures_analyzer.logging import get_logger

    class _CaptureHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    logger_name = "futures_analyzer.tests.logging"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = _CaptureHandler()
    logger.addHandler(handler)

    get_logger(logger_name).warning("history.save_skipped", symbol="BTCUSDT", error="boom")

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.getMessage() == "history.save_skipped"
    assert getattr(record, "_kv_symbol") == "BTCUSDT"
    assert getattr(record, "_kv_error") == "boom"


def test_cli_history_commands(monkeypatch) -> None:
    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: _FakeHistoryService())
    recent = runner.invoke(app, ["history", "recent", "--symbol", "BTCUSDT"])
    stats = runner.invoke(app, ["history", "stats", "--days", "30"])
    feedback = runner.invoke(app, ["history", "feedback", "--days", "30"])
    compare = runner.invoke(app, ["history", "compare", "--by", "mode", "--days", "30"])
    assert recent.exit_code == 0
    assert "Recent History" in recent.stdout
    assert "target_hit" in recent.stdout
    assert "BTC00001" in recent.stdout
    assert "style conservative | mode intraday | cmd analyse" in recent.stdout
    assert stats.exit_code == 0
    assert "Overall Feedback" in stats.stdout
    assert "Confidence Buckets" in stats.stdout
    assert "Quality Buckets" in stats.stdout
    assert "Regime Buckets" in stats.stdout
    assert feedback.exit_code == 0
    assert "Prediction Feedback" in feedback.stdout
    assert compare.exit_code == 0
    assert "History Compare By MODE" in compare.stdout


def test_cli_history_stats_export_markdown(monkeypatch) -> None:
    monkeypatch.setattr("futures_analyzer.cli._history_service", lambda: _FakeHistoryService())
    export_path = _scratch_path("history_stats_test.md")
    result = runner.invoke(app, ["history", "stats", "--days", "30", "--export", str(export_path)])
    assert result.exit_code == 0
    assert export_path.exists()
    markdown = export_path.read_text(encoding="utf-8")
    assert "# History Stats Report" in markdown
    assert "Confidence Buckets" in markdown


def test_analyse_async_keeps_owned_provider_open_through_replay(monkeypatch) -> None:
    class _Provider:
        def __init__(self) -> None:
            self.closed = False
            self.context_called = False

        async def fetch_market_meta(self, symbol: str):
            return MarketMeta(symbol=symbol, tick_size=0.1, step_size=0.001, mark_price=100.0, as_of=datetime(2026, 1, 1, tzinfo=UTC))

        async def fetch_klines(self, *, symbol: str, interval: str, limit: int = 300, start_time=None, end_time=None, min_required_candles: int = 30):
            base = datetime(2026, 1, 1, tzinfo=UTC)
            return [
                Candle(
                    open_time=base + timedelta(minutes=15 * idx),
                    close_time=base + timedelta(minutes=15 * (idx + 1)),
                    open=100.0 + idx,
                    high=100.5 + idx,
                    low=99.5 + idx,
                    close=100.2 + idx,
                    volume=100.0 + idx,
                )
                for idx in range(40)
            ]

        async def fetch_historical_market_context(self, *, symbol: str, as_of: datetime, interval: str = "5m"):
            assert self.closed is False
            self.context_called = True
            return (0.0001, 1000.0, 1.0)

        async def aclose(self) -> None:
            self.closed = True

    provider = _Provider()

    monkeypatch.setattr("futures_analyzer.cli.BinanceFuturesProvider", lambda: provider)

    result = asyncio.run(
        _analyze_async(
            symbol="BTCUSDT",
            risk_reward=None,
            style=StrategyStyle.CONSERVATIVE,
            market_mode=MarketMode.INTRADAY,
        )
    )

    assert provider.context_called is True
    assert provider.closed is True
    assert result.market_snapshot_meta.symbol == "BTCUSDT"
