import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
from futures_analyzer.history import HistoryCompareBy, HistoryService
from futures_analyzer.history.repository import HistoryRepository


class _FakeProvider:
    def __init__(self) -> None:
        self.last_interval: str | None = None
        self.last_limit: int | None = None

    async def fetch_klines(self, *, symbol: str, interval: str, limit: int = 300, start_time=None, end_time=None, min_required_candles: int = 30):
        self.last_interval = interval
        self.last_limit = limit
        start = start_time or datetime.now(UTC)
        return [
            Candle(
                open_time=start,
                close_time=start + timedelta(minutes=15),
                open=100.0,
                high=103.5,
                low=99.0,
                close=102.0,
                volume=100.0,
            ),
            Candle(
                open_time=start + timedelta(minutes=15),
                close_time=start + timedelta(minutes=30),
                open=102.0,
                high=104.0,
                low=101.5,
                close=103.0,
                volume=120.0,
            ),
        ]

    async def aclose(self) -> None:
        return


class _AmbiguousProvider:
    async def fetch_klines(self, *, symbol: str, interval: str, limit: int = 300, start_time=None, end_time=None, min_required_candles: int = 30):
        start = start_time or datetime.now(UTC)
        return [
            Candle(
                open_time=start,
                close_time=start + timedelta(minutes=15),
                open=100.0,
                high=104.5,
                low=98.0,
                close=101.0,
                volume=100.0,
            ),
            Candle(
                open_time=start + timedelta(minutes=15),
                close_time=start + timedelta(minutes=30),
                open=101.0,
                high=101.5,
                low=100.0,
                close=100.5,
                volume=100.0,
            ),
        ]

    async def aclose(self) -> None:
        return



def _contributor() -> ContributorDetail:
    return ContributorDetail(
        key="trend",
        label="Trend",
        value=1.2,
        impact=2.1,
        direction=ContributorDirection.POSITIVE,
        summary="Trend supports the setup.",
    )



def _result(as_of: datetime, *, trigger_timeframe: str = "15m", context_timeframe: str = "1h") -> AnalysisResult:
    setup = TradeSetup(
        side="long",
        entry_price=100.0,
        target_price=103.0,
        stop_loss=98.5,
        confidence=0.75,
        quality_label=QualityLabel.HIGH,
        quality_score=82.0,
        rationale="Synthetic result",
        top_positive_contributors=[_contributor()],
        top_negative_contributors=[],
        score_components={"trend": 2.1},
        structure_points={"support": 98.5, "resistance": 103.0, "atr": 1.2},
        risk_reward_ratio=2.0,
        stop_distance_pct=1.5,
        target_distance_pct=3.0,
        atr_multiple_to_stop=1.2,
        atr_multiple_to_target=2.4,
        invalidation_strength=0.8,
        evidence_agreement=6,
        evidence_total=7,
        deliberation_summary="6/7 aligned",
    )
    secondary = setup.model_copy(update={"side": "short", "quality_label": QualityLabel.MEDIUM, "quality_score": 60.0})
    return AnalysisResult(
        prediction_id="BTC00001",
        primary_setup=setup,
        secondary_context=secondary,
        timeframe_plan=TimeframePlan(
            profile_name="intraday_core",
            style=StrategyStyle.CONSERVATIVE,
            market_mode=MarketMode.INTRADAY,
            entry_timeframe="5m",
            trigger_timeframe=trigger_timeframe,
            context_timeframe=context_timeframe,
            higher_timeframe="4h",
            lookback_bars=600,
        ),
        market_snapshot_meta=MarketMeta(symbol="BTCUSDT", tick_size=0.1, step_size=0.001, mark_price=100.0, as_of=as_of),
        market_regime=MarketRegime.BULLISH_TREND,
        regime_confidence=0.8,
    )


def _result_for_symbol(
    symbol: str,
    as_of: datetime,
    *,
    is_tradable: bool = True,
) -> AnalysisResult:
    result = _result(as_of)
    return result.model_copy(
        update={
            "prediction_id": f"{symbol[:3]}00001",
            "primary_setup": result.primary_setup.model_copy(update={"is_tradable": is_tradable}),
            "market_snapshot_meta": result.market_snapshot_meta.model_copy(update={"symbol": symbol, "as_of": as_of}),
        }
    )



def _db_path(name: str) -> Path:
    data_dir = Path(".data")
    data_dir.mkdir(exist_ok=True)
    path = data_dir / name
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            pass
    return path



def test_history_records_and_evaluates_due_snapshots(monkeypatch) -> None:
    db_path = _db_path("history_test_one.db")
    service = HistoryService(db_path)
    monkeypatch.setattr("futures_analyzer.history.service.BinanceFuturesProvider", _FakeProvider)
    old_result = _result(datetime.now(UTC) - timedelta(hours=26))
    asyncio.run(service.record_results([old_result], command="analyze"))

    rows = service.recent(symbol="BTCUSDT", limit=5)
    assert len(rows) == 1
    assert rows[0].outcome == "target_hit"
    assert rows[0].is_profitable_at_24h_close is True
    assert rows[0].max_favorable_excursion_pct is not None
    assert json.loads(rows[0].analysis_json)["prediction_id"] == rows[0].prediction_id



def test_history_stats_bucket_aggregates(monkeypatch) -> None:
    db_path = _db_path("history_test_two.db")
    service = HistoryService(db_path)
    monkeypatch.setattr("futures_analyzer.history.service.BinanceFuturesProvider", _FakeProvider)
    base_time = datetime.now(UTC) - timedelta(hours=26)
    asyncio.run(
        service.record_results(
            [
                _result(base_time),
                _result(base_time - timedelta(days=1)),
            ],
            command="scan",
        )
    )
    report = service.stats(symbol="BTCUSDT", days=30)
    assert report.confidence_buckets
    assert report.quality_buckets
    assert report.regime_buckets
    assert report.confidence_buckets[0].sample_count >= 1


def test_history_compare_groups_by_mode_and_style(monkeypatch) -> None:
    db_path = _db_path("history_test_compare.db")
    service = HistoryService(db_path)
    monkeypatch.setattr("futures_analyzer.history.service.BinanceFuturesProvider", _FakeProvider)
    base_time = datetime.now(UTC) - timedelta(hours=26)
    intraday = _result(base_time)
    long_term = _result(base_time - timedelta(days=1)).model_copy(
        update={
            "timeframe_plan": _result(base_time - timedelta(days=1)).timeframe_plan.model_copy(
                update={
                    "profile_name": "long_term_swing",
                    "style": StrategyStyle.AGGRESSIVE,
                    "market_mode": MarketMode.LONG_TERM,
                    "entry_timeframe": "1h",
                    "trigger_timeframe": "4h",
                    "context_timeframe": "1d",
                    "higher_timeframe": "1w",
                    "lookback_bars": 900,
                }
            )
        }
    )
    asyncio.run(service.record_results([intraday], command="analyze"))
    asyncio.run(service.record_results([long_term], command="scan"))

    mode_report = service.compare(compare_by=HistoryCompareBy.MODE, days=30)
    style_report = service.compare(compare_by=HistoryCompareBy.STYLE, days=30)
    assert {bucket.bucket for bucket in mode_report.buckets} == {"intraday", "long_term"}
    assert {bucket.bucket for bucket in style_report.buckets} == {"conservative", "aggressive"}


def test_history_evaluation_uses_snapshot_trigger_timeframe() -> None:
    service = HistoryService(_db_path("history_test_interval.db"))
    provider = _FakeProvider()
    snapshot_id, _ = service.repository.save_result(
        _result(datetime.now(UTC) - timedelta(hours=26), trigger_timeframe="1h"),
        "analyze",
    )
    due = service.repository.due_for_evaluation(datetime.now(UTC), limit=5)
    assert due and due[0].id == snapshot_id
    evaluation = asyncio.run(service._evaluate_snapshot(provider, due[0]))
    assert evaluation is not None
    assert provider.last_interval == "1h"
    assert provider.last_limit == 26


def test_history_marks_same_bar_target_and_stop_as_ambiguous(monkeypatch) -> None:
    db_path = _db_path("history_test_ambiguous.db")
    service = HistoryService(db_path)
    monkeypatch.setattr("futures_analyzer.history.service.BinanceFuturesProvider", _AmbiguousProvider)
    old_result = _result(datetime.now(UTC) - timedelta(hours=26))
    asyncio.run(service.record_results([old_result], command="analyze"))

    rows = service.recent(symbol="BTCUSDT", limit=5)
    assert len(rows) == 1
    assert rows[0].outcome == "ambiguous_same_bar"


def test_history_repository_migrates_legacy_schema() -> None:
    db_path = _db_path("history_test_legacy.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE analysis_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            as_of TEXT NOT NULL,
            mode TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            target_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            confidence REAL NOT NULL,
            quality_label TEXT NOT NULL,
            quality_score REAL NOT NULL,
            regime TEXT NOT NULL,
            regime_confidence REAL NOT NULL,
            risk_reward_ratio REAL NOT NULL,
            stop_distance_pct REAL NOT NULL,
            target_distance_pct REAL NOT NULL,
            atr_multiple_to_stop REAL NOT NULL,
            atr_multiple_to_target REAL NOT NULL,
            invalidation_strength REAL NOT NULL,
            top_positive_contributors_json TEXT NOT NULL,
            top_negative_contributors_json TEXT NOT NULL,
            score_components_json TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            evaluation_status TEXT NOT NULL,
            outcome TEXT,
            resolved_at TEXT,
            max_favorable_excursion_pct REAL,
            max_adverse_excursion_pct REAL,
            pnl_at_24h_close_pct REAL,
            is_profitable_at_24h_close INTEGER
        )
        """
    )
    conn.execute(
        """
        INSERT INTO analysis_snapshots (
            symbol, as_of, mode, side, entry_price, target_price, stop_loss,
            confidence, quality_label, quality_score, regime, regime_confidence,
            risk_reward_ratio, stop_distance_pct, target_distance_pct,
            atr_multiple_to_stop, atr_multiple_to_target, invalidation_strength,
            top_positive_contributors_json, top_negative_contributors_json,
            score_components_json, analysis_json, evaluation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "BTCUSDT",
            datetime.now(UTC).isoformat(),
            "analyze",
            "long",
            100.0,
            103.0,
            98.5,
            0.7,
            "high",
            80.0,
            "bullish_trend",
            0.8,
            2.0,
            1.5,
            3.0,
            1.2,
            2.4,
            0.8,
            "[]",
            "[]",
            "{}",
            "{}",
            "resolved",
        ),
    )
    conn.commit()
    conn.close()

    repo = HistoryRepository(db_path)
    rows = repo.recent(limit=5)
    assert rows[0].prediction_id == "BTC00001"
    assert rows[0].command == "analyze"
    assert rows[0].style == "conservative"
    assert rows[0].market_mode == "intraday"
    assert rows[0].profile_name == "auto_core"
    assert rows[0].entry_timeframe == "5m"
    assert rows[0].trigger_timeframe == "15m"
    assert rows[0].context_timeframe == "1h"
    assert rows[0].higher_timeframe == "4h"

    conn = sqlite3.connect(db_path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    columns = {row[1] for row in conn.execute("PRAGMA table_info(analysis_snapshots)").fetchall()}
    conn.close()
    assert version == 6
    assert "mode" not in columns
    assert {
        "prediction_id",
        "command",
        "style",
        "market_mode",
        "profile_name",
        "entry_timeframe",
        "trigger_timeframe",
        "context_timeframe",
        "higher_timeframe",
    }.issubset(columns)


def test_history_repository_saves_after_legacy_mode_cleanup() -> None:
    db_path = _db_path("history_test_legacy_insert.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE analysis_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            as_of TEXT NOT NULL,
            mode TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            target_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            confidence REAL NOT NULL,
            quality_label TEXT NOT NULL,
            quality_score REAL NOT NULL,
            regime TEXT NOT NULL,
            regime_confidence REAL NOT NULL,
            risk_reward_ratio REAL NOT NULL,
            stop_distance_pct REAL NOT NULL,
            target_distance_pct REAL NOT NULL,
            atr_multiple_to_stop REAL NOT NULL,
            atr_multiple_to_target REAL NOT NULL,
            invalidation_strength REAL NOT NULL,
            top_positive_contributors_json TEXT NOT NULL,
            top_negative_contributors_json TEXT NOT NULL,
            score_components_json TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            evaluation_status TEXT NOT NULL,
            outcome TEXT,
            resolved_at TEXT,
            max_favorable_excursion_pct REAL,
            max_adverse_excursion_pct REAL,
            pnl_at_24h_close_pct REAL,
            is_profitable_at_24h_close INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

    repo = HistoryRepository(db_path)
    _, prediction_id = repo.save_result(_result(datetime.now(UTC)), "find")
    rows = repo.recent(limit=5)

    assert rows[0].prediction_id == prediction_id
    assert rows[0].command == "find"

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(analysis_snapshots)").fetchall()}
    stored = conn.execute(
        "SELECT command FROM analysis_snapshots WHERE prediction_id = ?",
        (prediction_id,),
    ).fetchone()[0]
    conn.close()
    assert "mode" not in columns
    assert stored == "find"


def test_history_repository_returns_latest_tradable_for_rule_scope() -> None:
    db_path = _db_path("history_test_latest_tradable.db")
    repo = HistoryRepository(db_path)
    now = datetime.now(UTC)
    repo.save_result(_result_for_symbol("BTCUSDT", now - timedelta(hours=3), is_tradable=True), "analyze")
    repo.save_result(_result_for_symbol("BTCUSDT", now - timedelta(hours=1), is_tradable=False), "analyze")
    repo.save_result(_result_for_symbol("ETHUSDT", now - timedelta(minutes=30), is_tradable=True), "scan")

    latest_btc = repo.latest_tradable(style="conservative", market_mode="intraday", symbol="BTCUSDT")
    latest_any = repo.latest_tradable(
        style="conservative",
        market_mode="intraday",
        symbols=["BTCUSDT", "ETHUSDT"],
    )

    assert latest_btc is not None
    assert latest_btc.symbol == "BTCUSDT"
    assert latest_btc.as_of == now - timedelta(hours=3)
    assert latest_any is not None
    assert latest_any.symbol == "ETHUSDT"


def test_history_repository_rejects_newer_unknown_schema() -> None:
    db_path = _db_path("history_test_future.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()

    try:
        HistoryRepository(db_path)
        raise AssertionError("Expected unsupported future schema version to fail")
    except RuntimeError as exc:
        assert "newer than supported" in str(exc)
