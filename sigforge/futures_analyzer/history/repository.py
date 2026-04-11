from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from futures_analyzer.analysis.models import AnalysisResult
from futures_analyzer.history.models import HistorySnapshot, SnapshotEvaluation
from futures_analyzer.logging import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 6

# How long (ms) a writer will wait for a lock before raising OperationalError.
# WAL mode allows concurrent readers alongside one writer, so 5 s is generous.
_BUSY_TIMEOUT_MS = 5_000


class HistoryRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.row_factory = sqlite3.Row
        # WAL journal: readers never block writers, writers never block readers.
        # Only one writer at a time, but concurrent reads are fully parallel.
        conn.execute("PRAGMA journal_mode = WAL")
        # Honour the busy timeout at the SQLite level too.
        conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        # Recommended companion settings for WAL.
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            version = self._detect_schema_version(conn)
            if version == 0:
                log.info("db.schema_create", db=str(self.db_path))
                self._create_schema(conn)
                version = SCHEMA_VERSION
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"History DB schema version {version} is newer than supported version {SCHEMA_VERSION}"
                )
            while version < SCHEMA_VERSION:
                log.info("db.migrate", from_version=version, to_version=version + 1, db=str(self.db_path))
                if version == 1:
                    self._migrate_v1_to_v2(conn)
                    version = 2
                elif version == 2:
                    self._migrate_v2_to_v3(conn)
                    version = 3
                elif version == 3:
                    self._migrate_v3_to_v4(conn)
                    version = 4
                elif version == 4:
                    self._migrate_v4_to_v5(conn)
                    version = 5
                elif version == 5:
                    self._migrate_v5_to_v6(conn)
                    version = 6
                else:
                    raise RuntimeError(f"Unsupported history schema version: {version}")
            self._ensure_indexes(conn)
            conn.execute(f"PRAGMA user_version = {version}")

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _detect_schema_version(self, conn: sqlite3.Connection) -> int:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version:
            return version
        if self._table_exists(conn, "analysis_snapshots"):
            return 1
        return 0

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                as_of TEXT NOT NULL,
                command TEXT NOT NULL,
                style TEXT NOT NULL DEFAULT 'conservative',
                market_mode TEXT NOT NULL DEFAULT 'intraday',
                profile_name TEXT NOT NULL DEFAULT 'auto_core',
                entry_timeframe TEXT NOT NULL DEFAULT '5m',
                trigger_timeframe TEXT NOT NULL DEFAULT '15m',
                context_timeframe TEXT NOT NULL DEFAULT '1h',
                higher_timeframe TEXT NOT NULL DEFAULT '4h',
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
                is_profitable_at_24h_close INTEGER,
                adjusted_rr_ratio REAL,
                total_slippage_pct REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_enhanced_metrics (
                snapshot_id          INTEGER PRIMARY KEY REFERENCES analysis_snapshots(id),
                rsi_14               REAL,
                macd_value           REAL,
                macd_signal          REAL,
                macd_histogram       REAL,
                stochastic_k         REAL,
                stochastic_d         REAL,
                bb_upper             REAL,
                bb_middle            REAL,
                bb_lower             REAL,
                bb_bandwidth_pct     REAL,
                bb_position          REAL,
                rsi_divergence       INTEGER,
                rsi_divergence_type  TEXT,
                order_book_imbalance REAL,
                volatility_rank      REAL,
                volatility_regime    TEXT,
                vwap                 REAL,
                vwap_deviation_pct   REAL,
                liquidity_score      REAL,
                slippage_estimate_pct REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_window_evaluations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id         INTEGER NOT NULL REFERENCES analysis_snapshots(id),
                window              TEXT NOT NULL,
                outcome             TEXT,
                pnl_pct             REAL,
                mfe_pct             REAL,
                mae_pct             REAL,
                rr_achieved         REAL,
                bars_to_resolution  INTEGER,
                UNIQUE(snapshot_id, window)
            )
            """
        )

    @staticmethod
    def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(str(row[1]) == column for row in rows)

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        if not self._column_exists(conn, "analysis_snapshots", "profile_name"):
            conn.execute(
                "ALTER TABLE analysis_snapshots ADD COLUMN profile_name TEXT NOT NULL DEFAULT 'auto_core'"
            )
        if not self._column_exists(conn, "analysis_snapshots", "trigger_timeframe"):
            conn.execute(
                "ALTER TABLE analysis_snapshots ADD COLUMN trigger_timeframe TEXT NOT NULL DEFAULT '15m'"
            )
        if not self._column_exists(conn, "analysis_snapshots", "context_timeframe"):
            conn.execute(
                "ALTER TABLE analysis_snapshots ADD COLUMN context_timeframe TEXT NOT NULL DEFAULT '1h'"
            )

    def _migrate_v2_to_v3(self, conn: sqlite3.Connection) -> None:
        if not self._column_exists(conn, "analysis_snapshots", "prediction_id"):
            conn.execute("ALTER TABLE analysis_snapshots ADD COLUMN prediction_id TEXT")
        if not self._column_exists(conn, "analysis_snapshots", "entry_timeframe"):
            conn.execute(
                "ALTER TABLE analysis_snapshots ADD COLUMN entry_timeframe TEXT NOT NULL DEFAULT '5m'"
            )
        if not self._column_exists(conn, "analysis_snapshots", "higher_timeframe"):
            conn.execute(
                "ALTER TABLE analysis_snapshots ADD COLUMN higher_timeframe TEXT NOT NULL DEFAULT '4h'"
            )

        counters: dict[str, int] = {}
        rows = conn.execute(
            "SELECT id, symbol FROM analysis_snapshots ORDER BY as_of ASC, id ASC"
        ).fetchall()
        for row in rows:
            prefix = self._prediction_prefix(str(row["symbol"]))
            counters[prefix] = counters.get(prefix, 0) + 1
            conn.execute(
                "UPDATE analysis_snapshots SET prediction_id = ? WHERE id = ?",
                (f"{prefix}{counters[prefix]:05d}", int(row["id"])),
            )

    def _migrate_v3_to_v4(self, conn: sqlite3.Connection) -> None:
        if not self._column_exists(conn, "analysis_snapshots", "command"):
            conn.execute("ALTER TABLE analysis_snapshots ADD COLUMN command TEXT")
        if not self._column_exists(conn, "analysis_snapshots", "style"):
            conn.execute(
                "ALTER TABLE analysis_snapshots ADD COLUMN style TEXT NOT NULL DEFAULT 'conservative'"
            )
        if not self._column_exists(conn, "analysis_snapshots", "market_mode"):
            conn.execute(
                "ALTER TABLE analysis_snapshots ADD COLUMN market_mode TEXT NOT NULL DEFAULT 'intraday'"
            )

        rows = conn.execute(
            "SELECT id, mode, analysis_json FROM analysis_snapshots ORDER BY id ASC"
        ).fetchall()
        for row in rows:
            command = str(row["mode"]) if "mode" in row.keys() and row["mode"] else "analyze"
            style = "conservative"
            market_mode = "intraday"
            analysis_json = row["analysis_json"] if "analysis_json" in row.keys() else None
            if analysis_json:
                try:
                    payload = json.loads(str(analysis_json))
                    timeframe_plan = payload.get("timeframe_plan", {}) if isinstance(payload, dict) else {}
                    if isinstance(timeframe_plan, dict):
                        style = str(timeframe_plan.get("style") or style)
                        market_mode = str(timeframe_plan.get("market_mode") or market_mode)
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            conn.execute(
                """
                UPDATE analysis_snapshots
                SET command = ?,
                    style = ?,
                    market_mode = ?
                WHERE id = ?
                """,
                (command, style, market_mode, int(row["id"])),
            )

    def _migrate_v4_to_v5(self, conn: sqlite3.Connection) -> None:
        conn.execute("ALTER TABLE analysis_snapshots RENAME TO analysis_snapshots_legacy")
        self._create_schema(conn)
        conn.execute(
            """
            INSERT INTO analysis_snapshots (
                id, prediction_id, symbol, as_of, command, style, market_mode, profile_name,
                entry_timeframe, trigger_timeframe, context_timeframe, higher_timeframe,
                side, entry_price, target_price, stop_loss,
                confidence, quality_label, quality_score, regime, regime_confidence,
                risk_reward_ratio, stop_distance_pct, target_distance_pct,
                atr_multiple_to_stop, atr_multiple_to_target, invalidation_strength,
                top_positive_contributors_json, top_negative_contributors_json,
                score_components_json, analysis_json, evaluation_status,
                outcome, resolved_at, max_favorable_excursion_pct, max_adverse_excursion_pct,
                pnl_at_24h_close_pct, is_profitable_at_24h_close
            )
            SELECT
                id,
                COALESCE(prediction_id, 'UNK00000'),
                symbol,
                as_of,
                COALESCE(command, mode, 'analyze'),
                COALESCE(style, 'conservative'),
                COALESCE(market_mode, 'intraday'),
                COALESCE(profile_name, 'auto_core'),
                COALESCE(entry_timeframe, '5m'),
                COALESCE(trigger_timeframe, '15m'),
                COALESCE(context_timeframe, '1h'),
                COALESCE(higher_timeframe, '4h'),
                side,
                entry_price,
                target_price,
                stop_loss,
                confidence,
                quality_label,
                quality_score,
                regime,
                regime_confidence,
                risk_reward_ratio,
                stop_distance_pct,
                target_distance_pct,
                atr_multiple_to_stop,
                atr_multiple_to_target,
                invalidation_strength,
                top_positive_contributors_json,
                top_negative_contributors_json,
                score_components_json,
                analysis_json,
                evaluation_status,
                outcome,
                resolved_at,
                max_favorable_excursion_pct,
                max_adverse_excursion_pct,
                pnl_at_24h_close_pct,
                is_profitable_at_24h_close
            FROM analysis_snapshots_legacy
            ORDER BY id ASC
            """
        )
        conn.execute("DROP TABLE analysis_snapshots_legacy")

    def _migrate_v5_to_v6(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_enhanced_metrics (
                snapshot_id          INTEGER PRIMARY KEY REFERENCES analysis_snapshots(id),
                rsi_14               REAL,
                macd_value           REAL,
                macd_signal          REAL,
                macd_histogram       REAL,
                stochastic_k         REAL,
                stochastic_d         REAL,
                bb_upper             REAL,
                bb_middle            REAL,
                bb_lower             REAL,
                bb_bandwidth_pct     REAL,
                bb_position          REAL,
                rsi_divergence       INTEGER,
                rsi_divergence_type  TEXT,
                order_book_imbalance REAL,
                volatility_rank      REAL,
                volatility_regime    TEXT,
                vwap                 REAL,
                vwap_deviation_pct   REAL,
                liquidity_score      REAL,
                slippage_estimate_pct REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_window_evaluations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id         INTEGER NOT NULL REFERENCES analysis_snapshots(id),
                window              TEXT NOT NULL,
                outcome             TEXT,
                pnl_pct             REAL,
                mfe_pct             REAL,
                mae_pct             REAL,
                rr_achieved         REAL,
                bars_to_resolution  INTEGER,
                UNIQUE(snapshot_id, window)
            )
            """
        )
        if not self._column_exists(conn, "analysis_snapshots", "adjusted_rr_ratio"):
            conn.execute("ALTER TABLE analysis_snapshots ADD COLUMN adjusted_rr_ratio REAL")
        if not self._column_exists(conn, "analysis_snapshots", "total_slippage_pct"):
            conn.execute("ALTER TABLE analysis_snapshots ADD COLUMN total_slippage_pct REAL")

    @staticmethod
    def _ensure_indexes(conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_symbol_as_of ON analysis_snapshots(symbol, as_of)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_snapshots_eval_status ON analysis_snapshots(evaluation_status, as_of)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_snapshots_prediction_id ON analysis_snapshots(prediction_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_enhanced_metrics_rsi ON snapshot_enhanced_metrics(rsi_14)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_enhanced_metrics_ob_imbalance ON snapshot_enhanced_metrics(order_book_imbalance)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_window_evals_snapshot ON snapshot_window_evaluations(snapshot_id, window)"
        )
    def clear_all(self) -> int:
        """Delete all rows from all history tables. Returns total rows deleted."""
        with self._connect() as conn:
            deleted = 0
            deleted += conn.execute("DELETE FROM snapshot_window_evaluations").rowcount
            deleted += conn.execute("DELETE FROM snapshot_enhanced_metrics").rowcount
            deleted += conn.execute("DELETE FROM analysis_snapshots").rowcount
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('analysis_snapshots', 'snapshot_window_evaluations')")
            return deleted

    @staticmethod
    def _prediction_prefix(symbol: str) -> str:
        for suffix in ("USDT", "BUSD", "USDC", "FDUSD", "BTC", "ETH"):
            if symbol.endswith(suffix) and len(symbol) > len(suffix):
                symbol = symbol[: -len(suffix)]
                break
        cleaned = "".join(ch for ch in symbol.upper() if ch.isalnum())
        return (cleaned[:3] or "PRD").ljust(3, "X")

    def _next_prediction_id(self, conn: sqlite3.Connection, symbol: str) -> str:
        prefix = self._prediction_prefix(symbol)
        row = conn.execute(
            "SELECT prediction_id FROM analysis_snapshots WHERE prediction_id LIKE ? ORDER BY prediction_id DESC LIMIT 1",
            (f"{prefix}%",),
        ).fetchone()
        next_number = 1
        if row is not None and row["prediction_id"]:
            suffix = str(row["prediction_id"])[len(prefix) :]
            if suffix.isdigit():
                next_number = int(suffix) + 1
        return f"{prefix}{next_number:05d}"

    def _ensure_prediction_id(self, conn: sqlite3.Connection, symbol: str, requested: str | None) -> str:
        """Ensure a unique prediction ID, with retry logic for race conditions.
        
        Returns a unique prediction ID. If the requested ID is already taken,
        generates a new one. Handles concurrent inserts gracefully.
        """
        if requested:
            exists = conn.execute(
                "SELECT 1 FROM analysis_snapshots WHERE prediction_id = ?",
                (requested,),
            ).fetchone()
            if exists is None:
                return requested
        return self._next_prediction_id(conn, symbol)

    def _insert_snapshot(self, conn: sqlite3.Connection, result: AnalysisResult, command: str, prediction_id: str) -> int:
        """Execute the INSERT and return the new row id."""
        setup = result.primary_setup
        cursor = conn.execute(
            """
            INSERT INTO analysis_snapshots (
                prediction_id, symbol, as_of, command, style, market_mode, profile_name,
                entry_timeframe, trigger_timeframe, context_timeframe, higher_timeframe,
                side, entry_price, target_price, stop_loss,
                confidence, quality_label, quality_score, regime, regime_confidence,
                risk_reward_ratio, stop_distance_pct, target_distance_pct,
                atr_multiple_to_stop, atr_multiple_to_target, invalidation_strength,
                top_positive_contributors_json, top_negative_contributors_json,
                score_components_json, analysis_json, evaluation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                result.market_snapshot_meta.symbol,
                result.market_snapshot_meta.as_of.isoformat(),
                command,
                result.timeframe_plan.style.value,
                result.timeframe_plan.market_mode.value,
                result.timeframe_plan.profile_name,
                result.timeframe_plan.entry_timeframe,
                result.timeframe_plan.trigger_timeframe,
                result.timeframe_plan.context_timeframe,
                result.timeframe_plan.higher_timeframe,
                setup.side,
                setup.entry_price,
                setup.target_price,
                setup.stop_loss,
                setup.confidence,
                setup.quality_label.value,
                setup.quality_score,
                result.market_regime.value,
                result.regime_confidence,
                setup.risk_reward_ratio,
                setup.stop_distance_pct,
                setup.target_distance_pct,
                setup.atr_multiple_to_stop,
                setup.atr_multiple_to_target,
                setup.invalidation_strength,
                json.dumps([item.model_dump(mode="json") for item in setup.top_positive_contributors]),
                json.dumps([item.model_dump(mode="json") for item in setup.top_negative_contributors]),
                json.dumps(setup.score_components),
                result.model_dump_json(),
                "unresolved",
            ),
        )
        return int(cursor.lastrowid)

    def save_result(self, result: AnalysisResult, command: str) -> tuple[int, str]:
        with self._connect() as conn:
            prediction_id = self._ensure_prediction_id(conn, result.market_snapshot_meta.symbol, result.prediction_id)
            result.prediction_id = prediction_id
            try:
                row_id = self._insert_snapshot(conn, result, command, prediction_id)
            except sqlite3.IntegrityError as exc:
                # Duplicate prediction_id due to race condition — generate a fresh one and retry once
                log.warning("db.prediction_id_collision", prediction_id=prediction_id, symbol=result.market_snapshot_meta.symbol, error=str(exc))
                prediction_id = self._next_prediction_id(conn, result.market_snapshot_meta.symbol)
                result.prediction_id = prediction_id
                row_id = self._insert_snapshot(conn, result, command, prediction_id)
            log.debug(
                "db.snapshot_saved",
                prediction_id=prediction_id,
                symbol=result.market_snapshot_meta.symbol,
                command=command,
                row_id=row_id,
            )
            return row_id, prediction_id

    def due_for_evaluation(self, now: datetime, *, limit: int = 100) -> list[HistorySnapshot]:
        cutoff = (now - timedelta(hours=24)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM analysis_snapshots
                WHERE evaluation_status = 'unresolved' AND as_of <= ?
                ORDER BY as_of ASC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def update_evaluation(self, snapshot_id: int, evaluation: SnapshotEvaluation) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE analysis_snapshots
                SET evaluation_status = ?,
                    outcome = ?,
                    resolved_at = ?,
                    max_favorable_excursion_pct = ?,
                    max_adverse_excursion_pct = ?,
                    pnl_at_24h_close_pct = ?,
                    is_profitable_at_24h_close = ?
                WHERE id = ?
                """,
                (
                    evaluation.evaluation_status,
                    evaluation.outcome,
                    evaluation.resolved_at.isoformat(),
                    evaluation.max_favorable_excursion_pct,
                    evaluation.max_adverse_excursion_pct,
                    evaluation.pnl_at_24h_close_pct,
                    1 if evaluation.is_profitable_at_24h_close else 0,
                    snapshot_id,
                ),
            )

    def recent(self, *, symbol: str | None = None, limit: int = 20) -> list[HistorySnapshot]:
        query = "SELECT * FROM analysis_snapshots"
        params: list[object] = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY as_of DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def latest_tradable(
        self,
        *,
        style: str,
        market_mode: str,
        symbol: str | None = None,
        symbols: list[str] | None = None,
    ) -> HistorySnapshot | None:
        query = "SELECT * FROM analysis_snapshots WHERE style = ? AND market_mode = ?"
        params: list[object] = [style, market_mode]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        elif symbols is not None:
            if not symbols:
                return None
            placeholders = ", ".join("?" for _ in symbols)
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
        query += " ORDER BY as_of DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        for row in rows:
            if self._analysis_json_is_tradable(row["analysis_json"] if "analysis_json" in row.keys() else None):
                return self._row_to_snapshot(row)
        return None

    def evaluated(
        self,
        *,
        symbol: str | None = None,
        days: int | None = None,
        is_tradable: bool | None = None,
        limit: int | None = None,
    ) -> list[HistorySnapshot]:
        query = "SELECT * FROM analysis_snapshots WHERE evaluation_status != 'unresolved'"
        params: list[object] = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            query += " AND as_of >= ?"
            params.append(cutoff)
        if is_tradable is not None:
            # Filter on the JSON field directly; avoids loading all blobs into Python
            if is_tradable:
                query += " AND json_extract(analysis_json, '$.primary_setup.is_tradable') = 1"
            else:
                query += " AND (json_extract(analysis_json, '$.primary_setup.is_tradable') IS NULL OR json_extract(analysis_json, '$.primary_setup.is_tradable') != 1)"
        query += " ORDER BY as_of DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    @staticmethod
    def _analysis_json_is_tradable(analysis_json: object) -> bool:
        if analysis_json is None:
            return False
        try:
            payload = json.loads(str(analysis_json))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        primary_setup = payload.get("primary_setup")
        if not isinstance(primary_setup, dict):
            return False
        return bool(primary_setup.get("is_tradable"))

    def save_enhanced_metrics(self, snapshot_id: int, metrics: "EnhancedMetrics") -> None:
        from futures_analyzer.analysis.models import EnhancedMetrics
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO snapshot_enhanced_metrics (
                    snapshot_id, rsi_14, macd_value, macd_signal, macd_histogram,
                    stochastic_k, stochastic_d, bb_upper, bb_middle, bb_lower,
                    bb_bandwidth_pct, bb_position, rsi_divergence, rsi_divergence_type,
                    order_book_imbalance, volatility_rank, volatility_regime,
                    vwap, vwap_deviation_pct, liquidity_score, slippage_estimate_pct
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    snapshot_id,
                    metrics.rsi_14, metrics.macd_value, metrics.macd_signal, metrics.macd_histogram,
                    metrics.stochastic_k, metrics.stochastic_d,
                    metrics.bollinger_upper, metrics.bollinger_middle, metrics.bollinger_lower,
                    metrics.bollinger_bandwidth_pct, metrics.bollinger_position,
                    1 if metrics.rsi_divergence else 0, metrics.rsi_divergence_type,
                    metrics.order_book_imbalance, metrics.volatility_rank, metrics.volatility_regime,
                    metrics.vwap, metrics.vwap_deviation_pct, metrics.liquidity_score,
                    metrics.slippage_estimate_pct,
                ),
            )

    def get_enhanced_metrics(self, snapshot_id: int) -> "EnhancedMetrics | None":
        from futures_analyzer.analysis.models import EnhancedMetrics
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM snapshot_enhanced_metrics WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            return None
        return EnhancedMetrics(
            rsi_14=float(row["rsi_14"] or 50.0),
            macd_value=float(row["macd_value"] or 0.0),
            macd_signal=float(row["macd_signal"] or 0.0),
            macd_histogram=float(row["macd_histogram"] or 0.0),
            stochastic_k=float(row["stochastic_k"] or 50.0),
            stochastic_d=float(row["stochastic_d"] or 50.0),
            bollinger_upper=float(row["bb_upper"] or 0.0),
            bollinger_middle=float(row["bb_middle"] or 0.0),
            bollinger_lower=float(row["bb_lower"] or 0.0),
            bollinger_bandwidth_pct=float(row["bb_bandwidth_pct"] or 0.0),
            bollinger_position=float(row["bb_position"] or 0.5),
            rsi_divergence=bool(row["rsi_divergence"]),
            rsi_divergence_type=str(row["rsi_divergence_type"] or "none"),
            order_book_imbalance=float(row["order_book_imbalance"] or 0.0),
            volatility_rank=float(row["volatility_rank"] or 50.0),
            volatility_regime=str(row["volatility_regime"] or "normal"),
            vwap=float(row["vwap"] or 0.0),
            vwap_deviation_pct=float(row["vwap_deviation_pct"] or 0.0),
            liquidity_score=float(row["liquidity_score"] or 50.0),
            slippage_estimate_pct=float(row["slippage_estimate_pct"] or 0.0),
        )

    def save_window_evaluations(self, snapshot_id: int, evals: dict) -> None:
        with self._connect() as conn:
            for window, metrics in evals.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO snapshot_window_evaluations
                        (snapshot_id, window, outcome, pnl_pct, mfe_pct, mae_pct, rr_achieved, bars_to_resolution)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        snapshot_id,
                        window,
                        getattr(metrics, "outcome", None),
                        getattr(metrics, "profit_loss_pct", None),
                        getattr(metrics.drawdown_metrics, "max_favorable_excursion_pct", None) if hasattr(metrics, "drawdown_metrics") else None,
                        getattr(metrics.drawdown_metrics, "max_adverse_excursion_pct", None) if hasattr(metrics, "drawdown_metrics") else None,
                        getattr(metrics, "risk_reward_achieved", None),
                        getattr(metrics, "bars_to_resolution", None),
                    ),
                )

    def get_window_evaluations(self, snapshot_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshot_window_evaluations WHERE snapshot_id = ? ORDER BY window",
                (snapshot_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_slippage_adjusted(
        self,
        snapshot_id: int,
        *,
        adjusted_rr_ratio: float,
        total_slippage_pct: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE analysis_snapshots
                SET adjusted_rr_ratio = ?, total_slippage_pct = ?
                WHERE id = ?
                """,
                (adjusted_rr_ratio, total_slippage_pct, snapshot_id),
            )

    def evaluated_with_filter(
        self,
        *,
        symbol: str | None = None,
        days: int | None = None,
        min_rsi: float | None = None,
        max_rsi: float | None = None,
        regime: str | None = None,
        min_ob_imbalance: float | None = None,
        volatility_regime: str | None = None,
    ) -> list[HistorySnapshot]:
        query = (
            "SELECT s.*, em.rsi_14, em.macd_histogram, em.bb_position, "
            "em.order_book_imbalance, em.volatility_regime AS em_volatility_regime "
            "FROM analysis_snapshots s "
            "LEFT JOIN snapshot_enhanced_metrics em ON em.snapshot_id = s.id "
            "WHERE s.evaluation_status != 'unresolved'"
        )
        params: list[object] = []
        if symbol:
            query += " AND s.symbol = ?"
            params.append(symbol)
        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            query += " AND s.as_of >= ?"
            params.append(cutoff)
        if min_rsi is not None:
            query += " AND em.rsi_14 >= ?"
            params.append(min_rsi)
        if max_rsi is not None:
            query += " AND em.rsi_14 <= ?"
            params.append(max_rsi)
        if regime is not None:
            query += " AND s.regime = ?"
            params.append(regime)
        if min_ob_imbalance is not None:
            query += " AND em.order_book_imbalance >= ?"
            params.append(min_ob_imbalance)
        if volatility_regime is not None:
            query += " AND em.volatility_regime = ?"
            params.append(volatility_regime)
        query += " ORDER BY s.as_of DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_snapshot(row) for row in rows]

    def window_evaluations_for_symbol(
        self,
        *,
        symbol: str | None = None,
        days: int | None = None,
    ) -> list[dict]:
        query = (
            "SELECT we.window, we.outcome, we.pnl_pct, we.mfe_pct "
            "FROM snapshot_window_evaluations we "
            "JOIN analysis_snapshots s ON s.id = we.snapshot_id "
            "WHERE 1=1"
        )
        params: list[object] = []
        if symbol:
            query += " AND s.symbol = ?"
            params.append(symbol)
        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            query += " AND s.as_of >= ?"
            params.append(cutoff)
        query += " ORDER BY we.window, s.as_of DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> HistorySnapshot:
        resolved_at = row["resolved_at"]
        return HistorySnapshot(
            id=int(row["id"]),
            prediction_id=str(row["prediction_id"]) if "prediction_id" in row.keys() and row["prediction_id"] else "UNK00000",
            symbol=str(row["symbol"]),
            as_of=datetime.fromisoformat(str(row["as_of"])),
            command=(
                str(row["command"])
                if "command" in row.keys() and row["command"] is not None
                else str(row["mode"])
                if "mode" in row.keys() and row["mode"] is not None
                else "unknown"
            ),
            style=str(row["style"]) if "style" in row.keys() and row["style"] is not None else "conservative",
            market_mode=(
                str(row["market_mode"])
                if "market_mode" in row.keys() and row["market_mode"] is not None
                else "intraday"
            ),
            profile_name=str(row["profile_name"]) if "profile_name" in row.keys() else "auto_core",
            entry_timeframe=str(row["entry_timeframe"]) if "entry_timeframe" in row.keys() else "5m",
            trigger_timeframe=str(row["trigger_timeframe"]) if "trigger_timeframe" in row.keys() else "15m",
            context_timeframe=str(row["context_timeframe"]) if "context_timeframe" in row.keys() else "1h",
            higher_timeframe=str(row["higher_timeframe"]) if "higher_timeframe" in row.keys() else "4h",
            side=str(row["side"]),
            entry_price=float(row["entry_price"]),
            target_price=float(row["target_price"]),
            stop_loss=float(row["stop_loss"]),
            confidence=float(row["confidence"]),
            quality_label=str(row["quality_label"]),
            quality_score=float(row["quality_score"]),
            regime=str(row["regime"]),
            regime_confidence=float(row["regime_confidence"]),
            risk_reward_ratio=float(row["risk_reward_ratio"]),
            stop_distance_pct=float(row["stop_distance_pct"]),
            target_distance_pct=float(row["target_distance_pct"]),
            atr_multiple_to_stop=float(row["atr_multiple_to_stop"]),
            atr_multiple_to_target=float(row["atr_multiple_to_target"]),
            invalidation_strength=float(row["invalidation_strength"]),
            top_positive_contributors_json=str(row["top_positive_contributors_json"]),
            top_negative_contributors_json=str(row["top_negative_contributors_json"]),
            score_components_json=str(row["score_components_json"]),
            analysis_json=str(row["analysis_json"]),
            evaluation_status=str(row["evaluation_status"]),
            outcome=str(row["outcome"]) if row["outcome"] is not None else None,
            resolved_at=datetime.fromisoformat(str(resolved_at)) if resolved_at else None,
            max_favorable_excursion_pct=float(row["max_favorable_excursion_pct"]) if row["max_favorable_excursion_pct"] is not None else None,
            max_adverse_excursion_pct=float(row["max_adverse_excursion_pct"]) if row["max_adverse_excursion_pct"] is not None else None,
            pnl_at_24h_close_pct=float(row["pnl_at_24h_close_pct"]) if row["pnl_at_24h_close_pct"] is not None else None,
            is_profitable_at_24h_close=bool(row["is_profitable_at_24h_close"]) if row["is_profitable_at_24h_close"] is not None else None,
            rsi_14=float(row["rsi_14"]) if "rsi_14" in row.keys() and row["rsi_14"] is not None else None,
            macd_histogram=float(row["macd_histogram"]) if "macd_histogram" in row.keys() and row["macd_histogram"] is not None else None,
            bb_position=float(row["bb_position"]) if "bb_position" in row.keys() and row["bb_position"] is not None else None,
            order_book_imbalance=float(row["order_book_imbalance"]) if "order_book_imbalance" in row.keys() and row["order_book_imbalance"] is not None else None,
            volatility_regime=(
                str(row["em_volatility_regime"])
                if "em_volatility_regime" in row.keys() and row["em_volatility_regime"] is not None
                else str(row["volatility_regime"]) if "volatility_regime" in row.keys() and row["volatility_regime"] is not None
                else None
            ),
            adjusted_rr_ratio=float(row["adjusted_rr_ratio"]) if "adjusted_rr_ratio" in row.keys() and row["adjusted_rr_ratio"] is not None else None,
            total_slippage_pct=float(row["total_slippage_pct"]) if "total_slippage_pct" in row.keys() and row["total_slippage_pct"] is not None else None,
        )
