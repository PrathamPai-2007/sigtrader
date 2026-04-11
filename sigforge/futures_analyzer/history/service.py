from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
import os

from futures_analyzer.analysis.models import AnalysisResult
from futures_analyzer.history.models import (
    EvaluationOutcome,
    HistoryCompareBy,
    HistoryCompareReport,
    HistorySnapshot,
    HistoryStatsReport,
    SnapshotEvaluation,
    StatsBucket,
)
from futures_analyzer.history.repository import HistoryRepository
from futures_analyzer.logging import get_logger
from futures_analyzer.providers import BinanceFuturesProvider

log = get_logger(__name__)


def default_history_db_path() -> Path:
    override = os.environ.get("FUTURES_ANALYZER_HISTORY_DB", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    data_dir = Path.cwd() / ".data"
    data_dir.mkdir(exist_ok=True)
    return (data_dir / "history.db").resolve()


class HistoryService:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or default_history_db_path()
        self.repository = HistoryRepository(self.db_path)

    async def record_results(self, results: list[AnalysisResult], *, command: str) -> list[str]:
        prediction_ids: list[str] = []
        for result in results:
            row_id, prediction_id = self.repository.save_result(result, command)
            result.prediction_id = prediction_id
            prediction_ids.append(prediction_id)
            try:
                self.repository.save_enhanced_metrics(row_id, result.enhanced_metrics)
            except Exception as exc:
                log.warning("history.enhanced_metrics_skipped", prediction_id=prediction_id, error=str(exc))
        try:
            await self.evaluate_due_snapshots()
        except Exception as exc:
            log.warning("history.evaluation_skipped", error=str(exc))
        return prediction_ids

    async def evaluate_due_snapshots(self, *, limit: int = 100) -> int:
        """Evaluate snapshots that are due for resolution.
        
        Returns the number of snapshots successfully evaluated.
        Ensures provider is always closed, even on exceptions.
        """
        due = self.repository.due_for_evaluation(datetime.now(UTC), limit=limit)
        if not due:
            return 0
        provider = BinanceFuturesProvider()
        updated = 0
        try:
            for snapshot in due:
                try:
                    evaluation = await self._evaluate_snapshot(provider, snapshot)
                    if evaluation is None:
                        log.debug("history.evaluation_no_data", symbol=snapshot.symbol, prediction_id=snapshot.prediction_id)
                        continue
                    self.repository.update_evaluation(snapshot.id, evaluation)
                    log.info(
                        "history.evaluated",
                        symbol=snapshot.symbol,
                        prediction_id=snapshot.prediction_id,
                        outcome=evaluation.outcome,
                    )
                    updated += 1
                except Exception as exc:
                    log.warning(
                        "history.evaluation_failed",
                        symbol=snapshot.symbol,
                        prediction_id=snapshot.prediction_id,
                        error=str(exc),
                    )
                    continue
        finally:
            await provider.aclose()
        return updated

    async def _evaluate_snapshot(
        self,
        provider: BinanceFuturesProvider,
        snapshot: HistorySnapshot,
    ) -> SnapshotEvaluation | None:
        interval = snapshot.trigger_timeframe or "15m"
        eval_hours = self._evaluation_hours_for_interval(interval)
        start_time = snapshot.as_of
        end_time = snapshot.as_of + timedelta(hours=eval_hours)
        candles = await provider.fetch_klines(
            symbol=snapshot.symbol,
            interval=interval,
            limit=self._evaluation_limit_for_interval(interval),
            start_time=start_time,
            end_time=end_time,
            min_required_candles=2,
        )
        if not candles:
            return None

        entry = snapshot.entry_price
        target = snapshot.target_price
        stop = snapshot.stop_loss
        side = snapshot.side
        outcome = EvaluationOutcome.NEITHER.value

        if entry <= 0:
            log.warning(
                "history.evaluation_invalid_entry",
                symbol=snapshot.symbol,
                prediction_id=snapshot.prediction_id,
                entry_price=entry,
            )
            return None

        if side == "long":
            max_favorable = max(((candle.high - entry) / entry) * 100.0 for candle in candles)
            max_adverse = max(((entry - candle.low) / entry) * 100.0 for candle in candles)
            for candle in candles:
                hit_target = candle.high >= target
                hit_stop = candle.low <= stop
                if hit_target and hit_stop:
                    outcome = EvaluationOutcome.AMBIGUOUS_SAME_BAR.value
                    break
                if hit_target:
                    outcome = EvaluationOutcome.TARGET_HIT.value
                    break
                if hit_stop:
                    outcome = EvaluationOutcome.STOP_HIT.value
                    break
            pnl_close = ((candles[-1].close - entry) / entry) * 100.0
        else:
            max_favorable = max(((entry - candle.low) / entry) * 100.0 for candle in candles)
            max_adverse = max(((candle.high - entry) / entry) * 100.0 for candle in candles)
            for candle in candles:
                hit_target = candle.low <= target
                hit_stop = candle.high >= stop
                if hit_target and hit_stop:
                    outcome = EvaluationOutcome.AMBIGUOUS_SAME_BAR.value
                    break
                if hit_target:
                    outcome = EvaluationOutcome.TARGET_HIT.value
                    break
                if hit_stop:
                    outcome = EvaluationOutcome.STOP_HIT.value
                    break
            pnl_close = ((entry - candles[-1].close) / entry) * 100.0

        primary_eval = SnapshotEvaluation(
            evaluation_status="resolved",
            outcome=outcome,
            resolved_at=datetime.now(UTC),
            max_favorable_excursion_pct=round(max_favorable, 4),
            max_adverse_excursion_pct=round(max_adverse, 4),
            pnl_at_24h_close_pct=round(pnl_close, 4),
            is_profitable_at_24h_close=pnl_close > 0,
        )

        # Multi-window evaluation
        try:
            from futures_analyzer.history.evaluation import MultiWindowEvaluator
            window_evals = MultiWindowEvaluator.evaluate_across_windows(
                candles,
                entry_price=entry,
                target_price=target,
                stop_price=stop,
                side=side,
                entry_time=snapshot.as_of,
            )
            self.repository.save_window_evaluations(snapshot.id, window_evals)
        except Exception as exc:
            log.warning("history.window_eval_skipped", prediction_id=snapshot.prediction_id, error=str(exc))

        # Slippage-adjusted R:R
        try:
            from futures_analyzer.history.evaluation import SlippageCalculator
            em = self.repository.get_enhanced_metrics(snapshot.id)
            if em and em.slippage_estimate_pct > 0:
                exec_metrics = SlippageCalculator.adjust_prices(
                    entry, target, stop,
                    em.slippage_estimate_pct,
                    em.slippage_estimate_pct,
                    em.slippage_estimate_pct,
                    side,
                )
                self.repository.update_slippage_adjusted(
                    snapshot.id,
                    adjusted_rr_ratio=exec_metrics.adjusted_rr_ratio,
                    total_slippage_pct=exec_metrics.total_slippage_pct,
                )
        except Exception as exc:
            log.warning("history.slippage_adjust_skipped", prediction_id=snapshot.prediction_id, error=str(exc))

        return primary_eval

    @staticmethod
    def _evaluation_hours_for_interval(interval: str) -> float:
        """Derive the evaluation time window from the trigger timeframe.

        Scalper (≤5m):  30 min
        Intraday (≤1h): 6h
        Swing (≤4h):    3 days
        Long-term (>4h): 5 days
        """
        minutes_map = {
            "1m": 1, "3m": 3, "5m": 5,
            "15m": 15, "30m": 30, "1h": 60,
            "2h": 120, "4h": 240,
            "6h": 360, "8h": 480, "12h": 720, "1d": 1440,
        }
        minutes = minutes_map.get(interval, 15)
        if minutes <= 5:
            return 0.5       # 30 min
        if minutes <= 60:
            return 6.0       # 6h
        if minutes <= 240:
            return 72.0      # 3 days
        return 120.0         # 5 days

    @staticmethod
    def _evaluation_limit_for_interval(interval: str) -> int:
        minutes_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
        }
        minutes = minutes_map.get(interval)
        if minutes is None or minutes <= 0:
            return 120
        eval_hours = HistoryService._evaluation_hours_for_interval(interval)
        return min(max(int((eval_hours * 60) / minutes) + 2, 2), 1500)

    def recent(self, *, symbol: str | None = None, limit: int = 20) -> list[HistorySnapshot]:
        return self.repository.recent(symbol=symbol, limit=limit)

    def latest_tradable_snapshot(
        self,
        *,
        style: str,
        market_mode: str,
        symbol: str | None = None,
        symbols: list[str] | None = None,
    ) -> HistorySnapshot | None:
        return self.repository.latest_tradable(
            style=style,
            market_mode=market_mode,
            symbol=symbol,
            symbols=symbols,
        )

    def feedback(self, *, symbol: str | None = None, limit: int = 20, days: int | None = None) -> list[HistorySnapshot]:
        return self.repository.evaluated(symbol=symbol, days=days)[:limit]

    def feedback_overview(self, *, symbol: str | None = None, days: int | None = None) -> StatsBucket | None:
        snapshots = self.repository.evaluated(symbol=symbol, days=days)
        if not snapshots:
            return None
        return self._aggregate_bucket("overall", snapshots)

    def stats(self, *, symbol: str | None = None, days: int | None = None) -> HistoryStatsReport:
        snapshots = self.repository.evaluated(symbol=symbol, days=days)
        return HistoryStatsReport(
            overall_feedback=self._aggregate_bucket("overall", snapshots) if snapshots else None,
            confidence_buckets=self._bucket_stats(snapshots, self._confidence_bucket),
            quality_buckets=self._bucket_stats(snapshots, lambda snap: snap.quality_label),
            regime_buckets=self._bucket_stats(snapshots, lambda snap: snap.regime),
        )

    def stats_with_filter(
        self,
        *,
        symbol: str | None = None,
        days: int | None = None,
        min_rsi: float | None = None,
        max_rsi: float | None = None,
        regime: str | None = None,
        min_ob_imbalance: float | None = None,
        volatility_regime: str | None = None,
    ) -> HistoryStatsReport:
        snapshots = self.repository.evaluated_with_filter(
            symbol=symbol,
            days=days,
            min_rsi=min_rsi,
            max_rsi=max_rsi,
            regime=regime,
            min_ob_imbalance=min_ob_imbalance,
            volatility_regime=volatility_regime,
        )
        return HistoryStatsReport(
            overall_feedback=self._aggregate_bucket("overall", snapshots) if snapshots else None,
            confidence_buckets=self._bucket_stats(snapshots, self._confidence_bucket),
            quality_buckets=self._bucket_stats(snapshots, lambda snap: snap.quality_label),
            regime_buckets=self._bucket_stats(snapshots, lambda snap: snap.regime),
        )

    def window_stats(
        self,
        *,
        symbol: str | None = None,
        days: int | None = None,
    ) -> list["WindowEvaluationBucket"]:
        from collections import defaultdict
        from futures_analyzer.history.models import WindowEvaluationBucket
        rows = self.repository.window_evaluations_for_symbol(symbol=symbol, days=days)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["window"]].append(row)
        buckets: list[WindowEvaluationBucket] = []
        for window in sorted(grouped):
            window_rows = grouped[window]
            n = len(window_rows)
            wins = sum(1 for r in window_rows if (r.get("pnl_pct") or 0.0) > 0)
            avg_pnl = sum((r.get("pnl_pct") or 0.0) for r in window_rows) / n
            avg_mfe = sum((r.get("mfe_pct") or 0.0) for r in window_rows) / n
            buckets.append(WindowEvaluationBucket(
                window=window,
                sample_count=n,
                win_rate=round(wins / n, 4),
                avg_pnl_pct=round(avg_pnl, 4),
                avg_mfe_pct=round(avg_mfe, 4),
            ))
        return buckets

    def backfill_enhanced_metrics(self, *, limit: int = 500) -> int:
        """Backfill enhanced metrics for snapshots that have analysis_json but no metrics row."""
        from futures_analyzer.analysis.models import AnalysisResult
        with self.repository._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.analysis_json FROM analysis_snapshots s
                WHERE NOT EXISTS (
                    SELECT 1 FROM snapshot_enhanced_metrics em WHERE em.snapshot_id = s.id
                )
                ORDER BY s.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        filled = 0
        for row in rows:
            try:
                result = AnalysisResult.model_validate_json(str(row["analysis_json"]))
                self.repository.save_enhanced_metrics(int(row["id"]), result.enhanced_metrics)
                filled += 1
            except Exception as exc:
                log.warning("history.backfill_metrics_failed", snapshot_id=int(row["id"]), error=str(exc))
                continue
        return filled

    def compare(
        self,
        *,
        compare_by: HistoryCompareBy,
        symbol: str | None = None,
        days: int | None = None,
    ) -> HistoryCompareReport:
        snapshots = self.repository.evaluated(symbol=symbol, days=days)
        key_map = {
            HistoryCompareBy.SYMBOL: lambda snap: snap.symbol,
            HistoryCompareBy.STYLE: lambda snap: snap.style,
            HistoryCompareBy.MODE: lambda snap: snap.market_mode,
            HistoryCompareBy.CONFIDENCE: self._confidence_bucket,
        }
        return HistoryCompareReport(
            compare_by=compare_by,
            overall_feedback=self._aggregate_bucket("overall", snapshots) if snapshots else None,
            buckets=self._bucket_stats(snapshots, key_map[compare_by]),
        )

    @staticmethod
    def _confidence_bucket(snapshot: HistorySnapshot) -> str:
        if snapshot.confidence < 0.4:
            return "0.00-0.39"
        if snapshot.confidence < 0.7:
            return "0.40-0.69"
        return "0.70-1.00"

    @staticmethod
    def _bucket_stats(
        snapshots: list[HistorySnapshot],
        key_fn,
    ) -> list[StatsBucket]:
        grouped: dict[str, list[HistorySnapshot]] = defaultdict(list)
        for snapshot in snapshots:
            grouped[key_fn(snapshot)].append(snapshot)
        return sorted(
            [HistoryService._aggregate_bucket(bucket, rows) for bucket, rows in grouped.items()],
            key=lambda item: item.bucket,
        )

    @staticmethod
    def _aggregate_bucket(bucket: str, rows: list[HistorySnapshot]) -> StatsBucket:
        sample_count = len(rows)
        if sample_count == 0:
            return StatsBucket(
                bucket=bucket,
                sample_count=0,
                target_hit_rate=0.0,
                stop_hit_rate=0.0,
                profitable_at_24h_rate=0.0,
                average_24h_pnl=0.0,
                average_mfe=0.0,
                average_mae=0.0,
            )
        target_hits = sum(1 for row in rows if row.outcome == EvaluationOutcome.TARGET_HIT.value)
        stop_hits = sum(1 for row in rows if row.outcome == EvaluationOutcome.STOP_HIT.value)
        profitable = sum(1 for row in rows if row.is_profitable_at_24h_close)
        avg_pnl = sum((row.pnl_at_24h_close_pct or 0.0) for row in rows) / sample_count
        avg_mfe = sum((row.max_favorable_excursion_pct or 0.0) for row in rows) / sample_count
        avg_mae = sum((row.max_adverse_excursion_pct or 0.0) for row in rows) / sample_count
        return StatsBucket(
            bucket=bucket,
            sample_count=sample_count,
            target_hit_rate=round(target_hits / sample_count, 4),
            stop_hit_rate=round(stop_hits / sample_count, 4),
            profitable_at_24h_rate=round(profitable / sample_count, 4),
            average_24h_pnl=round(avg_pnl, 4),
            average_mfe=round(avg_mfe, 4),
            average_mae=round(avg_mae, 4),
        )

    @staticmethod
    def decode_contributors(raw_json: str) -> list[dict[str, object]]:
        return json.loads(raw_json)

    def recent_drawdown_state(
        self,
        *,
        symbol: str | None = None,
        lookback: int | None = None,
    ) -> "DrawdownState":
        """Compute rolling drawdown state from the most recent resolved predictions.

        Returns a DrawdownState with severity classification driven by the
        thresholds in AppConfig.drawdown.
        """
        from futures_analyzer.config import load_app_config
        from futures_analyzer.history.models import DrawdownState

        cfg = load_app_config().drawdown
        n = lookback if lookback is not None else cfg.lookback

        snapshots = self.repository.evaluated(symbol=symbol, limit=n)
        # Most-recent first; take up to n (already limited by SQL)
        recent = snapshots[:n]
        sample_count = len(recent)

        if sample_count == 0:
            return DrawdownState(
                lookback=n,
                cumulative_pnl_pct=0.0,
                max_drawdown_pct=0.0,
                current_drawdown_pct=0.0,
                consecutive_losses=0,
                severity="none",
                sample_count=0,
            )

        # Reverse so oldest-first for the equity curve calculation
        pnls = [s.pnl_at_24h_close_pct or 0.0 for s in reversed(recent)]

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        current_dd = max(peak - cumulative, 0.0)

        # Consecutive losses at the tail (most-recent first)
        consecutive = 0
        for s in recent:
            if (s.pnl_at_24h_close_pct or 0.0) < 0:
                consecutive += 1
            else:
                break

        # Severity classification
        if (
            current_dd >= cfg.severe_threshold_pct
            or consecutive >= cfg.severe_consecutive_losses
        ):
            severity = "severe"
        elif (
            current_dd >= cfg.moderate_threshold_pct
            or consecutive >= cfg.moderate_consecutive_losses
        ):
            severity = "moderate"
        elif (
            current_dd >= cfg.mild_threshold_pct
            or consecutive >= cfg.mild_consecutive_losses
        ):
            severity = "mild"
        else:
            severity = "none"

        return DrawdownState(
            lookback=n,
            cumulative_pnl_pct=round(cumulative, 4),
            max_drawdown_pct=round(max_dd, 4),
            current_drawdown_pct=round(current_dd, 4),
            consecutive_losses=consecutive,
            severity=severity,
            sample_count=sample_count,
        )

    def clear_all(self) -> int:
        """Delete all history records. Returns total rows deleted."""
        return self.repository.clear_all()

    def kelly_inputs(
        self,
        *,
        symbol: str | None = None,
        lookback: int = 50,
    ) -> tuple[float, float, float]:
        """Return (win_rate, avg_win_pct, avg_loss_pct) from recent resolved trades.

        Used by PortfolioRiskManager for Kelly-fraction position sizing.
        Returns (0.5, 1.0, 1.0) as a neutral fallback when history is thin.
        """
        snapshots = self.repository.evaluated(symbol=symbol, limit=lookback)
        if not snapshots:
            return 0.5, 1.0, 1.0

        wins = [s.pnl_at_24h_close_pct for s in snapshots if (s.pnl_at_24h_close_pct or 0.0) > 0]
        losses = [abs(s.pnl_at_24h_close_pct) for s in snapshots if (s.pnl_at_24h_close_pct or 0.0) <= 0]

        win_rate = len(wins) / len(snapshots)
        avg_win = (sum(wins) / len(wins)) if wins else 1.0
        avg_loss = (sum(losses) / len(losses)) if losses else 1.0

        return round(win_rate, 4), round(avg_win, 4), round(avg_loss, 4)
