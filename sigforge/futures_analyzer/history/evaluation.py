"""Enhanced evaluation system for improved prediction assessment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from futures_analyzer.analysis.models import Candle


class EvaluationWindow(str, Enum):
    """Evaluation time windows."""
    SHORT_TERM = "4h"  # 4 hours
    MEDIUM_TERM = "24h"  # 24 hours
    LONG_TERM = "7d"  # 7 days
    EXTENDED = "30d"  # 30 days


class SlippageModel(str, Enum):
    """Slippage estimation models."""
    CONSERVATIVE = "conservative"  # 0.1% per 1% of daily volume
    MODERATE = "moderate"  # 0.05% per 1% of daily volume
    OPTIMISTIC = "optimistic"  # 0.02% per 1% of daily volume


@dataclass
class ExecutionMetrics:
    """Execution quality metrics."""
    entry_slippage_pct: float
    target_slippage_pct: float
    stop_slippage_pct: float
    total_slippage_pct: float
    adjusted_entry_price: float
    adjusted_target_price: float
    adjusted_stop_price: float
    adjusted_rr_ratio: float


@dataclass
class DrawdownMetrics:
    """Drawdown analysis metrics."""
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    consecutive_losses: int
    max_consecutive_losses: int
    recovery_bars: int | None
    profit_factor: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    outcome: str  # "win", "loss", "breakeven", "ambiguous"
    profit_loss_pct: float
    profit_loss_absolute: float
    risk_reward_achieved: float
    execution_metrics: ExecutionMetrics
    drawdown_metrics: DrawdownMetrics
    evaluation_window: str
    bars_to_resolution: int
    confidence_score: float


class SlippageCalculator:
    """Calculates realistic slippage based on order size and market conditions."""
    
    @staticmethod
    def calculate_slippage(
        order_size_pct: float,
        bid_ask_spread_pct: float,
        liquidity_score: float,
        model: SlippageModel = SlippageModel.MODERATE,
    ) -> float:
        """Calculate slippage for an order.
        
        Args:
            order_size_pct: Order size as % of daily volume
            bid_ask_spread_pct: Current bid-ask spread %
            liquidity_score: Liquidity score (0-100)
            model: Slippage model to use
        
        Returns:
            Estimated slippage %
        """
        # Base slippage from spread
        spread_slippage = bid_ask_spread_pct * 0.5
        
        # Order size impact
        if model == SlippageModel.CONSERVATIVE:
            size_impact = order_size_pct * 0.001
        elif model == SlippageModel.OPTIMISTIC:
            size_impact = order_size_pct * 0.0002
        else:  # MODERATE
            size_impact = order_size_pct * 0.0005
        
        # Liquidity adjustment
        liquidity_factor = max(0.5, 1.0 - (liquidity_score / 100.0))
        
        total_slippage = (spread_slippage + size_impact) * liquidity_factor
        return max(0.0, total_slippage)
    
    @staticmethod
    def adjust_prices(
        entry_price: float,
        target_price: float,
        stop_price: float,
        entry_slippage_pct: float,
        target_slippage_pct: float,
        stop_slippage_pct: float,
        side: str,
    ) -> ExecutionMetrics:
        """Adjust prices for slippage.
        
        Args:
            entry_price: Original entry price
            target_price: Original target price
            stop_price: Original stop price
            entry_slippage_pct: Entry slippage %
            target_slippage_pct: Target slippage %
            stop_slippage_pct: Stop slippage %
            side: "long" or "short"
        
        Returns:
            ExecutionMetrics with adjusted prices
        """
        if side == "long":
            # Slippage works against us on entry (higher), helps on target (lower), hurts on stop (lower)
            adj_entry = entry_price * (1.0 + entry_slippage_pct / 100.0)
            adj_target = target_price * (1.0 - target_slippage_pct / 100.0)
            adj_stop = stop_price * (1.0 - stop_slippage_pct / 100.0)
        else:  # short
            # Slippage works against us on entry (lower), helps on target (higher), hurts on stop (higher)
            adj_entry = entry_price * (1.0 - entry_slippage_pct / 100.0)
            adj_target = target_price * (1.0 + target_slippage_pct / 100.0)
            adj_stop = stop_price * (1.0 + stop_slippage_pct / 100.0)
        
        # Calculate adjusted R:R
        if side == "long":
            risk = adj_entry - adj_stop
            reward = adj_target - adj_entry
        else:
            risk = adj_stop - adj_entry
            reward = adj_entry - adj_target
        
        adjusted_rr = reward / risk if risk > 0 else 0.0
        total_slippage = entry_slippage_pct + target_slippage_pct + stop_slippage_pct
        
        return ExecutionMetrics(
            entry_slippage_pct=entry_slippage_pct,
            target_slippage_pct=target_slippage_pct,
            stop_slippage_pct=stop_slippage_pct,
            total_slippage_pct=total_slippage,
            adjusted_entry_price=adj_entry,
            adjusted_target_price=adj_target,
            adjusted_stop_price=adj_stop,
            adjusted_rr_ratio=adjusted_rr,
        )


class MultiWindowEvaluator:
    """Evaluates predictions across multiple time windows."""
    
    @staticmethod
    def evaluate_across_windows(
        candles: list[Candle],
        entry_price: float,
        target_price: float,
        stop_price: float,
        side: str,
        entry_time: datetime,
    ) -> dict[str, PerformanceMetrics]:
        """Evaluate prediction across multiple time windows.
        
        Args:
            candles: List of candles after entry
            entry_price: Entry price
            target_price: Target price
            stop_price: Stop price
            side: "long" or "short"
            entry_time: Entry timestamp
        
        Returns:
            Dict mapping window name to PerformanceMetrics
        """
        results = {}
        
        for window in EvaluationWindow:
            metrics = MultiWindowEvaluator._evaluate_window(
                candles,
                entry_price,
                target_price,
                stop_price,
                side,
                entry_time,
                window,
            )
            results[window.value] = metrics
        
        return results
    
    @staticmethod
    def _evaluate_window(
        candles: list[Candle],
        entry_price: float,
        target_price: float,
        stop_price: float,
        side: str,
        entry_time: datetime,
        window: EvaluationWindow,
    ) -> PerformanceMetrics:
        """Evaluate prediction within a specific window."""
        
        # Parse window duration
        if window == EvaluationWindow.SHORT_TERM:
            duration = timedelta(hours=4)
        elif window == EvaluationWindow.MEDIUM_TERM:
            duration = timedelta(hours=24)
        elif window == EvaluationWindow.LONG_TERM:
            duration = timedelta(days=7)
        else:  # EXTENDED
            duration = timedelta(days=30)
        
        window_end = entry_time + duration
        
        # Filter candles within window
        window_candles = [
            c for c in candles
            if entry_time <= c.close_time <= window_end
        ]
        
        if not window_candles:
            return PerformanceMetrics(
                outcome="no_data",
                profit_loss_pct=0.0,
                profit_loss_absolute=0.0,
                risk_reward_achieved=0.0,
                execution_metrics=ExecutionMetrics(0, 0, 0, 0, entry_price, target_price, stop_price, 0),
                drawdown_metrics=DrawdownMetrics(0, 0, 0, 0, None, 0),
                evaluation_window=window.value,
                bars_to_resolution=0,
                confidence_score=0.0,
            )
        
        # Analyze price action
        outcome, profit_loss_pct, bars_to_resolution = MultiWindowEvaluator._analyze_price_action(
            window_candles,
            entry_price,
            target_price,
            stop_price,
            side,
        )
        
        # Calculate drawdown metrics
        drawdown = MultiWindowEvaluator._calculate_drawdown(
            window_candles,
            entry_price,
            side,
        )
        
        # Calculate achieved R:R
        if side == "long":
            risk = entry_price - stop_price
            reward = max(c.high for c in window_candles) - entry_price
        else:
            risk = stop_price - entry_price
            reward = entry_price - min(c.low for c in window_candles)
        
        rr_achieved = reward / risk if risk > 1e-10 else 0.0
        
        # Confidence score based on outcome clarity
        if outcome in ["win", "loss"]:
            confidence = 0.95
        elif outcome == "breakeven":
            confidence = 0.70
        else:
            confidence = 0.50
        
        return PerformanceMetrics(
            outcome=outcome,
            profit_loss_pct=profit_loss_pct,
            profit_loss_absolute=profit_loss_pct * entry_price / 100.0,
            risk_reward_achieved=rr_achieved,
            execution_metrics=ExecutionMetrics(0, 0, 0, 0, entry_price, target_price, stop_price, rr_achieved),
            drawdown_metrics=drawdown,
            evaluation_window=window.value,
            bars_to_resolution=bars_to_resolution,
            confidence_score=confidence,
        )
    
    @staticmethod
    def _analyze_price_action(
        candles: list[Candle],
        entry_price: float,
        target_price: float,
        stop_price: float,
        side: str,
    ) -> tuple[str, float, int]:
        """Analyze price action to determine outcome.
        
        Returns:
            Tuple of (outcome, profit_loss_pct, bars_to_resolution)
        """
        for i, candle in enumerate(candles):
            if side == "long":
                if candle.high >= target_price and candle.low <= stop_price:
                    return "ambiguous_same_bar", 0.0, i
                if candle.high >= target_price:
                    profit_pct = ((target_price - entry_price) / entry_price) * 100.0
                    return "win", profit_pct, i
                if candle.low <= stop_price:
                    loss_pct = ((stop_price - entry_price) / entry_price) * 100.0
                    return "loss", loss_pct, i
            else:  # short
                if candle.low <= target_price and candle.high >= stop_price:
                    return "ambiguous_same_bar", 0.0, i
                if candle.low <= target_price:
                    profit_pct = ((entry_price - target_price) / entry_price) * 100.0
                    return "win", profit_pct, i
                if candle.high >= stop_price:
                    loss_pct = ((entry_price - stop_price) / entry_price) * 100.0
                    return "loss", loss_pct, i
        
        # No resolution within window
        if side == "long":
            current_price = candles[-1].close
            profit_pct = ((current_price - entry_price) / entry_price) * 100.0
        else:
            current_price = candles[-1].close
            profit_pct = ((entry_price - current_price) / entry_price) * 100.0
        
        if abs(profit_pct) < 0.1:
            return "breakeven", profit_pct, len(candles)
        elif profit_pct > 0:
            return "partial_win", profit_pct, len(candles)
        else:
            return "partial_loss", profit_pct, len(candles)
    
    @staticmethod
    def _calculate_drawdown(
        candles: list[Candle],
        entry_price: float,
        side: str,
    ) -> DrawdownMetrics:
        """Calculate drawdown metrics."""
        
        if not candles:
            return DrawdownMetrics(
                max_favorable_excursion_pct=0.0,
                max_adverse_excursion_pct=0.0,
                consecutive_losses=0,
                max_consecutive_losses=0,
                recovery_bars=None,
                profit_factor=0.0,
            )

        if side == "long":
            prices = [c.close for c in candles]
            running_max = entry_price
            max_dd = 0.0
            max_favorable = max((c.high - entry_price) / entry_price * 100.0 for c in candles)
            max_adverse = max((entry_price - c.low) / entry_price * 100.0 for c in candles)
            
            for price in prices:
                running_max = max(running_max, price)
                dd = (running_max - price) / running_max * 100.0
                max_dd = max(max_dd, dd)
        else:  # short
            prices = [c.close for c in candles]
            running_min = entry_price
            max_dd = 0.0
            max_favorable = max((entry_price - c.low) / entry_price * 100.0 for c in candles)
            max_adverse = max((c.high - entry_price) / entry_price * 100.0 for c in candles)
            
            for price in prices:
                running_min = min(running_min, price)
                dd = (price - running_min) / running_min * 100.0
                max_dd = max(max_dd, dd)
        
        # Count consecutive losses (simplified)
        consecutive_losses = 0
        max_consecutive = 0
        for i in range(1, len(candles)):
            if candles[i].close < candles[i - 1].close:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # Calculate profit factor (simplified)
        gains = sum(max(0, candles[i].close - candles[i - 1].close) for i in range(1, len(candles)))
        losses = sum(max(0, candles[i - 1].close - candles[i].close) for i in range(1, len(candles)))
        profit_factor = gains / losses if losses > 0 else 0.0
        
        return DrawdownMetrics(
            max_favorable_excursion_pct=max_favorable,
            max_adverse_excursion_pct=max_adverse,
            consecutive_losses=consecutive_losses,
            max_consecutive_losses=max_consecutive,
            recovery_bars=None,
            profit_factor=profit_factor,
        )


class EvaluationComparator:
    """Compares evaluations across different scenarios."""
    
    @staticmethod
    def compare_with_slippage(
        metrics_no_slippage: PerformanceMetrics,
        metrics_with_slippage: PerformanceMetrics,
    ) -> dict[str, Any]:
        """Compare metrics with and without slippage.
        
        Returns:
            Dict with comparison results
        """
        return {
            "outcome_changed": metrics_no_slippage.outcome != metrics_with_slippage.outcome,
            "profit_loss_impact_pct": (
                metrics_with_slippage.profit_loss_pct - metrics_no_slippage.profit_loss_pct
            ),
            "rr_ratio_impact": (
                metrics_with_slippage.risk_reward_achieved - metrics_no_slippage.risk_reward_achieved
            ),
            "confidence_impact": (
                metrics_with_slippage.confidence_score - metrics_no_slippage.confidence_score
            ),
        }
    
    @staticmethod
    def aggregate_window_results(
        window_results: dict[str, PerformanceMetrics],
    ) -> dict[str, Any]:
        """Aggregate results across all windows.
        
        Returns:
            Dict with aggregated statistics
        """
        outcomes = [m.outcome for m in window_results.values()]
        profit_losses = [m.profit_loss_pct for m in window_results.values()]
        rr_ratios = [m.risk_reward_achieved for m in window_results.values()]
        
        return {
            "consistent_outcome": len(set(outcomes)) == 1,
            "most_common_outcome": max(set(outcomes), key=outcomes.count),
            "avg_profit_loss_pct": sum(profit_losses) / len(profit_losses),
            "avg_rr_ratio": sum(rr_ratios) / len(rr_ratios),
            "window_count": len(window_results),
            "outcome_distribution": {o: outcomes.count(o) for o in set(outcomes)},
        }
