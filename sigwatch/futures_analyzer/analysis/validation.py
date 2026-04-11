"""Data validation and quality checks for analysis."""

from __future__ import annotations

from typing import Any

from futures_analyzer.analysis.models import Candle


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class CandleValidator:
    """Validates candle data quality and completeness."""
    
    MIN_CANDLES_REQUIRED = 30
    MAX_CANDLES_RECOMMENDED = 1000
    
    @staticmethod
    def validate_candles(
        candles: list[Candle],
        *,
        min_required: int = MIN_CANDLES_REQUIRED,
        symbol: str = "unknown",
    ) -> tuple[bool, list[str]]:
        """Validate candle data and return (is_valid, warnings).
        
        Args:
            candles: List of Candle objects to validate
            min_required: Minimum number of candles required
            symbol: Symbol name for error messages
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        if not candles:
            return False, [f"No candle data available for {symbol}"]
        
        if len(candles) < min_required:
            return False, [
                f"Insufficient data for {symbol}: {len(candles)} candles, "
                f"need at least {min_required}"
            ]
        
        if len(candles) > CandleValidator.MAX_CANDLES_RECOMMENDED:
            warnings.append(
                f"Large dataset for {symbol}: {len(candles)} candles. "
                f"Consider using a smaller lookback window for performance."
            )
        
        # Check for data quality issues
        for i, candle in enumerate(candles):
            if candle.high < candle.low:
                return False, [
                    f"Invalid candle at index {i} for {symbol}: "
                    f"high ({candle.high}) < low ({candle.low})"
                ]
            
            if candle.close < candle.low or candle.close > candle.high:
                return False, [
                    f"Invalid candle at index {i} for {symbol}: "
                    f"close ({candle.close}) outside high/low range"
                ]
            
            if candle.open < candle.low or candle.open > candle.high:
                return False, [
                    f"Invalid candle at index {i} for {symbol}: "
                    f"open ({candle.open}) outside high/low range"
                ]
            
            if candle.volume < 0:
                return False, [
                    f"Invalid candle at index {i} for {symbol}: "
                    f"negative volume ({candle.volume})"
                ]
        
        # Check for time ordering
        for i in range(1, len(candles)):
            if candles[i].open_time <= candles[i - 1].close_time:
                warnings.append(
                    f"Candle time ordering issue at index {i} for {symbol}: "
                    f"timestamps not strictly increasing"
                )
                break
        
        # Check for gaps in data
        if len(candles) >= 2:
            time_diffs = [
                (candles[i].open_time - candles[i - 1].close_time).total_seconds()
                for i in range(1, len(candles))
            ]
            
            if time_diffs:
                avg_diff = sum(time_diffs) / len(time_diffs)
                max_diff = max(time_diffs)
                
                # Allow 10% variance in time differences
                if max_diff > avg_diff * 1.1:
                    warnings.append(
                        f"Potential data gaps detected for {symbol}: "
                        f"max interval {max_diff}s vs avg {avg_diff}s"
                    )
        
        return True, warnings
    
    @staticmethod
    def validate_symbol(symbol: str) -> tuple[bool, str]:
        """Validate symbol format.
        
        Args:
            symbol: Symbol to validate (e.g., 'BTCUSDT')
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not symbol:
            return False, "Symbol cannot be empty"
        
        if not isinstance(symbol, str):
            return False, f"Symbol must be string, got {type(symbol)}"
        
        if len(symbol) < 3:
            return False, f"Symbol too short: {symbol}"
        
        if len(symbol) > 20:
            return False, f"Symbol too long: {symbol}"
        
        if not symbol.isupper():
            return False, f"Symbol must be uppercase: {symbol}"
        
        if not symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "").isalpha():
            return False, f"Symbol contains invalid characters: {symbol}"
        
        return True, ""
    
    @staticmethod
    def validate_market_meta(
        mark_price: float | None,
        tick_size: float | None,
        symbol: str = "unknown",
    ) -> tuple[bool, list[str]]:
        """Validate market metadata.
        
        Args:
            mark_price: Current mark price
            tick_size: Minimum price increment
            symbol: Symbol name for error messages
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        if mark_price is None or mark_price <= 0:
            return False, [f"Invalid mark price for {symbol}: {mark_price}"]
        
        if tick_size is None or tick_size <= 0:
            warnings.append(f"Invalid tick size for {symbol}: {tick_size}, using default precision")
        
        return True, warnings


class DataQualityMonitor:
    """Monitors data quality across multiple requests."""
    
    def __init__(self, max_consecutive_failures: int = 3):
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures: dict[str, int] = {}
        self.last_error: dict[str, str] = {}
    
    def record_success(self, symbol: str) -> None:
        """Record successful data fetch."""
        self.consecutive_failures[symbol] = 0
        self.last_error.pop(symbol, None)
    
    def record_failure(self, symbol: str, error: str) -> None:
        """Record failed data fetch."""
        self.consecutive_failures[symbol] = self.consecutive_failures.get(symbol, 0) + 1
        self.last_error[symbol] = error
    
    def should_retry(self, symbol: str) -> bool:
        """Check if symbol should be retried."""
        failures = self.consecutive_failures.get(symbol, 0)
        return failures < self.max_consecutive_failures
    
    def get_status(self, symbol: str) -> dict[str, Any]:
        """Get current status for a symbol."""
        return {
            "symbol": symbol,
            "consecutive_failures": self.consecutive_failures.get(symbol, 0),
            "last_error": self.last_error.get(symbol),
            "should_retry": self.should_retry(symbol),
        }
