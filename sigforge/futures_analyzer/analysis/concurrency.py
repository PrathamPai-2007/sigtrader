"""Concurrency utilities for parallel and batch analysis."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ParallelAnalyzer:
    """Manages parallel analysis with configurable concurrency."""
    
    def __init__(self, max_concurrent: int = 10):
        """Initialize parallel analyzer.
        
        Args:
            max_concurrent: Maximum number of concurrent operations
        """
        self.max_concurrent = max(1, min(max_concurrent, 50))  # Clamp between 1-50
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
    
    async def analyze_batch(
        self,
        items: list[T],
        analyze_func: Callable[[T], Any],
        *,
        fail_fast: bool = False,
    ) -> list[tuple[T, Any | None, str | None]]:
        """Analyze items in parallel with concurrency control.
        
        Args:
            items: List of items to analyze
            analyze_func: Async function to apply to each item
            fail_fast: If True, stop on first error
        
        Returns:
            List of (item, result, error) tuples
        """
        tasks = [
            self._run_with_semaphore(item, analyze_func, fail_fast)
            for item in items
        ]
        
        if fail_fast:
            return await asyncio.gather(*tasks)
        else:
            return await asyncio.gather(*tasks, return_exceptions=True)
    async def _run_with_semaphore(
        self,
        item: T,
        analyze_func: Callable[[T], Any],
        fail_fast: bool,
    ) -> tuple[T, Any | None, str | None]:
        """Run analysis with semaphore control."""
        async with self.semaphore:
            try:
                result = await analyze_func(item)
                return item, result, None
            except Exception as exc:
                error_msg = str(exc)
                if fail_fast:
                    raise
                return item, None, error_msg


class CacheOptimizer:
    """Optimizes cache usage to reduce redundant API calls."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self.cache: dict[str, tuple[float, Any]] = {}
        self.access_count: dict[str, int] = {}
    
    def get(self, key: str, ttl_seconds: float) -> Any | None:
        """Get cached value if still valid.
        
        Args:
            key: Cache key
            ttl_seconds: Time-to-live in seconds
        
        Returns:
            Cached value or None if expired/missing
        """
        import time
        
        if key not in self.cache:
            return None
        
        ts, value = self.cache[key]
        if (time.monotonic() - ts) > ttl_seconds:
            del self.cache[key]
            return None
        
        self.access_count[key] = self.access_count.get(key, 0) + 1
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set cache value.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        import time
        
        if len(self.cache) >= self.max_cache_size:
            # Evict least frequently used item
            lfu_key = min(self.access_count, key=self.access_count.get, default=None)
            if lfu_key:
                del self.cache[lfu_key]
                del self.access_count[lfu_key]
        
        self.cache[key] = (time.monotonic(), value)
        self.access_count[key] = 0
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
        self.access_count.clear()
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_cache_size,
            "total_accesses": sum(self.access_count.values()),
            "unique_keys": len(self.access_count),
        }


class BatchProcessor:
    """Processes items in batches for efficient API usage."""
    
    def __init__(self, batch_size: int = 10, delay_between_batches: float = 0.1):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            delay_between_batches: Delay in seconds between batches
        """
        self.batch_size = max(1, batch_size)
        self.delay_between_batches = max(0.0, delay_between_batches)
    
    async def process_batches(
        self,
        items: list[T],
        process_func: Callable[[list[T]], Any],
    ) -> list[Any]:
        """Process items in batches.
        
        Args:
            items: List of items to process
            process_func: Async function to process batch
        
        Returns:
            List of results from each batch
        """
        import asyncio
        
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            result = await process_func(batch)
            results.append(result)
            
            # Delay between batches to avoid rate limiting
            if i + self.batch_size < len(items):
                await asyncio.sleep(self.delay_between_batches)
        
        return results


class QueryOptimizer:
    """Optimizes API queries to reduce calls and improve performance."""
    
    @staticmethod
    def deduplicate_symbols(symbols: list[str]) -> list[str]:
        """Remove duplicate symbols while preserving order.
        
        Args:
            symbols: List of symbols
        
        Returns:
            Deduplicated list
        """
        seen = set()
        result = []
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                result.append(symbol)
        return result
    
    @staticmethod
    def filter_by_liquidity(
        symbols: list[str],
        volumes: dict[str, float],
        min_volume: float = 1_000_000.0,
    ) -> list[str]:
        """Filter symbols by minimum trading volume.
        
        Args:
            symbols: List of symbols
            volumes: Dict mapping symbol to 24h volume
            min_volume: Minimum volume threshold
        
        Returns:
            Filtered list of symbols
        """
        return [s for s in symbols if volumes.get(s, 0) >= min_volume]
    
    @staticmethod
    def prioritize_by_volatility(
        symbols: list[str],
        volatilities: dict[str, float],
        ascending: bool = False,
    ) -> list[str]:
        """Sort symbols by volatility.
        
        Args:
            symbols: List of symbols
            volatilities: Dict mapping symbol to volatility
            ascending: If True, sort low to high; else high to low
        
        Returns:
            Sorted list of symbols
        """
        return sorted(
            symbols,
            key=lambda s: volatilities.get(s, 0),
            reverse=not ascending,
        )
