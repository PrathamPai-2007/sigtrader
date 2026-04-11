def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == "BinanceFuturesProvider":
        from futures_analyzer.providers.binance_futures import BinanceFuturesProvider
        return BinanceFuturesProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BinanceFuturesProvider"]


