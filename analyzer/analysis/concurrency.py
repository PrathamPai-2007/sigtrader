# Backward-compatibility shim — canonical location is analyzer.analysis.utils.concurrency
from analyzer.analysis.utils.concurrency import *  # noqa: F401, F403
from analyzer.analysis.utils.concurrency import ParallelAnalyzer, CacheOptimizer, BatchProcessor, QueryOptimizer
