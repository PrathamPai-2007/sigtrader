"""Structured JSON logging for futures_analyzer.

Usage:
    from futures_analyzer.logging import get_logger
    log = get_logger(__name__)
    log.info("analysis.complete", symbol="BTCUSDT", confidence=0.82)
    log.warning("history.save_skipped", error=str(exc))
    log.error("api.request_failed", url=url, attempts=4)

Log output goes to stderr by default (so it doesn't pollute CLI stdout).
Set FUTURES_ANALYZER_LOG_LEVEL=DEBUG/INFO/WARNING/ERROR to control verbosity.
Set FUTURES_ANALYZER_LOG_FILE=/path/to/app.log to also write to a file.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any


# ── JSON formatter ────────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """Emits one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        # Extra structured fields attached via log.info("event", key=val)
        for key, val in record.__dict__.items():
            if key.startswith("_kv_"):
                payload[key[4:]] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ── Adapter that accepts keyword args as structured fields ────────────────────

class _StructuredAdapter(logging.LoggerAdapter):
    """Wraps a Logger so callers can pass keyword context:

        log.info("event.name", symbol="BTCUSDT", confidence=0.82)
    """

    def process(self, msg: object, kwargs: dict[str, Any]) -> tuple[object, dict[str, Any]]:
        extra = kwargs.pop("extra", {})
        # Move any remaining kwargs into extra as _kv_ prefixed keys
        for key, val in list(kwargs.items()):
            if key not in ("exc_info", "stack_info", "stacklevel"):
                extra[f"_kv_{key}"] = val
                del kwargs[key]
        kwargs["extra"] = extra
        return msg, kwargs

    # Convenience: forward stack level so line numbers stay correct
    def _log(self, level: int, msg: object, **kwargs: Any) -> None:  # type: ignore[override]
        kwargs.setdefault("stacklevel", 3)
        msg, kwargs = self.process(msg, kwargs)
        self.logger.log(level, msg, **kwargs)

    def debug(self, msg: object, **kwargs: Any) -> None:  # type: ignore[override]
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: object, **kwargs: Any) -> None:  # type: ignore[override]
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: object, **kwargs: Any) -> None:  # type: ignore[override]
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: object, **kwargs: Any) -> None:  # type: ignore[override]
        self._log(logging.ERROR, msg, **kwargs)

    def exception(self, msg: object, **kwargs: Any) -> None:  # type: ignore[override]
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, **kwargs)


# ── Root setup (called once) ──────────────────────────────────────────────────

_configured = False


def configure_logging() -> None:
    """Configure the root logger for futures_analyzer.

    Safe to call multiple times — only runs once.
    Reads env vars:
      FUTURES_ANALYZER_LOG_LEVEL  (default: WARNING)
      FUTURES_ANALYZER_LOG_FILE   (default: none / stderr only)
    """
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("FUTURES_ANALYZER_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)

    root = logging.getLogger("futures_analyzer")
    root.setLevel(level)
    root.propagate = False

    formatter = _JsonFormatter()

    # Always log to stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    # Optionally also log to a rotating file
    log_file = os.environ.get("FUTURES_ANALYZER_LOG_FILE", "").strip()
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> _StructuredAdapter:
    """Return a structured logger for the given module name.

    Example:
        log = get_logger(__name__)
        log.info("fetch.complete", symbol="BTCUSDT", bars=600)
    """
    configure_logging()
    return _StructuredAdapter(logging.getLogger(name), extra={})
