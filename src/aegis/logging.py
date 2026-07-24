"""Structured logging.

Human-readable by default; set ``AEGIS_LOG_JSON=1`` for one JSON object per line, which is what
the container images use. Run and agent ids ride along via a ``contextvars`` binding so that
every log line inside an agent is attributable without threading a logger through call sites.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Default is None, not {}: a mutable default on a ContextVar is shared across every context,
# so anything that mutated it in place would leak fields between unrelated runs.
_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "aegis_ctx", default=None
)


def _current() -> dict[str, Any]:
    return _context.get() or {}


_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


@contextmanager
def bind(**fields: Any) -> Iterator[None]:
    """Attach fields to every log line emitted inside the block."""
    token = _context.set({**_current(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _current().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(
            {
                k: v
                for k, v in record.__dict__.items()
                if k not in _RESERVED and not k.startswith("_")
            }
        )
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        return base


def configure(level: str = "INFO", json_output: bool = False) -> None:
    """Idempotently install the root handler."""
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(
        _JsonFormatter()
        if json_output
        else _TextFormatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are chatty at DEBUG and never tell us anything we want.
    for noisy in ("httpx", "httpcore", "urllib3", "anthropic._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
