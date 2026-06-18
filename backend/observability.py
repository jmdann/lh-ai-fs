"""Structured JSON logging. Twelve-Factor XI: one event per line, JSON, to
stdout.

Public surface kept minimal for spec 001 (codex round 2 + size discipline):

* ``configure_logging(level)`` — idempotent root-logger install.
* ``JsonFormatter`` — stdlib ``Formatter`` subclass that renders
  ``record.msg`` as the ``event`` field and merges any ``extra`` kwargs
  into the JSON payload, while excluding the LogRecord housekeeping
  fields so they cannot collide with payload keys.
* ``log_agent_run(logger, **fields)`` — convenience wrapper that emits
  the canonical ``agent.run`` event.

Spec 003 may add a typed ``AgentRunLog`` contract; for spec 001 the eval
harness parses raw JSON lines and does not need Pydantic validation on
log entries.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Fields the stdlib LogRecord always carries. Anything else on the record
# is treated as caller payload and merged into the JSON line.
_BUILTIN_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _BUILTIN_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install ``JsonFormatter`` on the root logger. Idempotent."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level)
        return
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


def _reset_logging_for_tests() -> None:
    """Test helper. Production code does not call this."""
    global _CONFIGURED
    _CONFIGURED = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)


def log_agent_run(logger: logging.Logger, **fields: Any) -> None:
    """Emit the canonical ``agent.run`` event. Required fields by convention:
    ``agent``, ``prompt_version``, ``outcome``, ``latency_ms``, ``trace_id``.
    Optional: ``prompt_tokens``, ``completion_tokens``, ``error``,
    ``finding_count``. Not validated at runtime — the orchestrator is
    trusted to populate the right shape."""
    logger.info("agent.run", extra=fields)
