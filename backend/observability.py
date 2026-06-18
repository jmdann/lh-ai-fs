"""Structured logging. Twelve-Factor XI: one structured event per line, JSON,
to stdout. The eval harness greps the canonical ``agent.run`` lines to build
its latency histogram and to populate ``agent_results`` in
``VerificationReport`` for debugging.

Two surfaces are public:

* ``configure_logging(level)`` — installs ``JsonFormatter`` on the root
  logger. Idempotent; safe to call from ``main.py`` startup and from
  ``run_evals.py``.
* ``emit_agent_run(logger, entry)`` — emits the canonical ``agent.run``
  line. Every agent call goes through this so the event shape is
  uniform across the pipeline.

Token counts (``prompt_tokens`` / ``completion_tokens``) are optional in
spec 001 because the ``LLMClient`` Protocol does not yet surface OpenAI's
``Usage`` object; spec 003 (orchestrator hardening) wires that through.
Latency, outcome, agent, and trace id are mandatory and always present.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AgentOutcome = Literal["success", "failure", "partial"]


# ── Canonical agent.run event ──────────────────────────────────────────────


class AgentRunLog(BaseModel):
    """Fields of the canonical ``agent.run`` log line. Required fields are the
    ones the eval harness depends on; optional ones get included when
    available (e.g. token usage once the LLM client surfaces it)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: str = Field(min_length=1)
    prompt_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    outcome: AgentOutcome
    latency_ms: int = Field(ge=0)
    trace_id: str = Field(min_length=1)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    error: str | None = None
    finding_count: int | None = Field(default=None, ge=0)


# ── JsonFormatter ──────────────────────────────────────────────────────────


# Fields the stdlib LogRecord always carries; anything else is treated as
# extra payload and merged into the JSON line.
_BUILTIN_LOGRECORD_FIELDS = frozenset(
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
    """Emits one JSON object per log record. The record's ``msg`` becomes the
    ``event`` field — callers pass dot-namespaced event names
    (``agent.run``, ``orchestrator.boundary.drop`` etc.) rather than
    human-readable strings, so log aggregators can group cleanly."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _BUILTIN_LOGRECORD_FIELDS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ── Setup ──────────────────────────────────────────────────────────────────


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install ``JsonFormatter`` on the root logger. Idempotent — calling
    twice does not double up handlers, so ``main.py`` and ``run_evals.py``
    can both invoke without coordination."""
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


# ── Helpers ────────────────────────────────────────────────────────────────


def emit_agent_run(logger: logging.Logger, entry: AgentRunLog) -> None:
    """Emit the canonical ``agent.run`` event with ``entry`` flattened into
    extras. ``model_dump(exclude_none=True)`` keeps optional fields out
    when the agent has nothing to report (e.g. token counts missing)."""
    logger.info("agent.run", extra=entry.model_dump(exclude_none=True))
