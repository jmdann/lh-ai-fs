"""Tests for backend.observability — minimal coverage of the load-bearing
contract: JsonFormatter emits valid JSON with merged extras, configure_logging
is idempotent, and log_agent_run round-trips the canonical event name."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from io import StringIO

import pytest

from backend.observability import (
    JsonFormatter,
    _reset_logging_for_tests,
    configure_logging,
    log_agent_run,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    _reset_logging_for_tests()
    yield
    _reset_logging_for_tests()


def _capture(logger: logging.Logger) -> StringIO:
    captured = StringIO()
    handler = logging.StreamHandler(captured)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return captured


def test_emits_valid_json_with_event_field() -> None:
    logger = logging.getLogger("test1")
    captured = _capture(logger)
    logger.info("agent.run", extra={"agent": "X", "latency_ms": 42})
    payload = json.loads(captured.getvalue())
    assert payload["event"] == "agent.run"
    assert payload["agent"] == "X"
    assert payload["latency_ms"] == 42
    assert payload["level"] == "INFO"
    assert payload["ts"].endswith("+00:00")


def test_builtin_logrecord_fields_excluded() -> None:
    """Stdlib LogRecord carries ~25 housekeeping fields. They must not leak
    into the payload — a hypothetical caller key like 'levelname' in extras
    would silently shadow the level we set."""
    logger = logging.getLogger("test2")
    captured = _capture(logger)
    logger.info("e", extra={"agent": "X"})
    payload = json.loads(captured.getvalue())
    for field in ("pathname", "lineno", "thread", "module", "msg"):
        assert field not in payload


def test_unserializable_objects_fall_back_to_str() -> None:
    class _Opaque:
        def __str__(self) -> str:
            return "fallback"

    logger = logging.getLogger("test3")
    captured = _capture(logger)
    logger.info("e", extra={"opaque": _Opaque()})
    assert "fallback" in captured.getvalue()


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG


def test_log_agent_run_writes_canonical_event() -> None:
    logger = logging.getLogger("test_canonical")
    captured = _capture(logger)
    log_agent_run(
        logger,
        agent="CitationExtractor",
        prompt_version="1.0.0",
        outcome="success",
        latency_ms=412,
        trace_id="trace-abc",
    )
    payload = json.loads(captured.getvalue())
    assert payload["event"] == "agent.run"
    assert payload["agent"] == "CitationExtractor"
    assert payload["outcome"] == "success"
