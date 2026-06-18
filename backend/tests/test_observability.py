"""Tests for backend.observability.

Properties under test are the contract the eval harness depends on:
canonical ``agent.run`` lines have the right fields, the JSON line
parses cleanly, configure_logging is idempotent, and the formatter
never lets a built-in LogRecord field leak into the payload (which
would silently overshadow our own keys)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from io import StringIO

import pytest
from pydantic import ValidationError

from backend.observability import (
    AgentRunLog,
    JsonFormatter,
    _reset_logging_for_tests,
    configure_logging,
    emit_agent_run,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    _reset_logging_for_tests()
    yield
    _reset_logging_for_tests()


# ── AgentRunLog ────────────────────────────────────────────────────────────


class TestAgentRunLog:
    def test_minimum_required_fields(self) -> None:
        entry = AgentRunLog(
            agent="CitationExtractor",
            prompt_version="1.0.0",
            outcome="success",
            latency_ms=412,
            trace_id="trace-abc",
        )
        assert entry.agent == "CitationExtractor"
        assert entry.prompt_tokens is None  # optional, not populated yet

    def test_outcome_literal_enforced(self) -> None:
        with pytest.raises(ValidationError):
            AgentRunLog(
                agent="X",
                prompt_version="1.0.0",
                outcome="mystery",  # type: ignore[arg-type]
                latency_ms=0,
                trace_id="t",
            )

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentRunLog(
                agent="X",
                prompt_version="1.0.0",
                outcome="success",
                latency_ms=-1,
                trace_id="t",
            )

    def test_prompt_version_pattern(self) -> None:
        with pytest.raises(ValidationError):
            AgentRunLog(
                agent="X",
                prompt_version="1.0",
                outcome="success",
                latency_ms=0,
                trace_id="t",
            )

    def test_negative_token_counts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentRunLog(
                agent="X",
                prompt_version="1.0.0",
                outcome="success",
                latency_ms=0,
                trace_id="t",
                prompt_tokens=-1,
            )

    def test_is_frozen(self) -> None:
        entry = AgentRunLog(
            agent="X",
            prompt_version="1.0.0",
            outcome="success",
            latency_ms=0,
            trace_id="t",
        )
        with pytest.raises(ValidationError):
            entry.agent = "Y"  # type: ignore[misc]


# ── JsonFormatter ─────────────────────────────────────────────────────────


def _format_one(record: logging.LogRecord) -> dict[str, object]:
    payload: dict[str, object] = json.loads(JsonFormatter().format(record))
    return payload


class TestJsonFormatter:
    def test_emits_valid_json(self) -> None:
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="x",
            lineno=1,
            msg="event.name",
            args=(),
            exc_info=None,
        )
        payload = _format_one(record)
        assert payload["event"] == "event.name"
        assert payload["level"] == "INFO"
        assert "ts" in payload

    def test_iso_8601_utc_timestamp(self) -> None:
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="x",
            lineno=1,
            msg="e",
            args=(),
            exc_info=None,
        )
        payload = _format_one(record)
        # Two cheap acceptance checks rather than parsing the timestamp.
        assert isinstance(payload["ts"], str)
        assert payload["ts"].endswith("+00:00")

    def test_extra_fields_merged(self) -> None:
        logger = logging.getLogger("test_extra")
        captured = StringIO()
        handler = logging.StreamHandler(captured)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("agent.run", extra={"agent": "X", "latency_ms": 100})
        payload = json.loads(captured.getvalue())
        assert payload["agent"] == "X"
        assert payload["latency_ms"] == 100
        assert payload["event"] == "agent.run"

    def test_builtin_fields_excluded(self) -> None:
        """Stdlib LogRecord carries ~25 housekeeping fields (pathname, lineno,
        thread, etc.). Letting them through would bury our payload in noise
        and risk colliding with our own keys (e.g. a hypothetical 'level' in
        extras would silently overwrite the levelname). Assert at least a
        few canonical built-ins do not leak."""
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="/x.py",
            lineno=42,
            msg="e",
            args=(),
            exc_info=None,
        )
        payload = _format_one(record)
        for field in ("pathname", "lineno", "thread", "process", "module", "msg"):
            assert field not in payload

    def test_exception_serialized(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="t",
                level=logging.ERROR,
                pathname="x",
                lineno=1,
                msg="e",
                args=(),
                exc_info=sys.exc_info(),
            )
        payload = _format_one(record)
        assert "exc" in payload
        assert "ValueError: boom" in str(payload["exc"])

    def test_unserializable_objects_fall_back_to_str(self) -> None:
        class _NotJson:
            def __str__(self) -> str:
                return "fallback"

        logger = logging.getLogger("test_fallback")
        captured = StringIO()
        handler = logging.StreamHandler(captured)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.info("e", extra={"opaque": _NotJson()})
        payload = json.loads(captured.getvalue())
        assert payload["opaque"] == "fallback"


# ── configure_logging ────────────────────────────────────────────────────


class TestConfigureLogging:
    def test_installs_handler_with_json_formatter(self) -> None:
        configure_logging("INFO")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.INFO

    def test_idempotent_no_handler_duplication(self) -> None:
        configure_logging("INFO")
        configure_logging("INFO")
        configure_logging("DEBUG")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG

    def test_level_can_be_changed_on_reconfigure(self) -> None:
        configure_logging("INFO")
        configure_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING


# ── emit_agent_run ───────────────────────────────────────────────────────


class TestEmitAgentRun:
    def test_writes_canonical_line(self) -> None:
        logger = logging.getLogger("test_emit")
        captured = StringIO()
        handler = logging.StreamHandler(captured)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        emit_agent_run(
            logger,
            AgentRunLog(
                agent="CitationExtractor",
                prompt_version="1.0.0",
                outcome="success",
                latency_ms=412,
                trace_id="trace-abc",
                finding_count=3,
            ),
        )
        payload = json.loads(captured.getvalue())
        assert payload["event"] == "agent.run"
        assert payload["agent"] == "CitationExtractor"
        assert payload["prompt_version"] == "1.0.0"
        assert payload["outcome"] == "success"
        assert payload["latency_ms"] == 412
        assert payload["trace_id"] == "trace-abc"
        assert payload["finding_count"] == 3

    def test_excludes_unset_optional_fields(self) -> None:
        logger = logging.getLogger("test_emit_partial")
        captured = StringIO()
        handler = logging.StreamHandler(captured)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        emit_agent_run(
            logger,
            AgentRunLog(
                agent="X",
                prompt_version="1.0.0",
                outcome="success",
                latency_ms=10,
                trace_id="t",
            ),
        )
        payload = json.loads(captured.getvalue())
        assert "prompt_tokens" not in payload
        assert "completion_tokens" not in payload
        assert "error" not in payload
        assert "finding_count" not in payload
