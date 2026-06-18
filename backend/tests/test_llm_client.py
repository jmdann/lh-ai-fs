"""Tests for backend.llm.client.

Unit tests do not hit the network (pytest-socket blocks it). The
``OpenAIClient`` is exercised only at construction and protocol-conformance
level here; real API behavior is integration-tested when an OpenAI key is
available, not in CI.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, SecretStr

from backend.config import Settings
from backend.llm.client import FakeLLMClient, LLMClient, OpenAIClient


class _DummyEcho(BaseModel):
    message: str


class _DummyOther(BaseModel):
    number: int


def _make_settings() -> Settings:
    return Settings.model_construct(
        openai_api_key=SecretStr("sk-test"),
        openai_model="gpt-4o-2024-08-06",
        openai_temperature=0.0,
        eval_seed=42,
        log_level="INFO",
    )


# ── FakeLLMClient ──────────────────────────────────────────────────────────


class TestFakeLLMClient:
    async def test_returns_queued_response(self) -> None:
        client = FakeLLMClient()
        client.queue(system="sys", user="usr", response=_DummyEcho(message="ok"))
        result = await client.complete(system="sys", user="usr", schema=_DummyEcho)
        assert result.message == "ok"

    async def test_records_calls(self) -> None:
        client = FakeLLMClient()
        client.queue(system="s1", user="u1", response=_DummyEcho(message="a"))
        client.queue(system="s2", user="u2", response=_DummyEcho(message="b"))
        await client.complete(system="s1", user="u1", schema=_DummyEcho)
        await client.complete(system="s2", user="u2", schema=_DummyEcho)
        assert client.calls == [
            ("s1", "u1", _DummyEcho),
            ("s2", "u2", _DummyEcho),
        ]

    async def test_raises_on_unexpected_call(self) -> None:
        client = FakeLLMClient()
        with pytest.raises(KeyError, match="no canned response"):
            await client.complete(system="sys", user="usr", schema=_DummyEcho)

    async def test_raises_on_schema_mismatch(self) -> None:
        client = FakeLLMClient()
        client.queue(system="sys", user="usr", response=_DummyEcho(message="ok"))
        with pytest.raises(TypeError, match="canned response is _DummyEcho"):
            await client.complete(system="sys", user="usr", schema=_DummyOther)

    async def test_constructor_seeded_responses(self) -> None:
        seeded = {("sys", "usr"): _DummyEcho(message="seeded")}
        client = FakeLLMClient(responses=seeded)
        result = await client.complete(system="sys", user="usr", schema=_DummyEcho)
        assert result.message == "seeded"

    async def test_repeated_call_returns_same_response(self) -> None:
        client = FakeLLMClient()
        client.queue(system="sys", user="usr", response=_DummyEcho(message="hi"))
        first = await client.complete(system="sys", user="usr", schema=_DummyEcho)
        second = await client.complete(system="sys", user="usr", schema=_DummyEcho)
        assert first.message == "hi"
        assert second.message == "hi"
        assert len(client.calls) == 2


# ── OpenAIClient ───────────────────────────────────────────────────────────


class TestOpenAIClientConstruction:
    def test_constructs_with_settings_no_network(self) -> None:
        # AsyncOpenAI defers all network calls to method invocations;
        # construction is pure-Python wiring. This test asserts that wiring
        # works without ever touching the network.
        client = OpenAIClient(_make_settings())
        assert client is not None

    def test_distinct_settings_produce_distinct_clients(self) -> None:
        s1 = _make_settings()
        s2 = _make_settings()
        a = OpenAIClient(s1)
        b = OpenAIClient(s2)
        assert a is not b


# ── Protocol conformance ───────────────────────────────────────────────────


class TestProtocolConformance:
    def test_fake_llm_client_satisfies_protocol(self) -> None:
        client: LLMClient = FakeLLMClient()
        assert isinstance(client, LLMClient)

    def test_openai_client_satisfies_protocol(self) -> None:
        client: LLMClient = OpenAIClient(_make_settings())
        assert isinstance(client, LLMClient)

    def test_object_without_complete_does_not_satisfy_protocol(self) -> None:
        class _NotAClient:
            pass

        assert not isinstance(_NotAClient(), LLMClient)
