"""LLM client layer. STANDARDS § 3.2: agents take an ``LLMClient`` Protocol,
never the concrete OpenAI SDK. Provider swap = new ``Client`` class, zero
agent code changes. FastAPI ``Depends`` lives at the API edge; agents
receive the client as a constructor argument.

Three implementations live here:

* ``LLMClient`` — the Protocol agents type against.
* ``OpenAIClient`` — the production implementation. Uses OpenAI's
  structured-outputs ``parse`` endpoint so the model is forced to return
  JSON matching a Pydantic schema. Tenacity-wrapped for transient errors
  (twelve-factor IX).
* ``FakeLLMClient`` — test double. Returns pre-canned Pydantic objects
  keyed by ``(system, user)``. Records every call for assertion. Raises
  loudly on unexpected calls so a test that hits the LLM unintentionally
  fails fast.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypeVar, runtime_checkable

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import Settings

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """Single-purpose LLM facade: hand it a system prompt, a user prompt, and
    a Pydantic schema; get back a parsed instance of that schema. The model is
    a per-call override; the default lives on the concrete implementation."""

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        model: str | None = None,
    ) -> T: ...


# ── Production ─────────────────────────────────────────────────────────────


_TRANSIENT_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


class OpenAIClient:
    """OpenAI-backed implementation. Uses the SDK's ``parse`` helper so the
    model is forced to JSON matching the schema; we never hand-roll the
    ``response_format`` payload here.

    Tenacity wraps the API call with exponential backoff on transient errors.
    Non-transient errors (e.g. 400 bad request) propagate immediately — they
    indicate a prompt or schema bug, not flakiness.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._model = settings.openai_model
        self._temperature = settings.openai_temperature

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        reraise=True,
    )
    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        model: str | None = None,
    ) -> T:
        completion = await self._client.beta.chat.completions.parse(
            model=model or self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
            temperature=self._temperature,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError(
                f"OpenAIClient: parse returned None for schema={schema.__name__}; "
                "model refusal or schema-validation failure"
            )
        return parsed


# ── Test double ────────────────────────────────────────────────────────────


class FakeLLMClient:
    """In-memory client used by every unit test. Two contracts:

    1. **Loud about unexpected calls.** A test that triggers an agent path
       not exercised by the canned-response map fails with ``KeyError``,
       not a silent ``None`` or a real network attempt.
    2. **Type-checks the canned response.** If a test seeds the wrong type
       (e.g. a ``Citation`` where the agent expects ``QuoteCheck``), the
       client raises ``TypeError`` — surfaces gold-set / fixture drift
       early.
    """

    def __init__(
        self,
        responses: Mapping[tuple[str, str], BaseModel] | None = None,
    ) -> None:
        self._responses: dict[tuple[str, str], BaseModel] = dict(responses or {})
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    def queue(self, *, system: str, user: str, response: BaseModel) -> None:
        """Seed a canned response for the next call matching ``(system, user)``."""
        self._responses[(system, user)] = response

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        model: str | None = None,
    ) -> T:
        self.calls.append((system, user, schema))
        key = (system, user)
        if key not in self._responses:
            raise KeyError(
                "FakeLLMClient: no canned response for the given (system, user). "
                "Did the test forget to call .queue(...) for this prompt?"
            )
        response = self._responses[key]
        if not isinstance(response, schema):
            raise TypeError(
                f"FakeLLMClient: canned response is {type(response).__name__}, "
                f"agent asked for {schema.__name__}"
            )
        return response
