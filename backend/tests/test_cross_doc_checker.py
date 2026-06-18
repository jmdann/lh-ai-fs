"""Tests for ``CrossDocConsistencyChecker``.

Two layers:

1. Prompt module shape — version pattern, ≤ 40 lines, build_user_prompt
   produces expected sectioning per record document.
2. ``CrossDocConsistencyChecker.run`` — wired with ``FakeLLMClient``; the
   happy path, the failure path, and the "no contradictions found" path
   all return ``DiscrepanciesResult`` envelopes with deterministic
   ``disc-N`` ids.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.agents.cross_doc_checker import (
    CrossDocConsistencyChecker,
    _ExtractedDiscrepancies,
    _LLMFactDiscrepancy,
)
from backend.agents.prompts.cross_doc_checker import (
    PROMPT_VERSION,
    SYSTEM,
    build_user_prompt,
)
from backend.llm.client import FakeLLMClient
from backend.models import DiscrepanciesResult, Document, DocumentKind, FactDiscrepancy, Span

# ── Helpers ────────────────────────────────────────────────────────────────


def _doc(doc_id: str, kind: DocumentKind, text: str = "irrelevant text") -> Document:
    return Document(id=doc_id, kind=kind, text=text)


def _motion(text: str = "The plaintiff was not wearing a safety harness.") -> Document:
    return _doc("motion", DocumentKind.MOTION, text)


def _police_report(
    text: str = "Officer Reyes observed the harness was properly secured.",
) -> Document:
    return _doc("police_report", DocumentKind.POLICE_REPORT, text)


def _medical(text: str = "No fall-related abrasions on the patient's torso.") -> Document:
    return _doc("medical_records_excerpt", DocumentKind.MEDICAL_RECORDS, text)


def _witness(text: str = "I saw him on the scaffolding with his harness on.") -> Document:
    return _doc("witness_statement", DocumentKind.WITNESS_STATEMENT, text)


def _seeded_one_discrepancy() -> _ExtractedDiscrepancies:
    return _ExtractedDiscrepancies(
        discrepancies=[
            _LLMFactDiscrepancy(
                motion_claim=Span(
                    doc_id="motion",
                    quote="The plaintiff was not wearing a safety harness.",
                ),
                contradicting_evidence=[
                    Span(
                        doc_id="police_report",
                        quote="Officer Reyes observed the harness was properly secured.",
                    )
                ],
                description="Motion says no harness; police report records harness as secured.",
                severity="material",
            )
        ]
    )


# ── Prompt module ─────────────────────────────────────────────────────────


class TestPromptModule:
    def test_prompt_version_pattern(self) -> None:
        import re

        assert re.fullmatch(r"\d+\.\d+\.\d+", PROMPT_VERSION)

    def test_system_prompt_under_40_lines(self) -> None:
        assert len(SYSTEM.splitlines()) <= 40

    def test_build_user_prompt_emits_record_headers(self) -> None:
        prompt = build_user_prompt(
            motion_id="motion",
            motion_text="motion body text",
            records=[
                ("police_report", "POLICE REPORT", "police body text"),
                ("medical_records_excerpt", "MEDICAL RECORDS", "medical body text"),
                ("witness_statement", "WITNESS STATEMENT", "witness body text"),
            ],
        )
        assert "MOTION (id: motion)" in prompt
        assert "RECORD: POLICE REPORT (id: police_report)" in prompt
        assert "RECORD: MEDICAL RECORDS (id: medical_records_excerpt)" in prompt
        assert "RECORD: WITNESS STATEMENT (id: witness_statement)" in prompt
        assert "police body text" in prompt
        assert "medical body text" in prompt
        assert "witness body text" in prompt

    def test_build_user_prompt_no_records_block(self) -> None:
        # Edge case: a motion with no record docs. The prompt should still
        # be well-formed (the LLM will just emit []).
        prompt = build_user_prompt(motion_id="motion", motion_text="body", records=[])
        assert "MOTION (id: motion)" in prompt
        assert "RECORD" not in prompt


# ── LLM I/O schema ────────────────────────────────────────────────────────


class TestLLMSchema:
    def test_requires_contradicting_evidence_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            _LLMFactDiscrepancy(
                motion_claim=Span(doc_id="motion", quote="claim"),
                contradicting_evidence=[],
                description="x",
                severity="material",
            )

    def test_severity_literal_enforced(self) -> None:
        with pytest.raises(ValidationError):
            _LLMFactDiscrepancy(
                motion_claim=Span(doc_id="motion", quote="claim"),
                contradicting_evidence=[Span(doc_id="police_report", quote="x")],
                description="x",
                severity="critical",  # type: ignore[arg-type]
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _LLMFactDiscrepancy(
                motion_claim=Span(doc_id="motion", quote="claim"),
                contradicting_evidence=[Span(doc_id="police_report", quote="x")],
                description="x",
                severity="material",
                surprise="bad",  # type: ignore[call-arg]
            )


# ── Agent.run happy path ──────────────────────────────────────────────────


class TestRunHappyPath:
    async def test_returns_discrepancies_with_deterministic_ids(self) -> None:
        motion = _motion()
        records = [_police_report(), _medical(), _witness()]
        record_triples = [
            ("police_report", "POLICE REPORT", _police_report().text),
            ("medical_records_excerpt", "MEDICAL RECORDS", _medical().text),
            ("witness_statement", "WITNESS STATEMENT", _witness().text),
        ]
        fake = FakeLLMClient()
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt("motion", motion.text, record_triples),
            response=_seeded_one_discrepancy(),
        )
        result = await CrossDocConsistencyChecker(fake).run(motion, records)
        assert isinstance(result, DiscrepanciesResult)
        assert result.outcome == "success"
        assert result.agent == CrossDocConsistencyChecker.AGENT_NAME
        assert result.prompt_version == PROMPT_VERSION
        assert [d.id for d in result.data] == ["disc-1"]
        assert all(isinstance(d, FactDiscrepancy) for d in result.data)

    async def test_agent_makes_exactly_one_llm_call(self) -> None:
        motion = _motion()
        records = [_police_report()]
        record_triples = [("police_report", "POLICE REPORT", _police_report().text)]
        fake = FakeLLMClient()
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt("motion", motion.text, record_triples),
            response=_seeded_one_discrepancy(),
        )
        await CrossDocConsistencyChecker(fake).run(motion, records)
        assert len(fake.calls) == 1
        system, user, schema = fake.calls[0]
        assert system == SYSTEM
        assert schema is _ExtractedDiscrepancies
        assert "POLICE REPORT" in user

    async def test_empty_record_set_still_works(self) -> None:
        motion = _motion()
        fake = FakeLLMClient()
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt("motion", motion.text, []),
            response=_ExtractedDiscrepancies(discrepancies=[]),
        )
        result = await CrossDocConsistencyChecker(fake).run(motion, [])
        assert result.outcome == "success"
        assert result.data == []

    async def test_no_contradictions_returns_empty_data(self) -> None:
        motion = _motion()
        records = [_police_report()]
        record_triples = [("police_report", "POLICE REPORT", _police_report().text)]
        fake = FakeLLMClient()
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt("motion", motion.text, record_triples),
            response=_ExtractedDiscrepancies(discrepancies=[]),
        )
        result = await CrossDocConsistencyChecker(fake).run(motion, records)
        assert result.outcome == "success"
        assert result.data == []
        # latency_ms should still be populated even on the empty path
        assert result.latency_ms >= 0

    async def test_multiple_discrepancies_get_sequential_ids(self) -> None:
        motion = _motion()
        records = [_police_report(), _witness()]
        record_triples = [
            ("police_report", "POLICE REPORT", _police_report().text),
            ("witness_statement", "WITNESS STATEMENT", _witness().text),
        ]
        seeded = _ExtractedDiscrepancies(
            discrepancies=[
                _LLMFactDiscrepancy(
                    motion_claim=Span(doc_id="motion", quote="claim 1"),
                    contradicting_evidence=[Span(doc_id="police_report", quote="contra 1")],
                    description="d1",
                    severity="minor",
                ),
                _LLMFactDiscrepancy(
                    motion_claim=Span(doc_id="motion", quote="claim 2"),
                    contradicting_evidence=[Span(doc_id="witness_statement", quote="contra 2")],
                    description="d2",
                    severity="dispositive",
                ),
            ]
        )
        fake = FakeLLMClient()
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt("motion", motion.text, record_triples),
            response=seeded,
        )
        result = await CrossDocConsistencyChecker(fake).run(motion, records)
        assert [d.id for d in result.data] == ["disc-1", "disc-2"]
        assert [d.severity for d in result.data] == ["minor", "dispositive"]


# ── Agent.run failure path ────────────────────────────────────────────────


class TestRunFailurePath:
    async def test_unexpected_llm_error_maps_to_failure(self) -> None:
        fake = FakeLLMClient()
        # Do NOT seed any response → FakeLLMClient raises KeyError.
        result = await CrossDocConsistencyChecker(fake).run(_motion(), [_police_report()])
        assert result.outcome == "failure"
        assert result.data == []
        assert result.error is not None
        assert "KeyError" in result.error or "no canned response" in result.error

    async def test_schema_mismatch_from_llm_maps_to_failure(self) -> None:
        class _Wrong(BaseModel):
            model_config = ConfigDict(extra="forbid")
            stuff: str = Field(default="")

        motion = _motion()
        record_triples = [("police_report", "POLICE REPORT", _police_report().text)]
        fake = FakeLLMClient()
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt("motion", motion.text, record_triples),
            response=_Wrong(),
        )
        result = await CrossDocConsistencyChecker(fake).run(motion, [_police_report()])
        assert result.outcome == "failure"
        assert result.error is not None
        assert "TypeError" in result.error
