"""Tests for ``CitationExtractor``.

Two layers:

1. ``extract_authority_strings`` — pure regex; tested against synthetic
   inputs and against the real Rivera MSJ to defend against format drift.
2. ``CitationExtractor.run`` — wired with ``FakeLLMClient``; asserts
   AgentResult envelope, id assignment, error path, and prompt sequencing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from backend.agents.citation_extractor import (
    CitationExtractor,
    _ExtractedCitations,
    _LLMCitation,
    extract_authority_strings,
)
from backend.agents.prompts.citation_extractor import (
    PROMPT_VERSION,
    SYSTEM,
    build_user_prompt,
)
from backend.llm.client import FakeLLMClient
from backend.models import Citation, CitationsResult, Document, DocumentKind, Span

MSJ_PATH = Path(__file__).resolve().parents[1] / "documents" / "motion_for_summary_judgment.txt"


@pytest.fixture(scope="module")
def msj_text() -> str:
    return MSJ_PATH.read_text()


# ── Regex extraction ──────────────────────────────────────────────────────


class TestRegexExtraction:
    def test_simple_cal_supreme_citation(self) -> None:
        text = (
            "The court relied on Privette v. Superior Court, 5 Cal.4th 689, 695 "
            "(1993). End of paragraph."
        )
        assert extract_authority_strings(text) == [
            "Privette v. Superior Court, 5 Cal.4th 689, 695 (1993)"
        ]

    def test_federal_district_citation(self) -> None:
        text = (
            "See also Whitmore v. Delgado Scaffolding Co., 334 F. Supp. 2d 1189, "
            "1195 (C.D. Cal. 2004) (granting summary judgment)."
        )
        result = extract_authority_strings(text)
        assert any("Whitmore v. Delgado" in c for c in result)

    def test_federal_circuit_citation(self) -> None:
        text = "See Kellerman v. Pacific Coast Construction, Inc., 887 F.2d 1204, 1209 (9th Cir. 1991) (parens)."
        result = extract_authority_strings(text)
        assert any("Kellerman" in c and "9th Cir." in c and "1991" in c for c in result)

    def test_dedup_preserves_first_occurrence(self) -> None:
        text = (
            "Privette v. Superior Court, 5 Cal.4th 689, 695 (1993). "
            "Later, Privette v. Superior Court, 5 Cal.4th 689, 695 (1993)."
        )
        result = extract_authority_strings(text)
        assert len(result) == 1

    def test_empty_text_returns_empty(self) -> None:
        assert extract_authority_strings("") == []

    def test_no_citations_returns_empty(self) -> None:
        assert extract_authority_strings("Nothing legal here, just a sentence.") == []

    def test_returns_in_document_order(self) -> None:
        text = (
            "First, Alpha v. Beta, 1 Cal.4th 1 (2000); "
            "next Gamma v. Delta, 2 Cal.4th 2 (2001); "
            "finally Theta v. Iota, 3 Cal.4th 3 (2002)."
        )
        result = extract_authority_strings(text)
        # Lowercase narrative connectors ("First, ", "next ", "finally ")
        # leave the case-name boundary clean. Capitalized leading words
        # like "See"/"Then" would absorb into the plaintiff — the LLM
        # proposition step still works because the prompt instructs it to
        # use only the listed authorities, but this test pins the clean
        # narrative case to keep the regex contract obvious.
        assert [r.split(" v. ")[0].split()[-1] for r in result] == [
            "Alpha",
            "Gamma",
            "Theta",
        ]


class TestAgainstRealMSJ:
    """Defends against format drift in the actual case-file MSJ."""

    def test_extracts_known_citations(self, msj_text: str) -> None:
        result = extract_authority_strings(msj_text)
        joined = " | ".join(result)
        # Authorities the regex MUST catch — if any of these regress, the
        # eval baseline depends on them and would silently misreport recall.
        for expected in [
            "Privette v. Superior Court",
            "Whitmore v. Delgado Scaffolding",
            "Kellerman v. Pacific Coast Construction",
            "Seabright Insurance",
            "Torres v. Granite Falls",
        ]:
            assert expected in joined, f"missing: {expected}"

    def test_returns_at_least_threshold_citations(self, msj_text: str) -> None:
        result = extract_authority_strings(msj_text)
        # Threshold guards the LLM-fallback path; if regex drops below it,
        # the agent silently switches modes, which is worth knowing.
        assert len(result) >= CitationExtractor.regex_threshold()


# ── Agent.run with FakeLLMClient ──────────────────────────────────────────


def _seeded_response() -> _ExtractedCitations:
    return _ExtractedCitations(
        citations=[
            _LLMCitation(
                cited_authority="Privette v. Superior Court, 5 Cal.4th 689, 695 (1993)",
                proposition=Span(doc_id="motion", quote="The hirer is not liable."),
                quoted_text=None,
            ),
            _LLMCitation(
                cited_authority="Whitmore v. Delgado Scaffolding Co., 334 F. Supp. 2d 1189 (C.D. Cal. 2004)",
                proposition=Span(doc_id="motion", quote="Summary judgment granted to hirer."),
                quoted_text="granting summary judgment to hirer",
            ),
        ]
    )


def _motion(text: str = "Privette v. Superior Court, 5 Cal.4th 689, 695 (1993).") -> Document:
    return Document(id="motion", kind=DocumentKind.MOTION, text=text)


class TestRunHappyPath:
    async def test_returns_citations_with_deterministic_ids(self, msj_text: str) -> None:
        fake = FakeLLMClient()
        motion = Document(id="motion", kind=DocumentKind.MOTION, text=msj_text)
        candidates = extract_authority_strings(msj_text)
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt(msj_text, candidates),
            response=_seeded_response(),
        )
        result = await CitationExtractor(fake).run(motion)
        assert isinstance(result, CitationsResult)
        assert result.outcome == "success"
        assert result.prompt_version == PROMPT_VERSION
        assert result.agent == CitationExtractor.AGENT_NAME
        assert [c.id for c in result.data] == ["cite-1", "cite-2"]
        assert all(isinstance(c, Citation) for c in result.data)
        assert result.latency_ms >= 0

    async def test_agent_makes_exactly_one_llm_call(self, msj_text: str) -> None:
        fake = FakeLLMClient()
        motion = Document(id="motion", kind=DocumentKind.MOTION, text=msj_text)
        candidates = extract_authority_strings(msj_text)
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt(msj_text, candidates),
            response=_seeded_response(),
        )
        await CitationExtractor(fake).run(motion)
        assert len(fake.calls) == 1
        system, user, schema = fake.calls[0]
        assert system == SYSTEM
        assert schema is _ExtractedCitations
        assert "MOTION TEXT" in user
        assert "CANDIDATES" in user

    async def test_empty_llm_response_yields_empty_data(self) -> None:
        fake = FakeLLMClient()
        motion = _motion()
        candidates = extract_authority_strings(motion.text)
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt(motion.text, candidates),
            response=_ExtractedCitations(citations=[]),
        )
        result = await CitationExtractor(fake).run(motion)
        assert result.outcome == "success"
        assert result.data == []


class TestRunFailurePath:
    async def test_unexpected_llm_error_maps_to_failure_outcome(self) -> None:
        fake = FakeLLMClient()
        motion = _motion()
        # Do NOT seed a response: FakeLLMClient.complete raises KeyError.
        result = await CitationExtractor(fake).run(motion)
        assert result.outcome == "failure"
        assert result.data == []
        assert result.error is not None
        assert "KeyError" in result.error or "no canned response" in result.error

    async def test_schema_mismatch_from_llm_maps_to_failure(self) -> None:
        class _Wrong(BaseModel):
            model_config = ConfigDict(extra="forbid")
            stuff: str = Field(default="")

        fake = FakeLLMClient()
        motion = _motion()
        candidates = extract_authority_strings(motion.text)
        fake.queue(
            system=SYSTEM,
            user=build_user_prompt(motion.text, candidates),
            response=_Wrong(),
        )
        result = await CitationExtractor(fake).run(motion)
        assert result.outcome == "failure"
        assert result.error is not None
        assert "TypeError" in result.error


# ── Prompt module surface ─────────────────────────────────────────────────


class TestPromptModule:
    def test_prompt_version_pattern(self) -> None:
        import re

        assert re.fullmatch(r"\d+\.\d+\.\d+", PROMPT_VERSION)

    def test_build_user_prompt_includes_motion_and_candidates(self) -> None:
        prompt = build_user_prompt("MOTION TEXT BODY", ["Smith v. Jones", "Alpha v. Beta"])
        assert "MOTION TEXT BODY" in prompt
        assert "1. Smith v. Jones" in prompt
        assert "2. Alpha v. Beta" in prompt

    def test_system_prompt_under_40_lines(self) -> None:
        # STANDARDS § 3.4 hard cap; prompt longer than 40 lines means the
        # agent is wrong-sized and needs to be split.
        assert len(SYSTEM.splitlines()) <= 40
