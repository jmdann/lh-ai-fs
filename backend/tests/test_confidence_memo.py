"""Tests for ``ConfidenceScorer`` + ``JudicialMemoWriter``.

The load-bearing properties: ConfidenceScorer never mutates findings other
than confidence + reasoning, returns confidence ∈ [0,1] with reasoning
mandatory, and short-circuits to 0.0 when evidence is missing.
JudicialMemoWriter is rejected (outcome=partial) when the memo cites a
finding id that doesn't exist or busts the word cap.
"""

from __future__ import annotations

from pydantic import BaseModel

from backend.agents.confidence_scorer import ConfidenceScorer, _LLMVerdict
from backend.agents.judicial_memo_writer import JudicialMemoWriter, validate_memo
from backend.llm.client import FakeLLMClient
from backend.models import EvidenceRef, Finding, FindingKind, Span


def _finding(
    fid: str = "find-1",
    kind: FindingKind = FindingKind.FACT_DISCREPANCY,
    summary: str = "Motion contradicted by police report.",
    evidence: list[EvidenceRef] | None = None,
) -> Finding:
    return Finding(
        id=fid,
        kind=kind,
        summary=summary,
        evidence=evidence
        or [EvidenceRef(span=Span(doc_id="motion", quote="claim"), role="primary")],
        agent="X",
        prompt_version="1.0.0",
    )


# ── ConfidenceScorer ──────────────────────────────────────────────────────


class TestConfidenceScorer:
    async def test_rescores_in_bounds_with_reasoning(self) -> None:
        fake = FakeLLMClient()

        async def _stub(*, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
            return _LLMVerdict(confidence=0.85, reasoning="Three sources agree.")

        fake.complete = _stub  # type: ignore[method-assign, assignment]
        result = await ConfidenceScorer(fake).run([_finding()])
        assert result.outcome == "success"
        assert len(result.data) == 1
        assert result.data[0].confidence == 0.85
        assert result.data[0].confidence_reasoning == "Three sources agree."
        # Unchanged fields
        assert result.data[0].id == "find-1"
        assert result.data[0].kind is FindingKind.FACT_DISCREPANCY

    async def test_empty_findings_succeeds_empty(self) -> None:
        fake = FakeLLMClient()
        result = await ConfidenceScorer(fake).run([])
        assert result.outcome == "success"
        assert result.data == []
        assert len(fake.calls) == 0

    async def test_no_evidence_short_circuits_without_llm(self) -> None:
        """A Finding cannot validate with empty evidence, so we wedge in via
        model_construct to simulate the orchestrator passing a malformed
        finding. The scorer's short-circuit emits 0.0 without LLM call."""
        bad = Finding.model_construct(
            id="find-99",
            kind=FindingKind.FACT_DISCREPANCY,
            summary="x",
            evidence=[],
            agent="X",
            prompt_version="1.0.0",
        )
        fake = FakeLLMClient()
        result = await ConfidenceScorer(fake).run([bad])
        assert result.outcome == "success"
        assert result.data[0].confidence == 0.0
        assert "no evidence" in (result.data[0].confidence_reasoning or "").lower()
        assert len(fake.calls) == 0


# ── JudicialMemoWriter ────────────────────────────────────────────────────


class TestJudicialMemoWriter:
    async def test_empty_findings_returns_no_problems(self) -> None:
        fake = FakeLLMClient()
        result = await JudicialMemoWriter(fake).run("Rivera v. Harmon", [])
        assert result.outcome == "success"
        assert result.data == "No material problems identified in the motion."
        assert len(fake.calls) == 0

    async def test_valid_memo_passes(self) -> None:
        from backend.agents.judicial_memo_writer import _LLMMemo

        async def _stub(*, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
            return _LLMMemo(memo="The motion misrepresents the harness facts [find-1].")

        fake = FakeLLMClient()
        fake.complete = _stub  # type: ignore[method-assign, assignment]
        result = await JudicialMemoWriter(fake).run("Rivera", [_finding()])
        assert result.outcome == "success"
        assert "find-1" in (result.data or "")

    async def test_memo_with_unknown_id_marked_partial(self) -> None:
        from backend.agents.judicial_memo_writer import _LLMMemo

        async def _stub(*, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
            return _LLMMemo(memo="Citing [find-1] and [find-99] which does not exist.")

        fake = FakeLLMClient()
        fake.complete = _stub  # type: ignore[method-assign, assignment]
        result = await JudicialMemoWriter(fake).run("Rivera", [_finding()])
        assert result.outcome == "partial"
        assert result.error is not None
        assert "find-99" in result.error

    async def test_memo_over_word_cap_marked_partial(self) -> None:
        from backend.agents.judicial_memo_writer import _LLMMemo

        long_memo = " ".join(["word"] * 200) + " [find-1]"

        async def _stub(*, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
            return _LLMMemo(memo=long_memo)

        fake = FakeLLMClient()
        fake.complete = _stub  # type: ignore[method-assign, assignment]
        result = await JudicialMemoWriter(fake).run("Rivera", [_finding()])
        assert result.outcome == "partial"
        assert "word cap" in (result.error or "")


# ── validate_memo unit ────────────────────────────────────────────────────


def test_validate_memo_clean() -> None:
    assert validate_memo("Bare [find-1] and [find-2].", {"find-1", "find-2"}) == []


def test_validate_memo_unknown_id() -> None:
    issues = validate_memo("Cites [find-9] which doesn't exist.", {"find-1"})
    assert any("find-9" in i for i in issues)


def test_validate_memo_no_refs_ok() -> None:
    assert validate_memo("Plain prose with no brackets.", {"find-1"}) == []
