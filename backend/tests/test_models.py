"""Tests for the Pydantic IR. Every invariant the orchestrator + eval harness
depend on gets exercised here. See specs/001-foundation-evals-crossdoc/spec.md
§ 4 for the load-bearing decisions these tests defend."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from backend.models import (
    AgentResult,
    AuthorityCheck,
    AuthorityCheckResult,
    Citation,
    CitationsResult,
    Claim,
    ConfidenceResult,
    DiscrepanciesResult,
    Document,
    DocumentKind,
    EvidenceRef,
    FactDiscrepancy,
    Finding,
    FindingKind,
    MemoResult,
    Quote,
    QuoteCheck,
    QuoteCheckResult,
    Span,
    VerificationReport,
)


def _span(doc_id: str = "motion", quote: str = "alpha", grounded: bool = False) -> Span:
    if grounded:
        return Span(doc_id=doc_id, quote=quote, start=0, end=len(quote))
    return Span(doc_id=doc_id, quote=quote)


# ── Document ──────────────────────────────────────────────────────────────


class TestDocument:
    def test_constructs_with_id_kind_and_text(self) -> None:
        doc = Document(id="motion", kind=DocumentKind.MOTION, text="contents")
        assert doc.id == "motion"
        assert doc.kind is DocumentKind.MOTION
        assert doc.text == "contents"

    def test_is_frozen(self) -> None:
        doc = Document(id="motion", kind=DocumentKind.MOTION, text="contents")
        with pytest.raises(ValidationError):
            doc.id = "other"  # type: ignore[misc]

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Document(id="motion", kind=DocumentKind.MOTION, text="contents", extra="nope")  # type: ignore[call-arg]

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Document(id="", kind=DocumentKind.MOTION, text="contents")


# ── Span ───────────────────────────────────────────────────────────────────


class TestSpan:
    def test_minimal_emission_from_llm(self) -> None:
        span = Span(doc_id="motion", quote="The defendant was negligent.")
        assert span.is_grounded is False
        assert span.start is None and span.end is None

    def test_grounded_after_orchestrator_fills_offsets(self) -> None:
        span = Span(doc_id="motion", quote="alpha", start=10, end=15)
        assert span.is_grounded is True

    def test_start_without_end_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both be set or both be None"):
            Span(doc_id="motion", quote="alpha", start=10)

    def test_end_without_start_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both be set or both be None"):
            Span(doc_id="motion", quote="alpha", end=10)

    def test_end_must_be_greater_than_start(self) -> None:
        with pytest.raises(ValidationError, match="strictly greater"):
            Span(doc_id="motion", quote="alpha", start=10, end=10)
        with pytest.raises(ValidationError, match="strictly greater"):
            Span(doc_id="motion", quote="alpha", start=10, end=5)

    def test_negative_offsets_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Span(doc_id="motion", quote="alpha", start=-1, end=5)

    def test_empty_quote_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Span(doc_id="motion", quote="")


# ── Citation / Claim / Quote ──────────────────────────────────────────────


class TestCitationAndFriends:
    def test_citation_id_pattern_enforced(self) -> None:
        with pytest.raises(ValidationError):
            Citation(id="cite_1", proposition=_span(), cited_authority="Smith v. Jones")
        Citation(id="cite-1", proposition=_span(), cited_authority="Smith v. Jones")

    def test_citation_quoted_text_optional(self) -> None:
        c = Citation(id="cite-1", proposition=_span(), cited_authority="Smith v. Jones")
        assert c.quoted_text is None

    def test_claim_id_pattern_enforced(self) -> None:
        with pytest.raises(ValidationError):
            Claim(id="claim_1", proposition=_span())
        Claim(id="claim-2", proposition=_span())

    def test_quote_citation_id_pattern(self) -> None:
        Quote(citation_id="cite-1", text="quoted")
        with pytest.raises(ValidationError):
            Quote(citation_id="bogus", text="quoted")


# ── QuoteCheck verdict / matched_span pairing ──────────────────────────────


class TestQuoteCheckValidators:
    @pytest.mark.parametrize("verdict", ["exact", "paraphrase", "altered"])
    def test_grounded_verdicts_require_matched_span(self, verdict: str) -> None:
        with pytest.raises(ValidationError, match="requires matched_span"):
            QuoteCheck(citation_id="cite-1", verdict=verdict, reasoning="x")  # type: ignore[arg-type]

    @pytest.mark.parametrize("verdict", ["exact", "paraphrase", "altered"])
    def test_grounded_verdicts_accept_matched_span(self, verdict: str) -> None:
        QuoteCheck(
            citation_id="cite-1",
            verdict=verdict,  # type: ignore[arg-type]
            matched_span=_span(grounded=True),
            reasoning="aligned",
        )

    @pytest.mark.parametrize("verdict", ["fabricated", "unverifiable"])
    def test_ungrounded_verdicts_forbid_matched_span(self, verdict: str) -> None:
        with pytest.raises(ValidationError, match="forbids matched_span"):
            QuoteCheck(
                citation_id="cite-1",
                verdict=verdict,  # type: ignore[arg-type]
                matched_span=_span(grounded=True),
                reasoning="x",
            )

    def test_reasoning_required(self) -> None:
        with pytest.raises(ValidationError):
            QuoteCheck(citation_id="cite-1", verdict="fabricated", reasoning="")


# ── AuthorityCheck verdict / source_basis pairing ──────────────────────────


class TestAuthorityCheckValidators:
    @pytest.mark.parametrize("verdict", ["supports", "contradicts"])
    def test_grounded_verdicts_require_source_basis(self, verdict: str) -> None:
        with pytest.raises(ValidationError, match="requires source_basis"):
            AuthorityCheck(citation_id="cite-1", verdict=verdict, reasoning="x")  # type: ignore[arg-type]

    def test_unverifiable_forbids_source_basis(self) -> None:
        with pytest.raises(ValidationError, match="forbids source_basis"):
            AuthorityCheck(
                citation_id="cite-1",
                verdict="unverifiable",
                source_basis=_span(grounded=True),
                reasoning="x",
            )

    def test_supports_with_basis_ok(self) -> None:
        AuthorityCheck(
            citation_id="cite-1",
            verdict="supports",
            source_basis=_span(grounded=True),
            reasoning="aligned",
        )


# ── FactDiscrepancy ────────────────────────────────────────────────────────


class TestFactDiscrepancy:
    def test_requires_at_least_one_contradicting_span(self) -> None:
        with pytest.raises(ValidationError):
            FactDiscrepancy(
                id="disc-1",
                motion_claim=_span(),
                contradicting_evidence=[],
                description="x",
                severity="material",
            )

    def test_accepts_multiple_contradicting_spans(self) -> None:
        d = FactDiscrepancy(
            id="disc-1",
            motion_claim=_span(),
            contradicting_evidence=[_span(doc_id="police_report"), _span(doc_id="medical_records")],
            description="multi-doc contradiction",
            severity="dispositive",
        )
        assert len(d.contradicting_evidence) == 2

    def test_id_pattern(self) -> None:
        with pytest.raises(ValidationError):
            FactDiscrepancy(
                id="disc_1",
                motion_claim=_span(),
                contradicting_evidence=[_span()],
                description="x",
                severity="minor",
            )


# ── Finding ────────────────────────────────────────────────────────────────


class TestFinding:
    def test_minimal_finding(self) -> None:
        Finding(
            id="find-1",
            kind=FindingKind.FACT_DISCREPANCY,
            summary="A claim contradicted by the police report.",
            evidence=[EvidenceRef(span=_span(), role="primary")],
            agent="CrossDocConsistencyChecker",
            prompt_version="1.0.0",
        )

    def test_confidence_requires_reasoning(self) -> None:
        with pytest.raises(ValidationError, match="confidence_reasoning is required"):
            Finding(
                id="find-1",
                kind=FindingKind.FACT_DISCREPANCY,
                summary="x",
                evidence=[EvidenceRef(span=_span(), role="primary")],
                agent="X",
                prompt_version="1.0.0",
                confidence=0.8,
            )

    def test_confidence_with_reasoning_ok(self) -> None:
        f = Finding(
            id="find-1",
            kind=FindingKind.FACT_DISCREPANCY,
            summary="x",
            evidence=[EvidenceRef(span=_span(), role="primary")],
            agent="X",
            prompt_version="1.0.0",
            confidence=0.8,
            confidence_reasoning="Two independent sources contradict the motion.",
        )
        assert f.confidence == 0.8

    def test_prompt_version_pattern(self) -> None:
        with pytest.raises(ValidationError):
            Finding(
                id="find-1",
                kind=FindingKind.FACT_DISCREPANCY,
                summary="x",
                evidence=[EvidenceRef(span=_span(), role="primary")],
                agent="X",
                prompt_version="1.0",
            )


# ── AgentResult: discriminated union round-trip ───────────────────────────


_AGENT_RESULT_ADAPTER: TypeAdapter[AgentResult] = TypeAdapter(AgentResult)


class TestAgentResultUnion:
    def test_citations_result_roundtrip(self) -> None:
        payload: dict[str, Any] = {
            "agent": "CitationExtractor",
            "prompt_version": "1.0.0",
            "outcome": "success",
            "error": None,
            "latency_ms": 412,
            "kind": "citations",
            "data": [
                {
                    "id": "cite-1",
                    "proposition": {
                        "doc_id": "motion",
                        "quote": "alpha",
                        "start": None,
                        "end": None,
                    },
                    "cited_authority": "Privette v. Superior Court",
                    "quoted_text": None,
                }
            ],
        }
        result = _AGENT_RESULT_ADAPTER.validate_python(payload)
        assert isinstance(result, CitationsResult)
        assert _AGENT_RESULT_ADAPTER.dump_python(result, mode="json")["kind"] == "citations"

    def test_discriminator_picks_right_subtype(self) -> None:
        for kind, klass in [
            ("citations", CitationsResult),
            ("discrepancies", DiscrepanciesResult),
            ("quote_check", QuoteCheckResult),
            ("authority_check", AuthorityCheckResult),
            ("confidence", ConfidenceResult),
            ("memo", MemoResult),
        ]:
            payload = {
                "agent": "X",
                "prompt_version": "1.0.0",
                "outcome": "success",
                "latency_ms": 1,
                "kind": kind,
                "data": None if kind == "memo" else [],
            }
            result = _AGENT_RESULT_ADAPTER.validate_python(payload)
            assert isinstance(result, klass)

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _AGENT_RESULT_ADAPTER.validate_python(
                {
                    "agent": "X",
                    "prompt_version": "1.0.0",
                    "outcome": "success",
                    "latency_ms": 1,
                    "kind": "mystery",
                    "data": [],
                }
            )

    def test_extra_fields_rejected_on_concrete_subtype(self) -> None:
        with pytest.raises(ValidationError):
            CitationsResult(
                agent="X",
                prompt_version="1.0.0",
                outcome="success",
                latency_ms=1,
                kind="citations",
                data=[],
                surprise="bad",  # type: ignore[call-arg]
            )


# ── VerificationReport: end-to-end JSON round-trip ─────────────────────────


class TestVerificationReportRoundTrip:
    def test_mixed_agent_results_serialize_and_parse(self) -> None:
        report = VerificationReport(
            case_name="Rivera v. Harmon Construction Group",
            generated_at=datetime(2026, 6, 18, 12, 0, tzinfo=UTC),
            citations=[
                Citation(
                    id="cite-1",
                    proposition=_span(grounded=True),
                    cited_authority="Privette v. Superior Court",
                ),
            ],
            findings=[
                Finding(
                    id="find-1",
                    kind=FindingKind.FACT_DISCREPANCY,
                    summary="The MSJ says the harness was not worn; the police report says it was.",
                    evidence=[
                        EvidenceRef(span=_span(doc_id="motion", grounded=True), role="primary"),
                        EvidenceRef(
                            span=_span(doc_id="police_report", grounded=True),
                            role="contradicting",
                        ),
                    ],
                    agent="CrossDocConsistencyChecker",
                    prompt_version="1.0.0",
                )
            ],
            agent_results=[
                CitationsResult(
                    agent="CitationExtractor",
                    prompt_version="1.0.0",
                    outcome="success",
                    latency_ms=312,
                    data=[],
                ),
                DiscrepanciesResult(
                    agent="CrossDocConsistencyChecker",
                    prompt_version="1.0.0",
                    outcome="success",
                    latency_ms=812,
                    data=[],
                ),
                MemoResult(
                    agent="JudicialMemoWriter",
                    prompt_version="1.0.0",
                    outcome="success",
                    latency_ms=412,
                    data="No material problems identified.",
                ),
            ],
        )
        payload = report.model_dump(mode="json")
        round_tripped = VerificationReport.model_validate(payload)
        assert round_tripped == report

    def test_empty_report_valid(self) -> None:
        report = VerificationReport(
            case_name="X v. Y",
            generated_at=datetime(2026, 6, 18, tzinfo=UTC),
        )
        assert report.citations == []
        assert report.findings == []
        assert report.agent_results == []
        assert report.judicial_memo is None
