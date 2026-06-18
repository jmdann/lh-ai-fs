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


def _span(doc_id: str = "motion", quote: str = "alpha") -> Span:
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


# ── Span: the type system rejects every offset attempt ───────────────────


class TestSpan:
    def test_minimal_construction(self) -> None:
        span = Span(doc_id="motion", quote="The defendant was negligent.")
        assert span.doc_id == "motion"
        assert span.quote == "The defendant was negligent."

    def test_rejects_offset_smuggling(self) -> None:
        """STANDARDS § 3.6: LLMs cannot emit offsets. The fix is structural —
        there is no integer field on Span. Any JSON the LLM emits with
        ``start``/``end`` fails ``extra="forbid"``."""
        with pytest.raises(ValidationError):
            Span.model_validate({"doc_id": "motion", "quote": "x", "start": 0})
        with pytest.raises(ValidationError):
            Span.model_validate({"doc_id": "motion", "quote": "x", "end": 5})
        with pytest.raises(ValidationError):
            Span.model_validate({"doc_id": "motion", "quote": "x", "start": 0, "end": 5})

    def test_rejects_offset_via_json(self) -> None:
        with pytest.raises(ValidationError):
            Span.model_validate_json('{"doc_id":"motion","quote":"x","start":0,"end":5}')

    def test_empty_doc_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Span(doc_id="", quote="x")

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

    def test_citation_rejects_empty_authority(self) -> None:
        with pytest.raises(ValidationError):
            Citation(id="cite-1", proposition=_span(), cited_authority="")

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
            matched_span=_span(),
            reasoning="aligned",
        )

    @pytest.mark.parametrize("verdict", ["fabricated", "unverifiable"])
    def test_ungrounded_verdicts_forbid_matched_span(self, verdict: str) -> None:
        with pytest.raises(ValidationError, match="forbids matched_span"):
            QuoteCheck(
                citation_id="cite-1",
                verdict=verdict,  # type: ignore[arg-type]
                matched_span=_span(),
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

    @pytest.mark.parametrize("verdict", ["supports", "contradicts"])
    def test_grounded_verdicts_accept_source_basis(self, verdict: str) -> None:
        AuthorityCheck(
            citation_id="cite-1",
            verdict=verdict,  # type: ignore[arg-type]
            source_basis=_span(),
            reasoning="aligned",
        )

    def test_unverifiable_forbids_source_basis(self) -> None:
        with pytest.raises(ValidationError, match="forbids source_basis"):
            AuthorityCheck(
                citation_id="cite-1",
                verdict="unverifiable",
                source_basis=_span(),
                reasoning="x",
            )

    def test_unverifiable_accepts_no_basis(self) -> None:
        AuthorityCheck(
            citation_id="cite-1",
            verdict="unverifiable",
            reasoning="External case-law authority; source text not in corpus.",
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
        """Spec 003: setting confidence without reasoning is rejected."""
        with pytest.raises(ValidationError, match="confidence_reasoning"):
            Finding(
                id="find-1",
                kind=FindingKind.FACT_DISCREPANCY,
                summary="x",
                evidence=[EvidenceRef(span=_span(), role="primary")],
                agent="ConfidenceScorer",
                prompt_version="1.0.0",
                confidence=0.8,
            )

    def test_confidence_with_reasoning_accepted(self) -> None:
        f = Finding(
            id="find-1",
            kind=FindingKind.FACT_DISCREPANCY,
            summary="x",
            evidence=[EvidenceRef(span=_span(), role="primary")],
            agent="ConfidenceScorer",
            prompt_version="1.0.0",
            confidence=0.85,
            confidence_reasoning="Three independent sources contradict the motion.",
        )
        assert f.confidence == 0.85

    def test_confidence_out_of_bounds_rejected(self) -> None:
        for bad in (-0.01, 1.01):
            with pytest.raises(ValidationError):
                Finding(
                    id="find-1",
                    kind=FindingKind.FACT_DISCREPANCY,
                    summary="x",
                    evidence=[EvidenceRef(span=_span(), role="primary")],
                    agent="X",
                    prompt_version="1.0.0",
                    confidence=bad,
                    confidence_reasoning="x",
                )

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

    def test_id_pattern(self) -> None:
        with pytest.raises(ValidationError):
            Finding(
                id="find_1",
                kind=FindingKind.FACT_DISCREPANCY,
                summary="x",
                evidence=[EvidenceRef(span=_span(), role="primary")],
                agent="X",
                prompt_version="1.0.0",
            )


# ── AgentResult: discriminated union round-trip via JSON and Python ───────


_AGENT_RESULT_ADAPTER: TypeAdapter[AgentResult] = TypeAdapter(AgentResult)


def _result_payload(kind: str, data: Any) -> dict[str, Any]:
    return {
        "agent": "X",
        "prompt_version": "1.0.0",
        "outcome": "success",
        "error": None,
        "latency_ms": 1,
        "kind": kind,
        "data": data,
    }


class TestAgentResultUnion:
    @pytest.mark.parametrize(
        "kind,klass",
        [
            ("citations", CitationsResult),
            ("discrepancies", DiscrepanciesResult),
            ("quote_check", QuoteCheckResult),
            ("authority_check", AuthorityCheckResult),
            ("confidence", ConfidenceResult),
        ],
    )
    def test_discriminator_picks_right_subtype_python(self, kind: str, klass: type) -> None:
        result = _AGENT_RESULT_ADAPTER.validate_python(_result_payload(kind, []))
        assert isinstance(result, klass)

    @pytest.mark.parametrize(
        "kind,klass",
        [
            ("citations", CitationsResult),
            ("discrepancies", DiscrepanciesResult),
            ("quote_check", QuoteCheckResult),
            ("authority_check", AuthorityCheckResult),
            ("confidence", ConfidenceResult),
        ],
    )
    def test_discriminator_picks_right_subtype_json(self, kind: str, klass: type) -> None:
        """JSON-string round-trip — distinct from validate_python because
        Pydantic v2 has separate code paths and a JSON-only failure could
        otherwise slip through unit tests (codex round B P2)."""
        json_blob = _AGENT_RESULT_ADAPTER.dump_json(
            _AGENT_RESULT_ADAPTER.validate_python(_result_payload(kind, []))
        )
        result = _AGENT_RESULT_ADAPTER.validate_json(json_blob)
        assert isinstance(result, klass)

    def test_citations_result_roundtrip_with_real_data(self) -> None:
        payload = _result_payload(
            "citations",
            [
                {
                    "id": "cite-1",
                    "proposition": {"doc_id": "motion", "quote": "alpha"},
                    "cited_authority": "Privette v. Superior Court",
                    "quoted_text": None,
                }
            ],
        )
        payload["agent"] = "CitationExtractor"
        payload["latency_ms"] = 412
        result = _AGENT_RESULT_ADAPTER.validate_python(payload)
        assert isinstance(result, CitationsResult)
        dumped = _AGENT_RESULT_ADAPTER.dump_python(result, mode="json")
        assert dumped["kind"] == "citations"

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _AGENT_RESULT_ADAPTER.validate_python(_result_payload("mystery", []))

    def test_outcome_timeout_rejected_in_spec_001(self) -> None:
        """Spec 003 will add ``outcome=timeout`` together with orchestrator
        timeout handling. Until then it must not validate."""
        payload = _result_payload("citations", [])
        payload["outcome"] = "timeout"
        with pytest.raises(ValidationError):
            _AGENT_RESULT_ADAPTER.validate_python(payload)

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


# ── VerificationReport: end-to-end JSON-string round-trip ──────────────────


class TestVerificationReportRoundTrip:
    def test_mixed_agent_results_json_roundtrip(self) -> None:
        report = VerificationReport(
            case_name="Rivera v. Harmon Construction Group",
            generated_at=datetime(2026, 6, 18, 12, 0, tzinfo=UTC),
            citations=[
                Citation(
                    id="cite-1",
                    proposition=_span(),
                    cited_authority="Privette v. Superior Court",
                ),
            ],
            findings=[
                Finding(
                    id="find-1",
                    kind=FindingKind.FACT_DISCREPANCY,
                    summary="The MSJ says the harness was not worn; the police report says it was.",
                    evidence=[
                        EvidenceRef(span=_span(doc_id="motion"), role="primary"),
                        EvidenceRef(
                            span=_span(doc_id="police_report"),
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
                QuoteCheckResult(
                    agent="QuoteChecker",
                    prompt_version="1.0.0",
                    outcome="success",
                    latency_ms=412,
                    data=[],
                ),
                AuthorityCheckResult(
                    agent="AuthoritySupportChecker",
                    prompt_version="1.0.0",
                    outcome="success",
                    latency_ms=512,
                    data=[],
                ),
            ],
        )
        # Round-trip through actual JSON string, not just a dict.
        json_blob = report.model_dump_json()
        round_tripped = VerificationReport.model_validate_json(json_blob)
        assert round_tripped == report

    def test_empty_report_valid(self) -> None:
        report = VerificationReport(
            case_name="X v. Y",
            generated_at=datetime(2026, 6, 18, tzinfo=UTC),
        )
        assert report.citations == []
        assert report.findings == []
        assert report.agent_results == []

    def test_judicial_memo_optional(self) -> None:
        """Spec 003 activated ``judicial_memo``. It stays optional so reports
        without the memo agent still validate."""
        with_memo = VerificationReport(
            case_name="X v. Y",
            generated_at=datetime(2026, 6, 18, tzinfo=UTC),
            judicial_memo="One-paragraph memo for the judge.",
        )
        assert with_memo.judicial_memo == "One-paragraph memo for the judge."

    def test_memo_result_subtype(self) -> None:
        """``MemoResult`` discriminates by kind=memo with data: str | None."""
        payload = {
            "agent": "JudicialMemoWriter",
            "prompt_version": "1.0.0",
            "outcome": "success",
            "latency_ms": 412,
            "kind": "memo",
            "data": "The motion misrepresents Privette.",
        }
        result = _AGENT_RESULT_ADAPTER.validate_python(payload)
        assert isinstance(result, MemoResult)
        assert result.data == "The motion misrepresents Privette."
