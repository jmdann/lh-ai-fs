"""Tests for ``Orchestrator``.

The grounding boundary is the load-bearing behavior here: spans the LLM
emits get verified against ``SourceRegistry`` and dropped if they don't
ground. The corresponding ``AgentResult`` flips to ``outcome="partial"``
so the eval harness can report ``grounding_integrity_failures``.
"""

from __future__ import annotations

import pytest

from backend.agents.citation_extractor import _ExtractedCitations, _LLMCitation
from backend.agents.cross_doc_checker import (
    _ExtractedDiscrepancies,
    _LLMFactDiscrepancy,
)
from backend.agents.prompts.citation_extractor import (
    SYSTEM as CITATION_SYSTEM,
)
from backend.agents.prompts.citation_extractor import (
    build_user_prompt as build_citation_prompt,
)
from backend.agents.prompts.cross_doc_checker import (
    SYSTEM as CROSS_DOC_SYSTEM,
)
from backend.agents.prompts.cross_doc_checker import (
    build_user_prompt as build_cross_doc_prompt,
)
from backend.llm.client import FakeLLMClient
from backend.models import (
    CitationsResult,
    DiscrepanciesResult,
    Document,
    DocumentKind,
    FindingKind,
    Span,
)
from backend.orchestrator import Orchestrator
from backend.sources import SourceRegistry

# ── Fixtures (plain helpers, not @fixture — they're cheap) ────────────────


def _motion() -> Document:
    return Document(
        id="motion",
        kind=DocumentKind.MOTION,
        text=(
            "The plaintiff was not wearing a safety harness at the time of "
            "the fall. Defendant relies on Privette v. Superior Court, "
            "5 Cal.4th 689, 695 (1993)."
        ),
    )


def _police_report() -> Document:
    return Document(
        id="police_report",
        kind=DocumentKind.POLICE_REPORT,
        text=(
            "Officer Reyes observed the harness was properly secured when first responders arrived."
        ),
    )


def _all_docs() -> dict[str, Document]:
    return {"motion": _motion(), "police_report": _police_report()}


def _seed_citations(fake: FakeLLMClient, motion: Document, candidates: list[str]) -> None:
    fake.queue(
        system=CITATION_SYSTEM,
        user=build_citation_prompt(motion.text, candidates),
        response=_ExtractedCitations(
            citations=[
                _LLMCitation(
                    cited_authority="Privette v. Superior Court, 5 Cal.4th 689, 695 (1993)",
                    proposition=Span(
                        doc_id="motion",
                        quote=(
                            "Defendant relies on Privette v. Superior Court, "
                            "5 Cal.4th 689, 695 (1993)."
                        ),
                    ),
                    quoted_text=None,
                )
            ]
        ),
    )


def _seed_cross_doc(
    fake: FakeLLMClient,
    motion: Document,
    record_triples: list[tuple[str, str, str]],
    *,
    motion_quote: str = ("The plaintiff was not wearing a safety harness at the time of the fall."),
    contradiction_quote: str = (
        "Officer Reyes observed the harness was properly secured when first responders arrived."
    ),
) -> None:
    fake.queue(
        system=CROSS_DOC_SYSTEM,
        user=build_cross_doc_prompt(motion.id, motion.text, record_triples),
        response=_ExtractedDiscrepancies(
            discrepancies=[
                _LLMFactDiscrepancy(
                    motion_claim=Span(doc_id="motion", quote=motion_quote),
                    contradicting_evidence=[
                        Span(doc_id="police_report", quote=contradiction_quote)
                    ],
                    description="Motion says no harness; police report disagrees.",
                    severity="material",
                )
            ]
        ),
    )


# ── Happy path ────────────────────────────────────────────────────────────


class TestHappyPath:
    async def test_report_has_citations_findings_and_agent_results(self) -> None:
        docs = _all_docs()
        motion = docs["motion"]
        from backend.agents.citation_extractor import extract_authority_strings

        candidates = extract_authority_strings(motion.text)
        record_triples = [
            ("police_report", "POLICE REPORT", docs["police_report"].text),
        ]
        fake = FakeLLMClient()
        _seed_citations(fake, motion, candidates)
        _seed_cross_doc(fake, motion, record_triples)
        orch = Orchestrator(fake, SourceRegistry())

        report = await orch.run(docs, case_name="Rivera v. Harmon")

        assert report.case_name == "Rivera v. Harmon"
        assert len(report.citations) == 1
        assert report.citations[0].id == "cite-1"
        assert len(report.findings) == 1
        assert report.findings[0].id == "find-1"
        assert report.findings[0].kind is FindingKind.FACT_DISCREPANCY
        assert len(report.findings[0].evidence) == 2
        assert report.findings[0].evidence[0].role == "primary"
        assert report.findings[0].evidence[1].role == "contradicting"
        assert len(report.agent_results) == 2
        kinds = {r.kind for r in report.agent_results}
        assert kinds == {"citations", "discrepancies"}
        for ar in report.agent_results:
            assert ar.outcome == "success"

    async def test_documents_auto_registered_in_source_registry(self) -> None:
        """The orchestrator does not assume the caller registered the input
        documents; it ensures grounding works regardless."""
        docs = _all_docs()
        motion = docs["motion"]
        from backend.agents.citation_extractor import extract_authority_strings

        candidates = extract_authority_strings(motion.text)
        record_triples = [
            ("police_report", "POLICE REPORT", docs["police_report"].text),
        ]
        fake = FakeLLMClient()
        _seed_citations(fake, motion, candidates)
        _seed_cross_doc(fake, motion, record_triples)
        registry = SourceRegistry()  # empty
        orch = Orchestrator(fake, registry)
        await orch.run(docs)
        assert registry.has("motion")
        assert registry.has("police_report")


# ── Grounding boundary ────────────────────────────────────────────────────


class TestGroundingBoundary:
    async def test_ungrounded_citation_dropped_marks_result_partial(self) -> None:
        docs = _all_docs()
        motion = docs["motion"]
        from backend.agents.citation_extractor import extract_authority_strings

        candidates = extract_authority_strings(motion.text)
        record_triples = [
            ("police_report", "POLICE REPORT", docs["police_report"].text),
        ]
        fake = FakeLLMClient()
        # Citation whose proposition is NOT in the motion text — should drop.
        fake.queue(
            system=CITATION_SYSTEM,
            user=build_citation_prompt(motion.text, candidates),
            response=_ExtractedCitations(
                citations=[
                    _LLMCitation(
                        cited_authority="Privette v. Superior Court, 5 Cal.4th 689, 695 (1993)",
                        proposition=Span(
                            doc_id="motion",
                            quote="This sentence does not appear anywhere in the motion text.",
                        ),
                        quoted_text=None,
                    )
                ]
            ),
        )
        _seed_cross_doc(fake, motion, record_triples)
        orch = Orchestrator(fake, SourceRegistry())

        report = await orch.run(docs)

        citations_result = next(r for r in report.agent_results if isinstance(r, CitationsResult))
        assert citations_result.outcome == "partial"
        assert "dropped" in (citations_result.error or "")
        assert report.citations == []

    async def test_ungrounded_discrepancy_dropped_marks_result_partial(self) -> None:
        docs = _all_docs()
        motion = docs["motion"]
        from backend.agents.citation_extractor import extract_authority_strings

        candidates = extract_authority_strings(motion.text)
        record_triples = [
            ("police_report", "POLICE REPORT", docs["police_report"].text),
        ]
        fake = FakeLLMClient()
        _seed_citations(fake, motion, candidates)
        # Discrepancy whose motion_claim quote is fabricated.
        fake.queue(
            system=CROSS_DOC_SYSTEM,
            user=build_cross_doc_prompt(motion.id, motion.text, record_triples),
            response=_ExtractedDiscrepancies(
                discrepancies=[
                    _LLMFactDiscrepancy(
                        motion_claim=Span(
                            doc_id="motion",
                            quote="A completely fabricated motion sentence.",
                        ),
                        contradicting_evidence=[
                            Span(
                                doc_id="police_report",
                                quote=(
                                    "Officer Reyes observed the harness was "
                                    "properly secured when first responders arrived."
                                ),
                            )
                        ],
                        description="Fake",
                        severity="material",
                    )
                ]
            ),
        )
        orch = Orchestrator(fake, SourceRegistry())

        report = await orch.run(docs)

        cross_doc_result = next(
            r for r in report.agent_results if isinstance(r, DiscrepanciesResult)
        )
        assert cross_doc_result.outcome == "partial"
        assert "dropped" in (cross_doc_result.error or "")
        assert report.findings == []

    async def test_mixed_grounded_and_ungrounded_keeps_grounded(self) -> None:
        docs = _all_docs()
        motion = docs["motion"]
        from backend.agents.citation_extractor import extract_authority_strings

        candidates = extract_authority_strings(motion.text)
        record_triples = [
            ("police_report", "POLICE REPORT", docs["police_report"].text),
        ]
        fake = FakeLLMClient()
        _seed_citations(fake, motion, candidates)
        # One real, one fabricated discrepancy.
        fake.queue(
            system=CROSS_DOC_SYSTEM,
            user=build_cross_doc_prompt(motion.id, motion.text, record_triples),
            response=_ExtractedDiscrepancies(
                discrepancies=[
                    _LLMFactDiscrepancy(
                        motion_claim=Span(
                            doc_id="motion",
                            quote=(
                                "The plaintiff was not wearing a safety harness "
                                "at the time of the fall."
                            ),
                        ),
                        contradicting_evidence=[
                            Span(
                                doc_id="police_report",
                                quote=(
                                    "Officer Reyes observed the harness was "
                                    "properly secured when first responders arrived."
                                ),
                            )
                        ],
                        description="real",
                        severity="material",
                    ),
                    _LLMFactDiscrepancy(
                        motion_claim=Span(doc_id="motion", quote="bogus quote"),
                        contradicting_evidence=[Span(doc_id="police_report", quote="also bogus")],
                        description="hallucinated",
                        severity="dispositive",
                    ),
                ]
            ),
        )
        orch = Orchestrator(fake, SourceRegistry())

        report = await orch.run(docs)

        assert len(report.findings) == 1
        assert report.findings[0].id == "find-1"
        assert report.findings[0].summary == "real"
        cross_doc_result = next(
            r for r in report.agent_results if isinstance(r, DiscrepanciesResult)
        )
        assert cross_doc_result.outcome == "partial"


# ── Failure propagation ──────────────────────────────────────────────────


class TestFailurePropagation:
    async def test_citation_agent_failure_does_not_break_report(self) -> None:
        docs = _all_docs()
        motion = docs["motion"]
        record_triples = [
            ("police_report", "POLICE REPORT", docs["police_report"].text),
        ]
        fake = FakeLLMClient()
        # Cross-doc seeded, citations NOT seeded → CitationExtractor.run
        # will fail on KeyError but the orchestrator must still return a
        # report.
        _seed_cross_doc(fake, motion, record_triples)
        orch = Orchestrator(fake, SourceRegistry())

        report = await orch.run(docs)

        assert report.citations == []
        assert len(report.findings) == 1
        citations_result = next(r for r in report.agent_results if isinstance(r, CitationsResult))
        assert citations_result.outcome == "failure"

    async def test_cross_doc_failure_does_not_break_report(self) -> None:
        docs = _all_docs()
        motion = docs["motion"]
        from backend.agents.citation_extractor import extract_authority_strings

        candidates = extract_authority_strings(motion.text)
        fake = FakeLLMClient()
        _seed_citations(fake, motion, candidates)
        # Cross-doc NOT seeded → will fail.
        orch = Orchestrator(fake, SourceRegistry())

        report = await orch.run(docs)

        assert len(report.citations) == 1
        assert report.findings == []
        cross_doc_result = next(
            r for r in report.agent_results if isinstance(r, DiscrepanciesResult)
        )
        assert cross_doc_result.outcome == "failure"

    async def test_both_agents_fail_still_returns_report(self) -> None:
        # Neither agent seeded.
        fake = FakeLLMClient()
        orch = Orchestrator(fake, SourceRegistry())
        report = await orch.run(_all_docs())
        assert report.citations == []
        assert report.findings == []
        outcomes = {r.outcome for r in report.agent_results}
        assert outcomes == {"failure"}


# ── Input validation ──────────────────────────────────────────────────────


class TestInputValidation:
    async def test_missing_motion_raises(self) -> None:
        # Only a record document — no motion.
        fake = FakeLLMClient()
        orch = Orchestrator(fake, SourceRegistry())
        with pytest.raises(ValueError, match="MOTION"):
            await orch.run({"police_report": _police_report()})

    async def test_multiple_motions_raises(self) -> None:
        m1 = _motion()
        m2 = Document(id="motion2", kind=DocumentKind.MOTION, text="also a motion")
        fake = FakeLLMClient()
        orch = Orchestrator(fake, SourceRegistry())
        with pytest.raises(ValueError, match="exactly one MOTION"):
            await orch.run({"motion": m1, "motion2": m2})
