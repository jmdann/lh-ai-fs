"""``Orchestrator`` — the pipeline runner.

One sentence: run the spec-001 agents concurrently, ground their spans,
assemble the ``VerificationReport``. Spec 003 adds rich failure
isolation; spec 001 keeps the orchestrator straight-line.

Responsibilities (specs/001 § 6, STANDARDS § 3.6):

1. **Fan out** the two independent agents via ``asyncio.gather`` so they
   share wall-clock. ``CitationExtractor`` and ``CrossDocConsistencyChecker``
   run in parallel from day one — Codex round 2 was explicit about this.
2. **Build findings** by wrapping ``FactDiscrepancy`` entries in typed
   ``Finding`` envelopes with deterministic ``find-N`` ids.
3. **Ground the spans** at the orchestrator boundary: every ``Span`` an
   agent emitted is looked up via ``SourceRegistry.find``. Ungrounded
   findings / citations are dropped before the report ships, and the
   corresponding ``AgentResult`` is updated to ``outcome="partial"`` so
   the eval harness can report ``grounding_integrity_failures``.

The grounding step is what makes the "no offsets in the IR" rule
actually safe: the type system stops the LLM from emitting integers,
and this boundary stops it from inventing quotes that don't exist in
any document we loaded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime

from backend.agents.citation_extractor import CitationExtractor
from backend.agents.cross_doc_checker import CrossDocConsistencyChecker
from backend.llm.client import LLMClient
from backend.models import (
    Citation,
    CitationsResult,
    DiscrepanciesResult,
    Document,
    DocumentKind,
    EvidenceRef,
    FactDiscrepancy,
    Finding,
    FindingKind,
    Span,
    VerificationReport,
)
from backend.sources import SourceRegistry


class Orchestrator:
    """Runs the spec-001 pipeline. ``case_name`` defaults to "untitled"
    but the API edge typically passes the case caption."""

    def __init__(self, llm: LLMClient, sources: SourceRegistry) -> None:
        self._extractor = CitationExtractor(llm)
        self._cross_doc = CrossDocConsistencyChecker(llm)
        self._sources = sources

    async def run(
        self,
        documents: Mapping[str, Document],
        *,
        case_name: str = "untitled",
    ) -> VerificationReport:
        motion = self._select_motion(documents)
        records = [doc for doc in documents.values() if doc.id != motion.id]
        # Ensure every document we were handed is grounded against — the
        # SourceRegistry the caller built may not include them yet.
        for doc in documents.values():
            self._sources.register(doc)

        citations_result, cross_doc_result = await asyncio.gather(
            self._extractor.run(motion),
            self._cross_doc.run(motion, records),
        )

        citations, citations_result = self._ground_citations(citations_result)
        findings, cross_doc_result = self._build_and_ground_findings(cross_doc_result)

        return VerificationReport(
            case_name=case_name,
            generated_at=datetime.now(UTC),
            citations=citations,
            findings=findings,
            agent_results=[citations_result, cross_doc_result],
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _select_motion(self, documents: Mapping[str, Document]) -> Document:
        motion_docs = [d for d in documents.values() if d.kind is DocumentKind.MOTION]
        if not motion_docs:
            raise ValueError("Orchestrator requires a document with kind=MOTION")
        if len(motion_docs) > 1:
            raise ValueError("Orchestrator expects exactly one MOTION document")
        return motion_docs[0]

    def _is_grounded(self, span: Span) -> bool:
        return self._sources.find(span.doc_id, span.quote).is_hit

    def _ground_citations(self, result: CitationsResult) -> tuple[list[Citation], CitationsResult]:
        if result.outcome == "failure":
            return [], result
        kept: list[Citation] = []
        dropped = 0
        for citation in result.data:
            if not self._is_grounded(citation.proposition):
                dropped += 1
                continue
            kept.append(citation)
        if dropped > 0:
            return kept, result.model_copy(
                update={
                    "outcome": "partial",
                    "data": kept,
                    "error": (f"{dropped} citation(s) dropped at the grounding boundary"),
                }
            )
        return kept, result

    def _build_and_ground_findings(
        self, result: DiscrepanciesResult
    ) -> tuple[list[Finding], DiscrepanciesResult]:
        if result.outcome == "failure":
            return [], result
        kept_findings: list[Finding] = []
        kept_discrepancies: list[FactDiscrepancy] = []
        dropped = 0
        for discrepancy in result.data:
            if not self._all_grounded(discrepancy):
                dropped += 1
                continue
            kept_discrepancies.append(discrepancy)
            finding_id = f"find-{len(kept_findings) + 1}"
            kept_findings.append(
                Finding(
                    id=finding_id,
                    kind=FindingKind.FACT_DISCREPANCY,
                    summary=discrepancy.description,
                    evidence=[
                        EvidenceRef(span=discrepancy.motion_claim, role="primary"),
                        *[
                            EvidenceRef(span=span, role="contradicting")
                            for span in discrepancy.contradicting_evidence
                        ],
                    ],
                    agent=result.agent,
                    prompt_version=result.prompt_version,
                )
            )
        if dropped > 0:
            return kept_findings, result.model_copy(
                update={
                    "outcome": "partial",
                    "data": kept_discrepancies,
                    "error": (
                        f"{dropped} discrepanc{'y' if dropped == 1 else 'ies'} "
                        "dropped at the grounding boundary"
                    ),
                }
            )
        return kept_findings, result

    def _all_grounded(self, discrepancy: FactDiscrepancy) -> bool:
        spans = [discrepancy.motion_claim, *discrepancy.contradicting_evidence]
        return all(self._is_grounded(span) for span in spans)
