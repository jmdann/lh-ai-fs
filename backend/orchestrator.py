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

from backend.agents.authority_support_checker import AuthoritySupportChecker
from backend.agents.citation_extractor import CitationExtractor
from backend.agents.confidence_scorer import ConfidenceScorer
from backend.agents.cross_doc_checker import CrossDocConsistencyChecker
from backend.agents.judicial_memo_writer import JudicialMemoWriter
from backend.agents.quote_checker import QuoteChecker
from backend.llm.client import LLMClient
from backend.models import (
    AuthorityCheckResult,
    Citation,
    CitationsResult,
    DiscrepanciesResult,
    Document,
    DocumentKind,
    EvidenceRef,
    FactDiscrepancy,
    Finding,
    FindingKind,
    QuoteCheckResult,
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
        self._quote_checker = QuoteChecker(llm, sources)
        self._authority_checker = AuthoritySupportChecker(llm, sources)
        self._confidence_scorer = ConfidenceScorer(llm)
        self._memo_writer = JudicialMemoWriter(llm)
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

        # Phase 1: extract + cross-doc (independent; fan out).
        citations_result, cross_doc_result = await asyncio.gather(
            self._extractor.run(motion),
            self._cross_doc.run(motion, records),
        )
        citations, citations_result = self._ground_citations(citations_result)

        # Phase 2: per-citation checks (also independent; fan out).
        quote_result, authority_result = await asyncio.gather(
            self._quote_checker.run(citations),
            self._authority_checker.run(citations),
        )

        findings, cross_doc_result = self._build_and_ground_findings(cross_doc_result)
        # Append quote / authority findings (kinds activated for spec 002).
        quote_findings, quote_result = self._build_quote_findings(quote_result, len(findings))
        findings.extend(quote_findings)
        authority_findings, authority_result = self._build_authority_findings(
            authority_result, len(findings)
        )
        findings.extend(authority_findings)

        # Phase 3 (spec 003): confidence scoring + judicial memo.
        # Both consume the assembled findings; they run sequentially because
        # the memo prefers confidence-rescored findings for its ranking.
        confidence_result = await self._confidence_scorer.run(findings)
        if confidence_result.outcome == "success":
            findings = confidence_result.data

        memo_result = await self._memo_writer.run(case_name, findings)

        return VerificationReport(
            case_name=case_name,
            generated_at=datetime.now(UTC),
            citations=citations,
            findings=findings,
            agent_results=[
                citations_result,
                cross_doc_result,
                quote_result,
                authority_result,
                confidence_result,
                memo_result,
            ],
            judicial_memo=memo_result.data if memo_result.outcome != "failure" else None,
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

    # ── Spec 002: quote + authority findings ──────────────────────────────

    def _build_quote_findings(
        self, result: QuoteCheckResult, finding_offset: int
    ) -> tuple[list[Finding], QuoteCheckResult]:
        """Materialize ``QuoteCheck`` entries with verdict ∈ {altered, fabricated}
        as ``Finding`` rows. ``exact``/``paraphrase``/``unverifiable`` stay in
        ``agent_results.data`` but do NOT become findings — they're not
        actionable for a reviewer triaging dishonesty.

        Parity with ``_build_and_ground_findings``: an ``altered`` verdict
        whose ``matched_span`` does not ground is silently dropped EXCEPT
        we flip ``outcome`` to ``partial`` with a count, so the eval harness
        sees the drop rather than missing it (Codex round 3, spec 002)."""
        if result.outcome == "failure":
            return [], result
        actionable = {"altered", "fabricated"}
        findings: list[Finding] = []
        dropped = 0
        for check in result.data:
            if check.verdict not in actionable:
                continue
            kind = (
                FindingKind.QUOTE_ALTERED
                if check.verdict == "altered"
                else FindingKind.QUOTE_FABRICATED
            )
            evidence: list[EvidenceRef] = []
            if check.matched_span is not None and self._is_grounded(check.matched_span):
                evidence.append(EvidenceRef(span=check.matched_span, role="primary"))
            if not evidence:
                dropped += 1
                continue  # cannot ship a finding without grounded evidence
            findings.append(
                Finding(
                    id=f"find-{finding_offset + len(findings) + 1}",
                    kind=kind,
                    summary=check.reasoning,
                    evidence=evidence,
                    agent=result.agent,
                    prompt_version=result.prompt_version,
                    citation_id=check.citation_id,
                )
            )
        if dropped > 0:
            return findings, result.model_copy(
                update={
                    "outcome": "partial",
                    "error": (f"{dropped} quote finding(s) dropped at the grounding boundary"),
                }
            )
        return findings, result

    def _build_authority_findings(
        self, result: AuthorityCheckResult, finding_offset: int
    ) -> tuple[list[Finding], AuthorityCheckResult]:
        """Materialize ``contradicts`` verdicts as Findings. ``supports`` and
        ``unverifiable`` stay in ``agent_results.data`` only — supports is
        the expected-good case; unverifiable is informational, not an
        actionable dishonesty signal.

        Parity with ``_build_and_ground_findings``: a ``contradicts`` whose
        ``source_basis`` does not ground is dropped but the result flips to
        ``partial`` so the eval harness sees the drop."""
        if result.outcome == "failure":
            return [], result
        findings: list[Finding] = []
        dropped = 0
        for check in result.data:
            if check.verdict != "contradicts":
                continue
            if check.source_basis is None or not self._is_grounded(check.source_basis):
                dropped += 1
                continue
            findings.append(
                Finding(
                    id=f"find-{finding_offset + len(findings) + 1}",
                    kind=FindingKind.AUTHORITY_UNSUPPORTED,
                    summary=check.reasoning,
                    evidence=[EvidenceRef(span=check.source_basis, role="primary")],
                    agent=result.agent,
                    prompt_version=result.prompt_version,
                    citation_id=check.citation_id,
                )
            )
        if dropped > 0:
            return findings, result.model_copy(
                update={
                    "outcome": "partial",
                    "error": (
                        f"{dropped} contradicts finding(s) dropped at the grounding boundary"
                    ),
                }
            )
        return findings, result
