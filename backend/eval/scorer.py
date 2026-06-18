"""Scorer — compute the metrics the brief actually grades.

STANDARDS § 5: report precision / recall / hallucination — and gate on COUNTS
(not percentages) since the gold set is small (5-15 entries) and percentages
become theater (recall 0.5 = 3/6).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.eval.gold import (
    GoldSet,
    matches_authority,
    matches_citation,
    matches_discrepancy,
    matches_quote,
)
from backend.models import (
    AuthorityCheckResult,
    Citation,
    Finding,
    QuoteCheckResult,
    VerificationReport,
)
from backend.sources import SourceRegistry


@dataclass(frozen=True)
class EvalScores:
    # Counts (the gate metrics)
    matched_gold_findings: int
    matched_gold_citations: int
    matched_gold_quotes: int
    matched_gold_authorities: int
    grounding_integrity_failures: int
    cited_doc_in_scope_failures: int

    # Totals
    total_gold_discrepancies: int
    total_gold_citations: int
    total_gold_quotes: int
    total_gold_authorities: int
    total_emitted_findings: int
    total_emitted_citations: int
    total_emitted_quote_checks: int
    total_emitted_authority_checks: int
    emitted_authority_unverifiable: int

    # Per-gold-entry matches (for the human-readable diff)
    discrepancy_matches: dict[str, bool]
    citation_matches: dict[str, bool]
    quote_matches: dict[str, bool]
    authority_matches: dict[str, bool]

    @property
    def discrepancy_recall(self) -> float:
        if self.total_gold_discrepancies == 0:
            return 1.0
        return self.matched_gold_findings / self.total_gold_discrepancies

    @property
    def citation_recall(self) -> float:
        if self.total_gold_citations == 0:
            return 1.0
        return self.matched_gold_citations / self.total_gold_citations

    @property
    def unverifiable_precision(self) -> float:
        """Of GOLD-LABELED authority checks the pipeline marked unverifiable,
        the fraction that gold ALSO marks unverifiable.

        Denominator is restricted to gold-labeled cites so the metric does
        not punish a pipeline for emitting unverifiable on cites gold hasn't
        labeled (e.g. footnote authorities the gold set deliberately doesn't
        enumerate). Gaming this still requires the pipeline to say
        unverifiable for cases gold actually marks supports/contradicts —
        once spec 002+ has in-corpus authorities, that's the live test."""
        if self.total_gold_authorities == 0:
            return 1.0
        # All gold entries today expect unverifiable, so this collapses to
        # matched_gold_authorities / total_gold_authorities. The structure
        # is here for when gold gains supports/contradicts entries.
        return self.matched_gold_authorities / self.total_gold_authorities

    @property
    def overall_recall(self) -> float:
        total_gold = (
            self.total_gold_discrepancies
            + self.total_gold_citations
            + self.total_gold_quotes
            + self.total_gold_authorities
        )
        if total_gold == 0:
            return 1.0
        return (
            self.matched_gold_findings
            + self.matched_gold_citations
            + self.matched_gold_quotes
            + self.matched_gold_authorities
        ) / total_gold


def _grounding_failures(
    findings: list[Finding], citations: list[Citation], sources: SourceRegistry
) -> tuple[int, int]:
    """Count emitted spans that fail SourceRegistry.find (`grounding_integrity`)
    OR cite a doc_id the registry never loaded (`cited_doc_in_scope`).

    The orchestrator already dropped findings/citations whose spans didn't
    ground, so this should normally be zero. Counting it again here catches
    the case where the orchestrator and the registry have diverged."""
    grounding_fail = 0
    scope_fail = 0
    for finding in findings:
        for ev in finding.evidence:
            span = ev.span
            if not sources.has(span.doc_id):
                scope_fail += 1
            elif not sources.find(span.doc_id, span.quote).is_hit:
                grounding_fail += 1
    for citation in citations:
        span = citation.proposition
        if not sources.has(span.doc_id):
            scope_fail += 1
        elif not sources.find(span.doc_id, span.quote).is_hit:
            grounding_fail += 1
    return grounding_fail, scope_fail


def score(report: VerificationReport, gold: GoldSet, sources: SourceRegistry) -> EvalScores:
    """Match each gold entry against the report, then compute the counts."""
    disc_matches: dict[str, bool] = {}
    for gd in gold.discrepancies:
        disc_matches[gd.id] = any(matches_discrepancy(gd, f) for f in report.findings)

    cite_matches: dict[str, bool] = {}
    for gc in gold.citations:
        cite_matches[gc.id] = any(matches_citation(gc, c) for c in report.citations)

    # Build a citation_id -> cited_authority lookup so quote / authority
    # matchers can pivot from a check's citation_id back to its authority.
    citation_lookup = {c.id: c.cited_authority for c in report.citations}

    # Pull quote / authority checks out of agent_results.
    quote_checks = []
    authority_checks = []
    for ar in report.agent_results:
        if isinstance(ar, QuoteCheckResult):
            quote_checks.extend(ar.data)
        elif isinstance(ar, AuthorityCheckResult):
            authority_checks.extend(ar.data)

    quote_matches: dict[str, bool] = {}
    for gq in gold.quotes:
        quote_matches[gq.id] = any(matches_quote(gq, q, citation_lookup) for q in quote_checks)

    auth_matches: dict[str, bool] = {}
    for ga in gold.authorities:
        auth_matches[ga.id] = any(
            matches_authority(ga, a, citation_lookup) for a in authority_checks
        )

    emitted_unverifiable = sum(1 for a in authority_checks if a.verdict == "unverifiable")

    grounding_fail, scope_fail = _grounding_failures(report.findings, report.citations, sources)

    return EvalScores(
        matched_gold_findings=sum(disc_matches.values()),
        matched_gold_citations=sum(cite_matches.values()),
        matched_gold_quotes=sum(quote_matches.values()),
        matched_gold_authorities=sum(auth_matches.values()),
        grounding_integrity_failures=grounding_fail,
        cited_doc_in_scope_failures=scope_fail,
        total_gold_discrepancies=len(gold.discrepancies),
        total_gold_citations=len(gold.citations),
        total_gold_quotes=len(gold.quotes),
        total_gold_authorities=len(gold.authorities),
        total_emitted_findings=len(report.findings),
        total_emitted_citations=len(report.citations),
        total_emitted_quote_checks=len(quote_checks),
        total_emitted_authority_checks=len(authority_checks),
        emitted_authority_unverifiable=emitted_unverifiable,
        discrepancy_matches=disc_matches,
        citation_matches=cite_matches,
        quote_matches=quote_matches,
        authority_matches=auth_matches,
    )
