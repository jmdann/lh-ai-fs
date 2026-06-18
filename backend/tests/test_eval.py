"""Tests for the eval harness — gold loader + matchers + scorer.

The matcher logic is what gates merges; if it drifts, recall numbers
silently shift. These tests pin the structural contract."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from backend.eval.gold import (
    GoldCitation,
    GoldDiscrepancy,
    load_gold_set,
    matches_citation,
    matches_discrepancy,
)
from backend.eval.scorer import score
from backend.models import (
    Citation,
    EvidenceRef,
    Finding,
    FindingKind,
    Span,
    VerificationReport,
)
from backend.sources import SourceRegistry

GOLD_PATH = Path("backend/tests/fixtures/gold_set.yaml")


# ── Loader ────────────────────────────────────────────────────────────────


def test_load_real_gold_set_smoke() -> None:
    gold = load_gold_set(GOLD_PATH)
    assert len(gold.discrepancies) >= 4
    assert len(gold.citations) >= 4
    assert all(d.severity in {"minor", "material", "dispositive"} for d in gold.discrepancies)
    assert all(d.evidence_docs for d in gold.discrepancies)


# ── matches_discrepancy ───────────────────────────────────────────────────


def _finding(motion_quote: str, evidence_doc_ids: list[str]) -> Finding:
    evidence = [EvidenceRef(span=Span(doc_id="motion", quote=motion_quote), role="primary")]
    evidence.extend(
        EvidenceRef(span=Span(doc_id=doc_id, quote="x"), role="contradicting")
        for doc_id in evidence_doc_ids
    )
    return Finding(
        id="find-1",
        kind=FindingKind.FACT_DISCREPANCY,
        summary="x",
        evidence=evidence,
        agent="X",
        prompt_version="1.0.0",
    )


def test_matcher_accepts_overlapping_motion_quote_and_correct_docs() -> None:
    gold = GoldDiscrepancy(
        id="gold-disc-1",
        motion_quote="Rivera was not wearing required personal protective equipment",
        evidence_docs=frozenset({"police_report"}),
        severity="material",
        notes="",
    )
    finding = _finding(
        motion_quote="Rivera was not wearing required PPE at the time",
        evidence_doc_ids=["police_report"],
    )
    assert matches_discrepancy(gold, finding) is True


def test_matcher_rejects_when_quote_overlap_too_low() -> None:
    gold = GoldDiscrepancy(
        id="gold-disc-1",
        motion_quote="Rivera was not wearing required personal protective equipment",
        evidence_docs=frozenset({"police_report"}),
        severity="material",
        notes="",
    )
    finding = _finding(motion_quote="completely unrelated text", evidence_doc_ids=["police_report"])
    assert matches_discrepancy(gold, finding) is False


def test_matcher_rejects_when_evidence_docs_missing() -> None:
    gold = GoldDiscrepancy(
        id="gold-disc-1",
        motion_quote="Rivera was not wearing required personal protective equipment",
        evidence_docs=frozenset({"police_report", "witness_statement"}),
        severity="material",
        notes="",
    )
    finding = _finding(
        motion_quote="Rivera was not wearing required PPE", evidence_doc_ids=["police_report"]
    )
    assert matches_discrepancy(gold, finding) is False


def test_matcher_rejects_wrong_finding_kind() -> None:
    gold = GoldDiscrepancy(
        id="gold-disc-1",
        motion_quote="x",
        evidence_docs=frozenset({"police_report"}),
        severity="material",
        notes="",
    )
    finding = Finding(
        id="find-1",
        kind=FindingKind.QUOTE_ALTERED,  # wrong kind
        summary="x",
        evidence=[EvidenceRef(span=Span(doc_id="motion", quote="x"), role="primary")],
        agent="X",
        prompt_version="1.0.0",
    )
    assert matches_discrepancy(gold, finding) is False


# ── matches_citation ──────────────────────────────────────────────────────


def test_citation_matcher_normalizes_case_and_whitespace() -> None:
    gold = GoldCitation(id="x", cited_authority_contains="Privette v. Superior Court", notes="")
    cite = Citation(
        id="cite-1",
        proposition=Span(doc_id="motion", quote="x"),
        cited_authority="PRIVETTE V.  SUPERIOR COURT, 5 Cal.4th 689 (1993)",
    )
    assert matches_citation(gold, cite) is True


def test_citation_matcher_rejects_missing_substring() -> None:
    gold = GoldCitation(id="x", cited_authority_contains="Privette v. Superior Court", notes="")
    cite = Citation(
        id="cite-1",
        proposition=Span(doc_id="motion", quote="x"),
        cited_authority="Smith v. Jones, 1 Cal.4th 1 (2000)",
    )
    assert matches_citation(gold, cite) is False


# ── score() integration ───────────────────────────────────────────────────


def test_score_zero_emitted_means_zero_matches() -> None:
    gold = load_gold_set(GOLD_PATH)
    empty_report = VerificationReport(
        case_name="x",
        generated_at=datetime(2026, 6, 18, tzinfo=UTC),
    )
    sources = SourceRegistry()
    s = score(empty_report, gold, sources)
    assert s.matched_gold_findings == 0
    assert s.matched_gold_citations == 0
    assert s.grounding_integrity_failures == 0
    assert s.cited_doc_in_scope_failures == 0


def test_score_grounding_failures_counted() -> None:
    """A finding whose evidence span doesn't ground gets counted as a
    grounding_integrity_failure even if the orchestrator forgot to drop it."""
    gold = load_gold_set(GOLD_PATH)
    finding = _finding(motion_quote="motion text", evidence_doc_ids=["police_report"])
    report = VerificationReport(
        case_name="x",
        generated_at=datetime(2026, 6, 18, tzinfo=UTC),
        findings=[finding],
    )
    sources = SourceRegistry()  # no docs registered → both spans miss
    s = score(report, gold, sources)
    assert s.cited_doc_in_scope_failures >= 2  # motion + police_report not registered
