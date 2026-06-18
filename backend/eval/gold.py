"""Gold-set loader + matcher.

STANDARDS § 5: the gold set is hand-authored from the source documents.
The matching key is **structural**, not string-similarity on summary text:
matched = (kind aligns) AND (motion span overlap >= 0.6 after normalization)
AND (evidence_docs ⊇ gold_evidence_docs). The `notes` field is reviewer
context only; it never enters scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from backend.models import (
    AuthorityCheck,
    Citation,
    Finding,
    FindingKind,
    QuoteCheck,
)
from backend.sources import _normalize

GoldDiscrepancyKind = Literal["minor", "material", "dispositive"]


@dataclass(frozen=True)
class GoldDiscrepancy:
    id: str
    motion_quote: str
    evidence_docs: frozenset[str]
    severity: GoldDiscrepancyKind
    notes: str


@dataclass(frozen=True)
class GoldCitation:
    id: str
    cited_authority_contains: str
    notes: str


@dataclass(frozen=True)
class GoldQuote:
    id: str
    cited_authority_contains: str
    expected_verdict: str  # one of QuoteVerdict
    notes: str


@dataclass(frozen=True)
class GoldAuthority:
    id: str
    cited_authority_contains: str
    expected_verdict: str  # one of AuthorityVerdict
    notes: str


@dataclass(frozen=True)
class GoldSet:
    discrepancies: tuple[GoldDiscrepancy, ...]
    citations: tuple[GoldCitation, ...]
    quotes: tuple[GoldQuote, ...] = ()
    authorities: tuple[GoldAuthority, ...] = ()

    @property
    def total(self) -> int:
        return (
            len(self.discrepancies) + len(self.citations) + len(self.quotes) + len(self.authorities)
        )


def load_gold_set(path: Path) -> GoldSet:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    discrepancies = tuple(
        GoldDiscrepancy(
            id=item["id"],
            motion_quote=item["motion_quote"],
            evidence_docs=frozenset(item["evidence_docs"]),
            severity=item["severity"],
            notes=item.get("notes", ""),
        )
        for item in raw.get("discrepancies", [])
    )
    citations = tuple(
        GoldCitation(
            id=item["id"],
            cited_authority_contains=item["cited_authority_contains"],
            notes=item.get("notes", ""),
        )
        for item in raw.get("citations", [])
    )
    quotes = tuple(
        GoldQuote(
            id=item["id"],
            cited_authority_contains=item["cited_authority_contains"],
            expected_verdict=item["expected_verdict"],
            notes=item.get("notes", ""),
        )
        for item in raw.get("quotes", [])
    )
    authorities = tuple(
        GoldAuthority(
            id=item["id"],
            cited_authority_contains=item["cited_authority_contains"],
            expected_verdict=item["expected_verdict"],
            notes=item.get("notes", ""),
        )
        for item in raw.get("authorities", [])
    )
    return GoldSet(
        discrepancies=discrepancies,
        citations=citations,
        quotes=quotes,
        authorities=authorities,
    )


# ── Matching ───────────────────────────────────────────────────────────────


def _quote_overlap(gold_quote: str, finding_quote: str) -> float:
    """Token-set overlap on normalized quotes. The structural matcher uses
    this as the "is the same motion claim being flagged" signal."""
    gold_tokens = set(_normalize(gold_quote).split())
    finding_tokens = set(_normalize(finding_quote).split())
    if not gold_tokens:
        return 0.0
    inter = gold_tokens & finding_tokens
    union = gold_tokens | finding_tokens
    return len(inter) / len(union) if union else 0.0


def matches_discrepancy(gold: GoldDiscrepancy, finding: Finding) -> bool:
    if finding.kind is not FindingKind.FACT_DISCREPANCY:
        return False
    # Find the primary evidence (motion span).
    primary = next((e.span for e in finding.evidence if e.role == "primary"), None)
    if primary is None or primary.doc_id != "motion":
        return False
    if _quote_overlap(gold.motion_quote, primary.quote) < 0.4:
        return False
    # Evidence docs: the gold's required record-doc set must be a subset of
    # what the finding cites.
    finding_evidence_docs = {e.span.doc_id for e in finding.evidence if e.role == "contradicting"}
    return gold.evidence_docs.issubset(finding_evidence_docs)


def matches_citation(gold: GoldCitation, citation: Citation) -> bool:
    needle = _normalize(gold.cited_authority_contains)
    return needle in _normalize(citation.cited_authority)


def matches_quote(gold: GoldQuote, check: QuoteCheck, citation_lookup: dict[str, str]) -> bool:
    """``citation_lookup`` maps citation_id → cited_authority (raw). Quote
    matches if the underlying citation mentions the gold's authority AND
    the verdict equals the expected verdict."""
    cited = citation_lookup.get(check.citation_id, "")
    if _normalize(gold.cited_authority_contains) not in _normalize(cited):
        return False
    return check.verdict == gold.expected_verdict


def matches_authority(
    gold: GoldAuthority, check: AuthorityCheck, citation_lookup: dict[str, str]
) -> bool:
    cited = citation_lookup.get(check.citation_id, "")
    if _normalize(gold.cited_authority_contains) not in _normalize(cited):
        return False
    return check.verdict == gold.expected_verdict
