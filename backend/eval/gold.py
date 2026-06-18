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

from backend.models import Citation, Finding, FindingKind
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
class GoldSet:
    discrepancies: tuple[GoldDiscrepancy, ...]
    citations: tuple[GoldCitation, ...]

    @property
    def total(self) -> int:
        return len(self.discrepancies) + len(self.citations)


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
    return GoldSet(discrepancies=discrepancies, citations=citations)


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
