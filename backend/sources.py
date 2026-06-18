"""``SourceRegistry`` — the grounding boundary.

STANDARDS § 3.6: agents emit ``Span(doc_id, quote)``. Before a finding ships
in the verification report, the orchestrator calls ``SourceRegistry.find``
to confirm the quote actually appears in the document. Hit → keep the
finding. Miss → drop it and log a ``grounding_integrity_failure``.

There are no character offsets in the IR; this registry is the only place
that knows how text is normalized to compare against source documents.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from backend.models import Document

# ── Normalization ──────────────────────────────────────────────────────────


# Normalization targets are deliberately ambiguous Unicode codepoints —
# RUF001/RUF003 are disabled for this file in pyproject because they fire
# on the very characters this dict exists to normalize away.
_SMART_TO_ASCII = str.maketrans(
    {
        "‘": "'",  # left single quotation mark
        "’": "'",  # right single quotation mark
        "“": '"',  # left double quotation mark
        "”": '"',  # right double quotation mark
        "–": "-",  # en dash
        "—": "-",  # em dash
        "…": "...",  # horizontal ellipsis
        "\xa0": " ",  # non-breaking space
    }
)


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace + replace smart punctuation. Symmetric
    on doc and quote so substring comparison is meaningful. Kept simple on
    purpose — fancier normalization (stemming, lemmatization) would mask
    real "altered quote" findings the eval is meant to catch."""
    text = text.translate(_SMART_TO_ASCII)
    text = " ".join(text.lower().split())
    return text


# ── Lookup result ──────────────────────────────────────────────────────────


class LookupKind(StrEnum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    MISS = "miss"
    DOC_NOT_IN_CORPUS = "doc_not_in_corpus"


class LookupResult(BaseModel):
    """Outcome of ``SourceRegistry.find``. Lives outside the IR — this is
    an orchestrator-internal type. ``score`` is 0-1; for ``EXACT`` it is
    1.0; for ``MISS`` / ``DOC_NOT_IN_CORPUS`` it is 0.0; for ``FUZZY`` it
    is the rapidfuzz partial ratio scaled into [0, 1]."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: LookupKind
    score: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def is_hit(self) -> bool:
        return self.kind in {LookupKind.EXACT, LookupKind.FUZZY}


# ── Registry ───────────────────────────────────────────────────────────────


class SourceRegistry:
    """Holds the loaded documents and looks up quotes against them.

    Two responsibilities:

    1. ``has(doc_id)`` — used by the eval harness to compute
       ``cited_doc_in_scope_failures``: did the LLM cite a document we never
       loaded?
    2. ``find(doc_id, quote, *, fuzzy_threshold)`` — used by the orchestrator
       at the span grounding boundary, and by spec 002's ``QuoteChecker`` as
       the deterministic-first step before calling the LLM.
    """

    def __init__(self, documents: Iterable[Document] = ()) -> None:
        self._docs: dict[str, Document] = {}
        self._normalized: dict[str, str] = {}
        for doc in documents:
            self.register(doc)

    def register(self, document: Document) -> None:
        """Add a document. Re-registering the same id replaces the previous
        entry (intentional — the eval harness rebuilds registries between
        runs and should not need to clear state)."""
        self._docs[document.id] = document
        self._normalized[document.id] = _normalize(document.text)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def find(
        self,
        doc_id: str,
        quote: str,
        *,
        fuzzy_threshold: float = 0.90,
    ) -> LookupResult:
        """Look up ``quote`` in document ``doc_id``.

        Order:
        1. Doc not in registry → ``DOC_NOT_IN_CORPUS``.
        2. Normalized substring match → ``EXACT`` (score 1.0).
        3. ``rapidfuzz.fuzz.partial_ratio`` ≥ ``fuzzy_threshold * 100``
           → ``FUZZY`` (score = ratio / 100).
        4. Otherwise → ``MISS`` (score = best partial_ratio / 100).
        """
        if not 0.0 <= fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be in [0, 1]")

        if doc_id not in self._docs:
            return LookupResult(kind=LookupKind.DOC_NOT_IN_CORPUS, score=0.0)

        norm_doc = self._normalized[doc_id]
        norm_quote = _normalize(quote)

        if not norm_quote:
            return LookupResult(kind=LookupKind.MISS, score=0.0)

        if norm_quote in norm_doc:
            return LookupResult(kind=LookupKind.EXACT, score=1.0)

        partial = fuzz.partial_ratio(norm_quote, norm_doc) / 100.0
        if partial >= fuzzy_threshold:
            return LookupResult(kind=LookupKind.FUZZY, score=partial)
        return LookupResult(kind=LookupKind.MISS, score=partial)
