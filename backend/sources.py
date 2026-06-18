"""``SourceRegistry`` — the grounding boundary.

STANDARDS § 3.6: agents emit ``Span(doc_id, quote)``. Before a finding
ships in the verification report, the orchestrator calls
``SourceRegistry.find`` to confirm the quote appears in the document.
Hit → keep. Miss → drop and log ``grounding_integrity_failure``.

There are no character offsets in the IR; this registry is the only
place that knows how text is normalized to compare against source docs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from rapidfuzz import fuzz

from backend.models import Document

# Normalization targets are deliberately ambiguous Unicode codepoints;
# pyproject ignores RUF001 / RUF003 for this file because the rules fire
# on exactly the chars this dict exists to normalize away.
_SMART_TO_ASCII = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
        "\xa0": " ",
    }
)


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace + replace smart punctuation. Symmetric
    on doc and quote so substring comparison is meaningful."""
    return " ".join(text.translate(_SMART_TO_ASCII).lower().split())


class LookupKind(StrEnum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    MISS = "miss"
    DOC_NOT_IN_CORPUS = "doc_not_in_corpus"


@dataclass(frozen=True)
class LookupResult:
    """Outcome of ``SourceRegistry.find``. Internal type, not part of the IR."""

    kind: LookupKind
    score: float = 0.0

    @property
    def is_hit(self) -> bool:
        return self.kind in {LookupKind.EXACT, LookupKind.FUZZY}


class SourceRegistry:
    """Holds the loaded documents and looks up quotes against them."""

    def __init__(self, documents: Iterable[Document] = ()) -> None:
        self._docs: dict[str, Document] = {}
        self._normalized: dict[str, str] = {}
        for doc in documents:
            self.register(doc)

    def register(self, document: Document) -> None:
        """Add a document. Re-registering the same id replaces the previous
        entry — the eval harness rebuilds registries between runs."""
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

        Order: doc-not-in-corpus → normalized substring (EXACT) →
        ``rapidfuzz.fuzz.partial_ratio`` above threshold (FUZZY) →
        MISS with best score for diagnostics."""
        if doc_id not in self._docs:
            return LookupResult(kind=LookupKind.DOC_NOT_IN_CORPUS)
        norm_quote = _normalize(quote)
        if not norm_quote:
            return LookupResult(kind=LookupKind.MISS)
        norm_doc = self._normalized[doc_id]
        if norm_quote in norm_doc:
            return LookupResult(kind=LookupKind.EXACT, score=1.0)
        partial = fuzz.partial_ratio(norm_quote, norm_doc) / 100.0
        if partial >= fuzzy_threshold:
            return LookupResult(kind=LookupKind.FUZZY, score=partial)
        return LookupResult(kind=LookupKind.MISS, score=partial)
