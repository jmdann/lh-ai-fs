"""Tests for SourceRegistry — the grounding boundary.

Minimal coverage of the load-bearing properties: registration,
out-of-corpus, normalized substring match, fuzzy threshold, and miss.
"""

from __future__ import annotations

from backend.models import Document, DocumentKind
from backend.sources import LookupKind, LookupResult, SourceRegistry


def _doc(doc_id: str = "motion", text: str = "The defendant was negligent.") -> Document:
    return Document(id=doc_id, kind=DocumentKind.MOTION, text=text)


def test_registration_and_has() -> None:
    reg = SourceRegistry([_doc("motion"), _doc("police_report")])
    assert reg.has("motion") is True
    assert reg.has("police_report") is True
    assert reg.has("missing") is False


def test_doc_not_in_corpus_short_circuits() -> None:
    reg = SourceRegistry([_doc()])
    result = reg.find("auth:privette", "anything")
    assert result.kind is LookupKind.DOC_NOT_IN_CORPUS
    assert result.is_hit is False


def test_exact_match_with_case_and_whitespace_normalization() -> None:
    reg = SourceRegistry([_doc(text="The   Defendant\nwas\tnegligent.")])
    result = reg.find("motion", "DEFENDANT was negligent")
    assert result.kind is LookupKind.EXACT
    assert result.score == 1.0
    assert result.is_hit is True


def test_smart_quotes_normalized_on_both_sides() -> None:
    reg = SourceRegistry([_doc(text='He said "hello" — politely.')])
    assert reg.find("motion", 'said "hello" - politely').kind is LookupKind.EXACT


def test_close_paraphrase_hits_fuzzy() -> None:
    reg = SourceRegistry(
        [
            _doc(
                text=(
                    "Plaintiff Carlos Rivera, a journeyman scaffolder, alleges that "
                    "he sustained injuries when a section of the scaffolding gave way."
                )
            )
        ]
    )
    result = reg.find(
        "motion",
        # one word changed: "sustained" -> "suffered"
        "Plaintiff Carlos Rivera, a journeyman scaffolder, alleges that "
        "he suffered injuries when a section of the scaffolding gave way.",
    )
    assert result.kind is LookupKind.FUZZY
    assert result.score >= 0.90


def test_unrelated_text_misses() -> None:
    reg = SourceRegistry([_doc(text="harness was not worn")])
    result = reg.find("motion", "Officer Reyes filed a report on Tuesday.")
    assert result.kind is LookupKind.MISS
    assert result.is_hit is False


def test_threshold_can_be_relaxed() -> None:
    reg = SourceRegistry([_doc(text="alpha beta gamma")])
    loose = "completely different content here"
    assert reg.find("motion", loose).kind is LookupKind.MISS
    assert reg.find("motion", loose, fuzzy_threshold=0.1).kind is LookupKind.FUZZY


def test_empty_quote_returns_miss() -> None:
    reg = SourceRegistry([_doc()])
    assert reg.find("motion", "   \n\t").kind is LookupKind.MISS


def test_re_register_replaces() -> None:
    reg = SourceRegistry()
    reg.register(_doc("motion", "first"))
    reg.register(_doc("motion", "second wins"))
    assert reg.find("motion", "second wins").kind is LookupKind.EXACT


def test_lookup_result_is_hit_semantics() -> None:
    assert LookupResult(kind=LookupKind.EXACT, score=1.0).is_hit is True
    assert LookupResult(kind=LookupKind.FUZZY, score=0.92).is_hit is True
    assert LookupResult(kind=LookupKind.MISS, score=0.5).is_hit is False
    assert LookupResult(kind=LookupKind.DOC_NOT_IN_CORPUS).is_hit is False
