"""Tests for backend.sources.SourceRegistry — the grounding boundary.

The properties under test are the load-bearing ones: a quote that exists
in the source after normalization is found; a quote that has been altered
beyond the fuzzy threshold is reported as a miss; a document the LLM cites
but we never loaded is reported as out-of-corpus.
"""

from __future__ import annotations

import pytest

from backend.models import Document, DocumentKind
from backend.sources import LookupKind, LookupResult, SourceRegistry


def _doc(doc_id: str = "motion", text: str = "The defendant was negligent.") -> Document:
    return Document(id=doc_id, kind=DocumentKind.MOTION, text=text)


# ── Registration ──────────────────────────────────────────────────────────


class TestRegistration:
    def test_constructs_empty(self) -> None:
        reg = SourceRegistry()
        assert reg.has("anything") is False

    def test_constructs_with_iterable(self) -> None:
        reg = SourceRegistry([_doc("motion"), _doc("police_report")])
        assert reg.has("motion") is True
        assert reg.has("police_report") is True

    def test_register_adds(self) -> None:
        reg = SourceRegistry()
        reg.register(_doc("motion"))
        assert reg.has("motion") is True

    def test_re_register_replaces(self) -> None:
        reg = SourceRegistry()
        reg.register(_doc("motion", "first text"))
        reg.register(_doc("motion", "second text replaces first"))
        result = reg.find("motion", "second text")
        assert result.kind is LookupKind.EXACT


# ── DOC_NOT_IN_CORPUS ─────────────────────────────────────────────────────


class TestDocNotInCorpus:
    def test_returns_doc_not_in_corpus(self) -> None:
        reg = SourceRegistry([_doc("motion", "...")])
        result = reg.find("auth:privette", "anything")
        assert result.kind is LookupKind.DOC_NOT_IN_CORPUS
        assert result.score == 0.0
        assert result.is_hit is False


# ── EXACT matching with normalization ─────────────────────────────────────


class TestExactMatchAfterNormalization:
    def test_substring_hit(self) -> None:
        reg = SourceRegistry([_doc(text="The defendant was negligent.")])
        assert reg.find("motion", "defendant was negligent").kind is LookupKind.EXACT

    def test_case_insensitive(self) -> None:
        reg = SourceRegistry([_doc(text="The Defendant Was Negligent.")])
        assert reg.find("motion", "DEFENDANT was negligent").kind is LookupKind.EXACT

    def test_whitespace_collapsing(self) -> None:
        reg = SourceRegistry([_doc(text="The   defendant\nwas\tnegligent.")])
        assert reg.find("motion", "The defendant was negligent.").kind is LookupKind.EXACT

    def test_smart_quotes_normalized(self) -> None:
        # Source has smart quotes; query uses ASCII (typical LLM emission)
        reg = SourceRegistry([_doc(text="He said “hello” to her.")])
        assert reg.find("motion", 'said "hello"').kind is LookupKind.EXACT

    def test_em_dash_normalized(self) -> None:
        reg = SourceRegistry([_doc(text="She arrived — he left.")])
        assert reg.find("motion", "she arrived - he left").kind is LookupKind.EXACT

    def test_nbsp_normalized(self) -> None:
        reg = SourceRegistry([_doc(text="word1\xa0word2")])
        assert reg.find("motion", "word1 word2").kind is LookupKind.EXACT

    def test_exact_match_score_is_one(self) -> None:
        reg = SourceRegistry([_doc(text="alpha beta gamma")])
        result = reg.find("motion", "beta gamma")
        assert result.score == 1.0
        assert result.is_hit is True


# ── FUZZY matching ────────────────────────────────────────────────────────


class TestFuzzyMatch:
    def test_close_paraphrase_above_threshold(self) -> None:
        """A near-exact quote with one swapped word should hit FUZZY at the
        default 0.90 threshold — the kind of "altered quote" QuoteChecker
        flags as ``altered`` (not ``fabricated``) in spec 002."""
        reg = SourceRegistry(
            [
                _doc(
                    text=(
                        "Plaintiff Carlos Rivera, a journeyman scaffolder employed by "
                        "subcontractor Apex Staffing Solutions, alleges that he sustained "
                        "injuries when a section of the scaffolding gave way."
                    )
                )
            ]
        )
        result = reg.find(
            "motion",
            # one word changed: "sustained" -> "suffered"
            "Plaintiff Carlos Rivera, a journeyman scaffolder employed by "
            "subcontractor Apex Staffing Solutions, alleges that he suffered "
            "injuries when a section of the scaffolding gave way.",
        )
        assert result.kind is LookupKind.FUZZY
        assert result.score >= 0.90

    def test_threshold_respected(self) -> None:
        reg = SourceRegistry([_doc(text="The harness was not worn properly.")])
        # A loose paraphrase that scores below default 0.90 should miss.
        loose = "The defendant alleges this is unrelated to safety equipment."
        result_default = reg.find("motion", loose)
        assert result_default.kind is LookupKind.MISS
        # But a low custom threshold should accept the same query.
        result_lowered = reg.find("motion", loose, fuzzy_threshold=0.10)
        assert result_lowered.kind is LookupKind.FUZZY

    def test_threshold_validation(self) -> None:
        reg = SourceRegistry([_doc()])
        with pytest.raises(ValueError, match="fuzzy_threshold"):
            reg.find("motion", "x", fuzzy_threshold=-0.1)
        with pytest.raises(ValueError, match="fuzzy_threshold"):
            reg.find("motion", "x", fuzzy_threshold=1.5)

    def test_fuzzy_score_in_unit_interval(self) -> None:
        reg = SourceRegistry([_doc(text="alpha beta gamma delta")])
        result = reg.find("motion", "alpha beta gamma delta epsilon")
        assert 0.0 <= result.score <= 1.0


# ── MISS ──────────────────────────────────────────────────────────────────


class TestMiss:
    def test_unrelated_text_misses(self) -> None:
        reg = SourceRegistry([_doc(text="The harness was not worn.")])
        result = reg.find("motion", "Officer Reyes filed his report on Tuesday.")
        assert result.kind is LookupKind.MISS
        assert result.is_hit is False

    def test_empty_quote_normalizes_to_miss(self) -> None:
        reg = SourceRegistry([_doc()])
        # Whitespace-only quotes normalize to empty; we report MISS rather
        # than letting "" be a substring of every document.
        result = reg.find("motion", "   \n\t  ")
        assert result.kind is LookupKind.MISS


# ── LookupResult ──────────────────────────────────────────────────────────


class TestLookupResult:
    def test_is_hit_for_exact_and_fuzzy(self) -> None:
        assert LookupResult(kind=LookupKind.EXACT, score=1.0).is_hit is True
        assert LookupResult(kind=LookupKind.FUZZY, score=0.92).is_hit is True

    def test_is_hit_false_for_miss_and_out_of_corpus(self) -> None:
        assert LookupResult(kind=LookupKind.MISS, score=0.1).is_hit is False
        assert LookupResult(kind=LookupKind.DOC_NOT_IN_CORPUS).is_hit is False

    def test_score_must_be_unit_interval(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LookupResult(kind=LookupKind.FUZZY, score=1.5)
        with pytest.raises(ValidationError):
            LookupResult(kind=LookupKind.FUZZY, score=-0.1)

    def test_lookup_result_is_frozen(self) -> None:
        from pydantic import ValidationError

        result = LookupResult(kind=LookupKind.EXACT, score=1.0)
        with pytest.raises(ValidationError):
            result.score = 0.5  # type: ignore[misc]
