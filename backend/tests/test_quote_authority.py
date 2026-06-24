"""Tests for ``QuoteChecker`` + ``AuthoritySupportChecker``.

The load-bearing rule: when the cited authority's source text is NOT in
``SourceRegistry``, both agents emit ``unverifiable`` WITHOUT calling the
LLM (STANDARDS § 3.5). That is the common case on the real Rivera MSJ.
"""

from __future__ import annotations

from pydantic import BaseModel

from backend.agents.authority_support_checker import AuthoritySupportChecker
from backend.agents.quote_checker import QuoteChecker, _authority_doc_id
from backend.llm.client import FakeLLMClient
from backend.models import Citation, Document, DocumentKind, Span
from backend.sources import SourceRegistry


def _cite(
    cid: str = "cite-1",
    cited: str = "Privette v. Superior Court, 5 Cal.4th 689 (1993)",
    quoted: str | None = "A hirer is never liable...",
    proposition: str = "Under California law, a hirer is presumptively not liable.",
) -> Citation:
    return Citation(
        id=cid,
        proposition=Span(doc_id="motion", quote=proposition),
        cited_authority=cited,
        quoted_text=quoted,
    )


# ── _authority_doc_id heuristic ───────────────────────────────────────────


def test_authority_doc_id_lowercases_plaintiff_head() -> None:
    assert _authority_doc_id("Privette v. Superior Court, 5 Cal.4th 689") == "auth:privette"
    assert _authority_doc_id("Kellerman v. Pacific Coast, 887 F.2d 1204") == "auth:kellerman"
    # Disambiguates two different authorities.
    a = _authority_doc_id("Smith v. Jones")
    b = _authority_doc_id("Brown v. Davis")
    assert a != b


# ── QuoteChecker ──────────────────────────────────────────────────────────


class TestQuoteCheckerOutOfCorpus:
    async def test_emits_unverifiable_without_calling_llm(self) -> None:
        fake = FakeLLMClient()  # No canned responses; if LLM is called → KeyError
        checker = QuoteChecker(fake, SourceRegistry())
        result = await checker.run([_cite()])
        assert result.outcome == "success"
        assert len(result.data) == 1
        check = result.data[0]
        assert check.verdict == "unverifiable"
        assert check.matched_span is None
        assert "not loaded" in check.reasoning
        assert len(fake.calls) == 0  # LOAD-BEARING: no LLM call when source absent

    async def test_skips_citations_without_quoted_text(self) -> None:
        fake = FakeLLMClient()
        checker = QuoteChecker(fake, SourceRegistry())
        cites = [
            _cite(cid="cite-1", quoted=None),
            _cite(cid="cite-2", quoted="A hirer is never liable..."),
        ]
        result = await checker.run(cites)
        assert len(result.data) == 1
        assert result.data[0].citation_id == "cite-2"


class TestQuoteCheckerInCorpus:
    def _source_doc(self) -> Document:
        return Document(
            id="auth:privette",
            kind=DocumentKind.EXTERNAL_AUTHORITY,
            text=(
                "A hirer is never liable for injuries sustained by an "
                "independent contractor's employees when the injuries arise "
                "from the contracted work. This is the rule of Privette."
            ),
        )

    async def test_exact_match_short_circuits_to_exact(self) -> None:
        fake = FakeLLMClient()
        sources = SourceRegistry([self._source_doc()])
        checker = QuoteChecker(fake, sources)
        result = await checker.run(
            [
                _cite(
                    quoted=(
                        "A hirer is never liable for injuries sustained by an "
                        "independent contractor's employees"
                    )
                )
            ]
        )
        assert result.data[0].verdict == "exact"
        assert result.data[0].matched_span is not None
        assert len(fake.calls) == 0  # no LLM call on exact

    async def test_fabricated_when_in_corpus_but_quote_absent(self) -> None:
        fake = FakeLLMClient()
        sources = SourceRegistry([self._source_doc()])
        checker = QuoteChecker(fake, sources)
        result = await checker.run(
            [_cite(quoted="The contractor must always inspect the scaffold.")]
        )
        assert result.data[0].verdict == "fabricated"
        assert result.data[0].matched_span is None

    async def test_fuzzy_match_invokes_llm(self) -> None:
        # Quote = source minus one word — partial_ratio >= 0.9 → fuzzy path.
        from backend.agents.quote_checker import _LLMVerdict

        fake = FakeLLMClient()
        sources = SourceRegistry([self._source_doc()])

        # Monkey-patch FakeLLMClient.complete to return a canned verdict.
        # We reuse the agent's own module-scope _LLMVerdict so this test
        # would fail loudly if that schema drifts.
        async def _stub_complete(*, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
            return _LLMVerdict(verdict="altered", reasoning="Material word dropped.")

        fake.complete = _stub_complete  # type: ignore[method-assign, assignment]
        checker = QuoteChecker(fake, sources)
        result = await checker.run(
            [
                _cite(
                    quoted=(
                        "A hirer is liable for injuries sustained by an "
                        "independent contractor's employees"
                    )
                )
            ]
        )
        assert result.data[0].verdict in {"altered", "paraphrase"}


# ── AuthoritySupportChecker ───────────────────────────────────────────────


class TestAuthorityCheckerOutOfCorpus:
    async def test_emits_unverifiable_without_calling_llm(self) -> None:
        fake = FakeLLMClient()
        checker = AuthoritySupportChecker(fake, SourceRegistry())
        result = await checker.run([_cite()])
        assert result.outcome == "success"
        assert result.data[0].verdict == "unverifiable"
        assert result.data[0].source_basis is None
        assert "hallucination" in result.data[0].reasoning.lower()
        assert len(fake.calls) == 0


class TestAuthorityCheckerInCorpus:
    async def test_supports_when_llm_says_supports(self) -> None:
        doc = Document(
            id="auth:privette",
            kind=DocumentKind.EXTERNAL_AUTHORITY,
            text=(
                "A hirer is never liable for injuries sustained by an "
                "independent contractor's employees when the injuries arise "
                "from the contracted work."
            ),
        )
        sources = SourceRegistry([doc])

        class _Verdict(BaseModel):
            verdict: str
            reasoning: str

        fake = FakeLLMClient()

        async def _stub(*, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
            return _Verdict(verdict="supports", reasoning="Source span establishes the rule.")

        fake.complete = _stub  # type: ignore[method-assign, assignment]
        checker = AuthoritySupportChecker(fake, sources)
        result = await checker.run([_cite()])
        check = result.data[0]
        assert check.verdict == "supports"
        assert check.source_basis is not None

    async def test_unverifiable_from_llm_strips_source_basis(self) -> None:
        doc = Document(id="auth:privette", kind=DocumentKind.EXTERNAL_AUTHORITY, text="x" * 200)
        sources = SourceRegistry([doc])

        class _Verdict(BaseModel):
            verdict: str
            reasoning: str

        fake = FakeLLMClient()

        async def _stub(*, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
            return _Verdict(verdict="unverifiable", reasoning="Span too vague.")

        fake.complete = _stub  # type: ignore[method-assign, assignment]
        checker = AuthoritySupportChecker(fake, sources)
        result = await checker.run([_cite()])
        check = result.data[0]
        assert check.verdict == "unverifiable"
        assert check.source_basis is None


# ── Orchestrator-side integration smoke ──────────────────────────────────


async def test_agents_consume_empty_citation_list_cleanly() -> None:
    """When CitationExtractor returns nothing, the per-citation agents
    should still succeed with empty data — not fail."""
    fake = FakeLLMClient()
    q = QuoteChecker(fake, SourceRegistry())
    a = AuthoritySupportChecker(fake, SourceRegistry())
    qr = await q.run([])
    ar = await a.run([])
    assert qr.outcome == "success"
    assert qr.data == []
    assert ar.outcome == "success"
    assert ar.data == []
    assert len(fake.calls) == 0
