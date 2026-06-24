"""``AuthoritySupportChecker`` — does the cited authority support the
proposition? Spec 002 § 5.2.

STANDARDS § 3.5 is the load-bearing rule here: when the authority's source
text is NOT in ``SourceRegistry`` (the common case in the Rivera MSJ — no
external case law is loaded), emit ``unverifiable`` without calling the
LLM. Reasoning about case law we do not have IS the hallucination the
grading rubric punishes hardest.

When an authority document IS registered (future PR with retrieval), the
agent passes the proposition + source span to the LLM and asks for
supports / contradicts / unverifiable.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from backend.agents.prompts.authority_support_checker import (
    PROMPT_VERSION,
    SYSTEM,
    build_user_prompt,
)
from backend.agents.quote_checker import _authority_doc_id, _best_fuzzy_window
from backend.llm.client import LLMClient
from backend.models import AuthorityCheck, AuthorityCheckResult, Citation, Span
from backend.sources import SourceRegistry


class _LLMVerdict(BaseModel):
    verdict: str
    reasoning: str


class AuthoritySupportChecker:
    AGENT_NAME = "AuthoritySupportChecker"
    PROMPT_VERSION = PROMPT_VERSION

    def __init__(self, llm: LLMClient, sources: SourceRegistry) -> None:
        self._llm = llm
        self._sources = sources

    async def run(self, citations: list[Citation]) -> AuthorityCheckResult:
        start = time.perf_counter()
        checks: list[AuthorityCheck] = []
        try:
            for citation in citations:
                checks.append(await self._check_one(citation))
            return AuthorityCheckResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="success",
                latency_ms=_elapsed_ms(start),
                data=checks,
            )
        except Exception as exc:
            return AuthorityCheckResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="failure",
                latency_ms=_elapsed_ms(start),
                data=checks,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _check_one(self, citation: Citation) -> AuthorityCheck:
        doc_id = _authority_doc_id(citation.cited_authority)

        # STANDARDS § 3.5: out-of-corpus → unverifiable, no LLM.
        if not self._sources.has(doc_id):
            return AuthorityCheck(
                citation_id=citation.id,
                verdict="unverifiable",
                source_basis=None,
                reasoning=(
                    f"Authority source for {citation.cited_authority!r} is not "
                    f"loaded (no document with id={doc_id!r}). Inferring "
                    "supports/contradicts from the cite alone would be "
                    "hallucination; deferring to unverifiable."
                ),
            )

        # In-corpus path: pull a relevant span, ask the LLM.
        source_text = self._sources._docs[doc_id].text
        snippet = _best_fuzzy_window(citation.proposition.quote, source_text)
        llm_out = await self._llm.complete(
            system=SYSTEM,
            user=build_user_prompt(citation.proposition.quote, snippet),
            schema=_LLMVerdict,
        )
        verdict = (
            llm_out.verdict
            if llm_out.verdict
            in {
                "supports",
                "contradicts",
                "unverifiable",
            }
            else "unverifiable"
        )
        if verdict == "unverifiable":
            return AuthorityCheck(
                citation_id=citation.id,
                verdict="unverifiable",
                source_basis=None,
                reasoning=llm_out.reasoning,
            )
        return AuthorityCheck(
            citation_id=citation.id,
            verdict=verdict,  # type: ignore[arg-type]
            source_basis=Span(doc_id=doc_id, quote=snippet),
            reasoning=llm_out.reasoning,
        )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
