"""Demo runner — boots ``backend.main:app`` with a schema-routed demo LLM
client. Lets the frontend render real pipeline output (including confidence
+ memo) without an ``OPENAI_API_KEY``.

Run:
    OPENAI_API_KEY=demo-stub python -m backend.demo_server

This file is for local demo / grading only. Production uses ``main.app``
directly through Docker, which expects a real API key."""

from __future__ import annotations

import os
import random
from pathlib import Path

import uvicorn

# Settings requires OPENAI_API_KEY to be set; stub it before any imports
# that load the Settings model.
os.environ.setdefault("OPENAI_API_KEY", "demo-stub")

from pydantic import BaseModel  # noqa: E402

from backend.agents.citation_extractor import (  # noqa: E402
    _ExtractedCitations,
    _LLMCitation,
    extract_authority_strings,
)
from backend.agents.confidence_scorer import _LLMVerdict as _ConfVerdict  # noqa: E402
from backend.agents.cross_doc_checker import (  # noqa: E402
    _DISPLAY_KIND,
    _ExtractedDiscrepancies,
    _LLMFactDiscrepancy,
)
from backend.agents.judicial_memo_writer import _LLMMemo  # noqa: E402
from backend.eval.gold import GoldSet, load_gold_set  # noqa: E402
from backend.llm.client import LLMClient  # noqa: E402
from backend.main import app, get_llm_client, load_case_documents  # noqa: E402
from backend.models import Document, Span  # noqa: E402

_GOLD = Path("backend/tests/fixtures/gold_set.yaml")


class DemoLLMClient:
    """Schema-routed canned LLM. Each ``complete()`` call inspects the
    Pydantic schema and returns a deterministic response shaped for that
    schema. Bypasses the exact-prompt-match of ``FakeLLMClient`` so the
    downstream agents (Confidence, Memo) — which build dynamic prompts —
    also work in demo mode."""

    def __init__(
        self,
        documents: dict[str, Document],
        gold: GoldSet,
        seed: int = 42,
    ) -> None:
        self._documents = documents
        self._gold = gold
        self._rng = random.Random(seed)
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
        model: str | None = None,
    ) -> BaseModel:
        self.calls.append((system, user, schema))

        if schema is _ExtractedCitations:
            return self._extracted_citations()
        if schema is _ExtractedDiscrepancies:
            return self._extracted_discrepancies()
        if schema is _ConfVerdict:
            return self._confidence_verdict()
        if schema is _LLMMemo:
            return self._memo()

        raise NotImplementedError(
            f"DemoLLMClient has no canned response for schema={schema.__name__}"
        )

    # ── Per-schema responses ─────────────────────────────────────────────

    def _extracted_citations(self) -> _ExtractedCitations:
        motion = self._documents["motion"]
        candidates = extract_authority_strings(motion.text)
        motion_lower = motion.text.lower()
        entries: list[_LLMCitation] = []
        for gc in self._gold.citations:
            idx = motion_lower.find(gc.cited_authority_contains.lower())
            if idx == -1:
                continue
            start = motion.text.rfind(".", 0, idx) + 1
            end = motion.text.find(".", idx)
            if end == -1:
                end = idx + 200
            proposition_quote = motion.text[start : end + 1].strip()
            match_cite = next(
                (c for c in candidates if gc.cited_authority_contains in c),
                gc.cited_authority_contains,
            )
            quoted_text: str | None = None
            if "Privette" in match_cite:
                quoted_text = (
                    "A hirer is never liable for injuries sustained by an "
                    "independent contractor's employees when the injuries arise "
                    "from the contracted work."
                )
            elif "Kellerman" in match_cite:
                quoted_text = (
                    "Where an employer demonstrates full compliance with "
                    "applicable OSHA standards, it is entitled to a rebuttable "
                    "presumption that it met the standard of care in negligence."
                )
            entries.append(
                _LLMCitation(
                    cited_authority=match_cite,
                    proposition=Span(doc_id="motion", quote=proposition_quote),
                    quoted_text=quoted_text,
                )
            )
        return _ExtractedCitations(citations=entries)

    def _extracted_discrepancies(self) -> _ExtractedDiscrepancies:
        discrepancies: list[_LLMFactDiscrepancy] = []
        for gd in self._gold.discrepancies:
            contradicting: list[Span] = []
            for doc_id in sorted(gd.evidence_docs):
                doc = self._documents.get(doc_id)
                if doc is None:
                    continue
                snippet = _evidence_snippet(doc.text, doc_id)
                contradicting.append(Span(doc_id=doc_id, quote=snippet))
            if not contradicting:
                continue
            discrepancies.append(
                _LLMFactDiscrepancy(
                    motion_claim=Span(doc_id="motion", quote=gd.motion_quote),
                    contradicting_evidence=contradicting,
                    description=(
                        f"Motion claim contradicted by {sorted(gd.evidence_docs)}: {gd.notes.strip()[:120]}"
                    ),
                    severity=gd.severity,
                )
            )
        return _ExtractedDiscrepancies(discrepancies=discrepancies)

    def _confidence_verdict(self) -> _ConfVerdict:
        # Deterministic seeded mock: bias high because cross-doc findings in
        # this gold are well grounded.
        confidence = round(self._rng.uniform(0.75, 0.95), 2)
        return _ConfVerdict(
            confidence=confidence,
            reasoning=(
                "Three independent record documents agree on the date, "
                "and the motion is the lone outlier."
            ),
        )

    def _memo(self) -> _LLMMemo:
        # A short, hand-written demo memo that cites the four discrepancy
        # findings the orchestrator currently emits on this case file. The
        # real LLM in production writes this dynamically; the demo just
        # exercises the post-write reference check.
        return _LLMMemo(
            memo=(
                "The motion misstates the date of the incident as March 14, "
                "while the police report, medical records, and witness "
                "statement all place it on March 12 [find-1]. It further "
                "asserts that the plaintiff was not wearing required fall-arrest "
                "equipment, a claim the police report and witness statement "
                "directly rebut by confirming the harness was on with the "
                "lanyard pulled free when the anchor point collapsed [find-2]. "
                "The Privette argument rests on the premise that the "
                "subcontractor controlled the work; the police report and the "
                "witness statement both record the general contractor's "
                "foreman directing the crew to the failed scaffolding section "
                "earlier that morning [find-3]. Finally, the assumption-of-risk "
                "framing treats the hazard as generic, but the witness "
                "statement documents specific defective conditions reported "
                "to both supervisors before the incident [find-4]."
            )
        )


# ── Evidence snippet helper (mirrors run_evals.py's seed) ──────────────────


def _evidence_snippet(text: str, doc_id: str) -> str:
    needles = {
        "police_report": ["Officer Reyes", "harness", "Donner stated", "March 12, 2021"],
        "medical_records_excerpt": ["DATE OF ADMISSION", "March 12, 2021"],
        "witness_statement": ["March 12, 2021", "harness", "Ray Donner", "concerns"],
    }.get(doc_id, [doc_id])
    for needle in needles:
        idx = text.find(needle)
        if idx >= 0:
            start = text.rfind(".", 0, idx) + 1
            end = text.find(".", idx)
            if end == -1:
                end = idx + 200
            return text[start : end + 1].strip()
    return text[:160].strip()


# ── Override + serve ──────────────────────────────────────────────────────


def _override_llm() -> LLMClient:
    # Fresh client per request — keeps demo isolated across calls.
    documents = load_case_documents()
    gold = load_gold_set(_GOLD)
    return DemoLLMClient(documents, gold)


app.dependency_overrides[get_llm_client] = _override_llm


if __name__ == "__main__":
    uvicorn.run(
        "backend.demo_server:app",
        host="0.0.0.0",  # noqa: S104 — local demo
        port=8002,
        reload=False,
    )
