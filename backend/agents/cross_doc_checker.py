"""``CrossDocConsistencyChecker`` — flag motion claims contradicted by record docs.

One sentence: Compare each factual claim in the motion against the police
report, medical records, and witness statement. Emit ``FactDiscrepancy``
ONLY when a record document directly contradicts the motion.

This is the fastest path to real signal in spec 001 (codex round 2):
record docs are local, so there is no source-retrieval tarpit. Spec 002's
``AuthoritySupportChecker`` is the harder agent because it has to reason
about external case law without retrieval.

Agent boundary discipline (STANDARDS § 3.3):
- Only emits ``FACT_DISCREPANCY`` findings.
- Does NOT touch legal authorities — that's
  ``AuthoritySupportChecker`` (spec 002).
- Does NOT judge quote fidelity — that's ``QuoteChecker`` (spec 002).
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.agents.prompts.cross_doc_checker import (
    PROMPT_VERSION,
    SYSTEM,
    build_user_prompt,
)
from backend.llm.client import LLMClient
from backend.models import DiscrepanciesResult, Document, DocumentKind, FactDiscrepancy, Span

# ── LLM I/O schema ────────────────────────────────────────────────────────


class _LLMFactDiscrepancy(BaseModel):
    """Single discrepancy as the LLM emits it. Agent code wraps it with a
    deterministic ``disc-N`` id."""

    model_config = ConfigDict(extra="forbid")

    motion_claim: Span
    contradicting_evidence: list[Span] = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Literal["minor", "material", "dispositive"]


class _ExtractedDiscrepancies(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discrepancies: list[_LLMFactDiscrepancy] = Field(default_factory=list)


# ── Display names per record kind ──────────────────────────────────────────


_DISPLAY_KIND: dict[DocumentKind, str] = {
    DocumentKind.POLICE_REPORT: "POLICE REPORT",
    DocumentKind.MEDICAL_RECORDS: "MEDICAL RECORDS",
    DocumentKind.WITNESS_STATEMENT: "WITNESS STATEMENT",
    DocumentKind.MOTION: "MOTION",
    DocumentKind.EXTERNAL_AUTHORITY: "EXTERNAL AUTHORITY",
}


# ── Agent ──────────────────────────────────────────────────────────────────


class CrossDocConsistencyChecker:
    """Wires the LLM call into a single
    ``run(motion, records) -> DiscrepanciesResult`` invocation."""

    AGENT_NAME = "CrossDocConsistencyChecker"
    PROMPT_VERSION = PROMPT_VERSION

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(
        self,
        motion: Document,
        records: list[Document],
    ) -> DiscrepanciesResult:
        start = time.perf_counter()
        try:
            record_triples = [
                (doc.id, _DISPLAY_KIND.get(doc.kind, doc.kind.value.upper()), doc.text)
                for doc in records
            ]
            user = build_user_prompt(motion.id, motion.text, record_triples)
            llm_response = await self._llm.complete(
                system=SYSTEM,
                user=user,
                schema=_ExtractedDiscrepancies,
            )
            discrepancies = [
                FactDiscrepancy(
                    id=f"disc-{idx}",
                    motion_claim=entry.motion_claim,
                    contradicting_evidence=entry.contradicting_evidence,
                    description=entry.description,
                    severity=entry.severity,
                )
                for idx, entry in enumerate(llm_response.discrepancies, start=1)
            ]
            return DiscrepanciesResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="success",
                latency_ms=_elapsed_ms(start),
                data=discrepancies,
            )
        except Exception as exc:
            return DiscrepanciesResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="failure",
                latency_ms=_elapsed_ms(start),
                data=[],
                error=f"{type(exc).__name__}: {exc}",
            )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
