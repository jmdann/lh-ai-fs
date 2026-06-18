"""``ConfidenceScorer`` — rates each Finding 0-1 with a one-sentence reason.

The agent operates on typed Findings only. It cannot introduce new
findings because Pydantic validation would reject a Finding without an
id / kind / evidence — the LLM gets back a verdict-style schema, the
agent code merges it into the existing Finding via model_copy.

If a Finding lacks grounded evidence (which the orchestrator should
have dropped upstream), the agent returns 0.0 confidence with reasoning
naming the missing ground rather than calling the LLM. STANDARDS § 3.5
discipline carries: do not infer when the data isn't there.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from backend.agents.prompts.confidence_scorer import (
    PROMPT_VERSION,
    SYSTEM,
    build_user_prompt,
)
from backend.llm.client import LLMClient
from backend.models import ConfidenceResult, Finding


class _LLMVerdict(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


class ConfidenceScorer:
    AGENT_NAME = "ConfidenceScorer"
    PROMPT_VERSION = PROMPT_VERSION

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, findings: list[Finding]) -> ConfidenceResult:
        start = time.perf_counter()
        rescored: list[Finding] = []
        try:
            for finding in findings:
                rescored.append(await self._rescore(finding))
            return ConfidenceResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="success",
                latency_ms=_elapsed_ms(start),
                data=rescored,
            )
        except Exception as exc:
            return ConfidenceResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="failure",
                latency_ms=_elapsed_ms(start),
                data=rescored,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _rescore(self, finding: Finding) -> Finding:
        if not finding.evidence:
            return finding.model_copy(
                update={
                    "confidence": 0.0,
                    "confidence_reasoning": "Finding has no evidence; cannot anchor confidence.",
                }
            )

        evidence_lines = [
            f"role={ev.role}  doc_id={ev.span.doc_id}  quote={ev.span.quote[:120]!r}"
            for ev in finding.evidence
        ]
        llm_out = await self._llm.complete(
            system=SYSTEM,
            user=build_user_prompt(
                finding_id=finding.id,
                kind=finding.kind.value,
                summary=finding.summary,
                evidence_lines=evidence_lines,
            ),
            schema=_LLMVerdict,
        )
        return finding.model_copy(
            update={
                "confidence": llm_out.confidence,
                "confidence_reasoning": llm_out.reasoning,
            }
        )


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
