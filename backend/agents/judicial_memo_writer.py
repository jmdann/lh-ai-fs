"""``JudicialMemoWriter`` — one-paragraph synthesis for a judge.

Text-out, but the input is structured (typed Findings). The
``validate_memo`` helper enforces two invariants post-write:

1. Every ``[find-N]`` reference in the memo body MUST appear in the
   findings list. Inventing ids is the obvious failure mode the brief
   would penalize hardest; the reference-check kills it deterministically.
2. Word count ≤ ``MAX_WORDS`` (180). The agent is instructed; the
   validator enforces.

Findings are ranked by ``(confidence x severity_weight)`` and the top K
are passed to the LLM. K defaults to 5; smaller K avoids the LLM
flattening the memo into generalities.
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel

from backend.agents.prompts.judicial_memo_writer import (
    MAX_WORDS,
    PROMPT_VERSION,
    SYSTEM,
    build_user_prompt,
)
from backend.llm.client import LLMClient
from backend.models import Finding, FindingKind, MemoResult

_REF_RE = re.compile(r"\[(find-\d+)\]")

_KIND_SEVERITY_FALLBACK: dict[FindingKind, float] = {
    FindingKind.FACT_DISCREPANCY: 0.8,
    FindingKind.QUOTE_FABRICATED: 0.9,
    FindingKind.QUOTE_ALTERED: 0.7,
    FindingKind.AUTHORITY_UNSUPPORTED: 0.85,
    FindingKind.AUTHORITY_UNVERIFIABLE: 0.3,
}


class _LLMMemo(BaseModel):
    memo: str


class JudicialMemoWriter:
    AGENT_NAME = "JudicialMemoWriter"
    PROMPT_VERSION = PROMPT_VERSION

    def __init__(self, llm: LLMClient, top_k: int = 5) -> None:
        self._llm = llm
        self._top_k = top_k

    async def run(self, case_name: str, findings: list[Finding]) -> MemoResult:
        start = time.perf_counter()
        try:
            if not findings:
                return MemoResult(
                    agent=self.AGENT_NAME,
                    prompt_version=self.PROMPT_VERSION,
                    outcome="success",
                    latency_ms=_elapsed_ms(start),
                    data="No material problems identified in the motion.",
                )
            top = _rank_findings(findings)[: self._top_k]
            finding_lines = [
                f"{f.id}  {f.kind.value}  conf={f.confidence if f.confidence is not None else '?'}  "
                f"{f.summary[:160]}"
                for f in top
            ]
            llm_out = await self._llm.complete(
                system=SYSTEM,
                user=build_user_prompt(case_name=case_name, finding_lines=finding_lines),
                schema=_LLMMemo,
            )
            memo = llm_out.memo.strip()
            valid_ids = {f.id for f in findings}
            issues = validate_memo(memo, valid_ids)
            if issues:
                return MemoResult(
                    agent=self.AGENT_NAME,
                    prompt_version=self.PROMPT_VERSION,
                    outcome="partial",
                    latency_ms=_elapsed_ms(start),
                    data=memo,
                    error="; ".join(issues),
                )
            return MemoResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="success",
                latency_ms=_elapsed_ms(start),
                data=memo,
            )
        except Exception as exc:
            return MemoResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="failure",
                latency_ms=_elapsed_ms(start),
                data=None,
                error=f"{type(exc).__name__}: {exc}",
            )


def validate_memo(memo: str, valid_finding_ids: set[str]) -> list[str]:
    """Return a list of validation issues. Empty list means the memo passes."""
    issues: list[str] = []
    words = len(memo.split())
    if words > MAX_WORDS:
        issues.append(f"memo over {MAX_WORDS}-word cap ({words} words)")
    cited = set(_REF_RE.findall(memo))
    unknown = cited - valid_finding_ids
    if unknown:
        issues.append(f"memo cites unknown finding ids: {sorted(unknown)}")
    return issues


def _rank_findings(findings: list[Finding]) -> list[Finding]:
    """confidence x severity-proxy. confidence defaults to 0.5 when None so
    pre-scored findings still get a deterministic order."""

    def score(f: Finding) -> float:
        conf = f.confidence if f.confidence is not None else 0.5
        sev = _KIND_SEVERITY_FALLBACK.get(f.kind, 0.5)
        return conf * sev

    return sorted(findings, key=score, reverse=True)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
