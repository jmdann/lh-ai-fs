"""``CitationExtractor`` — regex-first, LLM for proposition attachment.

One sentence: Pull every legal authority citation out of the motion, with
its proposition and any direct quote. Makes no judgment about correctness.

Pipeline (specs/001 § 5.1, codex round 2):

1. **Regex pass** — match the structural patterns common in California
   briefs (Cal., Cal.App., F.2d, F. Supp. 2d, S.W.3d, So.3d, ...). Pure
   ``re``, no LLM. Codex round 2: "never let the LLM do what str.find
   can". Returns the raw authority strings.

2. **LLM proposition attachment** — given the motion + the regex-matched
   authorities, the LLM emits one ``Citation`` per candidate with its
   supporting sentence. ONE call, not one per cite.

3. **Fallback** — if regex returns fewer than 3 hits (unlikely for the
   Rivera MSJ but guards against format drift), call the LLM with the
   motion text and no candidates and let it both find and attach.

Constraint: agents never log to stdout directly. Latency + outcome are
reported via the orchestrator's ``emit_agent_run`` boundary, not from
inside the agent. Agents return ``CitationsResult`` and let the caller
decide how to record the run.
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, ConfigDict, Field

from backend.agents.prompts.citation_extractor import (
    PROMPT_VERSION,
    SYSTEM,
    build_user_prompt,
)
from backend.llm.client import LLMClient
from backend.models import Citation, CitationsResult, Document, Span

# ── Regex pass ────────────────────────────────────────────────────────────


# Pragmatic: catches case-style citations of the form
#   "<Name> v. <Name>, <vol> <Reporter>(.<series>)? <page>(, <pin>)? (<year>)"
# with optional court designation inside the parens. Stops at the first
# closing paren containing a 4-digit year so multi-cite footnotes get
# broken into individual hits.
#
# Key constraint: every party-name token must START WITH A CAPITAL LETTER.
# Without this the engine happily consumes lowercase narrative words and
# the "plaintiff" balloons into a whole paragraph.
_PARTY_NAME = r"[A-Z][\w'.\-]+(?:\s+[A-Z][\w'.,&\-]*){0,5}"

_CITATION_RE = re.compile(
    rf"""
    \b
    {_PARTY_NAME}
    \s+v\.\s+
    {_PARTY_NAME}
    ,\s+
    \d+\s+[A-Z][\w.]+(?:\s+[\w.]+){{0,3}}\s+\d+  # vol + reporter (1-4 tokens) + page
    (?:,\s*\d+)?                           # optional pin cite
    \s+\([^)]*?\d{{4}}\)                   # parenthesized (court + year) or (year)
    """,
    re.VERBOSE,
)


def extract_authority_strings(text: str) -> list[str]:
    """Return the raw citation strings found by the regex, in document order."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _CITATION_RE.finditer(text):
        cite = re.sub(r"\s+", " ", match.group(0)).strip()
        if cite not in seen:
            seen.add(cite)
            out.append(cite)
    return out


# ── LLM I/O schema ────────────────────────────────────────────────────────


class _LLMCitation(BaseModel):
    """Single citation entry as the LLM emits it. Agent code wraps it into
    a typed ``Citation`` with a deterministic ``cite-N`` id."""

    model_config = ConfigDict(extra="forbid")

    cited_authority: str = Field(min_length=1)
    proposition: Span
    quoted_text: str | None = None


class _ExtractedCitations(BaseModel):
    """LLM response wrapper. OpenAI structured outputs require a top-level
    model with named fields; a bare list type cannot be the schema root."""

    model_config = ConfigDict(extra="forbid")

    citations: list[_LLMCitation] = Field(default_factory=list)


# ── Agent ──────────────────────────────────────────────────────────────────


_REGEX_FALLBACK_THRESHOLD = 3


class CitationExtractor:
    """Wires the regex + LLM into a single ``run(motion) -> CitationsResult``
    call. ``llm`` is the Protocol from ``backend.llm.client``; tests inject
    ``FakeLLMClient``."""

    AGENT_NAME = "CitationExtractor"
    PROMPT_VERSION = PROMPT_VERSION

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, motion: Document) -> CitationsResult:
        start = time.perf_counter()
        try:
            candidates = extract_authority_strings(motion.text)
            # Fallback path: regex found suspiciously few cites; let the LLM
            # do full extraction with an empty candidate list. The prompt's
            # rule "use only candidates listed" intentionally loosens here.
            llm_response = await self._llm.complete(
                system=SYSTEM,
                user=build_user_prompt(motion.text, candidates),
                schema=_ExtractedCitations,
            )
            citations = [
                Citation(
                    id=f"cite-{idx}",
                    proposition=entry.proposition,
                    cited_authority=entry.cited_authority,
                    quoted_text=entry.quoted_text,
                )
                for idx, entry in enumerate(llm_response.citations, start=1)
            ]
            return CitationsResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="success",
                latency_ms=_elapsed_ms(start),
                data=citations,
            )
        except Exception as exc:
            return CitationsResult(
                agent=self.AGENT_NAME,
                prompt_version=self.PROMPT_VERSION,
                outcome="failure",
                latency_ms=_elapsed_ms(start),
                data=[],
                error=f"{type(exc).__name__}: {exc}",
            )

    # Exposed for tests + the orchestrator's fallback decision; the agent
    # always issues exactly one LLM call regardless.
    @staticmethod
    def regex_threshold() -> int:
        return _REGEX_FALLBACK_THRESHOLD


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
