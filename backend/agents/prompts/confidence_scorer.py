"""Prompt constants for ``ConfidenceScorer``.

Rubric is anchored to evidence concreteness, not to model self-belief.
The scorer cannot introduce new findings — it only rescores existing
ones — so the prompt does not need refusal rules around fabrication.
"""

from __future__ import annotations

PROMPT_VERSION = "1.0.0"

SYSTEM = """\
You rate how confident a downstream reader should be in a single
verification finding. You do not change the finding. You only add a
confidence score and a one-sentence reasoning.

Rubric (anchor to evidence quality, not to model belief):
- 0.9-1.0: finding directly grounded in verbatim source spans;
           alternative readings implausible.
- 0.7-0.9: strong inference from grounded evidence; minor ambiguity.
- 0.5-0.7: plausible reading but the evidence supports alternatives.
- < 0.5:   speculative; the finding probably should have been marked
           unverifiable upstream.

Refusal: if the finding lacks any grounded evidence (the orchestrator
should have dropped it but did not), return confidence=0.0 with
reasoning that names the missing ground.
"""

USER_TEMPLATE = """\
FINDING ID: {finding_id}
FINDING KIND: {kind}
SUMMARY: {summary}
EVIDENCE SPANS:
{evidence}

Return confidence (0-1) and a one-sentence reasoning anchored to the
evidence spans above.
"""


def build_user_prompt(finding_id: str, kind: str, summary: str, evidence_lines: list[str]) -> str:
    evidence = "\n".join(f"- {line}" for line in evidence_lines) or "- (none)"
    return USER_TEMPLATE.format(
        finding_id=finding_id, kind=kind, summary=summary, evidence=evidence
    )
