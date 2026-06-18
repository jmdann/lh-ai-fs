"""Prompt constants for ``CitationExtractor``.

PROMPT_VERSION is the load-bearing knob: bumping it forces every dependent
test snapshot to be regenerated. Treat the trio (SYSTEM, USER_TEMPLATE,
EXAMPLE) as one atomic surface — changing any of them is a version bump.

See STANDARDS § 3.4 for the prompt-shape rules: ≤ 40 lines, role + I/O
contract + refusal rule + one worked example, schema enforced via
``response_format`` at the API edge.
"""

from __future__ import annotations

PROMPT_VERSION = "1.0.0"

SYSTEM = """\
You map legal citations to the exact sentence in the motion they support.

Rules:
- Use ONLY the authorities listed under CANDIDATES. Do not invent new ones.
- For each candidate, find the sentence in the motion that cites it and
  copy that sentence verbatim into ``proposition.quote``. If the citation
  appears mid-sentence, return the whole sentence containing the citation.
- ``proposition.doc_id`` must be exactly "motion".
- ``quoted_text`` is the direct quotation attributed to the authority,
  if any. If the motion does not quote the authority, set it to null.
- If a candidate is mentioned but its supporting sentence is not clearly
  identifiable, emit it with proposition.quote set to the most likely
  surrounding sentence anyway — never invent text.
- Output exactly one citation entry per candidate, in the order given.
"""

USER_TEMPLATE = """\
MOTION TEXT
===========
{motion_text}

CANDIDATES
==========
{candidates}

Return one citation entry per candidate, in order.
"""

EXAMPLE = """\
EXAMPLE INPUT (excerpt)
-----------------------
MOTION TEXT
The defendant relies on Smith v. Jones, 1 Cal.4th 1 (2000), which held \
"contractors owe no duty to subcontractor employees."

CANDIDATES
1. Smith v. Jones, 1 Cal.4th 1 (2000)

EXAMPLE OUTPUT
--------------
{
  "citations": [
    {
      "cited_authority": "Smith v. Jones, 1 Cal.4th 1 (2000)",
      "proposition": {
        "doc_id": "motion",
        "quote": "The defendant relies on Smith v. Jones, 1 Cal.4th 1 \
(2000), which held \\"contractors owe no duty to subcontractor \
employees.\\""
      },
      "quoted_text": "contractors owe no duty to subcontractor employees"
    }
  ]
}
"""


def build_user_prompt(motion_text: str, candidates: list[str]) -> str:
    """Format the user message with the motion + a numbered list of candidates."""
    enumerated = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates, start=1))
    return USER_TEMPLATE.format(motion_text=motion_text, candidates=enumerated)
