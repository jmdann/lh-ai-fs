"""Prompt constants for ``CrossDocConsistencyChecker``.

Same atomic-trio rule as the other prompt modules: SYSTEM + USER_TEMPLATE +
EXAMPLE are one surface; bumping PROMPT_VERSION forces snapshot
regeneration. STANDARDS § 3.4 caps the system prompt at 40 lines.
"""

from __future__ import annotations

PROMPT_VERSION = "1.0.0"

SYSTEM = """\
You compare factual statements in a legal motion against record evidence.

Rules:
- Emit a discrepancy ONLY when a record document directly contradicts a
  factual statement in the motion. Do not flag absences ("the police
  report does not mention X" is silence, not contradiction).
- Do not flag legal arguments, characterizations, or opinions — only
  concrete factual claims (who, what, when, where, how many).
- ``motion_claim.doc_id`` must be exactly "motion".
- Each ``contradicting_evidence`` span's ``doc_id`` must be the id of a
  record document supplied below (e.g. "police_report",
  "medical_records_excerpt", "witness_statement").
- ``quote`` fields are verbatim text from the source documents. Do not
  paraphrase, do not invent.
- ``description`` is one sentence naming the disagreement.
- ``severity``:
  * "minor"      — peripheral detail; would not change the analysis.
  * "material"   — could affect the legal analysis or summary-judgment ruling.
  * "dispositive"— would directly change the outcome.

If no contradictions exist, return an empty list.
"""

USER_TEMPLATE = """\
MOTION (id: {motion_id})
{motion_divider}
{motion_text}

{records_block}

Return all factual discrepancies between the motion and the record
documents above. Use the exact doc ids shown in the headers.
"""

EXAMPLE = """\
EXAMPLE
=======

MOTION (id: motion)
"The plaintiff was not wearing a safety harness at the time of the fall."

RECORD: POLICE REPORT (id: police_report)
"Officer Reyes observed plaintiff wearing a properly secured safety
harness when first responders arrived."

OUTPUT
{
  "discrepancies": [
    {
      "motion_claim": {
        "doc_id": "motion",
        "quote": "The plaintiff was not wearing a safety harness at the time of the fall."
      },
      "contradicting_evidence": [
        {
          "doc_id": "police_report",
          "quote": "Officer Reyes observed plaintiff wearing a properly secured safety harness when first responders arrived."
        }
      ],
      "description": "Motion claims plaintiff was unharnessed; police report records harness as properly secured.",
      "severity": "material"
    }
  ]
}
"""


def build_user_prompt(motion_id: str, motion_text: str, records: list[tuple[str, str, str]]) -> str:
    """Format the user message.

    ``records`` is a list of ``(doc_id, display_kind, text)`` tuples — e.g.
    ``("police_report", "POLICE REPORT", "<text>")``. Each appears under a
    clearly labeled header so the LLM can ground each ``doc_id`` correctly.
    """
    motion_divider = "=" * (len(f"MOTION (id: {motion_id})"))
    sections = []
    for doc_id, display_kind, text in records:
        header = f"RECORD: {display_kind} (id: {doc_id})"
        sections.append(f"{header}\n{'=' * len(header)}\n{text}")
    return USER_TEMPLATE.format(
        motion_id=motion_id,
        motion_divider=motion_divider,
        motion_text=motion_text,
        records_block="\n\n".join(sections),
    )
