"""Prompt constants for ``JudicialMemoWriter``.

The memo is text-out, but the input is structured (typed Findings). The
post-write reference-check rejects memos that cite finding ids that
don't exist — that's the deterministic gate against the agent inventing
material the report doesn't actually support.
"""

from __future__ import annotations

PROMPT_VERSION = "1.0.0"

MAX_WORDS = 180

SYSTEM = f"""\
You write ONE paragraph for a judge summarizing the most material
problems with a Motion for Summary Judgment, anchored to the findings
listed below.

Rules:
- Single paragraph. Maximum {MAX_WORDS} words. Sentences may be long.
- Cite findings by id using square brackets, e.g. [find-3]. Every
  bracketed id MUST appear in the FINDINGS list below; do not invent
  ids.
- Do NOT introduce new factual claims, new authorities, or new
  contradictions. You synthesize from the FINDINGS list; you do not
  extend it.
- Do NOT give legal advice. Do NOT recommend a ruling. Describe what
  the motion misrepresents and how, judicially neutral.
- If FINDINGS is empty, the memo is exactly: "No material problems
  identified in the motion." (without brackets).
"""

USER_TEMPLATE = """\
CASE: {case_name}

FINDINGS (id, kind, severity-or-verdict, one-line summary)
{findings}

Write the memo.
"""


def build_user_prompt(case_name: str, finding_lines: list[str]) -> str:
    findings = "\n".join(f"- {line}" for line in finding_lines) or "(none)"
    return USER_TEMPLATE.format(case_name=case_name, findings=findings)
