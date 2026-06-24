"""Prompt constants for ``QuoteChecker``.

The LLM is called ONLY when the cited authority's source text is in
``SourceRegistry`` — i.e., for in-corpus authority documents
(spec 002+ adds these). For external case law not in corpus, the agent
emits ``unverifiable`` without ever calling the LLM (STANDARDS § 3.5).
"""

from __future__ import annotations

PROMPT_VERSION = "1.0.0"

SYSTEM = """\
You compare a quoted string with a source span and judge fidelity only.

Rules:
- Decide between "paraphrase" (faithful rewording) and "altered" (material
  change that changes meaning, omits a qualifier, or adds words).
- Prefer "altered" when uncertain — false-flag (altered → paraphrase) is
  much worse than missing a real alteration for legal review.
- Reason in one short sentence; do not restate the strings.
- You are NOT deciding whether the authority supports the proposition.
  That is a different agent. Stay on quote fidelity.
"""

USER_TEMPLATE = """\
QUOTED TEXT (as the motion presents it)
{quoted_text}

SOURCE SPAN (verbatim from the authority document)
{source_text}

Verdict?
"""


def build_user_prompt(quoted_text: str, source_text: str) -> str:
    return USER_TEMPLATE.format(quoted_text=quoted_text, source_text=source_text)
