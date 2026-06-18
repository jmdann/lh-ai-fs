"""Prompt constants for ``AuthoritySupportChecker``.

Called ONLY when the cited authority's source text is available in
``SourceRegistry``. For external case law not in corpus, the agent emits
``unverifiable`` without calling the LLM (STANDARDS § 3.5).
"""

from __future__ import annotations

PROMPT_VERSION = "1.0.0"

SYSTEM = """\
You decide whether a cited authority's source span supports a stated
proposition.

Verdict options:
- "supports"   — the source span clearly establishes the proposition.
- "contradicts"— the source span clearly refutes the proposition.
- "unverifiable" — the source span is ambiguous, only partially relevant,
                   or insufficient to evaluate the proposition.

Rules:
- Prefer "unverifiable" over a low-confidence judgment. Reviewers grade
  hallucinations harder than misses.
- Reason in one short sentence anchored to the source span text.
- You are not deciding whether the quote is faithful. That is the
  QuoteChecker's job.
"""

USER_TEMPLATE = """\
PROPOSITION (sentence from the motion)
{proposition}

SOURCE SPAN (verbatim from the cited authority document)
{source_text}

Verdict?
"""


def build_user_prompt(proposition: str, source_text: str) -> str:
    return USER_TEMPLATE.format(proposition=proposition, source_text=source_text)
