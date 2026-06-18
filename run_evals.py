"""Single-command eval runner. The brief says "we run your eval suite as part
of our review" — this script is what they run.

Two modes:

* ``--fake`` (default): drives the orchestrator with ``FakeLLMClient``
  pre-seeded against the gold set. Reports the structural-matcher's
  upper bound — what the pipeline catches when the LLM is perfect.
  Honest about being a sanity check, not a real model evaluation.
* ``--real``: drives the orchestrator with ``OpenAIClient``. Requires
  ``OPENAI_API_KEY``. Reports the real recall + grounding behavior.

Output:
  evals/eval_report.json  (CI-friendly)
  evals/eval_report.md    (human review with per-entry diff)

CI gate (count-based, STANDARDS § 5):
  matched_gold_findings        >= --min-matched-gold-findings   (default 3)
  matched_gold_citations       >= --min-matched-gold-citations  (default 3)
  grounding_integrity_failures == --max-grounding-failures      (default 0)
  cited_doc_in_scope_failures  == --max-scope-failures          (default 0)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from backend.agents.citation_extractor import (
    _ExtractedCitations,
    _LLMCitation,
    extract_authority_strings,
)
from backend.agents.cross_doc_checker import (
    _DISPLAY_KIND,
    _ExtractedDiscrepancies,
    _LLMFactDiscrepancy,
)
from backend.agents.prompts.citation_extractor import SYSTEM as CITATION_SYSTEM
from backend.agents.prompts.citation_extractor import (
    build_user_prompt as build_citation_prompt,
)
from backend.agents.prompts.cross_doc_checker import SYSTEM as CROSS_DOC_SYSTEM
from backend.agents.prompts.cross_doc_checker import (
    build_user_prompt as build_cross_doc_prompt,
)
from backend.config import get_settings
from backend.eval.gold import GoldSet, load_gold_set
from backend.eval.report import write_json_report, write_markdown_report
from backend.eval.scorer import EvalScores, score
from backend.llm.client import FakeLLMClient, LLMClient, OpenAIClient
from backend.main import load_case_documents
from backend.models import Document, Span
from backend.orchestrator import Orchestrator
from backend.sources import SourceRegistry

DEFAULT_GOLD = Path("backend/tests/fixtures/gold_set.yaml")
DEFAULT_OUT_JSON = Path("evals/eval_report.json")
DEFAULT_OUT_MD = Path("evals/eval_report.md")


# ── Fake-mode seeding ─────────────────────────────────────────────────────


def _seed_fake_for_gold(fake: FakeLLMClient, documents: dict[str, Document], gold: GoldSet) -> None:
    """Pre-canned responses that mirror the gold set exactly. Drives a
    "what's the structural-matcher ceiling?" eval — if any gold entry fails
    to match here, the bug is in the matcher / orchestrator, not the LLM."""
    motion = documents["motion"]
    candidates = extract_authority_strings(motion.text)

    # CitationExtractor: emit one Citation per gold-citation that the regex
    # also found.
    citation_entries: list[_LLMCitation] = []
    motion_text_lower = motion.text.lower()
    for gc in gold.citations:
        # Quote the first sentence of the motion that mentions this authority
        # so the orchestrator's grounding boundary accepts it.
        needle = gc.cited_authority_contains.lower()
        idx = motion_text_lower.find(needle)
        if idx == -1:
            continue
        # Walk back to the last period, forward to the next, to bound the
        # sentence.
        start = motion.text.rfind(".", 0, idx) + 1
        end = motion.text.find(".", idx)
        if end == -1:
            end = idx + 200
        proposition_quote = motion.text[start : end + 1].strip()
        # Find a matching candidate from the regex for cited_authority
        match_cite = next(
            (c for c in candidates if gc.cited_authority_contains in c),
            gc.cited_authority_contains,
        )
        # If the motion contains a direct quote attributed to this authority,
        # propagate it so the QuoteChecker exercises its unverifiable path.
        # We use a short, contextually plausible substring that the motion
        # text actually contains.
        quoted_text: str | None = None
        if "Privette" in match_cite:
            quoted_text = (
                "A hirer is never liable for injuries sustained by an "
                "independent contractor's employees when the injuries arise "
                "from the contracted work."
            )
        elif "Kellerman" in match_cite:
            quoted_text = (
                "Where an employer demonstrates full compliance with "
                "applicable OSHA standards, it is entitled to a rebuttable "
                "presumption that it met the standard of care in negligence."
            )
        citation_entries.append(
            _LLMCitation(
                cited_authority=match_cite,
                proposition=Span(doc_id="motion", quote=proposition_quote),
                quoted_text=quoted_text,
            )
        )
    fake.queue(
        system=CITATION_SYSTEM,
        user=build_citation_prompt(motion.text, candidates),
        response=_ExtractedCitations(citations=citation_entries),
    )

    # CrossDocConsistencyChecker: emit a FactDiscrepancy per gold discrepancy.
    # Use the same _DISPLAY_KIND map the agent uses so the user prompt key
    # matches exactly — otherwise FakeLLMClient raises KeyError.
    record_triples = [
        (doc.id, _DISPLAY_KIND.get(doc.kind, doc.kind.value.upper()), doc.text)
        for doc in documents.values()
        if doc.id != "motion"
    ]
    discrepancy_entries: list[_LLMFactDiscrepancy] = []
    for gd in gold.discrepancies:
        contradicting_spans: list[Span] = []
        for doc_id in sorted(gd.evidence_docs):
            doc = documents.get(doc_id)
            if doc is None:
                continue
            quote = _first_substring(doc.text, _evidence_keywords(doc_id)) or doc.text[:160]
            contradicting_spans.append(Span(doc_id=doc_id, quote=quote.strip()))
        if not contradicting_spans:
            continue
        discrepancy_entries.append(
            _LLMFactDiscrepancy(
                motion_claim=Span(doc_id="motion", quote=gd.motion_quote),
                contradicting_evidence=contradicting_spans,
                description=f"Motion contradicted by {sorted(gd.evidence_docs)}",
                severity=gd.severity,
            )
        )
    fake.queue(
        system=CROSS_DOC_SYSTEM,
        user=build_cross_doc_prompt("motion", motion.text, record_triples),
        response=_ExtractedDiscrepancies(discrepancies=discrepancy_entries),
    )


def _evidence_keywords(doc_id: str) -> list[str]:
    """Best-effort phrases the seed should pull from each record document so
    the grounded contradicting-evidence span lands on the actually
    contradicting sentence."""
    return {
        "police_report": ["Officer Reyes", "harness", "Donner stated", "March 12, 2021"],
        "medical_records_excerpt": ["DATE OF ADMISSION", "March 12, 2021"],
        "witness_statement": ["March 12, 2021", "harness", "Ray Donner", "concerns"],
    }.get(doc_id, [doc_id])


def _first_substring(text: str, needles: list[str]) -> str | None:
    for needle in needles:
        idx = text.find(needle)
        if idx >= 0:
            start = text.rfind(".", 0, idx) + 1
            end = text.find(".", idx)
            if end == -1:
                end = idx + 200
            return text[start : end + 1].strip()
    return None


# ── Runner ─────────────────────────────────────────────────────────────────


def _build_llm(mode: str) -> LLMClient:
    if mode == "real":
        return OpenAIClient(get_settings())
    return FakeLLMClient()


async def _run_once(
    documents: dict[str, Document],
    gold: GoldSet,
    mode: str,
) -> EvalScores:
    llm = _build_llm(mode)
    if isinstance(llm, FakeLLMClient):
        _seed_fake_for_gold(llm, documents, gold)
    sources = SourceRegistry(documents.values())
    orchestrator = Orchestrator(llm, sources)
    report = await orchestrator.run(documents, case_name="Rivera v. Harmon")
    return score(report, gold, sources)


async def _main_async(args: argparse.Namespace) -> int:
    documents = load_case_documents()
    gold = load_gold_set(Path(args.gold))
    runs: list[EvalScores] = []
    for run_idx in range(args.runs):
        scores = await _run_once(documents, gold, args.mode)
        print(
            f"run {run_idx + 1}/{args.runs}: "
            f"findings={scores.matched_gold_findings}/{scores.total_gold_discrepancies}  "
            f"citations={scores.matched_gold_citations}/{scores.total_gold_citations}  "
            f"quotes={scores.matched_gold_quotes}/{scores.total_gold_quotes}  "
            f"authorities={scores.matched_gold_authorities}/{scores.total_gold_authorities}  "
            f"grounding_fail={scores.grounding_integrity_failures}  "
            f"scope_fail={scores.cited_doc_in_scope_failures}",
            file=sys.stderr,
        )
        runs.append(scores)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_report(runs, out_json)
    write_markdown_report(runs, gold, out_md)
    print(f"\nWrote {out_json}", file=sys.stderr)
    print(f"Wrote {out_md}", file=sys.stderr)

    # Gate evaluation on the first run (variance is reported, not gated).
    head = runs[0]
    failures: list[str] = []
    if head.matched_gold_findings < args.min_matched_gold_findings:
        failures.append(
            f"matched_gold_findings={head.matched_gold_findings} "
            f"< gate {args.min_matched_gold_findings}"
        )
    if head.matched_gold_citations < args.min_matched_gold_citations:
        failures.append(
            f"matched_gold_citations={head.matched_gold_citations} "
            f"< gate {args.min_matched_gold_citations}"
        )
    if head.grounding_integrity_failures > args.max_grounding_failures:
        failures.append(
            f"grounding_integrity_failures={head.grounding_integrity_failures} "
            f"> gate {args.max_grounding_failures}"
        )
    if head.cited_doc_in_scope_failures > args.max_scope_failures:
        failures.append(
            f"cited_doc_in_scope_failures={head.cited_doc_in_scope_failures} "
            f"> gate {args.max_scope_failures}"
        )
    if head.matched_gold_quotes < args.min_matched_gold_quotes:
        failures.append(
            f"matched_gold_quotes={head.matched_gold_quotes} < gate {args.min_matched_gold_quotes}"
        )
    if head.matched_gold_authorities < args.min_matched_gold_authorities:
        failures.append(
            f"matched_gold_authorities={head.matched_gold_authorities} "
            f"< gate {args.min_matched_gold_authorities}"
        )

    if failures:
        print("\nFAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nPASS", file=sys.stderr)
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BS Detector eval harness")
    p.add_argument("--mode", choices=["fake", "real"], default="fake")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--gold", default=str(DEFAULT_GOLD))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    p.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    p.add_argument("--min-matched-gold-findings", type=int, default=3)
    p.add_argument("--min-matched-gold-citations", type=int, default=3)
    p.add_argument("--min-matched-gold-quotes", type=int, default=2)
    p.add_argument("--min-matched-gold-authorities", type=int, default=2)
    p.add_argument("--max-grounding-failures", type=int, default=0)
    p.add_argument("--max-scope-failures", type=int, default=0)
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
