"""Report writers — produce the JSON + Markdown artifacts the grader reads."""

from __future__ import annotations

import json
from pathlib import Path

from backend.eval.gold import GoldSet
from backend.eval.scorer import EvalScores


def write_json_report(scores_runs: list[EvalScores], out_path: Path) -> None:
    """Serialize N=K runs as JSON for CI / programmatic consumers."""
    runs = [
        {
            "matched_gold_findings": s.matched_gold_findings,
            "matched_gold_citations": s.matched_gold_citations,
            "grounding_integrity_failures": s.grounding_integrity_failures,
            "cited_doc_in_scope_failures": s.cited_doc_in_scope_failures,
            "total_emitted_findings": s.total_emitted_findings,
            "total_emitted_citations": s.total_emitted_citations,
            "discrepancy_matches": s.discrepancy_matches,
            "citation_matches": s.citation_matches,
            "overall_recall": s.overall_recall,
        }
        for s in scores_runs
    ]
    out_path.write_text(json.dumps({"runs": runs}, indent=2))


def write_markdown_report(scores_runs: list[EvalScores], gold: GoldSet, out_path: Path) -> None:
    """Human-readable Markdown report. Shows per-gold-entry matches + the
    count gates so a reviewer can see what we caught and what we missed
    without running anything."""
    head = scores_runs[0]
    lines: list[str] = []
    lines.append("# BS Detector — Eval Report\n")
    lines.append(
        "Hand-labeled gold set in `backend/tests/fixtures/gold_set.yaml`. "
        "Matching is structural (kind + motion-span overlap + evidence_docs "
        "superset); see `STANDARDS.md` § 5.\n"
    )

    lines.append("## Count gates\n")
    lines.append("| Metric | Value | Gate | Pass? |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| matched_gold_findings | {head.matched_gold_findings} / "
        f"{head.total_gold_discrepancies} | ≥ 3 | "
        f"{'✅' if head.matched_gold_findings >= 3 else '❌'} |"
    )
    lines.append(
        f"| matched_gold_citations | {head.matched_gold_citations} / "
        f"{head.total_gold_citations} | ≥ 3 | "
        f"{'✅' if head.matched_gold_citations >= 3 else '❌'} |"
    )
    lines.append(
        f"| grounding_integrity_failures | {head.grounding_integrity_failures} | == 0 | "
        f"{'✅' if head.grounding_integrity_failures == 0 else '❌'} |"
    )
    lines.append(
        f"| cited_doc_in_scope_failures | {head.cited_doc_in_scope_failures} | == 0 | "
        f"{'✅' if head.cited_doc_in_scope_failures == 0 else '❌'} |"
    )
    lines.append("")

    lines.append("## Variance across N runs\n")
    if len(scores_runs) > 1:
        lines.append("| Run | matched_findings | matched_citations | overall_recall |")
        lines.append("|---|---|---|---|")
        for i, s in enumerate(scores_runs, start=1):
            lines.append(
                f"| {i} | {s.matched_gold_findings} | {s.matched_gold_citations} | "
                f"{s.overall_recall:.2f} |"
            )
    else:
        lines.append("(single run — variance reporting requires N≥2)")
    lines.append("")

    lines.append("## Per-gold-entry matches (first run)\n")
    lines.append("### Discrepancies\n")
    lines.append("| ID | Matched? | Severity | Motion quote (truncated) |")
    lines.append("|---|---|---|---|")
    for gd in gold.discrepancies:
        ok = head.discrepancy_matches.get(gd.id, False)
        lines.append(
            f"| `{gd.id}` | {'✅' if ok else '❌'} | {gd.severity} | "
            f"{gd.motion_quote[:80]}{'…' if len(gd.motion_quote) > 80 else ''} |"
        )
    lines.append("")

    lines.append("### Citations\n")
    lines.append("| ID | Matched? | Cited authority contains |")
    lines.append("|---|---|---|")
    for gc in gold.citations:
        ok = head.citation_matches.get(gc.id, False)
        lines.append(f"| `{gc.id}` | {'✅' if ok else '❌'} | {gc.cited_authority_contains} |")
    lines.append("")

    lines.append("## Emitted counts (first run)\n")
    lines.append(f"- Emitted findings: **{head.total_emitted_findings}**")
    lines.append(f"- Emitted citations: **{head.total_emitted_citations}**")
    lines.append("")

    lines.append("## What `grounding_integrity` proves (and does not)\n")
    lines.append(
        "It proves the quote the LLM emitted actually appears in a document we "
        "loaded. It does NOT prove the quote semantically supports the finding. "
        "Real semantic grounding would require an LLM-judge layer; the brief's "
        "6-hour budget pays for the deterministic check + honest unverifiable "
        "outcome instead. See `REFLECTION.md` for the gap.\n"
    )

    out_path.write_text("\n".join(lines))
