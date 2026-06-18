# Eval suite

Hand-labeled gold set + structural matcher + count-based CI gates.

## Run

```bash
# Fake mode (default) — pre-canned responses; harness sanity check, not a real
# model evaluation. Useful in CI and to verify the pipeline + matcher are wired
# correctly without burning OpenAI credits.
python run_evals.py

# Real mode — requires OPENAI_API_KEY. Reports the actual model recall +
# grounding behavior on the four case-file documents.
python run_evals.py --mode real --runs 3
```

Default thresholds (count-based, STANDARDS § 5):

```
--min-matched-gold-findings   3    # gold has 4 discrepancies
--min-matched-gold-citations  3    # gold has 5 citations
--max-grounding-failures      0
--max-scope-failures          0
```

## Outputs

- `evals/eval_report.json` — machine-readable, one entry per run
- `evals/eval_report.md` — human review with per-gold-entry diff
- `evals/baseline_report.{json,md}` — committed real-run baseline

## What `grounding_integrity` actually proves

It proves the quote the LLM emitted **appears in a document we loaded**. It does
**not** prove the quote semantically supports the finding — that would require
an LLM-judge layer the 6-hour budget doesn't pay for. Real semantic grounding
is a known gap, documented in `REFLECTION.md`.

What it DOES prevent: the LLM cannot launder ungrounded text through the
orchestrator boundary. If it cites a quote not in the source, the finding is
dropped before it reaches the report.

## Why count gates, not percentages

With N=5-15 gold entries, percentages are theater:
- `recall ≥ 0.5` means "3 out of 6". You can hit that on luck.
- `matched_gold_findings ≥ 3` makes the floor explicit — and rises per PR.

Percentages are still reported in the Markdown for trend visibility across PRs.
