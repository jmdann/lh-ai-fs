# BS Detector — Eval Report

Hand-labeled gold set in `backend/tests/fixtures/gold_set.yaml`. Matching is structural (kind + motion-span overlap + evidence_docs superset); see `STANDARDS.md` § 5.

## Count gates

| Metric | Value | Gate | Pass? |
|---|---|---|---|
| matched_gold_findings | 4 / 4 | ≥ 3 | ✅ |
| matched_gold_citations | 5 / 5 | ≥ 3 | ✅ |
| grounding_integrity_failures | 0 | == 0 | ✅ |
| cited_doc_in_scope_failures | 0 | == 0 | ✅ |

## Variance across N runs

| Run | matched_findings | matched_citations | overall_recall |
|---|---|---|---|
| 1 | 4 | 5 | 1.00 |
| 2 | 4 | 5 | 1.00 |
| 3 | 4 | 5 | 1.00 |

## Per-gold-entry matches (first run)

### Discrepancies

| ID | Matched? | Severity | Motion quote (truncated) |
|---|---|---|---|
| `gold-disc-1` | ✅ | material | On or about March 14, 2021 |
| `gold-disc-2` | ✅ | dispositive | Rivera was not wearing required personal protective equipment |
| `gold-disc-3` | ✅ | dispositive | Apex Staffing Solutions — not Harmon — was the employer responsible for scaffold… |
| `gold-disc-4` | ✅ | material | The risks associated with working at height on scaffolding are inherent to his t… |

### Citations

| ID | Matched? | Cited authority contains |
|---|---|---|
| `gold-cite-1` | ✅ | Privette v. Superior Court |
| `gold-cite-2` | ✅ | Whitmore v. Delgado Scaffolding |
| `gold-cite-3` | ✅ | Kellerman v. Pacific Coast Construction |
| `gold-cite-4` | ✅ | Seabright Insurance |
| `gold-cite-5` | ✅ | Torres v. Granite Falls |

## Emitted counts (first run)

- Emitted findings: **4**
- Emitted citations: **5**

## What `grounding_integrity` proves (and does not)

It proves the quote the LLM emitted actually appears in a document we loaded. It does NOT prove the quote semantically supports the finding. Real semantic grounding would require an LLM-judge layer; the brief's 6-hour budget pays for the deterministic check + honest unverifiable outcome instead. See `REFLECTION.md` for the gap.
