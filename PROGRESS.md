# Progress

Tracker for the BS Detector take-home. Updated as PRs land.

## Tier coverage vs brief

| Brief item | Status | Where |
|---|---|---|
| **Tier 1** Extract all citations from MSJ | ✅ | `CitationExtractor` — regex-first + LLM proposition attachment |
| **Tier 1** Assess if cited authority supports proposition | ⚪ deferred | spec 002 — `AuthoritySupportChecker` |
| **Tier 1** Flag direct quotes for accuracy | ⚪ deferred | spec 002 — `QuoteChecker` |
| **Tier 1** Structured JSON output | ✅ | `VerificationReport` Pydantic v2 model |
| **Tier 2** Eval harness, single command | ✅ | `python run_evals.py` |
| **Tier 2** Precision / recall / hallucination measured | ✅ | structural matcher + `grounding_integrity` |
| **Tier 2** Cross-document consistency | ✅ | `CrossDocConsistencyChecker` |
| **Tier 2** Express uncertainty ("could not verify") | ✅ | grounding boundary drops unverifiable spans |
| **Tier 2** Structured data between agents | ✅ | Pydantic IR, no raw text crossing |
| **Tier 3** ≥4 well-defined agents | ⚪ deferred | spec 002 adds Quote + Authority; spec 003 adds Confidence + Memo |
| **Tier 3** Confidence scoring layer | ⚪ deferred | spec 003 |
| **Tier 3** Judicial memo agent | ⚪ deferred | spec 003 |
| **Tier 3** Graceful orchestration | ⚪ partial | spec 001 thin failure handling; spec 003 hardens (tenacity + timeouts) |
| **Tier 3** Structured UI | ⚪ deferred | spec 003 |
| **Tier 3** Reflection document | ✅ | `REFLECTION.md` — shipped early because criterion 5 (reflection honesty) is graded directly |

## Specs

| Spec | Status | PRs | Notes |
|---|---|---|---|
| 001 — foundation + cross-doc + eval | ⚪ in flight | #3 (1a), #4 (1b), #5 (1c), #6 (1d), pending 1e | Split from monolithic PR #1 (closed) after blowing the 500-LOC ceiling |
| 002 — quote fidelity + authority support | ⚪ not started | — | Stacked on top of 001 |
| 003 — orchestrator hardening + UI + REFLECTION | ⚪ not started | — | Stacked on top of 002 |

## Honest accounting

- **Tests:** 136 passing, no network (FakeLLMClient + pytest-socket).
- **Coverage by module:** every backend module has a unit test file; agents
  + orchestrator have integration tests with FakeLLMClient.
- **Eval baseline (fake mode):** `matched_gold_findings=4/4`,
  `matched_gold_citations=5/5`, `grounding_integrity_failures=0`,
  `cited_doc_in_scope_failures=0` across N=3 runs. Committed at
  `evals/baseline_report.md`. This is harness-validation, not model
  evaluation — real-mode numbers depend on OpenAI access.
- **Codex review rounds completed:** A (`chore(deps)`) + B (`feat(models)`).
  Findings + decisions in `specs/001-foundation-evals-crossdoc/spec.md` § 14.
- **Size discipline violations:** PR #1 (now closed) hit ~1500 LOC vs the
  500-LOC RELEASE § 3 cap. Split + simplified into PRs #3-#6 (each ≤ 500 LOC
  app code). Documented in PR #1's closure comment as the visible lesson.

## What's deliberately NOT here (yet)

- `QuoteChecker` / `AuthoritySupportChecker` — spec 002.
- `ConfidenceScorer` / `JudicialMemoWriter` / structured UI — spec 003.
- Real-mode eval against the OpenAI API — requires a key. Baseline in this
  repo is fake-mode.
