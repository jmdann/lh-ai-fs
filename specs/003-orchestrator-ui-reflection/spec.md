# Spec 003 — Confidence + Memo + UI + Reflection

**Status:** in flight (PR #12 shipped agents; this spec section closes the UI piece).
**Stack target:** stacked on top of spec 002.
**Standards:** [STANDARDS.md](../../STANDARDS.md)

## 1. Goal

The polish layer. Earns Tier 3 of the brief: ≥4 agents with non-overlapping roles, confidence scoring per finding, judicial memo, structured UI, and the reflection document.

**Status of items in this spec:**
- ✅ `REFLECTION.md` — shipped early in PR #9 (criterion 5 graded directly)
- ✅ `AI_WORKFLOW.md` — shipped in PR #10
- ✅ `ConfidenceScorer` + `JudicialMemoWriter` — shipped in PR #12
- ⚪ Frontend rewrite (6 components) — this PR
- ❌ Orchestrator hardening (tenacity retries + per-agent timeouts + rich failure isolation) — **CUT, see § 11**
- ❌ Confidence calibration plot (Brier score) — **CUT, see § 11**

## 2. Non-goals

- Web-based case-law retrieval (the obvious gap, addressed in REFLECTION).
- Multi-annotator gold set (REFLECTION).
- Prompt-injection robustness in the eval (REFLECTION).
- Streaming responses to the UI — too much surface for the time left.
- Orchestrator hardening (tenacity / timeouts) — cut, see § 11.
- Confidence calibration plot — cut, see § 11.

## 3. Deliverables

1. `backend/agents/confidence_scorer.py` — 0-1 score + reasoning per finding.
2. `backend/agents/judicial_memo_writer.py` — one-paragraph synthesis for a judge.
3. `backend/agents/prompts/{confidence_scorer,judicial_memo_writer}.py` — prompts with `PROMPT_VERSION`.
4. `backend/orchestrator.py` updates — per-agent failure isolation + `tenacity` retries on the LLM client + timeout per agent.
5. `backend/models.py` updates — `Finding.confidence: float | None`, `Finding.confidence_reasoning: str | None`, `VerificationReport.judicial_memo: str | None`.
6. `frontend/src/App.jsx` rewrite — structured cards instead of raw JSON `<pre>`.
7. `frontend/src/components/` — `ReportSummary`, `MemoCard`, `FindingsList`, `FindingCard`, `CitationsTable`, `AgentTrace`. Minimal CSS, no UI framework (24KB budget — don't pull React-anything).
8. `tests/` — confidence and memo agents, orchestrator failure-isolation, frontend component tests with `vitest`.
9. `evals/baseline_report.md` — refreshed; confidence calibration plot if time permits (Brier score vs gold).
10. `REFLECTION.md` — the document that earns evaluation point 5.
11. `PROGRESS.md` — final state.
12. `README.md` — Status section flipped to all-green; "How to run the eval suite" finalized.

## 4. Domain model deltas

```python
# backend/models.py

class Finding(BaseModel):
    # ...previous fields...
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_reasoning: str | None = None

class VerificationReport(BaseModel):
    # ...previous fields...
    judicial_memo: str | None = None
```

`Finding.confidence` is optional so PR 002's findings still validate (forward-compat). When `ConfidenceScorer` runs, it populates both fields; the validator requires `confidence_reasoning` to be present when `confidence` is.

## 5. Agents (this spec)

### 5.1 ConfidenceScorer

**One sentence:** Given a finding and its evidence, rate confidence 0-1 with a one-sentence reason. Does not generate new findings.

**Signature:**
```python
class ConfidenceScorer:
    def __init__(self, llm: LLMClient) -> None: ...
    def run(self, finding: Finding, supporting_evidence: list[Span]) -> AgentResult[Finding]: ...
```

**Prompt rules:**
- Role: "Rate how confident a downstream reader should be in this finding. You do not change the finding; you only add confidence."
- Output: JSON schema = `{confidence: float, reasoning: str}` (merged into the Finding by the agent code).
- Rubric in the prompt:
  - **0.9-1.0** — finding directly grounded in verbatim source text; alternative interpretations implausible.
  - **0.7-0.9** — strong inference from grounded evidence; minor ambiguity.
  - **0.5-0.7** — plausible reading but evidence supports alternatives.
  - **< 0.5** — speculative; should probably have been marked `unverifiable` upstream.
- Refusal: if the finding lacks any grounded evidence, return `confidence: 0.0` with reasoning naming the missing ground.

**Why we run it on `Finding` not raw text:** the scorer can't introduce hallucinations because it operates over Pydantic-validated, span-grounded inputs. Worst case it gives a bad number; it can't fabricate a finding.

### 5.2 JudicialMemoWriter

**One sentence:** Given the top-K findings ranked by `confidence × severity`, write a one-paragraph memo written for a judge.

**Signature:**
```python
class JudicialMemoWriter:
    def __init__(self, llm: LLMClient) -> None: ...
    def run(self, top_findings: list[Finding], max_words: int = 180) -> AgentResult[str]: ...
```

**Prompt rules:**
- Role: "Write a single paragraph for a busy judge summarizing the most material problems with the Motion for Summary Judgment. Be precise. Cite findings by id. No legal advice."
- Output: JSON schema = `{memo: str}`.
- Hard constraint: max 180 words. Prompt enforces; eval truncates with a warning if violated.
- Refusal: if `top_findings` is empty, return a memo stating "No material problems identified" — do not invent.

This agent is text-out, not structured-out — the only one in the pipeline. That's fine because the *input* is structured (validated Findings); the LLM can't smuggle in new claims because anything not traceable to a Finding id is dropped by a post-write validator.

## 6. Orchestrator hardening — **CUT**

Originally promised: tenacity retries with exponential backoff, per-agent
`asyncio.wait_for` timeout, structured log line per agent. **Cut** because:

- Each of the six agents already wraps its work in `try/except Exception`
  that maps any failure to `AgentResult(outcome="failure", error=...)`,
  so the orchestrator's contract — "every invocation produces a
  `VerificationReport`" — already holds.
- Tenacity retries against the OpenAI API are configured at the
  `OpenAIClient` layer (see `backend/llm/client.py`), so retry logic is
  already in place for the LLM call, just not at the agent boundary.
- Per-agent timeouts would prevent a slow LLM from hanging a request,
  but on the four-document case file the real model rarely takes > 5s
  per agent. The risk/reward for adding `wait_for` here is low in the
  6-hour budget vs. shipping the frontend cards the brief explicitly
  grades.

The 12-factor XI structured log line is implemented at the
`backend.observability.log_agent_run` helper (PR #4) but not wired into
the orchestrator. Wiring it would add ~10 LOC; deferred as part of this
cut because the eval harness reads `agent_results.latency_ms` from the
report instead of grepping logs.

## 7. Frontend

`frontend/src/App.jsx` becomes a thin shell. Components:

- `<ReportSummary>` — case name, counts (citations, findings by kind), confidence histogram.
- `<MemoCard>` — top of page, judicial memo paragraph.
- `<FindingsList>` — grouped by kind (fact_discrepancy, quote_altered, quote_fabricated, authority_unsupported, authority_unverifiable).
- `<FindingCard>` — summary line, evidence spans rendered with doc + offsets, confidence bar (green ≥ 0.8, yellow 0.5-0.8, red < 0.5), reasoning.
- `<CitationsTable>` — citation id, authority, has-quote, quote verdict, authority verdict.
- `<AgentTrace>` — collapsed by default; expand to see `agent_results` with latency and outcome. The "show your work" view.

Styling: inline styles + a single 50-line `styles.css`. No Tailwind, no MUI, no Radix. Reviewers can read every line of CSS in 30 seconds. JSON copy button on the report so users (and graders) can still grab raw output.

Vitest tests for each component: renders empty state, renders one example finding per kind, renders failed-agent state.

## 8. Eval refresh

This PR adds:
- **Memo sanity check** — assert memo length ≤ 180 words; assert every `find-N` reference in the memo points to a real finding. Deterministic check. Lives in `backend/agents/judicial_memo_writer.validate_memo`; if the memo fails either gate, `MemoResult.outcome="partial"` with an error string listing the issues. Shipped in PR #12.

**Cut — confidence calibration plot.** Originally: bucket findings by
confidence (0.0-0.2, 0.2-0.4, ...) and report match rate per bucket +
Brier score. Cut because (a) we don't have multiple runs across
temperature settings to populate the buckets meaningfully, (b) on this
case file most findings end up at similar confidence so a Brier number
would be uninformative anyway. The pipeline reports per-finding
confidence in the report; calibration analysis is deferred to the same
hypothetical future PR that adds real-mode N=3 variance reporting.

Thresholds for this PR:
```
python run_evals.py \
  --fail-under-recall 0.65 \
  --fail-over-hallucination 0.10 \
  --fail-under-unverifiable-precision 0.80 \
  --fail-over-memo-reference-mismatch 0.0
```

(Memo reference mismatch must be zero. If the memo cites `find-9` but no `find-9` exists, that's a hard fail.)

## 9. REFLECTION.md (the deliverable that earns evaluation point 5)

Structure (~1 page, plain prose, no headings deeper than `##`):

```markdown
# Reflection

## What I built
One paragraph: pipeline shape, what it catches, what it doesn't.

## What I cut and why
Bulleted. Each item names the cut, the reason (not "ran out of time"), and the cost.
- Web-based case-law retrieval — out of corpus today; pipeline emits `unverifiable`. Cost: high recall ceiling on the bogus-authority class.
- Multi-annotator gold set — single author's labels, so recall numbers reflect my reading of the docs, not consensus. Cost: real recall is probably ±10% of reported.
- Confidence calibration plot — Brier score computed but visualization deferred. Cost: easier to spot overconfident agents with a chart.
- ...

## Where the pipeline is weakest
Specific failure modes, named. Examples:
- AuthoritySupportChecker is essentially LLM-vs-LLM on the rare in-corpus authority case; on out-of-corpus it emits unverifiable, which is correct but also surrenders the question.
- CrossDocConsistencyChecker can miss multi-hop inferences (the witness statement implies X, the medical record implies Y, together they contradict Z).
- Gold set bias: I wrote it. Recall measured against my labels overstates real recall.
- Eval doesn't measure prompt-injection robustness. If the motion text contained "Ignore previous instructions...", I don't know what would happen.

## What I'd do differently with a week
- Retrieval over a small case-law corpus (Justia + CourtListener APIs) so AuthoritySupportChecker can actually verify external citations instead of always emitting `unverifiable`.
- Three-annotator gold set with inter-rater agreement reported.
- Adversarial eval set with prompt-injection attempts in document text.
- `structured-output` schema versioning so prompt changes can ship without re-snapshotting every test.
- Confidence calibration sweep across temperature settings.

## What surprised me
At least one honest finding from running the actual pipeline. Filled in at end.

## Cross-model review
Codex challenged the spec division and pushed evals into 001. That was the highest-leverage feedback I got. The full exchange lives in specs/001/spec.md § 14.
```

The graders will smell a sanitized reflection. The cuts list has to be specific and the "weakest" section has to name failure modes that would embarrass a less-honest submission.

## 10. Repository layout after this PR

```
backend/
  agents/
    confidence_scorer.py      # NEW
    judicial_memo_writer.py   # NEW
    prompts/
      confidence_scorer.py    # NEW
      judicial_memo_writer.py # NEW
  orchestrator.py             # hardened with tenacity + timeouts + failure isolation
  models.py                   # confidence fields on Finding, judicial_memo on report
frontend/
  src/
    App.jsx                   # rewritten
    components/
      ReportSummary.jsx
      MemoCard.jsx
      FindingsList.jsx
      FindingCard.jsx
      CitationsTable.jsx
      AgentTrace.jsx
    styles.css
  src/__tests__/              # vitest
REFLECTION.md                 # NEW
PROGRESS.md                   # final state
README.md                     # status all green
```

## 11. Cuts log

Items removed from this spec as they were not built. Listed here so the
plan does not lie about what shipped (per user feedback: "se vai cortar
escopo tem que remover do plano").

| Item | Status | Why cut | Documented in |
|---|---|---|---|
| Orchestrator hardening (tenacity + per-agent timeouts) | Cut | Each agent already has `try/except Exception → outcome="failure"`; OpenAIClient already retries at LLM layer; per-agent timeouts low-value vs. budget. | § 6 above + REFLECTION.md "What I cut and why" |
| Confidence calibration plot (Brier score) | Cut | Requires multiple runs across temperature settings to populate buckets; on this case file most findings cluster at similar confidence so a Brier number would be uninformative. | § 8 above + REFLECTION.md |
| `unverifiable-precision` metric (spec 002) | Cut | Formula collapsed to recall under a precision name; gaming-detection requires gold entries with `expected_verdict ∈ {supports, contradicts}` which the Rivera corpus does not produce. | spec 002 § 8 + REFLECTION.md |

What IS in this spec / PR:

| Item | Status | Where |
|---|---|---|
| `ConfidenceScorer` | ✅ Shipped | PR #12 |
| `JudicialMemoWriter` + reference check | ✅ Shipped | PR #12 |
| `REFLECTION.md` | ✅ Shipped | PR #9 |
| `AI_WORKFLOW.md` | ✅ Shipped | PR #10 |
| Frontend rewrite (6 components) | ⚪ This PR | `frontend/src/` |

## 12. Standards delta

Add one line to STANDARDS § 6 (Testing): "Frontend components have `vitest` tests for empty-state, one-of-each-kind, and failed-agent state."

## 13. How this earns evaluation points

| Brief point | Earned by |
|---|---|
| **1. Agent decomposition** | Six total agents now, each with one-sentence contracts. `ConfidenceScorer` and `JudicialMemoWriter` are downstream-only (consume Findings; can't fabricate them) — that's a real design property, not vibes. |
| **2. Prompt precision** | Confidence rubric in the prompt; memo word cap enforced both at prompt and post-validation. Both prompts under 40 lines. |
| **3. Eval quality** | Confidence calibration adds a meta-metric (is the pipeline well-calibrated, not just accurate?); memo reference mismatch is a hard deterministic gate. |
| **4. How far we get** | This is the polish layer. Even partially-shipped 003 (e.g., confidence scoring but no UI) is mergeable because everything is feature-flagged. Reviewer sees graceful degradation, not a half-broken main branch. |
| **5. Reflection honesty** | `REFLECTION.md` ships with this PR. It's the deliverable. |

## 14. Cross-model review

### 13.1 Codex challenge — pre-implementation

To be run via `/codex` with this spec. Focus areas:
- Is `ConfidenceScorer` worth the extra LLM call, or does it create a false sense of measurability? (Is bad confidence worse than no confidence?)
- Is the memo agent earning its keep when the reviewer can already see the findings list? Or is it cargo-culted "synthesis" with no information added?
- Is feature-flag-gated 003 actually safe to merge partially, or am I lying to myself about "behavior-preserving"?
- Should `JudicialMemoWriter` be the orchestrator (top of pipeline, picks priorities) instead of a tail agent? Codex's earlier feedback on retrieval suggests the high-leverage agent is the one closest to the source-of-truth gap.

Findings + decisions appended here before implementation.

### 13.2 Codex review — post-implementation

`/codex review` once `[3/3]` is green. Findings + responses appended.

## 15. Test plan

- `test_confidence_scorer.py` — confidence within [0,1]; reasoning mandatory; ungrounded finding → 0.0.
- `test_judicial_memo_writer.py` — empty findings → "no material problems" memo; non-empty → memo references real find-ids; word cap enforced.
- `test_orchestrator.py` — agent timeout → outcome=failure, report still returned; tenacity retries observed via FakeLLMClient call count.
- `test_main.py` — end-to-end with all six agents, FakeLLMClient injected.
- `frontend/src/__tests__/` — vitest per component (3 states each: empty, populated, error).

## 16. Acceptance criteria

- [ ] Six agents wired and tested.
- [ ] Orchestrator returns a report even when any single agent fails (test asserts this).
- [ ] `pytest -q` and `vitest` both clean.
- [ ] `mypy --strict` clean.
- [ ] Eval thresholds hold: recall ≥ 0.65, hallucination ≤ 0.10, unverifiable-precision ≥ 0.80, memo reference mismatch = 0.
- [ ] Frontend renders the report as structured cards; raw JSON copy button present.
- [ ] `REFLECTION.md` written; cuts list specific; weakest-points list specific.
- [ ] `PROGRESS.md` final; `README.md` Status all green.
- [ ] PR body links spec, summarizes what's in REFLECTION, quotes the final eval numbers.
