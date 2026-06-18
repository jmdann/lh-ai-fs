# Spec 002 — Quote Fidelity + Authority Support + Uncertainty Discipline

**Status:** draft
**Stack target:** `[2/3]` — depends on `[1/3]`
**Target time:** ~2h of the 6h budget
**Standards:** [STANDARDS.md](../../STANDARDS.md)

## 1. Goal

After this PR, the pipeline catches three new failure modes the brief explicitly grades: **altered quotes**, **fabricated quotes**, and **citations whose authority does not support (or contradicts) the stated proposition**. When the cited source text isn't available — which is the common case for case-law citations — the pipeline emits `unverifiable` with reasoning instead of guessing. Eval thresholds tighten to reflect the harder workload.

This is the spec where the brief's "express uncertainty appropriately — 'could not verify' rather than fabricating a finding" requirement gets fully earned.

## 2. Non-goals (deferred to 003)

- Confidence scoring layer → spec 003
- Judicial memo synthesis → spec 003
- Orchestrator failure isolation + retries → spec 003
- Frontend cards → spec 003

## 3. Deliverables

1. `backend/agents/quote_checker.py` — fidelity-only quote comparison.
2. `backend/agents/authority_support_checker.py` — supports / contradicts / unverifiable judgment.
3. `backend/agents/prompts/{quote_checker,authority_support_checker}.py` — prompts with `PROMPT_VERSION`.
4. `backend/sources.py` — minimal source-text registry. Holds the four case-file documents and is the ONLY thing `QuoteChecker` consults for "is this quote in the source?". Returns a typed `SourceLookup` result: `found` / `not_in_corpus`.
5. `backend/models.py` updates — activate `FindingKind.{QUOTE_ALTERED, QUOTE_FABRICATED, AUTHORITY_UNSUPPORTED, AUTHORITY_UNVERIFIABLE}`. Add `QuoteCheck` and `AuthorityCheck` envelope models.
6. `backend/orchestrator.py` updates — fan-out: after `CitationExtractor`, run `QuoteChecker` on every `Citation.quoted_text` and `AuthoritySupportChecker` on every `Citation` in parallel via `asyncio.gather`.
7. `tests/fixtures/gold_set.yaml` — expand to ~12-15 entries adding altered-quote, fabricated-quote, bogus-authority, and unverifiable-authority cases.
8. `tests/` — unit + integration tests for both new agents; prompt snapshots; orchestrator fan-out test.
9. `run_evals.py` updates — per-agent breakdown now covers four agents; thresholds bumped.
10. `evals/baseline_report.md` — refreshed with the new numbers.
11. `PROGRESS.md` — updated.
12. `README.md` — agent table updated.

## 4. Domain model deltas

```python
# additions / activations in backend/models.py

class QuoteVerdict(str, Enum):
    EXACT = "exact"              # byte-for-byte (or whitespace-normalized) match in source corpus
    PARAPHRASE = "paraphrase"    # not exact, but semantically faithful to source
    ALTERED = "altered"          # in source but with material words changed/removed
    FABRICATED = "fabricated"    # quoted attribution does not appear in source corpus
    UNVERIFIABLE = "unverifiable"  # source text not available to us

class QuoteCheck(BaseModel):
    citation_id: str
    quoted_text: str             # what the motion claims is the quote
    verdict: QuoteVerdict
    matched_span: Span | None    # populated when verdict in {EXACT, PARAPHRASE, ALTERED}
    reasoning: str               # one sentence; mandatory when verdict ∈ {ALTERED, FABRICATED, UNVERIFIABLE}

class AuthorityVerdict(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    UNVERIFIABLE = "unverifiable"

class AuthorityCheck(BaseModel):
    citation_id: str
    verdict: AuthorityVerdict
    source_basis: Span | None    # the source span the judgment is grounded in
    reasoning: str               # mandatory
```

Validator additions:
- `QuoteCheck`: if verdict ∈ {EXACT, PARAPHRASE, ALTERED}, `matched_span` MUST be non-null and ground in the source corpus.
- `AuthorityCheck`: if verdict ∈ {SUPPORTS, CONTRADICTS}, `source_basis` MUST be non-null and ground. If `UNVERIFIABLE`, `source_basis` MUST be null AND `reasoning` MUST name the missing source.
- These invariants are enforced in `models.py` Pydantic validators — not in agent code. Bad output from the LLM → Pydantic raises → orchestrator records `outcome="failure"` and continues.

## 5. Agents (this spec)

### 5.1 QuoteChecker

**One sentence:** Given a citation with a quoted string, decide whether the quoted string appears verbatim / paraphrased / altered / fabricated in the source corpus we have. Fidelity only.

**Signature:**
```python
class QuoteChecker:
    def __init__(self, llm: LLMClient, sources: SourceRegistry) -> None: ...
    def run(self, citation: Citation) -> AgentResult[QuoteCheck]: ...
```

**Hybrid pipeline (deterministic first, then LLM):**
1. **Deterministic pre-check** — `SourceRegistry.contains(quoted_text)` does a normalized substring search (whitespace + case + smart-quote normalized) across the four record documents. A hit short-circuits to `EXACT` with the matched span. No LLM call.
2. **Fuzzy match** — if no exact hit, compute the best fuzzy alignment (rapidfuzz token-set ratio ≥ 90) against each source document. Hit → `LLM judges {PARAPHRASE, ALTERED}` given the original + match.
3. **No fuzzy hit, source unavailable** — if the citation refers to a document we have (one of the four), emit `FABRICATED`. If it refers to external case law not in the corpus, emit `UNVERIFIABLE` with reasoning naming the missing source.

This is the only agent where we use code-as-judge before the LLM. Two reasons: (a) substring matching is faster, free, and deterministic — never let the LLM do what `str.find` can; (b) it caps the surface area where the LLM can hallucinate "exact" matches.

**Prompt rules** (only invoked for the `PARAPHRASE | ALTERED` decision):
- Role: "Compare two strings. Decide if string A is a faithful paraphrase of string B or a material alteration."
- Output: JSON schema = `{verdict: "paraphrase" | "altered", reasoning: str}`.
- Refusal: if uncertain, prefer `altered` — it's the safer side for the user.

### 5.2 AuthoritySupportChecker

**One sentence:** Given a proposition span + cited authority + (optionally) source text we have, decide whether the authority supports / contradicts / cannot verify the proposition.

**Signature:**
```python
class AuthoritySupportChecker:
    def __init__(self, llm: LLMClient, sources: SourceRegistry) -> None: ...
    def run(self, citation: Citation) -> AgentResult[AuthorityCheck]: ...
```

**Discipline (the load-bearing rule):**
- If `SourceRegistry.lookup(citation.cited_authority)` returns `not_in_corpus`, the agent MUST emit `UNVERIFIABLE` without calling the LLM. No exceptions. We don't "reason about what the case probably says" — that's hallucination dressed up.
- If the source IS in corpus (e.g., the motion cites the police report as authority for a fact), the LLM is called with the proposition + source span and asked for `supports | contradicts | unverifiable`.
- Even with source in hand, `unverifiable` remains a legal output — the prompt encourages it when the source is ambiguous.

**Prompt rules:**
- Role: "Decide whether a source span supports, contradicts, or is insufficient to evaluate a stated proposition."
- Output: JSON schema = `AuthorityCheck` (excluding `citation_id`, filled by agent).
- Refusal: prefer `unverifiable` over a low-confidence judgment. Reviewers grade hallucinations harder than misses.

### 5.3 Why the rename matters

`CitationVerifier` (original) collapsed three jobs: extract, fidelity, support. Codex flagged that as a seam-ambiguity bug. `AuthoritySupportChecker` does exactly one of those things and is forbidden from quote parsing — quote fidelity is `QuoteChecker`'s job. The seams in STANDARDS § 3.3 enforce this.

## 6. Orchestrator updates

```python
async def run(self, documents: ...) -> VerificationReport:
    motion = documents[DocumentKind.MOTION]
    records = [d for kind, d in documents.items() if kind != DocumentKind.MOTION]

    citations_result = self.extractor.run(motion)
    cross_doc_result = self.cross_doc.run(motion, records)

    # New in 002: fan out per-citation work
    quote_results, authority_results = await asyncio.gather(
        asyncio.gather(*(self.quote_checker.run(c) for c in citations_result.data or [])),
        asyncio.gather(*(self.authority_checker.run(c) for c in citations_result.data or [])),
    )

    findings = self._build_findings(
        cross_doc_result, quote_results, authority_results
    )
    findings = self._validate_span_grounding(findings, documents)  # drop ungrounded
    return VerificationReport(...)
```

Bounded by `asyncio.Semaphore(value=settings.max_concurrent_llm_calls)`, configurable via env (12-factor III + VIII). Default 5.

This spec keeps the orchestrator straight-line on failures — a single agent erroring still surfaces in `agent_results` with `outcome="failure"` and the report still returns. Rich retry / circuit-breaker behavior is 003.

## 7. Gold set expansion

Target ~12-15 entries total. New entries cover:

```yaml
- id: gold-quote-altered-1
  kind: quote_altered
  citation_contains: "Privette v. Superior Court"
  motion_quote: "...landowner owes no duty to the employees of an independent contractor..."
  source_doc: <external; mark unverifiable>     # or, if we stage a stub source, the real text
  expected_verdict: altered
  notes: "Motion drops the qualifier; source text says 'generally owes no duty'."

- id: gold-quote-fabricated-1
  kind: quote_fabricated
  citation_contains: "Officer Reyes"
  motion_quote: "I observed the worker was not wearing fall protection."
  source_doc: police_report
  expected_verdict: fabricated
  notes: "Police report contains no such statement attributed to Officer Reyes."

- id: gold-authority-unsupported-1
  kind: authority_unsupported
  citation_contains: "Hooker v. Department of Transportation"
  proposition: "...affirmative contribution from the hirer is required..."
  expected_verdict: <supports | contradicts | unverifiable>
  notes: "External authority — pipeline MUST emit unverifiable, NOT a guess."

- id: gold-authority-unverifiable-1
  kind: authority_unverifiable
  citation_contains: "Smith v. Doe Trucking"  # the brief's fake citation, if present
  expected_verdict: unverifiable
  notes: "Cite does not exist; agent must emit unverifiable with reasoning naming the missing source."
```

Gold entries reference text by **quote**, never by offset (STANDARDS § 3.6). The eval harness uses `SourceRegistry.find()` to ground both gold quotes and emitted spans, then compares grounded ranges with the structural match function from spec 001 § 7.2 (`kind + doc_id + span overlap ≥ 0.6`).

**The gold set is written by reading the documents BEFORE the prompts are tuned for this PR** (STANDARDS § 5). Any drift between agent behavior and gold is fixed by changing the agent, not the gold.

## 8. Eval thresholds (tightened)

This PR's CI gate (count-based per STANDARDS § 5, matching spec 001's pattern):

```
python run_evals.py \
  --min-matched-gold 7 \
  --max-grounding-failures 0 \
  --max-cited-doc-scope-failures 0 \
  --min-unverifiable-precision 0.80
```

Counts rise from 3 → 7 because the gold set roughly doubles. `unverifiable-precision` stays as a ratio because it's about *which* unverifiable findings are right, not how many — the gaming failure mode here is "say unverifiable to everything", and the count-gate doesn't catch that.

**New metric — `unverifiable-precision`**: of findings where the pipeline emitted `unverifiable`, the fraction that the gold also marks unverifiable. Below 0.80 means the pipeline is hiding behind "I don't know" to dodge the hallucination metric.

## 9. Repository layout after this PR

```
backend/
  sources.py                    # NEW — SourceRegistry, normalized lookup
  agents/
    quote_checker.py            # NEW
    authority_support_checker.py # NEW
    prompts/
      quote_checker.py          # NEW
      authority_support_checker.py # NEW
  orchestrator.py               # updated — async fan-out
  models.py                     # updated — new envelopes, validators
tests/
  fixtures/
    gold_set.yaml               # expanded
    sources/                    # NEW — any stubbed source excerpts we use
  test_quote_checker.py         # NEW
  test_authority_support_checker.py # NEW
  test_sources.py               # NEW — substring + fuzzy match correctness
  test_orchestrator.py          # updated — fan-out test
run_evals.py                    # updated — new metric, new thresholds
evals/baseline_report.md        # refreshed
```

## 10. Standards delta

No new standards; this PR is the first one that fully exercises STANDARDS § 3.3 (four-agent seams) and § 3.5 (source-retrieval discipline). The deterministic-first quote-check pattern is worth a one-line note added to STANDARDS § 4: "Determinism: prefer code-as-judge over LLM-as-judge whenever string ops can answer the question."

## 11. How this earns evaluation points

| Brief point | Earned by |
|---|---|
| **1. Agent decomposition** | Two more agents with crisp seams; `QuoteChecker` ≠ `AuthoritySupportChecker` ≠ `CrossDocChecker` enforced by code AND by the prompts. Code-as-judge for substring fidelity demonstrates we know when LLMs are wrong tool. |
| **2. Prompt precision** | Prompts are minimal — `QuoteChecker`'s LLM call only sees two strings and emits 2-value verdict; `AuthoritySupportChecker` is forbidden from running when source is absent. Smaller prompts = smaller hallucination surface. |
| **3. Eval quality** | New `unverifiable-precision` metric catches the "say IDK to everything" failure mode. Gold set now covers all four agents. Threshold-bump documents progress. |
| **4. How far we get** | This PR is the bulk of the actual quality work. Submitted alone (without 003), still a strong Tier 1+2 result with all four core agents + measured uncertainty. |
| **5. Reflection honesty** | `unverifiable` as first-class is itself the most defensible design choice in the project. REFLECTION.md in 003 will lean on this. |

## 12. Cross-model review

### 12.1 Codex challenge — pre-implementation

To be run: `/codex` with this spec.md as input, focusing on:
- Is the deterministic-first quote pipeline actually safer, or does it introduce its own failure mode (normalization bugs hiding real alterations)?
- Is `AuthoritySupportChecker` skipping the LLM entirely on out-of-corpus citations the right call, or should it at least try semantic matching against a vetted citation list?
- Are the new gold entries adversarial enough, or am I testing the prompts I already wrote?
- Is `unverifiable-precision` a meaningful metric or a gamable one?

Findings + decisions appended here before implementation starts.

### 12.2 Codex review — post-implementation

`/codex review` once PR `[2/3]` is green. Findings + responses appended.

## 13. Test plan

- `test_sources.py` — substring normalization (whitespace, smart quotes, case); fuzzy ratio threshold; `not_in_corpus` path.
- `test_quote_checker.py` — exact-match short-circuits without LLM (assert LLM not called); fuzzy → LLM paraphrase/altered; no match → fabricated; external → unverifiable.
- `test_authority_support_checker.py` — out-of-corpus citation never calls LLM, emits unverifiable; in-corpus citation calls LLM and validates verdict invariants; bad LLM output (e.g., `supports` with null source_basis) → Pydantic raises → `outcome=failure`.
- `test_orchestrator.py` — fan-out with `asyncio.gather`; one agent failure doesn't kill the report.
- `test_prompts.py` — snapshots for both new prompts.

## 14. Acceptance criteria

- [ ] All four agents wired through orchestrator with async fan-out.
- [ ] `pytest -q` passes; zero network in unit tests.
- [ ] `mypy --strict backend/` clean.
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `python run_evals.py` meets recall ≥ 0.65, hallucination ≤ 0.10, unverifiable-precision ≥ 0.80.
- [ ] `evals/baseline_report.md` refreshed; commit shows the delta vs 001.
- [ ] `PROGRESS.md` updated.
- [ ] PR body links spec, quotes deltas from 001's baseline, notes any gold entries that the pipeline misses (honesty).
