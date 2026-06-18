# Spec 001 — Foundation, Cross-Doc Contradictions, Thin Eval Harness

**Status:** draft
**Stack target:** `[1/3]` — base of the stack on `origin/main`
**Target time:** ~4h of the 6h budget. Codex round 2 called the original 2.5h estimate unrealistic for what this PR ships (full IR + two agents + eval harness + tests + baseline + ceremony). Scope kept intact — we eat the budget rather than the discipline.
**Standards:** [STANDARDS.md](../../STANDARDS.md)

## 1. Goal

After this PR, `POST /analyze` returns a real structured `VerificationReport` derived from the four case-file documents, and `python run_evals.py` reports precision / recall / hallucination rate against a hand-labeled gold set. The pipeline catches cross-document factual contradictions (the highest-signal, lowest-uncertainty class of flaw). Everything is grounded in typed Pydantic models with no raw text crossing agent boundaries.

This is the slice that, **if everything else fails**, is still a credible submission. It satisfies the brief's Tier 1 core (structured output, citation extraction) plus a chunk of Tier 2 (eval harness, cross-doc, "could not verify" discipline).

## 2. Non-goals (deferred to later specs)

- Quote fidelity checking → spec 002
- Authority-support judgment ("does the case actually say this?") → spec 002
- Confidence scoring layer → spec 003
- Judicial memo synthesis → spec 003
- Orchestrator with rich failure isolation → spec 003 (this PR has a straight-line orchestrator)
- Frontend cards → spec 003 (this PR keeps the existing raw-JSON `<pre>`)

## 3. Deliverables

Full scope. Codex flagged some of these as cuttable; we kept them because the discipline is the point.

1. `backend/models.py` — full stable IR (Pydantic v2). Includes the spec-002 types (`Claim`, `Quote`, `QuoteCheck`, `AuthorityCheck`, `EvidenceRef`) as stubs so the IR doesn't churn across PRs.
2. `backend/llm/client.py` — `LLMClient` Protocol + `OpenAIClient` + `FakeLLMClient`.
3. `backend/config.py` — `Settings(BaseSettings)` (12-factor III).
4. `backend/sources.py` — `SourceRegistry` with normalized substring + fuzzy lookup. Lands in 001 because `Span` grounding depends on it (STANDARDS § 3.6).
5. `backend/agents/citation_extractor.py` — regex-first + LLM proposition attachment.
6. `backend/agents/cross_doc_checker.py` — motion claims vs record documents.
7. `backend/agents/prompts/{citation_extractor,cross_doc_checker}.py` — prompts as constants with `PROMPT_VERSION`.
8. `backend/orchestrator.py` — async fan-out (`asyncio.gather`) over the two independent agents; span grounding boundary.
9. `backend/main.py` — `POST /analyze` wired through orchestrator, `Depends(get_llm_client)` at the route only.
10. `backend/observability.py` — structured-JSON `logging.Formatter` + helper to emit one canonical line per agent invocation (12-factor XI).
11. `tests/fixtures/gold_set.yaml` — hand-labeled, 5-8 entries. Quote-based.
12. `tests/fixtures/llm/*.json` — canned LLM responses for `FakeLLMClient`.
13. `tests/test_prompts.py` — snapshots every `PROMPT_VERSION` constant; changing a prompt without bumping the version fails CI (STANDARDS § 3.4).
14. `tests/` — unit (FakeLLMClient), one integration per agent against snapshot, one end-to-end POST `/analyze`.
15. `run_evals.py` — single command. Runs N=3 with `temperature=0` to report variance even when "deterministic". Structural matching (§ 7.2). Count-based CI gates (§ 7.4). Per-agent latency histogram in the markdown report.
16. `evals/baseline_report.md` — committed real-run baseline (JSON + Markdown).
17. `evals/README.md` — how to read the metrics; what `grounding_integrity` does and doesn't prove.
18. `PROGRESS.md` — root-level "what's done / what's deferred / why" tracker, updated end-of-PR.
19. `README.md` updates — Status section, eval-run instructions, agent table.

## 4. Domain models (the load-bearing decision)

`backend/models.py`. Everything else is downstream of these. This section was rewritten after Codex round 2 — see § 14.2 for what changed and why.

**Core rule**: the LLM emits quotes, not offsets. `Span(doc_id, quote)` is what agents return; the orchestrator grounds the quote to character offsets in code via `SourceRegistry.find()`. See STANDARDS § 3.6.

```python
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

# ─── Documents ─────────────────────────────────────────────────────────────

class DocumentKind(str, Enum):
    MOTION = "motion_for_summary_judgment"
    POLICE_REPORT = "police_report"
    MEDICAL_RECORDS = "medical_records_excerpt"
    WITNESS_STATEMENT = "witness_statement"
    EXTERNAL_AUTHORITY = "external_authority"   # reserved for spec 002+

class Document(BaseModel):
    id: str                  # stable identifier, e.g. "motion", "police_report", "auth:privette"
    kind: DocumentKind
    text: str

# ─── Spans: quote is authoritative, offsets are derived ────────────────────

class Span(BaseModel):
    """A pointer into a document. The LLM provides doc_id + quote.
    start/end are filled by the orchestrator via SourceRegistry.find().
    Ungrounded spans (where find() fails) are dropped before the report returns.
    """
    doc_id: str
    quote: str                # what the LLM said is in the document
    start: int | None = None  # filled in code, not by the LLM
    end: int | None = None

# ─── Claims, Citations, Quotes ─────────────────────────────────────────────

class Claim(BaseModel):
    """A factual proposition asserted in the motion. Spec 002+ uses this; in spec 001
    CrossDocConsistencyChecker emits FactDiscrepancy directly without a Claim hop,
    but the type exists now so we don't churn the IR in the next PR."""
    id: str
    proposition: Span         # location of the claim in the motion

class Citation(BaseModel):
    """A legal authority cited by the motion. No judgment in spec 001."""
    id: str                                # cite-{n}
    proposition: Span                      # the sentence the cite is offered to support
    cited_authority: str                   # raw text e.g. "Privette v. Superior Court (1993) 5 Cal. 4th 689"
    quoted_text: str | None = None         # direct quote attributed to the authority, if any

class Quote(BaseModel):
    """A direct quotation attributed to a cited authority. Spec 002+ populates."""
    citation_id: str
    text: str
    attributed_doc_hint: str | None = None  # e.g. "Privette" — used by SourceRegistry

class QuoteCheck(BaseModel):
    """Spec 002+ populates. Stubbed here so IR is stable across PRs."""
    citation_id: str
    verdict: Literal["exact", "paraphrase", "altered", "fabricated", "unverifiable"]
    matched_span: Span | None = None
    reasoning: str

class AuthorityCheck(BaseModel):
    """Spec 002+ populates."""
    citation_id: str
    verdict: Literal["supports", "contradicts", "unverifiable"]
    source_basis: Span | None = None
    reasoning: str

class EvidenceRef(BaseModel):
    """A typed reference from a Finding back to its supporting span(s)."""
    span: Span
    role: Literal["primary", "supporting", "contradicting"]

# ─── Findings ──────────────────────────────────────────────────────────────

class FindingKind(str, Enum):
    FACT_DISCREPANCY = "fact_discrepancy"
    # activated in spec 002:
    QUOTE_ALTERED = "quote_altered"
    QUOTE_FABRICATED = "quote_fabricated"
    AUTHORITY_UNSUPPORTED = "authority_unsupported"
    AUTHORITY_UNVERIFIABLE = "authority_unverifiable"

class FactDiscrepancy(BaseModel):
    """Factual claim in the motion contradicted by record evidence."""
    id: str
    motion_claim: Span
    contradicting_evidence: list[Span]   # was singular in v1; Codex flagged — multi-source contradictions are common
    description: str
    severity: Literal["minor", "material", "dispositive"]

class Finding(BaseModel):
    id: str                              # find-{n}
    kind: FindingKind
    summary: str                         # human-readable; eval matching does NOT use this field
    evidence: list[EvidenceRef]
    agent: str
    prompt_version: str
    citation_id: str | None = None       # populated when kind ∈ {QUOTE_*, AUTHORITY_*}
    claim_id: str | None = None
    # confidence + reasoning added in spec 003

# ─── Agent results: discriminated union, not Generic[T] ────────────────────

class _AgentResultBase(BaseModel):
    agent: str
    prompt_version: str
    outcome: Literal["success", "failure", "partial", "timeout"]
    error: str | None = None
    latency_ms: int

class CitationsResult(_AgentResultBase):
    kind: Literal["citations"] = "citations"
    data: list[Citation] = []

class DiscrepanciesResult(_AgentResultBase):
    kind: Literal["discrepancies"] = "discrepancies"
    data: list[FactDiscrepancy] = []

class QuoteCheckResult(_AgentResultBase):
    kind: Literal["quote_check"] = "quote_check"
    data: list[QuoteCheck] = []

class AuthorityCheckResult(_AgentResultBase):
    kind: Literal["authority_check"] = "authority_check"
    data: list[AuthorityCheck] = []

class ConfidenceResult(_AgentResultBase):
    kind: Literal["confidence"] = "confidence"
    data: list[Finding] = []          # rescored findings

class MemoResult(_AgentResultBase):
    kind: Literal["memo"] = "memo"
    data: str | None = None

AgentResult = Annotated[
    Union[CitationsResult, DiscrepanciesResult, QuoteCheckResult,
          AuthorityCheckResult, ConfidenceResult, MemoResult],
    Field(discriminator="kind"),
]

# ─── Report ────────────────────────────────────────────────────────────────

class VerificationReport(BaseModel):
    case_name: str
    generated_at: datetime
    citations: list[Citation]
    findings: list[Finding]
    agent_results: list[AgentResult]
    judicial_memo: str | None = None     # populated in spec 003
```

Invariants enforced in code:

1. **Span grounding** (orchestrator boundary): for every `Span` emitted by an agent, `SourceRegistry.find(doc_id, quote)` must return a hit. If not, the Finding owning that span is dropped and an entry is logged on the corresponding `AgentResult` with `outcome="partial"`. This is the `grounding_integrity` metric the eval reports.
2. **Stable IDs**: `Citation.id = "cite-{n}"`, `Finding.id = "find-{n}"`, deterministic numbering, so eval matching across runs is stable.
3. **Discriminated union, not Generic[T]**: Codex flagged Pydantic v2 generic envelopes as awkward for schema generation. Each result kind is its own class; `Annotated[Union[...], Field(discriminator="kind")]` gives clean JSON schema for OpenAPI without runtime gymnastics.
4. **Document.id ≠ DocumentKind**: in spec 001 the four case-file docs use ids matching `kind.value`; spec 002 adds authority documents whose `id` is distinct (e.g. `auth:privette`). The IR doesn't need to change.

## 5. Agents (this spec)

### 5.1 CitationExtractor — regex first, LLM fallback

**One sentence:** Pull every legal authority citation out of the motion, with its proposition and any direct quote. No judgment.

Codex round 2 flagged spending an LLM call on citation extraction as bad ROI — California citation formats are regular enough to catch with a few patterns. The agent goes regex-first; the LLM handles only the proposition span attachment (mapping each cite to its surrounding sentence) and any cites the regex misses.

**Signature:**
```python
class CitationExtractor:
    def __init__(self, llm: LLMClient) -> None: ...
    def run(self, motion: Document) -> CitationsResult: ...
```

**Pipeline:**
1. **Regex pass** (deterministic) — match common California citation forms:
   - `<Plaintiff> v. <Defendant>(, ...)? \(<year>\) \d+ Cal\.( \d+(th|st|nd|rd))? \d+`
   - Statute forms (`Cal. <Code>, § \d+(\.\d+)?`)
   - Restatement / treatise patterns as needed.
   Each match yields a candidate `(authority_text, char_offset)`.
2. **LLM proposition attachment** (one call total, not per-cite) — given the motion + the list of regex-matched authorities, the LLM emits `{authority_text, proposition_quote, quoted_text | null}` per cite. Quote-based span shape (STANDARDS § 3.6); no offsets.
3. **LLM-only fallback** — if regex returns < 3 cites (unlikely given the document, but guards against format drift), run a full-extraction LLM call with the same output schema.

**Prompt rules** (for steps 2 / 3):
- Role: "You map legal citations to the sentence they support. You do not invent citations."
- Output: JSON schema = `list[Citation]` (without `id` — filled by agent code).
- Refusal: if a regex-matched authority has no clear supporting sentence within 200 chars, emit it with `proposition.quote = ""` and let the orchestrator log it; do not guess.
- Worked example included.

### 5.2 CrossDocConsistencyChecker

**One sentence:** Compare each factual claim in the motion against the police report, medical records, and witness statement. Emit `FactDiscrepancy` only when a record document directly contradicts the motion.

**Signature:**
```python
class CrossDocConsistencyChecker:
    def __init__(self, llm: LLMClient) -> None: ...
    def run(self, motion: Document, records: list[Document]) -> AgentResult[list[FactDiscrepancy]]: ...
```

**Prompt rules:**
- Role: "You compare factual statements in a legal motion against record evidence."
- Output: JSON schema = `list[FactDiscrepancy]`.
- Refusal: if a motion fact has no corresponding statement in any record document, emit NOTHING for that fact. Do not flag absences.
- Severity rubric in the prompt: minor (peripheral detail), material (could affect judgment), dispositive (would change the outcome).
- Worked example included.

## 6. Orchestrator

Codex round 2 pushed for parallelism in PR 1: `CitationExtractor` and `CrossDocConsistencyChecker` are independent and both expensive. Running them serially leaves wall-clock on the table for no benefit.

```python
class Orchestrator:
    def __init__(self, llm: LLMClient, sources: SourceRegistry) -> None:
        self.extractor = CitationExtractor(llm)
        self.cross_doc = CrossDocConsistencyChecker(llm)
        self.sources = sources

    async def run(self, documents: dict[str, Document]) -> VerificationReport:
        motion = documents["motion"]
        records = [d for k, d in documents.items() if k != "motion"]

        # Independent agents — fan out from the start.
        citations_result, cross_doc_result = await asyncio.gather(
            self._run_agent(self.extractor.run, motion),
            self._run_agent(self.cross_doc.run, motion, records),
        )

        findings = self._build_findings(cross_doc_result)
        findings = self._ground_and_drop(findings, documents)   # invariant 1
        return VerificationReport(...)
```

Spec 001 keeps the failure handling intentionally thin (try/except → `outcome="failure"` on the result; report still returns). Tenacity retries + timeouts arrive in spec 003.

**On context size** (Codex's secondary concern): cross-doc over the motion + three records is ~2200 words total, fits in one prompt comfortably with `gpt-4o`. If a future iteration loads larger records, the right move is a `ClaimExtractor` pre-pass that emits `Claim` objects, then per-claim cross-doc. The `Claim` model exists in the IR for this reason.

## 7. Eval harness

`run_evals.py` is the highest-leverage deliverable. The brief literally says they run it.

### 7.1 Gold set

`tests/fixtures/gold_set.yaml` — hand-authored by reading the four documents end-to-end. Target 5-8 entries in this PR (we add more in spec 002).

Schema per entry:
```yaml
- id: gold-fact-1
  kind: fact_discrepancy
  expected_finding:
    motion_quote: "Rivera was wearing his harness incorrectly..."  # substring from MSJ
    contradicting_doc: police_report
    contradicting_quote: "Worker observed wearing safety harness properly secured..."
    severity: material
  notes: "Police report directly contradicts motion's safety-harness narrative."

- id: gold-cite-1
  kind: citation_present
  expected_citation:
    contains: "Privette v. Superior Court"
  notes: "Defendant relies on Privette doctrine; must be extracted."
```

The gold set is committed in this PR. Author reads the docs, labels flaws, commits — then writes the prompts. **Never the other way around** (STANDARDS § 5).

### 7.2 Metrics — structural, honest about what they prove

Codex round 2 killed the original Jaccard-on-summary approach. Match by structure; name the offset/text check for what it is.

**Match function** (gold entry vs. emitted findings):
```python
def matches(gold: GoldEntry, finding: Finding) -> bool:
    if gold.kind != finding.kind:
        return False
    # Motion span: did the pipeline locate the claim in the same neighborhood?
    if span_overlap(gold.motion_span, primary_evidence(finding)) < 0.6:
        return False
    # Evidence docs: did the pipeline cite at least the documents gold cites?
    if not gold.evidence_docs.issubset(evidence_docs_of(finding)):
        return False
    return True
```

`summary` text is reported in the human-readable diff but never enters the score — that prevents prompt phrasing from leaking into the metric.

**Reported numbers**:
```
matched_gold_findings   = |{ g ∈ gold : ∃ f ∈ emitted . matches(g, f) }|
total_emitted           = |emitted|
total_gold              = |gold|

precision               = matched_gold_findings / total_emitted
recall                  = matched_gold_findings / total_gold

grounding_integrity_failures = |{ f ∈ emitted : ∃ span ∈ f.evidence . sources.find(span.doc_id, span.quote) is None }|
cited_doc_in_scope_failures  = |{ f ∈ emitted : ∃ span ∈ f.evidence . span.doc_id ∉ loaded_documents }|
```

**Honest naming** (the change Codex demanded):
- `grounding_integrity` ≠ hallucination check. It proves the quote the LLM emitted actually appears in a document we loaded. It does **not** prove the quote semantically supports the finding. The real semantic-grounding gap is documented in `REFLECTION.md` as a known limitation; without an LLM judge layer there's no honest way to close it inside the 6h budget.
- `cited_doc_in_scope` catches the failure mode where the LLM cites a document the orchestrator never loaded.

Per-agent breakdown + overall, plus a **latency histogram** (p50 / p95 / max per agent) parsed from the structured JSON logs. **N=3 variance run** with `temperature=0` — surfaces residual nondeterminism the docs say shouldn't exist but always does at the margin (tool-call ordering, top-k tie-breaking). Variance reported as `recall_stdev`, `precision_stdev`. If stdev > 0 at temp=0, that's worth a line in the baseline report.

### 7.3 Outputs

- `evals/eval_report.json` (CI-friendly)
- `evals/eval_report.md` (human review — per-finding diff)
- `evals/baseline_report.md` (the run we commit so graders see the actual number)

### 7.4 CI gates (this PR) — counts, not percentages

With N=5-8 gold entries, percentage thresholds are theater (`recall ≥ 0.5` = 3/6). Codex was right; gate on counts:

```
python run_evals.py \
  --min-matched-gold 3 \
  --max-grounding-failures 0 \
  --max-cited-doc-scope-failures 0
```

Percentages still appear in `eval_report.md` for trend visibility across PRs. The count gates rise per PR (spec 002: `--min-matched-gold 7`).

## 8. Config (12-factor III)

`backend/config.py`:
```python
class Settings(BaseSettings):
    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-2024-08-06"
    openai_temperature: float = 0.0
    eval_seed: int = 42
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
```

`.env.example` updated. `OPENAI_MODEL` is overridable so the eval CI can pin a specific version.

## 9. LLM client (Protocol layer, STANDARDS § 3.2)

```python
# backend/llm/client.py
class LLMClient(Protocol):
    def complete(
        self, *, system: str, user: str, schema: type[BaseModel], model: str | None = None
    ) -> BaseModel: ...

class OpenAIClient:
    def __init__(self, settings: Settings) -> None: ...
    def complete(self, *, system, user, schema, model=None) -> BaseModel:
        # Uses response_format={"type": "json_schema", "json_schema": {...}}
        # with strict=True so the model is forced to comply.

class FakeLLMClient:
    """Test double. Returns pre-canned Pydantic objects per (system, user) hash."""
    def __init__(self, responses: dict[str, BaseModel]) -> None: ...
```

FastAPI wiring (only at the route):
```python
def get_llm_client() -> LLMClient: ...
@app.post("/analyze")
def analyze(llm: LLMClient = Depends(get_llm_client)) -> VerificationReport: ...
```

`Depends` is forbidden inside `agents/`, `orchestrator.py`, and `run_evals.py`.

## 10. Logging (12-factor XI)

Structured JSON via stdlib `logging` + a small JSON formatter. One line per agent invocation:
```json
{"event":"agent.run","agent":"CitationExtractor","prompt_version":"1.0.0","outcome":"success","latency_ms":823,"prompt_tokens":612,"completion_tokens":188}
```

The eval harness captures these and includes a per-agent latency histogram in the report.

## 11. Repository layout after this PR

```
backend/
  main.py                 # FastAPI route + Depends wiring only
  config.py               # Settings(BaseSettings)
  models.py               # stable IR — full set (incl. spec-002 stubs)
  sources.py              # SourceRegistry: normalized substring + fuzzy lookup
  llm/
    __init__.py
    client.py             # LLMClient Protocol + OpenAIClient + FakeLLMClient
  agents/
    __init__.py
    citation_extractor.py # regex-first + LLM proposition attach
    cross_doc_checker.py
    prompts/
      __init__.py
      citation_extractor.py     # PROMPT_VERSION + SYSTEM + USER + EXAMPLE
      cross_doc_checker.py
  orchestrator.py         # async fan-out + span grounding boundary
  tests/
    fixtures/
      gold_set.yaml
      llm/                       # canned FakeLLMClient responses
    test_models.py               # IR invariants
    test_sources.py              # normalized lookup, fuzzy threshold
    test_citation_extractor.py   # regex coverage, LLM fallback
    test_cross_doc_checker.py
    test_orchestrator.py         # grounding boundary drops ungrounded
    test_prompts.py              # snapshot per PROMPT_VERSION
    test_observability.py        # structured-log fields
    test_main.py                 # end-to-end with FakeLLMClient
run_evals.py                     # N=3 variance, latency histogram, count gates
evals/
  baseline_report.md
  baseline_report.json
  README.md                      # how to read; what grounding_integrity proves
PROGRESS.md
STANDARDS.md
specs/
  001-foundation-evals-crossdoc/spec.md  (this file)
```

## 12. Standards delta

This PR introduces the rules now codified in `STANDARDS.md` §§ 3.1-3.5 and § 5. No exceptions.

## 13. How this earns evaluation points

| Brief point | Earned by |
|---|---|
| **1. Agent decomposition** | `models.py` + the seams table in STANDARDS § 3.3. Two crisp agents in this PR, both with one-sentence contracts. |
| **2. Prompt precision** | `agents/prompts/*.py` as constants with `PROMPT_VERSION`, JSON-schema response format, explicit refusal rules, snapshot tests. |
| **3. Eval quality** | `run_evals.py` reports precision / recall / hallucination + variance; gold set written before prompts; hallucination computed deterministically by span grounding, not LLM judge; baseline committed. |
| **4. How far we get** | This PR alone is a credible Tier 1+ submission. If 002 and 003 don't land, we still ship structured output, real findings, and honest metrics. |
| **5. Reflection honesty** | `PROGRESS.md` lands now; `REFLECTION.md` writes itself from these artifacts in spec 003. |

## 14. Cross-model review

### 14.1 Codex challenge — pre-implementation

Run via `/codex` with the three-spec division as input.

**Verbatim core findings** (full transcript in conversation):
1. **Front-loading the wrong uncertainty** — eval harness must move into 001. Without a gold set + scorer on day one, there's no brake on hallucinated findings.
2. **Architecture only scalable with a stable IR introduced immediately** — current `backend/llm.py` hardcodes OpenAI at module scope and returns raw strings. Domain models must come first; `Depends` stays at the API edge.
3. **Agent seams blurry** — `CitationVerifier` overlaps `QuoteChecker` (quote alteration affects "support") and overlaps `CrossDocConsistencyChecker` (when motion cites record evidence as authority). Rename + crisp four-way split.
4. **Highest risk = source retrieval, not orchestration** — cited case law is not in the repo. `AuthoritySupportChecker` becomes a hallucination generator with a nicer schema unless `unverifiable` is first-class.
5. **Stacked PRs partially overkill** — for a 6h take-home, three stacks is only worth the ceremony if reviewability is a deliverable.

### 14.2 Decisions taken in response

| Finding | Decision | Rationale |
|---|---|---|
| 1 | **Accepted** — moved eval harness into 001 | Brief says "we run your eval suite as part of our review". Quality signal from minute one. |
| 2 | **Accepted** — `models.py` is the first file written, `LLMClient` Protocol mandatory | Reduces future refactor cost to zero; provider swap = new Client class. |
| 3 | **Accepted** — renamed `CitationVerifier` → `AuthoritySupportChecker`, four-agent seam table codified in STANDARDS § 3.3 | Eliminates the ambiguity Codex flagged. |
| 4 | **Accepted** — `unverifiable` is a first-class `FindingKind`, prompts require it when source absent, hallucination rate measured deterministically | Reviewers forgive fewer agents; they don't forgive fake certainty. |
| 5 | **Rejected** — stacked PRs stay | Reviewability is itself a graded signal (decomposition + reflection). Cost (~30min) bought back via tighter seams. Discussed with user; explicit override. |

### 14.2 Codex challenge — round 2 (post-spec-draft)

Run via `/codex` with the full spec + STANDARDS as input. Six findings, all material.

**Verbatim core findings:**

1. **Time budget unrealistic.** "`~2.5h` is not real for what is listed. This is a `4-5h` PR unless most scaffolding already exists." Drop `PROGRESS.md`, per-agent latency histogram, N=3 variance, prompt snapshot tests, fancy per-finding diff. Make `CitationExtractor` regex-first — burning an LLM call on extraction is bad ROI.
2. **`Span(start, end, text)` is a footgun.** "LLMs are bad at exact char offsets... `text` and `start/end` will disagree; then you need a rule for which one wins." Better: LLM emits `Span(doc_id, quote)`, code grounds. Also: `DocumentKind` hardcoded to four docs is not stable IR (PR 2 adds authorities); `FactDiscrepancy.contradicting_evidence` should be `list[Span]`; spec 001 ships half the IR promised in STANDARDS § 3.1; `AgentResult[Generic[T]]` is awkward in Pydantic v2 schema generation — use a discriminated union.
3. **Eval methodology weak.** "Jaccard on `summary + evidence` is weak. Failure modes: correct finding different wording = false miss; wrong finding with overlapping nouns = false hit." Match on `(kind, doc_id, motion span overlap, evidence span overlap)`. **The "hallucination" check is offset/text consistency, not hallucination.** "The model can cite random real substrings and pass." Rename or strengthen.
4. **Serial orchestration unjustified.** `CitationExtractor` and `CrossDocConsistencyChecker` are independent and should fan out from day one.
5. **Acceptance thresholds are theater with N=5-8.** "`recall ≥ 0.5` means 3/6. That is fine as a smoke gate. It is not meaningful signal." Gate on counts: `matched_gold ≥ 3`, `ungrounded_findings == 0`. Report percentages in markdown, gate on counts.
6. **Amateur-hour callout.** "Expecting the LLM to emit exact character offsets, then calling offset agreement a hallucination metric." Secondary: STANDARDS promises full IR, spec ships half.

**Decisions taken in response:**

| # | Decision | Concrete change |
|---|---|---|
| 1 | **Partial accept** | Budget revised up to ~4h (honest). Regex-first `CitationExtractor` kept (it's a real ROI win). **Scope cuts rejected** — `PROGRESS.md`, prompt snapshot tests, latency histogram, and N=3 variance stay. Discipline is the deliverable; we eat the budget. |
| 2 | **Accept — biggest delta** | Full § 4 rewrite. `Span(doc_id, quote)` with offsets derived in code (STANDARDS § 3.6 added). `Document.id` separated from `DocumentKind`. `FactDiscrepancy.contradicting_evidence: list[Span]`. Full IR landed in spec 001 (`Claim, Quote, QuoteCheck, AuthorityCheck, EvidenceRef`). `AgentResult` → discriminated union. |
| 3 | **Accept** | § 7.2 rewritten. Structural matching; `summary` text excluded from score. "Hallucination check" renamed to `grounding_integrity` with honest semantics; added `cited_doc_in_scope`. Real semantic-grounding gap deferred to `REFLECTION.md`. |
| 4 | **Accept** | § 6 now uses `asyncio.gather` over the two independent agents from day one. |
| 5 | **Accept** | § 7.4 gates on counts (`--min-matched-gold 3 --max-grounding-failures 0 --max-cited-doc-scope-failures 0`). Percentages reported in markdown for trend. |
| 6 | **Accept — owned** | The offset-emitting LLM was the embarrassing thing. Whole IR rewrite + grounding-metric rename address it. STANDARDS now codifies "LLMs emit quotes, never offsets" (§ 3.6). |

### 14.3 Codex review — incremental (post-PR-open)

Per RELEASE § 8, `/codex review` runs on the open PR as commits land, not only at the end. Findings logged here with the commit that addressed them.

#### Round A — after commit `chore(deps)` (`05a90a3`)

Three findings, all accepted. Fixed in commit `fix(deps)` (`4c4eea1`).

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | P1 | `openai >= 1.30` lower bound predates `client.responses` / `client.beta.chat.completions.parse`; a clean install can satisfy the pin and then fail at the first structured-outputs call. Codex verified against openai-python v1.30.0 API reference. | Raised floor to `>= 1.50`. |
| 2 | P2 | `mypy.overrides.module = ["rapidfuzz.*", "pytest_socket.*"]` covers submodules only; top-level `from rapidfuzz import fuzz` still trips `--strict`. | Added bare module names alongside wildcards. |
| 3 | P2 | `--allow-unix-socket` in pytest addopts isn't needed for FastAPI `TestClient` (httpx-based, in-process). Broader than the stated policy. | Removed; `--disable-socket` alone enforces no-network policy. |

Commit-message review: clean.

#### Round B — after implementation commits

To be appended once `feat(models)` through `docs` commits land.

### 14.4 Codex review — final pre-merge

Run before flipping the PR from Draft to Ready-for-review. Same protocol as 14.3; the verbatim output gets pasted here in full.

## 15. Test plan

- `test_models.py` — discriminated-union AgentResult round-trips JSON; Pydantic validators fire on bad shapes.
- `test_sources.py` — normalized substring (whitespace, case, smart quotes); fuzzy ratio threshold; not-in-corpus path.
- `test_citation_extractor.py` — regex matches every known form in the MSJ; LLM proposition attachment runs via FakeLLMClient with canned response; LLM fallback fires when regex returns < 3.
- `test_cross_doc_checker.py` — FakeLLMClient returns canned discrepancies; severity literal enforced; missing-record case emits empty list (no fabrication).
- `test_orchestrator.py` — `asyncio.gather` runs both agents; ungrounded spans drop the finding and tag `outcome="partial"` on the agent result.
- `test_prompts.py` — snapshot test for every `PROMPT_VERSION` constant; rendering a prompt with a fixed input must equal the committed snapshot. Drift → fail. Update only by bumping `PROMPT_VERSION` and regenerating.
- `test_observability.py` — structured JSON formatter emits `agent_name`, `trace_id`, `prompt_version`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `outcome`; canonical fields present on every agent path.
- `test_main.py` — POST `/analyze`, assert `VerificationReport` schema, FakeLLMClient via dependency override.
- `pytest --disable-socket` (via `pytest-socket`) — zero network in unit tests.

## 16. Acceptance criteria

- [ ] `POST /analyze` returns a populated `VerificationReport` against the real four documents (manual smoke).
- [ ] `pytest -q` passes with sockets disabled.
- [ ] `mypy --strict backend/` clean.
- [ ] `ruff check . && ruff format --check .` clean.
- [ ] `python run_evals.py` runs end-to-end, writes JSON + Markdown reports.
- [ ] Baseline eval committed at `evals/baseline_report.md`.
- [ ] CI gates pass: `matched_gold_findings ≥ 3`, `grounding_integrity_failures == 0`, `cited_doc_in_scope_failures == 0`.
- [ ] N=3 variance run completes; `recall_stdev` and `precision_stdev` reported.
- [ ] Prompt snapshot tests pass; both prompts have `PROMPT_VERSION = "1.0.0"`.
- [ ] `PROGRESS.md` shipped with this PR.
- [ ] PR body links this spec and quotes the baseline counts (matched / total gold, emitted, grounding failures) + per-agent p95 latency.
