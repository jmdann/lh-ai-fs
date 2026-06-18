# Project Standards — BS Detector

Standards every PR is held to. Linked from each spec under "Standards". Updated after the spec-001 adversarial review with Codex (see `specs/001-foundation-evals-crossdoc/spec.md` § Cross-model review).

## 1. Python — PEP 8

- **PEP 8** is the baseline style guide for all Python code.
- **Tooling**:
  - `ruff check .` — lint (replaces flake8/pylint/isort).
  - `ruff format --check .` — formatting (PEP 8 compliant, black-compatible).
  - `mypy --strict backend/` — static types on the whole backend.
- **Type hints** mandatory on every function signature and class attribute.
- **Pydantic v2** models for all agent inputs/outputs and the public API schema. No raw dicts crossing agent boundaries.
- **Docstrings**: one-line summary on public functions when the name isn't enough. No multi-paragraph docstrings.
- **Naming**: `snake_case` for functions/vars, `PascalCase` for classes, `SCREAMING_SNAKE` for module constants.
- **Imports**: stdlib → third-party → local, grouped and sorted by ruff.
- **Line length**: 100 (ruff default override; PEP 8's 79 is too tight for typed signatures).

## 2. Twelve-Factor App (applicable subset)

Reference: <https://12factor.net/pt_br/>.

| # | Factor | How we apply it |
|---|--------|-----------------|
| **III** | Config in env | `OPENAI_API_KEY`, model name, temperature, eval thresholds via `os.environ` / `.env`. Loaded through Pydantic `BaseSettings`. Never hardcoded. |
| **IV** | Backing services as attached resources | OpenAI client behind an `LLMClient` Protocol. `Depends` only at the API edge; agents take the protocol, not the concrete client. Swappable fake for tests/evals. |
| **VI** | Stateless processes | Agents are pure functions over typed inputs. No module-level mutable state, no in-process caches that survive a request. |
| **VIII** | Concurrency | `uvicorn --workers N` for the API. Eval harness fans out per-item work with `asyncio.gather` + bounded `asyncio.Semaphore`. |
| **IX** | Disposability | Fast startup, graceful shutdown. Retries with exponential backoff (`tenacity`) on OpenAI calls. |
| **X** | Dev/prod parity | `docker compose` mirrors what the eval CI runs. Same Python version, same pinned model, `temperature=0` so dev runs == eval runs. |
| **XI** | Logs as event streams | Structured JSON to stdout. One log line per agent step: `agent_name, trace_id, prompt_version, latency_ms, prompt_tokens, completion_tokens, outcome`. |
| **XII** | Admin processes | `run_evals.py` lives in the same repo, same deps, same config loader as the API. |

**Not applicable at this scope:** I (one repo), II (covered by requirements + Docker), V (no deploy), VII (uvicorn binds).

## 3. Architectural rules (from Codex review of spec 001)

These are non-negotiable. They earn evaluation points 1 and 2 (decomposition + prompt precision).

### 3.1 Stable intermediate representation, first

Before any agent is written, `backend/models.py` defines the Pydantic v2 domain models that every agent consumes and emits:

```
Document, Span, Claim, Citation, Quote, QuoteCheck,
AuthorityCheck, FactDiscrepancy, Finding, EvidenceRef, AgentResult[T]
```

Agents take typed inputs and return typed outputs. **Never raw text blobs** between agents. The orchestrator is responsible for wiring these together, not the agents.

### 3.2 LLM client as Protocol, injected

```python
class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...
```

- Concrete `OpenAIClient` implements it.
- `FakeLLMClient` for tests returns pre-canned Pydantic objects.
- FastAPI `Depends(get_llm_client)` lives **at the route**, not inside agents. Agents receive the protocol as a constructor arg.
- Swapping providers = write a new `Client` class. No agent code changes.

### 3.3 Agent seams (no overlap)

| Agent | Job (one sentence) | Does NOT |
|---|---|---|
| `CitationExtractor` | Identify proposition spans + cited source references. | Make any judgment about correctness. |
| `QuoteChecker` | Compare quoted text in the motion against the cited source text — fidelity only. | Decide whether the quoted authority *supports* the proposition. |
| `AuthoritySupportChecker` | Given a proposition + validated source span, decide supports / contradicts / unverifiable. | Parse quotes or fetch sources. |
| `CrossDocConsistencyChecker` | Compare motion factual claims against record documents (police report / medical / witness). | Touch legal authorities. |
| `ConfidenceScorer` (Tier 3) | Rate each finding 0-1 with reasoning. | Generate new findings. |
| `JudicialMemoWriter` (Tier 3) | One-paragraph synthesis for a judge. | Re-rank or re-judge. |

Renaming from the original plan: `CitationVerifier` → `AuthoritySupportChecker` (Codex flagged the original name as overloaded).

### 3.4 Prompts as module constants

- Prompts live in `backend/agents/prompts/*.py` as module-level string constants. Never inline `f"""..."""` inside the agent function.
- Each prompt module exports `PROMPT_VERSION: str` (semver-ish). The eval harness logs the version with every run.
- A snapshot test (`tests/test_prompts.py`) freezes the rendered prompt for a fixed input. Changing the prompt without updating the snapshot fails CI.
- Hard cap: **40 lines** per prompt. Longer prompts mean the agent is wrong-sized — split it.
- Every prompt must include: role line, input contract (Pydantic schema), output contract (`response_format={"type": "json_schema", ...}`), refusal rules (when to emit `unverifiable`), one worked example.

### 3.6 Span identity — quote-only, no offsets in the IR

LLMs are bad at exact character offsets. Letting the model emit `start` / `end` integers is the fastest way to ship a pipeline whose "grounding" check passes a unit test and fails the first time a document gets normalized. **The type system, not policy, must rule this out.** Codex round B (PR #1 § 14.3) flagged the original "offsets are optional" shape as untrustworthy — the LLM could still emit integers and validate.

Rule:
- The `Span` type is exactly `{doc_id, quote}`. No `start`, no `end`, no offsets of any kind. Pydantic's `extra="forbid"` means any JSON the LLM emits with `start` / `end` fails at parse time.
- Grounding is a **boundary check** done by the orchestrator, never a `Span` field: `SourceRegistry.find(doc_id, quote)` returns hit / miss. On miss, the finding is dropped and a `grounding_integrity_failure` is logged on the corresponding `AgentResult`. Agents do not get to launder ungrounded quotes through.
- If a future spec (e.g. spec 003 UI rendering) genuinely needs offsets, it adds them as a separate `ResolvedSpan` type. The base `Span` stays narrow.

This caps how badly the LLM can lie: it has to produce a quote that actually appears in a document we loaded. "I cite a string that exists somewhere" is a much weaker hallucination than "I cite (137, 198) and the model gets to define what that means".

### 3.5 Source retrieval discipline

Codex's highest-risk callout: **the cited case law is not in the repo**. We have no retrieval layer in 6h.

Rule: `AuthoritySupportChecker` and `QuoteChecker` operate only on text we can ground. When the cited source text is unavailable, the agent **must** emit `unverifiable` with a reasoning sentence naming the missing source. Inferring meaning from the cite alone is a hallucination and counts against the eval `hallucination_rate`.

## 4. Determinism

- `temperature=0`, model pinned via `OPENAI_MODEL` env var.
- All sampling (eval shuffling, fixture order) seeds `random.Random(seed)` explicitly.
- Snapshot LLM responses for unit tests under `backend/tests/fixtures/llm/`. Unit tests never hit the network.
- Eval harness runs each case N=3 with the same seed and reports variance. Surfaces residual nondeterminism even at `temperature=0`.

## 5. Eval discipline (earns evaluation point 3)

- `tests/fixtures/gold_set.yaml` is **hand-authored by reading the source documents**, not by asking an LLM to generate it. Origin documented in `REFLECTION.md`.
- The gold set is written **before** the prompts. We don't rewrite gold entries to match observed agent behavior — that's the silent way to fake recall.
- `run_evals.py` reports per-agent and overall:
  - **Precision** — of findings emitted, fraction matching gold.
  - **Recall** — of gold flaws, fraction caught.
  - **Hallucination rate** — of findings, fraction citing spans / quotes / authorities that do not appear in the source documents (computed by string-grounding the cited span back to the source).
- Output: machine-readable `evals/eval_report.json` + human-readable `evals/eval_report.md` with per-finding diff.
- We commit a real baseline run as `evals/baseline_report.md` so reviewers see honest numbers without running anything.
- **Match findings by structure, not by string overlap on summary text.** Match key = `(kind, doc_id, motion_span_overlap ≥ 0.6, evidence_doc_set ⊇ gold_evidence_doc_set)`. Jaccard on the `summary` field rewards prompt phrasing over detection quality and fails honest paraphrases — banned.
- **Don't call offset/text consistency a "hallucination check".** It proves the agent didn't fabricate offsets; it does not prove the cited span actually supports the claim. The deterministic check we run is `grounding_integrity` (quote actually appears in `doc_id`). Real semantic grounding is documented as a known gap in REFLECTION.
- **CI gates on counts, not percentages, when the gold set is small.** With N ≈ 5-15 entries, `recall ≥ 0.5` is theater (3/6 satisfies it). Gate on `matched_gold_findings ≥ K` and `grounding_integrity_failures == 0`. Percentages live in the markdown report for trend visibility.

## 6. Testing

- `pytest` for backend, `vitest` for frontend.
- Unit tests use `FakeLLMClient` (rule 3.2).
- One integration test per agent against snapshotted LLM responses.
- One end-to-end test that POSTs `/analyze` and asserts the report shape.
- `tests/test_prompts.py` snapshot-tests every prompt (rule 3.4).
- Network calls in unit tests are blocked via `pytest-socket` or equivalent.

## 7. Spec structure (required sections)

Every spec under `specs/NNN-*/spec.md` must include these sections — they're what makes the slicing legible to a reviewer and force honesty up-front:

1. **Goal** — one paragraph; what's true after this PR lands.
2. **Non-goals** — explicit list of what's deferred, with a pointer to the spec that owns it.
3. **Deliverables** — numbered list of files / artifacts.
4. **Domain models / agent contracts** — Pydantic shapes touched, one-sentence contracts per agent.
5. **Repository layout after this PR** — tree of files added/changed.
6. **Standards delta** — anything new this PR introduces to `STANDARDS.md`.
7. **How this earns evaluation points 1-5** — table mapping brief criteria → concrete artifacts in this PR. Forces every spec to defend its scope against the rubric.
8. **Cross-model review** — two subsections:
   - **Pre-implementation Codex challenge** — run `/codex` with this spec as input; quote core findings verbatim; record accept/reject + rationale per finding.
   - **Post-implementation Codex review** — run `/codex review` once the PR is green; appended after the fact.
9. **Test plan** — bullets, one per test file.
10. **Acceptance criteria** — checkboxes the PR must satisfy before merge.

The "evaluation points" mapping (§7) and "Cross-model review" section (§8) are non-negotiable. They're the difference between a spec that looks engineered and one that *is* engineered. Reviewers can't see your thinking; the spec is the evidence.

## 8. Stacked PRs

This project uses [git-spice](https://github.com/abhinav/git-spice) stacked PRs over `origin/main`. Three stacks, one per spec:
- `[1/3] feat(001): foundation + cross-doc + thin eval`
- `[2/3] feat(002): quote fidelity + authority support`
- `[3/3] feat(003): orchestrator + UI + reflection`

Each PR must be independently mergeable, revertible, and testable. Behavior-preserving until the activation PR — Tier 3 polish features (confidence, memo, UI) are gated by feature flags / env vars and default off until 003 lands.

PR title carries `[N/3]`; PR body links the spec and the eval delta.

## 9. Enforcement (CI checklist)

Every PR must pass:

```bash
ruff check .
ruff format --check .
mypy --strict backend/
pytest -q
python run_evals.py --fail-under-recall 0.5 --fail-over-hallucination 0.10  # thresholds rise per PR
npm --prefix frontend run lint    # when frontend touched
npm --prefix frontend test -- --run
```

Bouncing reasons (non-exhaustive): Python without types, network in unit tests, raw dicts crossing agent boundaries, `Depends` inside agent code, inline prompts, prompt change without `PROMPT_VERSION` bump or snapshot update, gold set rewritten to match agent behavior.
