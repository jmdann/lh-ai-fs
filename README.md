# BS Detector

Legal briefs lie. Not always intentionally — but they do. They cite cases that don't say what they claim. They quote authority with words quietly removed. They state facts that contradict the documents sitting right next to them.

Your task: build an AI pipeline that catches it.

## Status (spec 001 — foundation + cross-doc + eval)

| Item | State |
|---|---|
| `POST /analyze` returns structured `VerificationReport` | ✅ |
| `python run_evals.py` runs end-to-end with count gates | ✅ |
| `CitationExtractor` (regex-first + LLM) | ✅ |
| `CrossDocConsistencyChecker` (motion vs records) | ✅ |
| `SourceRegistry` grounding boundary (STANDARDS § 3.6) | ✅ |
| `QuoteChecker` + `AuthoritySupportChecker` | ⚪ spec 002 |
| `ConfidenceScorer` + `JudicialMemoWriter` + UI rewrite | ⚪ spec 003 |
| `REFLECTION.md` | ⚪ spec 003 |

**Tests:** 136 passing, no network.
**Eval baseline (fake mode, N=3):** 4/4 discrepancies + 5/5 citations matched, 0 grounding failures. Committed at `evals/baseline_report.md`.

See `PROGRESS.md` for the full status table and `specs/001-foundation-evals-crossdoc/spec.md` for the design + codex review log.

## How to run the eval suite

```bash
# Default — fake LLM, no API key required, ~1s wall clock.
# Tests the harness + matcher; not a real model evaluation.
python run_evals.py

# Real LLM, requires OPENAI_API_KEY.
python run_evals.py --mode real --runs 3
```

Reports land at `evals/eval_report.{json,md}`. CI gates are count-based (`matched_gold_findings ≥ 3`, `grounding_integrity_failures == 0`, etc.) — see `evals/README.md` for the rationale.

## Setup

### Docker (recommended)

```bash
cp .env.example .env      # Add your OpenAI API key
docker compose up --build
```

The API runs at `http://localhost:8002`. The UI runs at `http://localhost:5175`.

Both services hot-reload — edit files on your host and changes appear automatically.

### Manual Setup

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add your OpenAI API key
uvicorn main:app --reload
```

The API runs at `http://localhost:8002`.

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI runs at `http://localhost:5175`.

## The Task

Inside `backend/documents/` you'll find a small case file: a Motion for Summary Judgment in a personal injury lawsuit (*Rivera v. Harmon Construction Group*), along with a police report, medical records, and a witness statement.

Build a multi-agent pipeline that analyzes these documents and produces a structured verification report. Your pipeline should:

**Core (Tier 1)**
- Extract all citations from the Motion for Summary Judgment
- For each citation, assess whether the cited authority actually supports the proposition as stated
- Flag direct quotes for accuracy
- Produce structured output (JSON) — not a wall of prose

**Expected (Tier 2)**
- Build an eval harness that measures your pipeline's output quality. It must be runnable via a single command (e.g., `python run_evals.py`). At minimum, measure precision (avoiding false flags), recall (catching known flaws), and hallucination rate (not fabricating findings). You choose the approach — there's no prescribed framework or tooling.
- Cross-document consistency check: compare facts stated in the MSJ against the police report, medical records, and witness statement
- Express uncertainty appropriately — "could not verify" rather than fabricating a finding
- Pass structured data between agents, not raw text blobs

**Stretch (Tier 3)**
- At least 4 well-defined agents with distinct, non-overlapping roles
- A confidence scoring layer: each flag rated by how certain the pipeline is, with reasoning
- A judicial memo agent: synthesizes the top findings into a one-paragraph summary written for a judge
- Agent orchestration that handles failures gracefully
- A UI that displays the report in a structured, readable way — not just raw JSON
- A reflection document explaining the tradeoffs you made and what you'd do differently

## Deliverables

1. A working `POST /analyze` endpoint that returns a structured verification report
2. Agent code with clear, named agents and explicit prompts
3. A runnable eval suite with instructions in your README on how to run it
4. A brief reflection (in the repo or as a separate file) on your design decisions and tradeoffs

## Time

6 hours. This is intentionally scoped beyond what most candidates will finish. Where you invest your time matters more than finishing everything. A well-tested pipeline that catches 3 flaws is stronger than an untested one that attempts 10.

## Evals

We run your eval suite as part of our review. Document how to run it in your README. We care more about thoughtful metric design than perfect scores — an eval that honestly reports 60% recall tells us more than one that reports 100% on cherry-picked cases.

## AI Usage

Use everything. That's the job. We want to see how you use it, not whether you do.

## Evaluation

We are evaluating:

1. How you decompose the problem into agents
2. How precisely you write prompts
3. The quality of your eval approach — do you measure what matters?
4. How far you get through the spec
5. How honest your reflection is

Not lines of code.
