# AI-Based Development Cycle (used for this challenge)

How I actually used AI to build this submission. Honest about what worked,
what was wasteful, and what I'd keep / drop next time. Companion to
`REFLECTION.md` — that file is about the code; this one is about the
methodology.

## Tool stack

| Tool | Role | When |
|---|---|---|
| **Claude (Sonnet/Opus via Claude Code)** | Primary author. Reads spec, writes code, runs tests, drives git. | Continuously — every commit on this repo has Claude as the writing agent. |
| **OpenAI Codex CLI** | Adversarial reviewer. Second opinion on plan, IR, and per-PR diffs. | Before each spec is implemented; after each major commit lands. |
| **git-spice** (`gs`) | Stacked-PR manager. | Branch creation, restack on merge, `gs branch submit`. |
| **gh CLI** | GitHub operations (PR open / close / ready). | Manual cleanup when `gs` and bare `git push` collided. |
| **Python + Pydantic v2 + FastAPI** | Implementation stack — same as the brief's scaffold. | The actual product. |
| **ruff + mypy --strict + pytest + pytest-socket** | Local CI gates. | Pre-commit, pre-push, before flipping each PR to Ready. |

I did NOT use: GitHub Actions / external CI (skipped to keep budget intact),
Cursor / GitHub Copilot inline, voice input, or any agent orchestration
framework beyond the harness I built in this repo.

## The cycle, per spec

```
1. Brainstorm + draft spec.md
   ↓
2. Codex challenge on the draft  ←──┐  ("/codex challenge"; adversarial mode,
   ↓                                │   tries to break the design)
3. Update spec; log decisions ──────┘
   ↓
4. Implement, one commit per logical unit
   ↓
5. Codex review per commit OR per logical group  ←──┐  ("/codex review"; reads
   ↓                                                 │   the diff vs base, flags
6. Fix commits address findings ────────────────────┘   P1/P2 with severity)
   ↓
7. Final pre-merge sweep (all gates green + last codex pass)
   ↓
8. Flip Draft → Ready
```

Each codex round produces a verbatim transcript in
`specs/NNN-*/spec.md § 14`. The decisions taken (accept / reject / partial)
are recorded alongside the findings — so the next reader can see whether
I rationalized away a real critique.

## Anti-patterns I deliberately avoided

These are the failure modes of AI-assisted development I've watched happen
to myself and others. Calling them out so this file isn't just a Vendor of
Methods piece.

- **"Looks plausible" merge.** Code that compiles, tests that pass, but the
  test covers the wrong invariant. Counter: `STANDARDS.md` says every
  load-bearing rule is exercised by a test that fails when the rule is
  broken — not just a happy path.

- **AI rationalizing its own taste.** Asking Claude "is this a good
  design?" gets you "yes, here's why" 100% of the time. Counter: Codex
  is a separate process, separate model family, instructed to be
  adversarial. It disagreed with me on real things (offsets in `Span`,
  spec 003 surface forward-ship, the original time budget for spec 001).

- **Test theater.** Adding tests because "more tests good". Counter:
  during the simplification pass I dropped 41 tests across config,
  observability, sources, documents_io. Each dropped test had no
  load-bearing assertion — it tested implementation details (cache
  identity, frozen semantics, missing-file path).

- **Spec drift.** Code lands; spec stays at the "this is what I'll
  build" state. Counter: after every significant Codex round, spec § 4
  (the IR), § 5 (agents), § 7 (eval) get rewritten in place. The spec
  is the design document of what shipped, not what I planned.

## What worked

1. **Spec-first, with Codex challenging the spec before code.** Codex
   round 1 (pre-implementation) shifted the eval harness from spec 002
   into spec 001 and renamed `CitationVerifier` to
   `AuthoritySupportChecker`. Both decisions saved more time than the
   review took. Round 2 caught the offset-emitting `Span` design before
   it had any code attached — fix cost is order-of-magnitude lower
   pre-implementation than post.

2. **Typed IR as the API between agents.** No raw text crosses an agent
   boundary in this project. Every agent input + output is Pydantic v2.
   When the discriminated `AgentResult` union forced me to give each
   agent a stable result kind, the eval harness wrote itself — there
   was no string parsing surface to get wrong.

3. **Code-as-judge wherever possible.** Citation regex catches 10/10
   real cites; LLM only attaches propositions. `SourceRegistry.find`
   uses normalized substring + `rapidfuzz.partial_ratio`; no LLM judge
   needed for grounding. Both decisions came from the Codex prompt
   *"never let the LLM do what `str.find` can"*.

4. **Eval baseline committed.** A reviewer opens the repo and sees
   numbers without running anything. The numbers are explicitly labeled
   `--mode fake` so they can't be mistaken for model evaluation. The
   honest framing IS a feature.

## What was wasteful

These are real time sinks I'd cut next round.

- **STANDARDS.md + RELEASE.md at ~1500 lines combined.** They're useful
  evidence of engineering discipline for a grader, but on a 6h budget
  they ate the equivalent of an entire agent. Half their content would
  have served the same purpose.

- **Three rounds of Codex review with full appendix logs.** Each round
  costs ~10 minutes wall clock + ~10 minutes of read-and-respond.
  Rounds A and B caught real bugs (worth the cost). The "post-each-
  commit" review cadence was overkill — one round per spec at the
  inflection point (mid-implementation) would have caught the same bugs.

- **PR-size violation followed by retroactive split.** I knew the 500 LOC
  ceiling was in my own `RELEASE.md`. I built past it anyway. The fix —
  splitting into 4 stacked PRs + simplifying — cost ~75 min that should
  have been agents. Lesson: the AI tools make it easy to keep typing;
  the discipline that catches "this commit is now too big" is on me.

- **Stacked-PR ceremony for a 6h take-home.** `git-spice` is the right
  tool for multi-week refactors with multiple reviewers; for a solo
  6h project, 2-4 clean commits on one branch would have been enough.
  I spent ~30 min on the split / restack / submit dance that wasn't
  product work.

## What I'd change next round

Concrete, in order of leverage:

1. **One Codex round per spec at the inflection point, not per commit.**
   Two AI reviews (pre-implementation challenge + final pre-merge) is
   the right cadence. Per-commit reviews catch the same things, slower.

2. **Cap process docs at 200 lines total.** STANDARDS + RELEASE compressed
   to a single 200-line `CONTRIBUTING.md` would carry the same signal.
   The current document corpus is useful as evidence but the marginal
   word stopped earning grader signal somewhere around line 600.

3. **Run a budget tracker as a first-class artifact.** A `TIME_LOG.md`
   that records "20 min spec drafting / 15 min Codex round / 1.5h
   implementation" per slice. I lost track of time twice during this
   submission and only noticed at the simplification pass. Without the
   tracker, the "is this over-engineering?" question only gets asked at
   end of day.

4. **AI-pair sanity prompt at every hour mark.** Hand the AI a fresh
   context with "I've built X, the brief asks Y, I have Z time left.
   Tell me bluntly: am I optimizing for the right thing?" This is the
   AI version of the bathroom mirror — works much better when the
   model has no investment in continuing the current direction.

## Closing note

The cycle in this document is honest about what I followed, not
prescriptive. The right cycle for a different problem (longer budget,
multiple reviewers, production stakes) would weight the ceremony
differently. For a 6h adversarial-verification take-home, half the
process I ran was correctly leveraged and half was self-imposed
overhead. `REFLECTION.md` enumerates the technical consequences;
this file enumerates the process consequences.
