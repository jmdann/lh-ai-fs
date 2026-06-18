# Release & Commit Strategy — BS Detector

How this take-home gets from spec files to a submitted artifact. Single source of truth for branch shape, commit conventions, PR flow, and what the reviewer ultimately reads.

## 1. Repositories

| Role | URL | Use |
|---|---|---|
| **Fork (origin)** | <https://github.com/jmdann/lh-ai-fs> | Where we push. All PRs land here. The grader gets this URL. |
| **Upstream** | <https://github.com/JuanVazTor/lh-ai-fs> | Original challenge repo. Read-only — only useful if upstream updates the docs / starter code. Never push. |

Local remotes (already configured):
```
origin    https://github.com/jmdann/lh-ai-fs.git    (fetch + push)
upstream  https://github.com/JuanVazTor/lh-ai-fs.git (fetch + push, but we never push)
```

## 2. Trunk

`main` is trunk on the fork. Every stacked PR rebases onto `main`. No long-lived `develop` — we're not Phoenix-shop; this is a 6h sprint.

```
upstream/main  ──●─────────────────────────────● (frozen reference)
                                                \
fork/main      ──●──[1/3]──[2/3]──[3/3]──●──── v1.0-submission (tag)
                                          \
                                           release notes
```

## 3. Branch shape

Three stacked feature branches plus `main`. Names match spec dirs so they're trivially searchable.

```
main
└── feat/001-foundation-evals-crossdoc      ← stack base
    └── feat/002-quote-authority             ← parent: feat/001
        └── feat/003-orchestrator-ui-reflection  ← parent: feat/002
```

Each branch:
- Carries multiple small commits (Conventional Commits, see § 5).
- Opens a PR whose **base** is its parent branch, not `main`. git-spice handles this.
- Stays ≤ 500 LOC of application code (excludes tests, fixtures, generated, lockfiles — same exclusions as `~/.claude/rules/git-workflow.md`). If a single spec blows the limit, it splits into `[Na/3]` + `[Nb/3]` sub-stacks.

## 4. git-spice setup

[git-spice](https://github.com/abhinav/git-spice) (`gs`) manages the stack. One-time:

```bash
gs auth login      # GitHub PAT — already done at user level
gs repo init       # detect trunk; pick `main`
```

Per-branch creation:
```bash
gs branch create feat/001-foundation-evals-crossdoc
# ... work ...
gs branch submit  # opens / updates PR with base = parent branch
```

Restack on top-of-stack updates:
```bash
gs repo sync       # ff main, prune merged branches
gs stack restack   # rebase every branch in the stack onto its (updated) parent
```

Navigation:
```bash
gs up / gs down    # move along the stack
gs ls              # see stack
```

## 5. Commit conventions

Conventional Commits. Subject ≤ 72 chars, imperative, lowercase scope. Body when the *why* isn't obvious.

```
<type>(<scope>): <subject>

<optional body>

<optional footer: refs spec, codex findings, etc.>
```

| Type | When |
|---|---|
| `feat` | New behavior visible to a user or downstream agent |
| `fix` | Bug fix (not "fix typo" — that's `chore`) |
| `test` | Tests only |
| `refactor` | Internal structure, no behavior change |
| `perf` | Performance change with a measurable delta |
| `docs` | Spec, README, RELEASE, STANDARDS, REFLECTION |
| `chore` | Tooling, deps, lockfiles, formatting-only commits |

Examples we'll actually use:
```
feat(models): add quote-based Span + discriminated AgentResult

LLM emits Span(doc_id, quote); orchestrator grounds to offsets in code.
Closes the offset-drift footgun Codex round 2 flagged.

Refs: specs/001-foundation-evals-crossdoc/spec.md § 4, § 14.2
```
```
feat(agents): add regex-first CitationExtractor

Regex covers Cal. citation forms; LLM only attaches propositions
and handles fallback when regex returns < 3 hits.

Refs: specs/001-foundation-evals-crossdoc/spec.md § 5.1
```
```
test(eval): add structural matcher + count-based CI gates

Match key is (kind, doc_id, motion_span_overlap ≥ 0.6).
summary text is excluded from scoring — prompt phrasing must not leak.

Refs: specs/001-foundation-evals-crossdoc/spec.md § 7.2, § 7.4
```

Commit boundaries inside a branch: one logical unit per commit so `git log` reads as a narrative. Reviewers (and graders) navigate by commit, not by diff.

## 6. PR conventions

### Title

```
[N/3] feat(NNN): <subject>
```

`N` = position in stack. `NNN` = spec number. Example: `[1/3] feat(001): foundation + cross-doc + thin eval harness`.

### Body template

```markdown
## Summary
<2-3 bullets: what behavior exists after this lands>

## Spec
- [specs/NNN-*/spec.md](path) — drafted before code.

## Eval delta vs previous PR
| Metric | Before | After |
|---|---|---|
| matched_gold_findings | N | M |
| grounding_integrity_failures | 0 | 0 |
| precision | x.xx | y.yy |
| recall | x.xx | y.yy |
| unverifiable_precision | — | y.yy (new) |

## Cross-model review
- `/codex` challenge findings + decisions: see § 14 of spec.
- `/codex review` on this branch: <pending | clean | N findings addressed>

## Test plan
- [ ] `ruff check . && ruff format --check .`
- [ ] `mypy --strict backend/`
- [ ] `pytest -q --disable-socket`
- [ ] `python run_evals.py` (gates: ...)
- [ ] Manual smoke: `curl -X POST localhost:8002/analyze | jq`

## What's intentionally not here
<bullets pointing to the next spec/PR>
```

### Labels (manual on the fork)

- `tier:1-core`, `tier:2-expected`, `tier:3-stretch`
- `agent:<name>` per agent introduced
- `codex:challenged`, `codex:reviewed`

Labels make `gh pr list --label tier:1-core` legible to graders who want to triage by what's done.

## 7. Per-PR pre-flight checklist

Before `gs branch submit`:

```bash
# 1. Lint + types
ruff check .
ruff format --check .
mypy --strict backend/

# 2. Tests (no network)
pytest -q --disable-socket

# 3. Eval suite (real LLM calls; this is the only place we hit OpenAI in local dev)
python run_evals.py --min-matched-gold <gate>

# 4. Manual smoke
docker compose up --build -d
curl -sS -X POST http://localhost:8002/analyze | jq '.findings | length'

# 5. Stack hygiene
gs repo sync
gs stack restack
gs ls   # eyeball stack state
```

If any step fails, the PR doesn't open. No "I'll fix it in review."

## 8. Codex gates

Per STANDARDS § 7, each spec has pre- and post-implementation Codex passes.

- **Pre-implementation** — already run (`/codex challenge`). Findings + decisions recorded in `specs/NNN/spec.md § 14.1, 14.2`. Status: spec 001 has rounds 1 + 2 logged; specs 002 + 003 still need their pre-pass.
- **Post-implementation** — run `/codex review` on the open PR before merge. Verbatim output goes into `specs/NNN/spec.md § 14.3`. P1 findings block merge; P2 findings get an issue link or a follow-up commit.

## 9. Merge strategy

Rebase-and-merge per PR on GitHub. Reasons:
- Stack stays linear after each merge — no "Merge branch 'feat/001' into main" noise.
- Each spec lands as a contiguous block of commits, navigable via `git log --oneline --grep="(001)"`.
- The grader can `git checkout v1.0-submission~N..v1.0-submission~M` to see exactly one spec's deltas.

We do **not** squash-merge. Squashing collapses the Conventional Commit narrative we deliberately wrote.

Merge order: `[1/3]` → restack → `[2/3]` → restack → `[3/3]`. Each merge triggers `gs repo sync` on the local stack.

## 10. Branch cleanup

After each PR merges:
```bash
gs repo sync       # deletes merged local branches; ff main
```

The fork's remote branches are kept until v1.0 tag (useful if the grader wants to compare specs as branches).

## 11. Versioning + tagging

The take-home is a one-shot, but the artifact should still be tagged:

```bash
# after [3/3] merges
git checkout main && git pull
git tag -a v1.0-submission -m "BS Detector — submission for lh-ai-fs take-home"
git push origin v1.0-submission
```

`v1.0-submission` is what the submission email points at. If we ship a fix after submitting (rare; we resist the urge), it goes as `v1.0.1-postsubmit` with a note in the submission thread.

## 12. Submission deliverable

What the grader actually receives:

1. **Repo URL** — <https://github.com/jmdann/lh-ai-fs>
2. **Tag** — `v1.0-submission`
3. **Reading guide** — one paragraph in the submission email pointing to:
   - `README.md` § Status — what's green
   - `specs/` — three specs, in order
   - `evals/baseline_report.md` — actual numbers
   - `REFLECTION.md` — honest cuts + weakest points
   - PRs list (`gh pr list --state merged --base main`) — three stacked PRs as the navigation path

## 13. What is intentionally NOT in this strategy

- **CI on the fork.** No GitHub Actions workflow. Setting one up costs 30min and the only consumer is us. We run gates locally and paste the output into PR bodies. If the grader runs the eval suite (per the brief), they do it locally per the README.
- **Pre-commit hooks.** Same calculus. Manual `ruff` + `mypy` invocation in the pre-flight is enough discipline for a 6h project; hook config is ceremony.
- **Changelog file.** PR bodies and commit subjects ARE the changelog. `git log v0..v1.0-submission` reads cleanly.
- **Issue tracker discipline.** Three PRs, three specs. No issues. Defects that codex flags during review become commits on the same branch.

## 14. Emergency cuts

If we hit the 5h mark with `[1/3]` not yet ready:

1. **Drop spec 003 entirely.** Don't open the PR. Update `README.md` Status to mark Tier 3 as "not implemented; see REFLECTION § cuts."
2. **Submit with only `[1/3]` merged** if necessary. Spec 001 alone is a credible Tier 1+ submission (the brief's whole point: "well-tested pipeline that catches 3 flaws > untested one attempting 10").
3. **Tag whatever state main is in** as `v1.0-submission`. Honest > complete.

The slicing was designed so this emergency cut is a one-line README change, not a refactor.
