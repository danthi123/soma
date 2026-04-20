# Gitea Actions — status + revival guide

**Status as of 2026-04-20:** Gitea Actions is **dormant**. No runners
are configured on `git.dant123.com/dant123/soma` (GET
`/api/v1/repos/dant123/soma/actions/runners` returns
`{"runners":[], "total_count":0}`). Any workflow file in this
directory that lands here is config-only — nothing executes it until
runners come online.

**Active CI platform:** GitHub Actions. See `.github/workflows/` at
the repo root. Both remotes (`origin` → Gitea, `github` → GitHub) get
pushes; GitHub Actions runs automatically on push to `main` and on
PRs.

## Why the test + lint CI lives on GitHub, not Gitea

- Zero-config for public repos; no runner to babysit.
- Matches the "one CI source of truth" principle.
- Can migrate to Gitea later without losing commit history (the
  removed `.gitea/workflows/ci.yml` lives in git history — see commit
  trail below).

## Why these workflow files are still here

`bench-regression.yml`, `client-ts.yml`, `helm.yml`, and
`helm-release.yml` target concerns that have no GitHub equivalent yet
(nightly benchmark regressions, TypeScript client CI, Helm chart
release publishing). They're kept dormant for future revival rather
than ported immediately, because porting each involves non-trivial
adaptation (e.g., bench-regression assumes benchmark fixtures that
may not be CI-friendly on GitHub runners).

## Reviving Gitea Actions — pre-enable checklist

Before re-enabling Gitea CI for any of these workflows, walk the
review-round checklist below. Each item exists because the workflow
was written at a point in time that may no longer match the repo's
current state.

### 1. Runner setup (platform-level, one-time)

- [ ] Provision a `gitea/act_runner` instance. Natural home: Unraid,
  using the same `claude-code-runner` pattern. Private-network-only
  is fine; runners poll Gitea for jobs.
- [ ] Register the runner against the repo (or, if you want it shared
  across repos, at org or system scope).
- [ ] Confirm via `curl -H "Authorization: token $TOKEN"
  https://git.dant123.com/api/v1/repos/dant123/soma/actions/runners`
  that the runner shows up with `status: "online"`.

### 2. Workflow-by-workflow revival review

When re-enabling any workflow, check these against `main` HEAD:

- [ ] **Marker list:** do the pytest marker excludes
  (`-m "not X and not Y ..."`) still match the marker definitions in
  `pyproject.toml [tool.pytest.ini_options]`? New markers added since
  the workflow was written should be added to the exclude list (or
  dropped into a separate, gated workflow).
- [ ] **Extras list:** do the `pip install -e ".[...]"` extras still
  exist in `pyproject.toml`? And, conversely, are there new extras
  the workflow ought to install (e.g., if a new optional feature
  ships)?
- [ ] **Pinned tool versions:** does the pinned `ruff==X.Y.Z` (or
  mypy version) still match the one used locally? Stale pins produce
  CI-only failures. Bump deliberately.
- [ ] **Assertion set:** the old `ci.yml` enforced
  `ruff format --check` and `mypy --strict`. Both were aspirational
  (the current repo has ~100 format-divergent files and mypy strict
  has not been run cleanly). Revival needs either a dedicated
  "repo-wide reformat + mypy-strict adoption" commit first, OR a
  decision to relax those checks in the revived Gitea CI to match
  what `.github/workflows/` enforces today.
- [ ] **Concurrency / cancellation:** add a `concurrency:` block
  grouped by `${{ github.ref }}` to avoid queueing stale jobs. The
  `.github/workflows/` files show the pattern.
- [ ] **Timeout budget:** nightly jobs (bench-regression) need a
  generous `timeout-minutes`; test jobs should aim for ≤10 min.

### 3. Diverge-or-mirror decision

Decide PER WORKFLOW whether Gitea CI should mirror GitHub CI or add
something GitHub can't do:

- **Mirror** — same matrix, same assertions. Low value (two CIs
  running the same tests) but does provide redundancy if GitHub ever
  has an outage.
- **Diverge** — Gitea runs things GitHub can't, e.g.:
  - GPU-bound tests (`-m "cuda"`) if the Gitea runner has a GPU
  - `SOMA_S3_INTEGRATION_BUCKET=...` real-S3 tests with private
    creds (GitHub secrets work but your ops may prefer self-hosted)
  - Longer bench-regression sweeps that exceed free-tier runner limits
  - `.gitea/workflows/bench-regression.yml` is closest to this pattern

My recommendation when revival day comes: **diverge**. Let GitHub
cover the fast feedback loop (tests + lint on every push) and let
Gitea cover the heavy / private / GPU workflows. That uses both
platforms for what each is best at.

## Historical context — the removed `ci.yml`

The original Gitea `ci.yml` ran ruff check + ruff format --check +
mypy (strict) + pytest. It was written 2026-04-15, dormant from day
one (no runners), and removed 2026-04-20 as part of the "GitHub as
the active CI platform" decision. Recoverable from git history:

```bash
git log --all --full-history -- .gitea/workflows/ci.yml
git show <sha>:.gitea/workflows/ci.yml
```

The GitHub replacement lives at `.github/workflows/test.yml` +
`.github/workflows/lint.yml`.
