# Phase 40: Third-Party Security Review Preparation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Prepare the auth / token-lifecycle / rate-limit / revocation
code for external security review. No code changes — deliverables are
(1) a review package doc, (2) an automated vuln-scan CI step, and (3)
a cryptographic inventory. Designed so a reviewer (internal senior or
external contractor) can read one document and find every load-bearing
surface without spelunking.

**Architecture:**
- `docs/security-review-package.md` — the handoff doc:
  * Scope (what's in, what's out)
  * Threat model (attacker capabilities, assumed trust boundaries)
  * Attack surface table (endpoint → auth class → scopes → storage
    touch)
  * Cryptographic inventory (algorithms, key sizes, key storage, KDF
    choices)
  * Known non-goals (not a WAF, not a secret vault, not SOC 2
    certified — these are operator's responsibility)
  * Explicit assumptions (e.g. "rate limiter is a defense-in-depth
    layer, not a WAF; deploy behind nginx/Cloudflare for public
    internet")
  * Known tech-debt / caveats the reviewer should know about
- CI: `.gitea/workflows/security-scan.yml` runs on every PR:
  * `pip-audit` — dependency CVE scan (blocks on HIGH/CRITICAL)
  * `bandit` — Python static security analyser (non-blocking warning
    mode; baseline of existing findings pinned in
    `.bandit.yml` so only new findings fail)
  * `ruff check --select S` — ruff's security lint subset
- No source changes. If the scanners surface real issues during this
  phase, they get fixed in a follow-up — this phase is scaffolding.

**Out-of-scope:**
- Actually performing the review.
- Remediation of whatever findings the review surfaces (separate
  phase with a name-and-number based on what gets found).
- SOC 2 / ISO 27001 / specific-regulation compliance work — those
  are operator concerns layered on top of the technical artifacts.

**Out-of-scope (central merge):** `CHANGELOG.md`, `deferred-items.md`.

---

### Task 1: Review package doc

**Files:**
- Create: `docs/security-review-package.md`

**Required sections:**
1. **Scope** — which files are in scope:
   * `src/soma/auth.py` (JWT issue/verify/refresh)
   * `src/soma/auth_revocation.py` (FileBlocklist + RedisBlocklist +
     hashed-token store)
   * `src/soma/rate_limit.py` (TokenBucket + RateLimiter)
   * `src/soma/serve.py` (all endpoints with `require_auth` decorator)
   * `src/soma/forget_audit.py` (audit trail integrity)
   * `src/soma/cli.py` (auth-related CLI verbs: issue, verify, revoke,
     rotate-secret, refresh)
2. **Out-of-scope** — vector backends, memory layer, LLM backends,
   bundle storage (except where they touch auth state).
3. **Threat model**:
   * Attacker has network access, no host access.
   * Attacker may have a valid token (compromised user).
   * Attacker may have a stale-but-revoked token.
   * Attacker controls arbitrary request bodies (but not headers
     beyond `Authorization`).
   * Attacker has access to this repo (public assumption).
4. **Trust boundaries** — diagram-ish text:
   * HTTP → FastAPI route: `require_auth` is the gate.
   * FastAPI → storage: auth'd principal's scopes control what
     bundle name + operation is allowed.
   * HS256 secret / RS256 private key are trusted (operator's
     responsibility to store securely).
5. **Attack surface table** — for each endpoint:
   * Method + path
   * Auth required (yes/no; which scopes)
   * Inputs (body fields, path params)
   * State touched (MemoryLayer, blocklist, audit sink)
   * Error cases that leak info vs. redacted
6. **Cryptographic inventory**:
   * HS256 via `pyjwt[crypto]>=2.8` — which 2024 hardening landed.
   * RS256 private-key file format, minimum recommended bit size.
   * SHA256 (jti hashing in FileBlocklist + RedisBlocklist).
   * No custom crypto anywhere — all via `pyjwt` / `hashlib`.
7. **Known non-goals** — things SOMA is NOT and doesn't claim to be.
8. **Known caveats** — things worth reviewer attention:
   * File-backed blocklist has a 30-second poll window (documented).
   * In-proc rate limiter is not a WAF.
   * `docs/gdpr.md` audit trail is append-only JSONL; rotation /
     retention is operator responsibility.
   * Refresh tokens don't rotate — same JTI new exp.
9. **How to run scanners locally** — one-liners for pip-audit +
   bandit + ruff security.

**Step 5:** `git commit -m "docs(security): review package handoff doc"`

---

### Task 2: CI security-scan workflow

**Files:**
- Create: `.gitea/workflows/security-scan.yml`
- Create: `.bandit.yml` (baseline config — pin current findings so
  only new ones fail the build)
- Optional: `scripts/security_scan.sh` — local reproduction of the CI
  steps for dev use.

**Workflow shape:**
```yaml
name: security-scan
on: [pull_request, workflow_dispatch]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -e ".[dev]" pip-audit bandit
      - name: pip-audit
        run: pip-audit --strict --desc
      - name: bandit
        run: bandit -r src/ -c .bandit.yml
      - name: ruff security subset
        run: ruff check --select S src/
```

**`.bandit.yml`:** start with `skips: []` and use `bandit -b baseline.json`
on a first run to capture the existing findings; commit the baseline;
CI runs with `-b baseline.json` so regressions surface but existing
findings don't block.

**Step 5:** `git commit -m "ci(security): pip-audit + bandit + ruff-S on PRs"`

---

### Task 3: Cryptographic inventory script (optional)

**Files:**
- Create: `scripts/crypto_inventory.py`
- Create: `tests/test_scripts/test_crypto_inventory.py`

**What it does:** greps src/soma for uses of known-dangerous crypto
primitives (`md5`, `sha1` for security use — not file-content
checksums, `DES`, ECB mode, `random.random` for secrets, etc.) and
emits a markdown report listing what it found with line references.

Low-priority — bandit covers most of this. Skip if time-boxed; ship
if it's quick (~1 hour).

**Step 5:** `git commit -m "tools(security): crypto inventory script"`

---

### Task 4: Docs cross-ref

**Files:**
- Modify: `docs/auth.md` — new "Security review" section linking to
  `docs/security-review-package.md` and explaining when operators
  should run the scan workflow locally.

**Step 5:** `git commit -m "docs(auth): cross-ref security review package"`

---

### Final sanity

```bash
ruff check src/soma tests
pip-audit --strict
bandit -r src/
# If any fail, decide: fix here vs. pin in baseline. Document the call.
```

Target: zero new tests (unless the inventory script lands as an
optional task with its unit test). Zero code changes to `src/soma/*`.

**Gotchas:**
- `bandit` produces a lot of noise on first run (B101 assert-in-tests
  is a false positive on dense test suites; B404/B603 subprocess is a
  false positive on tooling scripts). Use `-c .bandit.yml` to suppress
  the global false-positive classes and keep the real findings.
- `pip-audit` updates its advisory DB on every run; a transient
  network hiccup can fail CI. Use `--timeout 30` and an explicit
  non-zero exit only on HIGH/CRITICAL.
- The review package doc gets stale. Add a "last reviewed" field at
  the top and a quarterly refresh note to the author-of-record.
