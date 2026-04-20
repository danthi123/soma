# Phase 14: Docs + README Pass

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Bring user-facing docs in line with what we actually shipped today (Phases 1-13). README is currently stale — it pitches "drop-in vector DB replacement" without mentioning ConversationalMemory, multi-user scoping, Grafana dashboards, LanceDB, JWT revocation, or the `soma bundle` CLI group.

**Architecture:** Pure docs refresh — no code changes. Focus: README.md (front door), docs/quickstart.md (getting-started flow), docs/positioning.md (competitive positioning), docs/cookbook.md (recipes). CLAUDE.md only if it references deleted or renamed paths.

**Out-of-scope (I will do centrally):** `CHANGELOG.md` — it's already the authoritative what-shipped doc.

---

### Task 1: Audit pass (no writes yet)

**Read everything:**
1. `README.md` — entire file
2. `docs/positioning.md` — current pitch
3. `docs/quickstart.md` — getting-started
4. `docs/cookbook.md` — recipes
5. `CLAUDE.md` — project memory
6. `docs/observability.md` — already updated in Phase 8 merge
7. `docs/auth.md` — already updated in JWT revocation
8. `docs/conversational-memory.md` — check if exists and current

**For each, identify:**
- Feature mentions that no longer match code (e.g. "requires cloud provider" when we ship local-first)
- Missing features shipped since doc was last touched
- Stale CLI examples (e.g. no mention of `soma bundle`, `soma auth revoke`)
- Stale code snippets (outdated imports, deprecated APIs)
- Broken cross-references to renamed files

Record findings in a comment or throwaway note; no commits from Task 1.

---

### Task 2: README rewrite

**Files:**
- Modify: `README.md`

**Focus:**
- Headline pitch — "Local-first agent-memory layer" (from positioning.md), not "drop-in vector DB"
- Feature list: update feature comparison table to reflect everything shipped (JWT auth + revocation, ConversationalMemory + multi-user, Grafana dashboards, LanceDB, bundle CLI). Cross-check against the list in docs/positioning.md.
- Installation: `pip install soma`, `pip install "soma-memory[metrics]"`, `pip install "soma-memory[serve]"`, `pip install "soma-memory[lancedb]"` — all extras we now have
- Quickstart snippet: a 10-line example that exercises the happy path (create bundle, store, retrieve, forget)
- Link to docs/quickstart.md for the full flow (JWT → ConversationalMemory → retrieve → bundle mgmt)
- Link to deploy/grafana/ for the dashboards
- Commit message: `docs(readme): refresh for Phase 1-13 feature set`

---

### Task 3: Quickstart refresh

**Files:**
- Modify (or create): `docs/quickstart.md`

**Flow (end-to-end agent-memory use case):**
1. Install the extras needed for the full flow (`pip install 'soma-memory[serve,metrics]'`)
2. Start the server (`soma serve`)
3. Mint a JWT (`soma auth rotate-secret`, then `soma auth issue --sub alice --bundle alice:read,write --expires 30d`)
4. Create a bundle via REST (`POST /store`)
5. Use ConversationalMemory client-side for fact extraction (with optional `extractor_llm=` for small-model users)
6. Multi-user scoping (pass `user_id` through metadata)
7. Retrieve with filters
8. Check bundle state (`soma bundle info`)
9. Revoke a leaked token (`soma auth revoke --token <jwt>`)
10. Import Grafana dashboard for observability

Each step should be copy-paste-runnable. Use shell blocks for CLI, Python blocks for API calls.

Commit message: `docs(quickstart): end-to-end agent-memory flow`

---

### Task 4: Cookbook + positioning touch-up

**Files:**
- Modify: `docs/cookbook.md` (check each recipe for currency; update as needed)
- Modify: `docs/positioning.md` (competitive comparison row for Grafana dashboards, multi-user, LanceDB)
- Modify: `CLAUDE.md` (only if it references deleted paths — otherwise leave alone)

Commit message: `docs(cookbook, positioning): align with current feature set`

---

### Task 5: Sanity

```bash
# Verify no broken internal links
grep -rn "](docs/" README.md docs/*.md | grep -v "http" | while read ref; do
    # ... check each referenced file exists
done

# Verify every `soma <verb>` shown actually parses
python -c "from soma.cli import build_parser; p = build_parser(); print('OK')"

ruff check tests/  # no docs touch tests, but sanity
```

---

### Notes for the implementer

- Don't write a lot. Keep each doc focused. README max ~200 lines. Quickstart max ~300 lines with code blocks.
- Don't invent APIs. Every code snippet must be grep-confirmable against current src/.
- When rewriting positioning, lean on the actual metric numbers shipped in `benchmarks/reports/` (scale_enterprise_100k.md, backend_matrix.md, etc.) — don't vaguely claim "faster than X".
- Don't bullet-spam. Prose > endless bullet lists.
- Don't add emojis unless there's already an emoji convention in the file.
