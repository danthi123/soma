# Phase 15: Polish Audit + Small Cleanup

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Catch inconsistencies introduced during the 16-phase push and close two small deferred items.

**Architecture:** An audit pass followed by targeted fixes. Two discrete cleanup tasks bolted on: the pre-existing mypy errors in `api.py` and the TypeScript retry middleware deferred from Phase 5.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/*.md`, `README.md`. Phase 14 agent owns docs concurrently.

---

### Task 1: Audit pass

Find inconsistencies. Grep across `src/soma/`, `tests/`, `benchmarks/`. Look for:

**1.1 Stale metric names.** The Phase 9 agent caught `soma_retrieve_seconds` vs `soma_retrieve_latency_seconds`. Find other places where metric names were guessed rather than grepped. Verify every reference matches what's actually exported from `src/soma/metrics.py`.

```bash
# Collect real metric names:
grep -E "^\s*(COMPACTION|RETRIEVE|STORE|WAL|FAISS|BM25|CONSOLIDATE|AUTH|RELOAD|ENTRIES)\w*\s*=" src/soma/metrics.py
# Compare against references:
grep -rn "soma_\(store\|retrieve\|wal\|faiss\|consolidate\|compaction\|auth\|bm25\|entries\|reload\)_" src/ tests/ deploy/ docs/ benchmarks/ | sort -u
```

Flag every mismatch. Fix ones in `src/`, `tests/`, `deploy/grafana/*.json` (Phase 9 output) if they still have errors.

**1.2 Duplicate / dead code.** Phase 11 found duplicate `_add_fact`/`_reconcile`/`_reconcile_with_llm` in `conversational.py` (cleaned up in `43ce5a5`). Sweep the rest of `src/soma/` for similar accidental duplication:

```bash
# Find duplicate method-def lines per file
for f in $(find src/soma -name "*.py"); do
    python3 -c "
import ast
with open('$f') as fh: tree = ast.parse(fh.read())
from collections import Counter
defs = []
for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
    defs += [(cls.name, m.name) for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
dupes = {k: v for k, v in Counter(defs).items() if v > 1}
if dupes: print('$f:', dupes)
"
done
```

Remove dead duplicates; keep the authoritative copy.

**1.3 Deprecated API surfaces still referenced.** Grep for any import / call of removed or renamed functions. Use `git log --oneline -50 --name-only` to catch recent renames.

**1.4 Inconsistent error handling.** Phases shipped in parallel might differ on return-on-error vs raise-on-error conventions. Scan new files (`src/soma/bundle.py`, `src/soma/auth_revocation.py`, `src/soma/memory/backends/lancedb*.py`) for common-sense uniformity with the surrounding code. Note issues; fix only the clear-cut ones.

**1.5 TODO / FIXME / XXX comments.** `grep -rn "TODO\|FIXME\|XXX" src/soma/` — decide per-item whether to resolve, defer (keep comment), or delete.

**Scope:** Audit is research only. Each fix gets its own commit.

**Commit cadence:** one focused commit per logical fix cluster.

---

### Task 2: Close pre-existing mypy errors

**Files:**
- `src/soma/memory/api.py` — the ~5 errors flagged in deferred-items.md ("all from the `has_encoder` pattern")

**Step 1:** Run `mypy src/soma/memory/api.py --strict` (or `mypy src/soma/` if strict flag is too aggressive). Capture the error list.

**Step 2:** For each, understand the issue and pick the minimal fix. The `has_encoder` pattern is likely a `Optional[Thing]` where mypy can't narrow. Common fixes:
- `assert self._encoder is not None` after a caller-side check
- Hoist the None check into a property
- Use TypeGuard

**Step 3:** Re-run mypy; confirm clean on api.py.

**Step 4:** Don't break tests (`pytest tests/test_memory -q`).

**Commit:** `chore(memory): close pre-existing mypy errors in api.py`

---

### Task 3: TypeScript retry middleware

**Files:**
- Add: `clients/typescript/src/retry.ts` (or appropriate file)
- Modify: `clients/typescript/src/index.ts` (re-export)
- Add tests: `clients/typescript/tests/retry.test.ts`

**Contract (from deferred-items.md):**
> Retry middleware as optional re-exports. Not forks of generated code — simple fetch-level wrapper.

**Design:**
```typescript
export interface RetryOptions {
  maxRetries?: number;       // default 3
  backoff?: "linear" | "exponential";  // default "exponential"
  initialDelayMs?: number;   // default 200
  retryOn?: (response: Response) => boolean;  // default: status 502/503/504
}

export function withRetry(fetchImpl: typeof fetch, opts?: RetryOptions): typeof fetch
```

Tests should cover: retries on 503, gives up after maxRetries, respects retryOn predicate, doesn't retry on 4xx by default.

**Step 1-4: TDD.**

**Commit:** `feat(clients-ts): optional retry middleware for fetch`

---

### Task 4: Final sanity

```bash
ruff check src/ tests/ benchmarks/
SOMA_EMBED_MODEL=stub pytest tests/ benchmarks/tests/ -q
cd clients/typescript && npm test -- --run
```

Full suite stays green (1483+ passing). Ruff clean.

---

### Out-of-scope — do NOT touch

- `README.md`, `docs/*.md` — Phase 14 agent owns those concurrently
- `CHANGELOG.md` — central merge after both phases
- `clients/typescript/src/schema.d.ts` or other generated code — retry middleware is a standalone wrapper, not a schema edit
- `deploy/grafana/*.json` — don't retouch unless your audit finds a broken metric name
- Anything in `src/soma/memory/api.py` beyond the mypy fixes (no gratuitous refactors)
