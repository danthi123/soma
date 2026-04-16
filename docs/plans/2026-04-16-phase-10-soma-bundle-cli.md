# Phase 10: `soma bundle` Subcommand Group

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Add lifecycle verbs for bundles — `list`, `info`, `delete` — so operators aren't hand-writing `mem.save/load` scripts.

**Architecture:** New `soma bundle <verb>` parser group wired into `src/soma/cli.py`. Introspection helpers live in a new module `src/soma/bundle.py` (standalone — avoids cycles with `memory/api.py`). Helpers read `backend.json` and the store JSONL directly rather than instantiating `MemoryLayer`, keeping `list` fast across directories with hundreds of bundles.

**Tech Stack:** stdlib only for introspection (`json`, `os`, `pathlib`); `argparse` for parsing.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/cli.md`.

---

### Task 1: `src/soma/bundle.py` introspection helpers

**Files:**
- Create: `src/soma/bundle.py`
- Create: `tests/test_bundle.py`

**API:**
```python
def is_bundle_dir(p: Path) -> bool:
    """A directory is a SOMA bundle if it has backend.json + store.jsonl."""

@dataclass(frozen=True)
class BundleInfo:
    path: Path
    entries: int
    embed_dim: int
    backend: str          # "inproc-flat" / "qdrant-http" / ...
    last_modified: datetime
    wal_bytes: int        # 0 if no WAL
    corrupt: bool         # True if required files missing/malformed
    corrupt_reason: str   # populated when corrupt=True

def load_info(p: Path) -> BundleInfo: ...
def list_bundles(root: Path) -> list[BundleInfo]: ...
```

**Step 1: Write failing tests.** Build a fixture tmp dir containing 3 bundles of mixed states (healthy, missing backend.json, truncated store.jsonl). Assert:
- `is_bundle_dir` returns True/False correctly
- `load_info` populates every field for healthy bundle
- `load_info` sets `corrupt=True` with a non-empty reason for bad bundles
- `list_bundles` returns them sorted by `last_modified` desc

**Step 2:** Fails.

**Step 3: Implement.** Read `backend.json` for embed_dim + backend; count lines of `store.jsonl` for entries; stat the WAL file for `wal_bytes`. On any exception, populate `corrupt=True` with a truncated reason (cap 200 chars).

**Step 4:** Tests pass.

**Step 5:** `git commit -m "feat(bundle): BundleInfo introspection helpers"`

---

### Task 2: `soma bundle list`

**Files:**
- Modify: `src/soma/cli.py` — add `p_bundle = sub.add_parser("bundle")` → `bundle_sub` → `list`
- Modify: `tests/test_cli.py` — add tests

**Behavior:**
```
$ soma bundle list ./data
PATH                          ENTRIES  DIM  BACKEND        LAST-MODIFIED       WAL
./data/alex/memories           1,234   384  inproc-hnsw    2026-04-16 08:14   3.2 MB
./data/bobbi/memories          88,200  384  qdrant-http    2026-04-16 07:22   —
./data/broken                  CORRUPT                     2026-04-15 12:00   (missing backend.json)
```

Default root dir: `.` if omitted.

**Step 1: Write failing tests.**
- `test_bundle_list_prints_table` — seed two healthy bundles, assert stdout has both paths + entry counts
- `test_bundle_list_shows_corrupt_badge` — seed one broken bundle, assert "CORRUPT" appears
- `test_bundle_list_returns_2_for_missing_root`

**Step 2:** Fails.

**Step 3: Implement.** Walk the root up to depth=3 (prevent traversing users' entire home dir); call `list_bundles`; print aligned columns.

**Step 4:** Tests pass.

**Step 5:** `git commit -m "feat(cli): soma bundle list"`

---

### Task 3: `soma bundle info <path>`

**Files:**
- Modify: `src/soma/cli.py`, `tests/test_cli.py`

**Behavior:** Detailed view — entries, embed dim, backend, WAL path + size, last save timestamp, snapshot generation (from `backend.json.snapshot_ts` if present), bundle size on disk (recursive `du`).

**Step 1: Write failing tests.** Standard pattern: healthy bundle, missing bundle (exit 2), corrupt bundle (exit 2 with reason).

**Step 2-4:** TDD through to green.

**Step 5:** `git commit -m "feat(cli): soma bundle info"`

---

### Task 4: `soma bundle delete <path> [--yes]`

**Files:**
- Modify: `src/soma/cli.py`, `tests/test_cli.py`

**Behavior:**
- Without `--yes`: prompts `delete ./data/alex/memories with 1234 entries? [y/N] ` — read from stdin, accept y|yes (case-insensitive); anything else → abort with message, exit 0 (not an error).
- With `--yes`: delete immediately.
- Confirms by calling `is_bundle_dir` first; refuses to delete non-bundle paths even with `--yes` (safety — prevents `soma bundle delete ~`).

**Step 1: Write failing tests.**
- `test_bundle_delete_yes_flag_removes_bundle`
- `test_bundle_delete_interactive_y_removes`
- `test_bundle_delete_interactive_n_aborts`
- `test_bundle_delete_refuses_non_bundle_dir`

**Step 2-4:** TDD.

**Step 5:** `git commit -m "feat(cli): soma bundle delete with confirm"`

---

### Final sanity

```bash
ruff check src/soma/bundle.py src/soma/cli.py tests/
pytest tests/test_bundle.py tests/test_cli.py -q
```
