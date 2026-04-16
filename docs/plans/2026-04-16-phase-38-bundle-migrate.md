# Phase 38: `soma bundle migrate` Verb + Schema Version Registry

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Ship the infrastructure for cross-version bundle migration
**before** we break the bundle format for the first time. Low-urgency
today (format is stable) but high-cost-if-we-miss-it: once a bundle
written by an older SOMA can't be loaded by a newer one, users with
data on-disk are stuck. Phase 38 adds a version field, a migrator
registry, and a `soma bundle migrate` CLI verb. The v1 migrator list
is **empty** — the whole point is to have the plumbing ready for the
first real schema change.

**Architecture:**
- New `bundle_version: int` field in `memory_index.json` (or a new
  `bundle.json` manifest — agent's call based on what's cleanest
  given existing files). Reads default to `1` when the field is
  absent (pre-Phase-38 bundles).
- `src/soma/bundle_migrations.py` — migrator registry + runner:
  ```python
  Migrator = Callable[[ObjectStore], None]   # mutates store in place
  _MIGRATORS: dict[int, Migrator] = {}       # key = from_version

  def register(from_version: int) -> Callable[[Migrator], Migrator]: ...
  def migrate_bundle(
      store: ObjectStore, *, target: int | None = None,
      dry_run: bool = False,
  ) -> list[int]:
      """Apply migrators from current → target (default: latest
      registered). Returns the list of versions applied."""
  ```
- `soma bundle migrate <path> [--to N] [--dry-run]` CLI verb:
  * Accepts `file://`, `s3://`, `gs://`, or a plain path (leverages
    Phase 30-32's URL parser).
  * `--dry-run` lists migrators that would run; no writes.
  * Without `--dry-run`: snapshots the bundle to `<path>.pre-migrate/`
    first (via ObjectStore copy), runs migrators, swaps on success,
    leaves the backup for manual rollback.
- **The `_MIGRATORS` dict is empty in this phase.** We're shipping
  infrastructure. The first actual migrator lands with whatever phase
  breaks the schema — tests in this phase use a fake fixture migrator.

**Out-of-scope:**
- Actual data migrations. None exist yet.
- Auto-migrate on `MemoryLayer.load()`: tempting but risky. Operator
  should opt in via the explicit CLI verb so they can snapshot first.
- Downgrade path (N → N-1). Plan for up-only; down-migration is
  generally unsafe.

**Out-of-scope (central merge):** `CHANGELOG.md`, `deferred-items.md`
strikethrough (the item's been on the list since Phase 10).

---

### Task 1: Version field + reader tolerance

**Files:**
- Modify: `src/soma/memory/api.py` — `save` writes `bundle_version`
  into the manifest; `load` reads it (default `1` when absent).
- Extend: `tests/test_memory/test_api.py`

**Tests:**
```python
def test_save_writes_bundle_version():
    mem = MemoryLayer(...)
    mem.save(tmp_path / "b")
    manifest = json.loads((tmp_path / "b" / "memory_index.json").read_text())
    assert manifest["bundle_version"] >= 1

def test_load_tolerates_missing_bundle_version():
    # Pre-Phase-38 bundle: manifest has no bundle_version field.
    # load() should assume 1 and succeed.

def test_load_rejects_future_version():
    # Write a manifest with bundle_version=999 (far future).
    # load() should raise a clear error pointing at `soma bundle migrate`.
```

**Step 5:** `git commit -m "feat(api): bundle_version field + reader tolerance"`

---

### Task 2: Migrator registry + runner

**Files:**
- Create: `src/soma/bundle_migrations.py`
- Create: `tests/test_bundle_migrations.py`

**Tests (use a fake migrator fixture to exercise the runner without
relying on a real breaking change):**
```python
@pytest.fixture
def fake_migrator_registry(monkeypatch):
    migrators = {}
    monkeypatch.setattr("soma.bundle_migrations._MIGRATORS", migrators)
    return migrators

def test_register_and_run_single_migrator(fake_migrator_registry, ...):
    ran = []
    @register(from_version=1)
    def _m(store):
        ran.append(1)
        _bump_version(store, 2)
    migrate_bundle(store)
    assert ran == [1]
    assert _read_version(store) == 2

def test_chain_of_migrators(fake_migrator_registry, ...):
    # Register 1→2 and 2→3; assert both run in order.

def test_skip_when_already_at_target(fake_migrator_registry, ...): ...
def test_dry_run_does_not_mutate(fake_migrator_registry, ...): ...
def test_missing_migrator_raises_clear_error(fake_migrator_registry, ...):
    # Bundle at v1, registry has 2→3 only. Runner raises
    # KeyError("no migrator from v1").
```

**Step 5:** `git commit -m "feat(bundle): migrator registry + migrate_bundle runner"`

---

### Task 3: `soma bundle migrate` CLI verb

**Files:**
- Modify: `src/soma/cli.py` — add `migrate` subcommand under
  `soma bundle` (the subcommand group landed in Phase 10).
- Extend: `tests/test_cli.py`

**Shape:**
```
soma bundle migrate PATH [--to VERSION] [--dry-run] [--no-backup]
```
- `PATH` — local path or `file://` / `s3://` / `gs://` URL.
- `--to VERSION` — explicit target (default: highest registered).
- `--dry-run` — list migrators that would run; no writes.
- `--no-backup` — skip the `.pre-migrate/` snapshot (power-user flag).

**Tests:**
```python
def test_migrate_dry_run_lists_plan(tmp_path): ...
def test_migrate_applies_and_snapshots_backup(tmp_path, fake_migrator_registry): ...
def test_migrate_no_backup_skips_snapshot(...): ...
def test_migrate_on_s3_url(mock_aws, fake_migrator_registry): ...
def test_migrate_already_at_target_is_noop(...): ...
```

**Step 5:** `git commit -m "feat(cli): soma bundle migrate verb"`

---

### Task 4: Docs

**Files:**
- Modify: `docs/backends.md` — new "Bundle migrations" section
  explaining the version field, registry, and operator workflow.
- Modify: `docs/cookbook.md` — short recipe: "inspect bundle version"
  + "migrate before upgrade".

**Step 5:** `git commit -m "docs(bundle): migration workflow + version field"`

---

### Final sanity

```bash
ruff check src/soma tests
pytest tests/test_memory tests/test_bundle_migrations.py tests/test_cli.py -q
```

Baseline post-Phase-37: ~1500 tests across the full suite. Target
+~12 new (3 api + 6 migrations + 3 cli), 0 regressions.

**Gotchas:**
- `.pre-migrate/` backup on S3/GCS uses `list_prefix` + per-key copy;
  can be expensive for large bundles. Document the cost and the
  `--no-backup` escape hatch.
- The version field MUST be added even if no migrators exist yet —
  that's the whole "pay now so we don't pay later" play. A future
  phase that skips the version-stamp write breaks this phase's
  invariant.
- Don't auto-migrate on `load()`. An operator should see the
  migration happen explicitly, with a chance to snapshot.
