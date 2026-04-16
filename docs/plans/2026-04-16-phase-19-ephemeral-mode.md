# Phase 19: Ephemeral Mode Ergonomics

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Formalise the already-existing RAM-only mode with three ergonomic additions: `MemoryLayer.ephemeral()` classmethod, `soma chat --save-on-exit <path>` flag, and `POST /snapshot` REST endpoint. Right for notebooks / REPLs / short agent sessions; WAL stays the default elsewhere.

**Architecture:** All three are thin wrappers over existing capabilities. Ephemeral MemoryLayer = instantiate without `bundle_path` (no WAL). Chat save-on-exit = `atexit`-register a `mem.save(path)` call. REST `/snapshot` = one-shot `mem.save(...)` hit by the client at session end.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/cookbook.md`.

---

### Task 1: `MemoryLayer.ephemeral()` classmethod

**Files:**
- Modify: `src/soma/memory/api.py`
- Create: `tests/test_memory/test_ephemeral.py`

**API:**
```python
@classmethod
def ephemeral(
    cls,
    *,
    embed_fn: EmbedFn | None = None,
    embed_dim: int | None = None,
    # sbert convenience — same as with_sbert
    sbert_model: str | None = None,
    **kwargs: Any,
) -> "MemoryLayer":
    """Create a MemoryLayer with no WAL / no on-disk state.

    Reads never touch disk; writes stay in RAM until the caller
    opts into persistence via .save(path). Right for
    notebooks, REPLs, short agent runs. For long-lived services
    use the default MemoryLayer() with bundle_path=.
    """
```

Implementation: route through the existing constructor with `bundle_path=None`. If `sbert_model` passed, use `with_sbert` internals. Document clearly that `.save()` is the only path to persistence.

**Step 1: Write failing tests.**
```python
def test_ephemeral_no_wal_written(tmp_path, monkeypatch):
    mem = MemoryLayer.ephemeral(embed_fn=stub, embed_dim=8)
    mem.store("hello")
    # No WAL artifacts anywhere on disk.

def test_ephemeral_save_then_load_round_trip(tmp_path):
    mem = MemoryLayer.ephemeral(embed_fn=stub, embed_dim=8)
    nid = mem.store("hello")
    bundle = tmp_path / "b"
    mem.save(bundle)
    mem2 = MemoryLayer.load(bundle)
    assert nid in mem2

def test_ephemeral_sbert_variant():
    mem = MemoryLayer.ephemeral(sbert_model="all-MiniLM-L6-v2")
    # Smoke: store + retrieve runs; no disk state.
```

**Step 5:** `git commit -m "feat(memory): MemoryLayer.ephemeral() classmethod"`

---

### Task 2: `soma chat --save-on-exit <path>` flag

**Files:**
- Modify: `src/soma/cli.py` — `soma chat` parser gains `--save-on-exit` + `--ephemeral`
- Modify: `tests/test_cli.py`

**Contract:**
- `soma chat --ephemeral` — start an ephemeral chat (no WAL, no bundle on disk).
- `soma chat --save-on-exit path/to/bundle` — register an `atexit` handler that calls `mem.save(path)` at exit. Works with or without `--ephemeral`.

Both default off; existing `soma chat --bundle path/` keeps today's behaviour.

**Step 1: Tests.**
```python
def test_chat_ephemeral_flag_parsed():
    args = build_parser().parse_args(["chat", "--ephemeral"])
    assert args.ephemeral is True

def test_chat_save_on_exit_flag_parsed():
    args = build_parser().parse_args(["chat", "--save-on-exit", "out/bundle"])
    assert args.save_on_exit == Path("out/bundle")

def test_chat_ephemeral_and_bundle_mutually_exclusive():
    # Passing both should exit 2 with a clear error.
```

Implementation detail: current `soma chat` loads a bundle from `--bundle`. Ephemeral path creates a fresh MemoryLayer via `MemoryLayer.ephemeral()`.

**Step 5:** `git commit -m "feat(cli): soma chat --ephemeral + --save-on-exit"`

---

### Task 3: `POST /snapshot` REST endpoint

**Files:**
- Modify: `src/soma/serve.py`
- Modify: `tests/test_serve/test_serve_smoke.py` or new file

**Contract:**
```
POST /snapshot
{"path": "path/on/server/fs/bundle"}
→ 200 {"saved": true, "path": "...", "entries": 42}
```

Requires `write` perm on the target bundle when auth is on. When auth is off, open (matching current endpoints). Write path check — fail with 400 if path is relative outside cwd, for safety.

**Step 1: Tests.**
```python
def test_snapshot_writes_bundle_to_path(tmp_path):
    client = _client_with_stub_mem()
    client.post("/store", json={"text": "a"})
    r = client.post("/snapshot", json={"path": str(tmp_path / "b")})
    assert r.status_code == 200
    assert r.json()["saved"] is True
    assert (tmp_path / "b").exists()

def test_snapshot_returns_entries_count():
    ...

def test_snapshot_requires_write_perm_when_auth_on():
    ...
```

**Step 5:** `git commit -m "feat(serve): POST /snapshot for end-of-session dumps"`

---

### Final sanity

```bash
ruff check src/soma tests
SOMA_EMBED_MODEL=stub pytest tests -q
```

Baseline 1522 pass; target +~8 new tests, 0 regressions.
