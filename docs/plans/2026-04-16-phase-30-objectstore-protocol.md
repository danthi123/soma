# Phase 30: `ObjectStore` Protocol + `LocalFSObjectStore`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Foundation for the S3/GCS bundle-backend track (Phases 30–33).
Introduce an `ObjectStore` abstraction so `MemoryLayer.save/load` can
talk to either a local filesystem or a remote object store through
one interface. This phase ships only the Protocol + the local-FS
implementation — behaviour stays identical for every existing
caller. No S3/GCS code yet.

**Architecture:**
- `src/soma/storage/` new package:
  * `base.py` — `ObjectStore` Protocol.
  * `local.py` — `LocalFSObjectStore` (wraps `pathlib` / `shutil`).
  * `urls.py` — `parse_store_url("file:///path")` returns the right
    `ObjectStore`. Dispatch table extended in Phases 31/32.
- **Contract**: save/load semantics unchanged. A "bundle" is still a
  directory of files; the directory is now addressed by URL
  (`file:///abs/path` or a plain `/abs/path` that auto-prefixes to
  `file://`). `MemoryLayer.load(path)` accepts both a `str` URL and
  a `Path` (back-compat).
- **Why streaming in the Protocol**: bundles can be hundreds of MB
  (vectors.npy). Cloud stores download chunk-by-chunk; local FS just
  opens a file handle. The Protocol surfaces `get_stream` /
  `put_stream` so callers can pipe without slurping into RAM.
- **Explicitly out of scope for this phase**: S3, GCS, WAL-to-cloud,
  caching, atomicity across multiple PUTs. Those ship in 31 / 32 / 33.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, top-level
`docs/backends.md` section, `deferred-items.md` strikethrough (Phase 33
finishes the track; we strike through then).

---

### Task 1: `ObjectStore` Protocol + `LocalFSObjectStore`

**Files:**
- Create: `src/soma/storage/__init__.py` — re-exports Protocol + local impl + `parse_store_url`
- Create: `src/soma/storage/base.py`
- Create: `src/soma/storage/local.py`
- Create: `src/soma/storage/urls.py`
- Create: `tests/test_storage/__init__.py`
- Create: `tests/test_storage/test_protocol.py`
- Create: `tests/test_storage/test_local.py`
- Create: `tests/test_storage/test_urls.py`

**Protocol:**
```python
from __future__ import annotations
from typing import Protocol, BinaryIO, Iterator

class ObjectStore(Protocol):
    def get_bytes(self, key: str) -> bytes: ...
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_stream(self, key: str) -> BinaryIO: ...       # caller closes
    def put_stream(self, key: str, stream: BinaryIO) -> None: ...
    def list_prefix(self, prefix: str) -> Iterator[str]: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
```

Raises `KeyError` on missing key (for `get_*`, `delete` of absent key
is idempotent). `list_prefix` yields keys in the store-native order;
callers that need sorted output sort themselves.

**Tests (`test_protocol.py`):**
```python
def test_protocol_is_runtime_checkable():
    assert isinstance(LocalFSObjectStore(tmp_path), ObjectStore)

def test_all_methods_declared():
    # Reflects the Protocol; regression guard when adding methods.
    assert set(ObjectStore.__protocol_attrs__) >= {
        "get_bytes", "put_bytes", "get_stream", "put_stream",
        "list_prefix", "delete", "exists",
    }
```

**Tests (`test_local.py`):**
```python
def test_put_get_round_trip(tmp_path): ...
def test_stream_round_trip_large(tmp_path):
    # 10 MB payload; assert memory usage stays under 2 MB during transfer.
def test_list_prefix_returns_matching_keys(tmp_path): ...
def test_delete_missing_is_idempotent(tmp_path): ...
def test_exists_true_false(tmp_path): ...
def test_get_missing_raises_key_error(tmp_path): ...
def test_put_overwrites(tmp_path): ...
def test_nested_keys_create_parent_dirs(tmp_path):
    store.put_bytes("a/b/c.bin", b"x")
    assert (tmp_path / "a" / "b" / "c.bin").exists()
def test_stream_closes_file_handle(tmp_path): ...
```

**Tests (`test_urls.py`):**
```python
def test_file_url_parses(tmp_path):
    store = parse_store_url(f"file://{tmp_path}")
    assert isinstance(store, LocalFSObjectStore)

def test_plain_path_auto_prefixes(tmp_path):
    store = parse_store_url(str(tmp_path))
    assert isinstance(store, LocalFSObjectStore)

def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unsupported store scheme"):
        parse_store_url("s3://bucket/prefix")  # Phase 31 adds this

def test_pathlib_accepted(tmp_path):
    # Path object works same as str
    store = parse_store_url(tmp_path)
    assert isinstance(store, LocalFSObjectStore)
```

Note: the `s3://` rejection test is the contract-pin for Phase 31 to
flip (s3:// dispatch replaces the error).

**Step 1: Write tests.**
**Step 2-4: TDD.**
**Step 5:** `git commit -m "feat(storage): ObjectStore Protocol + LocalFSObjectStore"`

---

### Task 2: Route `MemoryLayer.save/load` through `ObjectStore`

**Files:**
- Modify: `src/soma/memory/api.py` — `save`/`load` accept URL or Path
- Modify: `tests/test_memory/test_api.py` (or wherever MemoryLayer
  save/load tests live) — add a parametrized test exercising the
  file-URL path alongside the existing path-based one
- Minimal changes elsewhere — all four backends (InProc, Qdrant,
  LanceDB, Chroma, pgvector) write directory-shaped bundles today;
  we keep that invariant, just route the writes through the new
  abstraction

**Change shape:**
```python
class MemoryLayer:
    def save(self, dest: str | Path | ObjectStore, /) -> None:
        store = _coerce_store(dest)
        # ... existing logic, but every open()/write() becomes
        # store.put_bytes / store.put_stream with a relative key
    @classmethod
    def load(cls, src: str | Path | ObjectStore) -> "MemoryLayer":
        store = _coerce_store(src)
        # ... existing logic inverted
```

`_coerce_store(x)` is the single conversion point: ObjectStore
instance → return as-is; str/Path → `parse_store_url(x)`.

**Tests:**
```python
def test_memory_layer_save_load_file_url(tmp_path):
    mem = MemoryLayer(...)
    mem.add(["hello"], embeddings)
    url = f"file://{tmp_path / 'bundle'}"
    mem.save(url)
    restored = MemoryLayer.load(url)
    assert restored.ntotal == 1

def test_memory_layer_save_accepts_object_store(tmp_path):
    store = LocalFSObjectStore(tmp_path / "b")
    mem = MemoryLayer(...)
    mem.save(store)
    # ... restore same way
```

**Invariants to pin:**
- Pre-existing Path-based callers (every MemoryLayer test + CLI verb
  + REST endpoint) keep working byte-identically. Run the full
  `tests/test_memory` suite to confirm.
- Backend-specific files (Qdrant snapshot, Lance table dir, Chroma
  persist dir) still land in the same relative layout inside the
  bundle. No backend code changes.

**Step 5:** `git commit -m "feat(api): MemoryLayer save/load routes through ObjectStore"`

---

### Final sanity

```bash
ruff check src/soma/storage src/soma/memory tests/test_storage tests/test_memory
pytest tests/test_storage -q
pytest tests/test_memory -q    # full suite; nothing should regress
```

Baseline post-Phase-29: 504 passed, 33 skipped in `tests/test_memory`.
Target for Phase 30: +~20 new `tests/test_storage` tests, 0
regressions in `tests/test_memory`.

**Gotchas:**
- On Windows, `pathlib.Path("/abs/posix/path")` doesn't normalise to
  a drive-letter path. The URL parser must handle POSIX-style URLs
  explicitly (`file://C:/Users/...` is the Windows form). Test both.
- `shutil.copyfileobj` in `put_stream` uses a default 16KB buffer
  which is fine; pin that in a test so nobody later bumps it to
  something memory-hostile.
- `list_prefix` on local FS is `os.walk`; empty dirs yield no keys
  (matches S3 semantics where empty "prefixes" don't exist).
