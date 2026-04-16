# Phase 1 — WAL + autosave

> **For Claude:** Execute task-by-task via TDD. Each task has a failing test first, minimal impl, passing test, commit.

**Goal:** MemoryLayer survives process crash (no loss of committed stores) and safely supports concurrent writers on the same bundle.

**Architecture:**
- Append-only WAL split into `memory_ops.wal.jsonl` (metadata) + `memory_embeddings.wal.bin` (length-prefixed + CRC32 binary framing).
- `portalocker.Lock` on sidecar `bundle.lock` held for: WAL append → fsync (if `durability="sync"`) → in-memory mutation. Released before retrieve.
- Schema v2 bundle: snapshot + WAL that replays on top.
- Compaction rewrites snapshot when `WAL > max(4 MB, 1.0 × snapshot) OR records > 10_000 OR age > 1 h`.
- Existing `save()` made atomic: tmp file + `os.replace` + dir fsync.
- Backward-compat: v1 bundles load without WAL; next save upgrades to v2.

**Tech stack:** `portalocker` (already a dep), stdlib `zlib.crc32`, stdlib `struct`, stdlib `os.fsync`.

---

### Task 1: Atomic save() fix (pre-WAL)

**Files:**
- Modify: `src/soma/memory/api.py:725-762` (`save()`)
- Test: `tests/test_memory/test_api.py`

**Why first:** WAL replay assumes the snapshot is consistent on disk. Current `save()` has three non-atomic `write_text`/`torch.save` calls — a crash mid-save leaves a half-written bundle.

**Step 1:** failing test `test_save_is_atomic_under_crash` — mock `torch.save` to raise mid-way through, assert original bundle (or no bundle) still parses.

**Step 2:** implement `_atomic_write_bytes(path, data)` helper: write `<path>.tmp`, `os.fsync(fd)`, `os.replace(tmp, path)`. Then `os.fsync` on the parent dir fd (Unix; Windows no-op).

**Step 3:** update `save()` to use `_atomic_write_bytes` for all three outputs. Wrap in try/except that deletes `.tmp` leftovers.

**Step 4:** test passes; commit `fix(memory): atomic save — tmp + rename + fsync`.

### Task 2: WAL record framing module

**Files:**
- Create: `src/soma/memory/wal.py`
- Test: `tests/test_memory/test_wal.py`

**Step 1:** failing tests in `test_wal.py`:
- `test_append_then_replay_roundtrip` — append 100 records, close, replay, assert equality.
- `test_torn_tail_line_is_truncated` — write 10 records then a half line, replay returns 10 records and file ends at last good offset.
- `test_torn_binary_record_truncated` — write 10 embeddings then partial bytes, replay returns 10, CRC mismatch triggers truncate.
- `test_crc_mismatch_triggers_truncate` — flip a byte in the middle record, replay stops at the first bad CRC (does NOT skip to next).
- `test_header_written_on_open` — fresh WAL has schema header line.
- `test_empty_wal_with_header_has_zero_records` — header only, replay returns [].

**Step 2:** implement `class WAL`:

```python
@dataclass(frozen=True)
class WalRecord:
    op: Literal["store", "forget"]
    node_id: str
    text: str | None  # None for forget
    metadata: dict[str, Any]
    timestamp_step: int
    embedding: torch.Tensor | None  # None for forget

class WAL:
    """Paired JSONL-metadata + CRC-framed-binary-embeddings WAL."""
    def __init__(self, bundle_dir: Path, embed_dim: int, durability: str = "sync") -> None
    def open(self) -> None  # opens files, writes header if new
    def append(self, record: WalRecord) -> None  # atomic lock-held write
    def replay(self) -> Iterator[WalRecord]  # yields validated records
    def truncate(self) -> None  # compaction: zero both files, rewrite header
    def flush(self) -> None  # fsync on demand
    def close(self) -> None
    @property
    def size_bytes(self) -> int
    @property
    def record_count(self) -> int
```

Binary frame: `struct.pack("<II", length, crc32(payload)) + payload` where payload is `dtype_tag_byte + embed_dim*4 bytes`.

Header JSON line: `{"schema": 1, "embed_dim": N, "created_at": ts, "bundle_version": V}`.

**Step 3:** tests pass. Commit `feat(memory): WAL record framing with CRC32 + torn-tail recovery`.

### Task 3: MemoryLayer WAL wiring — store / store_batch / forget

**Files:**
- Modify: `src/soma/memory/api.py` (init, store, store_batch, forget, load)
- Test: `tests/test_memory/test_wal_integration.py` (new)

**Step 1:** failing integration tests:
- `test_store_then_crash_then_load_recovers_entry` — store 3 entries without save(), simulate crash, reload same bundle dir, assert all 3 present via `__len__` and `get`.
- `test_forget_then_crash_then_load_replays_tombstone` — store 2, forget 1, crash, reload, assert forgotten.
- `test_concurrent_store_two_memorylayers_on_same_bundle` — two MemoryLayer instances pointing at same bundle dir, alternate stores, both reload correctly.
- `test_durability_async_can_lose_tail_without_flush` — confirms the docstring: `durability="async"` + crash loses in-flight; `flush()` makes it durable.

**Step 2:** add constructor arg `bundle_path: Path | None` and `durability: Literal["sync","batch","async"] = "sync"`. When `bundle_path` is provided at init (or attached via `attach_bundle(path)`), allocate a `WAL` under a `portalocker.Lock`.

**Step 3:** modify `store()`:
- Acquire lock.
- Build WAL record with embedding.
- `wal.append(record)` (fsync if sync).
- Commit to in-memory state (existing code).
- Release lock.

`store_batch` does one lock-acquire for N records.

`forget` appends a tombstone record (op=forget, node_id=X).

**Step 4:** modify `load()`:
- After loading snapshot, check for WAL sidecar files.
- If present, `wal.replay()` and apply each record to in-memory state.
- Keep WAL open for subsequent writes.

**Step 5:** tests pass. Commit `feat(memory): MemoryLayer persists through WAL; survives crashes`.

### Task 4: Schema v2 + backward compat

**Files:**
- Modify: `src/soma/memory/api.py` (`save()`, `load()`)
- Test: `tests/test_memory/test_schema_v2_migration.py` (new)

**Step 1:** failing tests:
- `test_loads_v1_bundle_without_wal` — saved bundle from before this PR loads cleanly.
- `test_save_writes_v2_and_starts_wal` — after save, `memory_index.json` has `schema_version: 2`, WAL files exist.
- `test_v1_bundle_upgraded_on_next_save` — load v1, store one more entry, save; resulting bundle is v2.

**Step 2:** bump `schema_version` to 2 in save. In load, accept 1 or 2:
- v1: no WAL; legacy mode.
- v2: check for WAL, replay if present.

**Step 3:** commit `feat(memory): schema v2 with WAL; v1 bundles auto-upgrade on save`.

### Task 5: Compaction

**Files:**
- Modify: `src/soma/memory/api.py` (new `_maybe_compact()` helper invoked from store/store_batch/forget)
- Test: `tests/test_memory/test_wal_compaction.py` (new)

**Step 1:** failing tests:
- `test_compaction_triggered_by_size` — store many records, assert WAL gets truncated and snapshot is refreshed.
- `test_compaction_triggered_by_record_count` — store 10_001 small records, trigger fires.
- `test_compaction_preserves_all_entries` — snapshot count + WAL count before/after identical.
- `test_compaction_under_concurrent_store` — two threads storing while compaction runs; no data loss, no deadlock (use threading.Event to sync).

**Step 2:** `_maybe_compact()` checks the three triggers; on hit, spawns a daemon thread that:
1. Acquires lock.
2. Copies in-memory state refs.
3. Releases lock (readers can continue).
4. Writes new snapshot atomically to a temp dir.
5. Re-acquires lock.
6. `os.replace` new snapshot into place.
7. `wal.truncate()` and rewrite fresh header.
8. Releases lock.

**Step 3:** commit `feat(memory): auto-compaction keeps WAL bounded`.

### Task 6: Multi-worker reload_if_stale

**Files:**
- Modify: `src/soma/memory/api.py` (`reload_if_stale()` method)
- Modify: `src/soma/serve.py` (call it before each retrieve)
- Test: `tests/test_memory/test_multiworker_staleness.py` (new)

**Step 1:** failing test:
- `test_second_process_sees_new_stores_on_retrieve` — one MemoryLayer stores; a *separate* MemoryLayer instance on the same bundle calls `retrieve` and gets the fresh entry. (Simulates two uvicorn workers.)

**Step 2:** add `reload_if_stale()`: checks WAL mtime/size since last read; if newer, acquires shared lock and replays new records on top of in-memory state. Guard with a `_last_wal_offset` cursor.

**Step 3:** wire `_get_mem()` in `serve.py` to call it on retrieve paths only (not on every store — stores already go through the lock).

**Step 4:** commit `feat(serve): multi-worker safe — readers catch up via WAL tail`.

### Task 7: Docs + README

**Files:**
- Modify: `docs/cookbook.md` (durability recipe)
- Modify: `README.md` (brief mention in the feature-comparison table)
- Modify: `CHANGELOG.md`
- Modify: `docs/positioning.md` (upgrade "survives model swaps" line)

**Step 1:** document `durability="sync"` default, `flush()`, crash-recovery guarantee, and multi-worker caveats.

**Step 2:** commit `docs: Phase 1 — WAL durability + concurrency story`.

### Task 8: End-to-end smoke via REST

**Files:**
- Modify: `tests/test_serve/test_serve_smoke.py`

**Step 1:** failing test:
- `test_rest_store_crash_reload_persists` — TestClient stores 3 entries, force `_mem_cache.clear()`, next `/retrieve` reloads from disk + WAL, gets the entries.

**Step 2:** commit `test(serve): end-to-end crash-and-reload via REST`.

---

## Open questions resolved during research

- **fsync policy default:** `sync` (matches user "I pressed Ctrl-C" expectation; at ~1k ops/s, never the bottleneck).
- **CRC on JSONL:** no (json.loads failure is equivalent signal).
- **Activations persisted in WAL:** no (derivable; regenerated on consolidate after reload).
- **NFS/SMB support:** out of scope, document as local-filesystem-only.

## Risks

1. **Lock starvation by consolidate()**: consolidate runs per-entry SOMA forward passes while holding the lock. *Mitigation:* release lock during forward passes; re-acquire only to swap `_soma_activations`. Called out in Task 3 (consolidate() edit).
2. **FAISS rebuild after large replay**: first retrieve post-load pays O(N·d) rebuild. Already happens today; just more often with WAL. Acceptable.
3. **Float16/int8 embeddings**: binary WAL tags payload with dtype byte; future-proofs but only float32 path tested in this phase.
