# Phase 6 — Pluggable VectorBackend + Qdrant adapter

> **For Claude:** Execute via TDD after Phase 1 (WAL) lands. Each task has failing tests first, then minimal impl, then commit.

**Goal:** Decouple MemoryLayer from FAISS/in-proc list storage. Ship `InProcBackend` (default, zero behavior change) + `QdrantBackend` (local-file + HTTP modes) behind a `VectorBackend` Protocol. Unlock billions-of-vectors scale without losing SOMA's hybrid BM25, cross-encoder rerank, query expansion, `where` filter semantics, bundle portability, or plastic-graph substrate.

**Architecture:** `MemoryLayer` keeps ids/texts/metadata/timestamps in Python and delegates every vector op to a `self._backend: VectorBackend`. Metadata filter dispatches: backend declares `supports_filter_pushdown` + raises `FilterPushdownUnsupported` for unsupported ops → MemoryLayer falls back to Python pre-filter + `search_subset`. `_soma_activations` switches from positional list to dict-by-id so backends can soft-delete or reorder safely. WAL (Phase 1) remains the source of truth; backends are query accelerators.

**Tech stack:** `numpy`, `qdrant-client` (optional extra `soma[qdrant]`), `portalocker` (already in tree).

---

### Task 1: Protocol definition + exception

**Files:**
- Create: `src/soma/memory/backend.py`
- Test: `tests/test_memory/test_backend_protocol.py`

**Step 1:** failing tests:
- `test_protocol_is_runtime_checkable` — `isinstance(obj, VectorBackend)` works.
- `test_filter_pushdown_unsupported_carries_fields` — exception has `op` and `field` attrs.

**Step 2:** implement the Protocol (see research report §2 for exact shape):
```python
from typing import Protocol, runtime_checkable
import numpy as np
from pathlib import Path

class FilterPushdownUnsupported(Exception):
    def __init__(self, *, op: str | None = None, field: str | None = None): ...

@runtime_checkable
class VectorBackend(Protocol):
    @property
    def ntotal(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def supports_filter_pushdown(self) -> bool: ...
    @property
    def name(self) -> str: ...
    def open(self) -> None: ...
    def close(self) -> None: ...
    def clear(self) -> None: ...
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def remove(self, ids: list[str]) -> None: ...
    def get_vectors(self, ids: list[str]) -> np.ndarray: ...
    def search(self, query: np.ndarray, k: int, *, exclude_ids: set[str] | None = None,
               where: dict[str, Any] | None = None) -> list[tuple[str, float]]: ...
    def search_subset(self, query: np.ndarray, candidate_ids: list[str],
                      k: int) -> list[tuple[str, float]]: ...
    def snapshot(self, bundle_dir: Path) -> None: ...
    def restore(self, bundle_dir: Path) -> None: ...
```

**Step 3:** commit `feat(memory): VectorBackend Protocol + FilterPushdownUnsupported`.

### Task 2: InProcBackend extraction

**Files:**
- Create: `src/soma/memory/backends/__init__.py`
- Create: `src/soma/memory/backends/inproc.py`
- Modify: `src/soma/memory/api.py` (partial — just extract; full wiring in Task 4)
- Test: `tests/test_memory/test_inproc_backend.py`

**Step 1:** failing tests (parametrized over `flat` and `hnsw` FAISS modes):
- `test_add_then_search_top_k` — basic round-trip.
- `test_search_respects_exclude_ids` — `exclude_ids={x}` omits x.
- `test_search_subset_restricts_to_candidates` — only scores ids in list.
- `test_remove_triggers_rebuild_on_next_search` — FAISS rebuild invalidated.
- `test_get_vectors_roundtrips_stored_vectors` — bit-exact.
- `test_snapshot_restore_roundtrip_preserves_ntotal` — save/load.
- `test_clear_empties_store` — `ntotal == 0` post-clear.
- `test_supports_filter_pushdown_false` — inproc uses Python-filter fallback.

**Step 2:** move the current FAISS+tensor-list logic from `api.py:1004-1086` (`_rank_linear`, `_rank_faiss`, `_maybe_build_faiss`, `_rebuild_faiss`) into `InProcBackend`. Preserve the lazy FAISS build semantics (`faiss_threshold`, `faiss_index_type="flat"|"hnsw"` + HNSW params) as constructor kwargs. `supports_filter_pushdown=False`. Snapshot writes `memory_embeddings.pt` (unchanged format); restore reads same.

**Step 3:** commit `feat(memory): InProcBackend — FAISS/tensor-list path extracted`.

### Task 3: Dict-keyed _soma_activations + consolidate cursor

**Files:**
- Modify: `src/soma/memory/api.py` (`__init__`, `store`, `store_batch`, `forget`, `consolidate`, `_recapture_activations_stable`, `_retrieve_with_rerank`)
- Test: `tests/test_memory/test_soma_activations_keying.py` (new)

**Why:** current `_soma_activations: list[Tensor|None]` indexes by position. If the backend soft-deletes or reorders, positional lookup breaks. Moving to `dict[str, Tensor|None]` is a one-time churn that unlocks every backend. `consolidate()` cursor (int into text list) stays valid because stores are append-only and forgets remove a single id.

**Step 1:** failing tests:
- `test_forget_then_store_keeps_activations_keyed_by_id` — forget a middle entry, activations for other ids survive unchanged.
- `test_retrieve_with_rerank_uses_id_keyed_activations` — activation lookup by node_id in `_retrieve_with_rerank`.
- `test_consolidate_cursor_still_advances` — after forget, consolidate still processes only new entries.

**Step 2:** change datatype; update all call sites. `_soma_activations[nid]` replaces `_soma_activations[idx]`. `consolidate` still uses the text-index cursor (unchanged) but writes into the dict by node_id.

**Step 3:** commit `refactor(memory): key _soma_activations by node_id for backend independence`.

### Task 4: MemoryLayer ↔ backend wiring

**Files:**
- Modify: `src/soma/memory/api.py` (constructor kwarg, all vector-op callsites)
- Test: extensions to `tests/test_memory/test_api.py`

**Step 1:** failing tests:
- `test_memory_layer_default_backend_is_inproc` — `mem._backend.name == "inproc"`.
- `test_all_existing_api_tests_still_pass_with_explicit_inproc` — pass `backend=InProcBackend(dim=32)` explicitly, identical behavior.

**Step 2:** add `backend: VectorBackend | None = None` to `MemoryLayer.__init__`. If None, instantiate `InProcBackend` with same faiss kwargs. Route every vector op through `self._backend`:
- `store` / `store_batch`: `self._backend.add([nid], vec[None])`
- `related`: `q = self._backend.get_vectors([nid])[0]; self._backend.search(q, k, exclude_ids={nid})`
- `forget`: `self._backend.remove([nid])`
- `_rank_subset`: `self._backend.search_subset(q_vec, candidate_ids, k)`
- Delete `_rank_linear`, `_rank_faiss`, `_maybe_build_faiss`, `_rebuild_faiss`, `_faiss_index`, `_embeddings_list` — moved to backend.
- `save`: `self._backend.snapshot(out)` replaces embeddings write.
- `load`: `self._backend.restore(src)` replaces embeddings read.

**Step 3:** run the entire `tests/test_memory/` suite — must pass with zero changes. Commit `feat(memory): MemoryLayer routes all vector ops through VectorBackend`.

### Task 5: Filter pushdown dispatch

**Files:**
- Modify: `src/soma/memory/api.py` (`retrieve` where-branch)
- Test: `tests/test_memory/test_filter_parity.py` (new, parametrized over backends)

**Step 1:** failing test:
- `test_retrieve_with_where_delegates_to_backend_when_supported` — mock backend with `supports_filter_pushdown=True`; assert `search(where=...)` called with the filter.
- `test_retrieve_falls_back_to_python_when_backend_rejects_filter` — backend raises `FilterPushdownUnsupported`; MemoryLayer catches + falls back to current pre-filter path.
- `test_filter_parity_inproc_vs_mock_pushdown_backend` — identical results on a 10-entry corpus with every supported operator.

**Step 2:** in `retrieve()`: if `where is not None`:
```python
if self._backend.supports_filter_pushdown:
    try:
        return self._backend.search(q_vec, k, where=where)
    except FilterPushdownUnsupported:
        pass  # fall through to Python pre-filter
# existing pre-filter path
```

**Step 3:** commit `feat(memory): filter pushdown with Python fallback`.

### Task 6: QdrantBackend — in-memory + local-file

**Files:**
- Create: `src/soma/memory/backends/qdrant.py`
- Create: `src/soma/memory/backends/qdrant_filter.py`
- Modify: `pyproject.toml` (add `qdrant = ["qdrant-client>=1.10"]` optional)
- Test: `tests/test_memory/test_qdrant_backend.py`

**Step 1:** failing tests (skipif qdrant-client not installed):
- `test_add_search_roundtrip_in_memory_mode` — client=":memory:", add 10, search.
- `test_add_search_roundtrip_local_file_mode` — client=tmp_path, add 10, search, reopen, search again.
- `test_remove_omits_from_search` — remove one, not in top-k.
- `test_get_vectors_roundtrips_stored_vectors` — via Qdrant's scroll API.
- `test_search_subset_uses_payload_id_filter` — restrict to candidate_ids list.
- `test_supports_filter_pushdown_true` — QdrantBackend declares True.
- `test_collection_per_bundle_name` — instance uses `collection_name=name`.
- `test_local_mode_warns_above_20k` — `ntotal > 20_000` triggers `warnings.warn`.

**Step 2:** implement `QdrantBackend(mode="memory"|"local"|"http", path=..., url=..., collection=...)`. Vector config: `VectorParams(size=dim, distance=Distance.COSINE)`. Use deterministic int64 point-ids mapped from node_id uuids (hash or generate on `add`); maintain internal `node_id ↔ point_id` dict for scroll/get paths. Snapshot in local mode: `client.create_snapshot(...)` copied into bundle dir as `qdrant.snapshot`.

**Step 3:** commit `feat(memory): QdrantBackend in-memory + local-file modes`.

### Task 7: QdrantBackend — HTTP mode + snapshot/restore

**Files:**
- Modify: `src/soma/memory/backends/qdrant.py`
- Test: `tests/test_memory/test_qdrant_http.py` (skip unless `SOMA_QDRANT_TEST_URL` env set)

**Step 1:** failing tests (conditional on QDRANT_URL):
- `test_http_mode_basic_roundtrip`
- `test_http_snapshot_produces_sidecar_json` — `backend.json` with url+collection.
- `test_http_restore_repoints_without_reingest` — new client, same collection → `ntotal` > 0.
- `test_http_snapshot_fallback_on_server_reset` — server state cleared; MemoryLayer replays WAL to reconstitute Qdrant collection.

**Step 2:** HTTP branch of QdrantBackend. Snapshot: write `backend.json` sidecar with `{mode: "qdrant-http", url, collection, qdrant_version}`. Restore: read sidecar, reconnect client, verify collection exists; if absent, rely on WAL replay (MemoryLayer handles this after `restore`).

**Step 3:** commit `feat(memory): QdrantBackend HTTP mode + bundle sidecar`.

### Task 8: Filter translator

**Files:**
- `src/soma/memory/backends/qdrant_filter.py`
- Test: `tests/test_memory/test_qdrant_filter.py`

**Step 1:** failing tests — every operator in `_COMPARE_OPS` + `$in`/`$nin`:
- `test_exact_match_to_match_value`
- `test_eq_ne_in_nin`
- `test_range_operators_gt_gte_lt_lte`
- `test_unsupported_op_raises_filter_pushdown_unsupported`
- `test_and_across_fields_populates_must_list`

**Step 2:** implement `to_qdrant_filter(where: dict) -> qm.Filter` per research report §3. Raise `FilterPushdownUnsupported(op=..., field=...)` for anything outside the translator. Wire into `QdrantBackend.search` behind `where=` kwarg.

**Step 3:** commit `feat(memory): Qdrant filter translator for Chroma-style where`.

### Task 9: pyproject.toml + import guards + docs

**Files:**
- Modify: `pyproject.toml` — `qdrant = ["qdrant-client>=1.10"]`
- Modify: `src/soma/memory/backends/qdrant.py` — gate imports with try/except ImportError, raise helpful message on instantiation
- Create: `docs/backends.md`
- Modify: `docs/positioning.md` (updated scale story)
- Modify: `CHANGELOG.md`

**Step 1:** test: `test_qdrant_backend_raises_clear_error_when_dep_missing` — monkeypatch `qdrant_client` import to fail; instantiation raises `ImportError` with "pip install soma-memory[qdrant]" message.

**Step 2:** implement guards. Write docs: protocol overview, InProc tradeoffs, Qdrant local vs HTTP positioning (20K cap for local, HTTP for scale).

**Step 3:** commit `docs: VectorBackend positioning + Qdrant setup`.

### Task 10: Benchmark matrix

**Files:**
- Create: `benchmarks/run_backend_matrix.py`
- Create: `benchmarks/reports/backend_matrix.md` (generated)
- Update: `benchmarks/reports/paper-draft.md` (fold new numbers in)

**Step 1:** failing test: `test_backend_matrix_runs_inproc_only_without_qdrant` — no Qdrant URL set, matrix still produces a valid report row for InProcFlat/HNSW.

**Step 2:** adapter-matrix harness: InProcFlat, InProcHNSW, QdrantLocal, QdrantHTTP (when URL set). Scale 100K and 1M, shared sbert cache (reuse `run_scale_enterprise`'s cache). Metrics: store_total, retrieve p50/p95, disk footprint, Recall@10 vs exact.

**Step 3:** run the bench (100K first; 1M when cache available). Commit `bench(backend): InProc vs Qdrant matrix at 100K and 1M`.

### Task 11: Paper-draft + CHANGELOG

**Files:**
- Modify: `benchmarks/reports/paper-draft.md`
- Modify: `CHANGELOG.md`
- Modify: `README.md` (feature table: pluggable backends yes for Chroma/Pinecone parity)

**Step 1:** integrate backend matrix findings into paper-draft §4. Call out QdrantHTTP positioning vs FAISS at 1M.

**Step 2:** commit `docs: Phase 6 — pluggable VectorBackend (InProc + Qdrant)`.

---

## Ship-blocker thresholds

- QdrantHTTP retrieve p50 ≤ 2× InProc at 1M.
- Recall@10 within 0.02 of InProc at same k.
- Store throughput ≥ 500 ops/s on HTTP mode.
- Full `tests/test_memory/` + `tests/test_serve/` suites green with default InProc backend.

## Risks

1. **Filter semantics drift:** add `test_filter_parity.py` parameterized over backends.
2. **Score scale drift:** protocol mandates cosine ∈ [-1, 1]; adapter contract tests pin.
3. **`related()` HTTP round-trip:** one extra `get_vectors` call; acceptable for v1. Future optimization: `backend.search_near_id`.
4. **consolidate cursor with forget:** cursor is text-list int; forget is a rare op; ordered id log already available via WAL.
5. **Qdrant local-mode 20K cap:** emit warning at MemoryLayer level; docs lead with HTTP as the scale story.
6. **Snapshot version skew:** bundle records `qdrant_version`; restore refuses incompatible snapshots.
7. **WAL across hosts with HTTP backend:** out of v1 scope — document.

## Open questions

- **Per-bundle Qdrant collection vs shared collection with bundle_name tag?** Per-bundle cleaner; shared reduces collection overhead at 1000+ bundles. Default: per-bundle; shared-mode flag for Stage 7+ if demand.
- **Async Qdrant client?** `qdrant-client>=1.10` has `AsyncQdrantClient`. Phase 6 ships sync; async is a follow-up once FastAPI routes become async.
- **LanceDB as second adapter instead of Qdrant?** Arrow-native embedded. Compelling for a different segment (multimodal, local-first-forever). Decision: Qdrant wins v1 because it validates the HTTP path; LanceDB is Phase 7.

## Related plans

- Phase 1 — `docs/plans/2026-04-16-phase-1-wal-autosave.md` (WAL lands first; Phase 6 consumes)
- Phase 2 — `docs/plans/2026-04-16-phase-2-conversational-memory.md` (independent; layers on top)
