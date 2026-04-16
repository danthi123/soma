# Phase 32: `GCSObjectStore`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Third implementation of the `ObjectStore` Protocol, backed
by Google Cloud Storage. Matches `S3ObjectStore`'s shape so every
bundle-I/O path already works; this phase is mostly about the GCS
client binding + URL dispatch.

**Architecture:**
- `src/soma/storage/gcs.py` — `GCSObjectStore(bucket, prefix, ...)`
  using `google-cloud-storage`. Same Protocol as S3 + Local;
  `KeyError` on missing; streaming read/write.
- `gs://bucket/prefix/` URL scheme wired into
  `src/soma/storage/urls.py`. Credentials come from the standard
  Google ADC chain (env var `GOOGLE_APPLICATION_CREDENTIALS`,
  metadata server on GCP, `gcloud auth application-default login`).
- `project=` kwarg + URL query param for setups that need it
  explicitly.
- **Testing**: `google-cloud-storage` is harder to mock than S3 —
  there's no canonical `moto`-equivalent. Use the officially-supported
  `google-cloud-storage`'s `testbench` emulator pattern OR the
  community `gcp-storage-emulator` package. Fall back to pure mocks
  for unit tests if the emulator is unavailable on CI.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/backends.md` big section, `deferred-items.md`.

---

### Task 1: `GCSObjectStore` implementation

**Files:**
- Create: `src/soma/storage/gcs.py`
- Create: `tests/test_storage/test_gcs.py`
- Modify: `pyproject.toml` — add `gcs = ["google-cloud-storage>=2.10"]`
  optional extra; add `gcs-test = ["gcp-storage-emulator>=2024.8"]`
  (or whichever emulator turns out viable — agent's judgment call).

**Skeleton:**
```python
try:
    from google.cloud import storage as gcs
    _HAS_GCS = True
except ImportError:
    _HAS_GCS = False

class GCSObjectStore:
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        client: Any = None,
    ) -> None:
        if not _HAS_GCS:
            raise ImportError(
                'google-cloud-storage not installed; pip install "soma[gcs]"'
            )
        self._prefix = prefix.rstrip("/")
        self._client = client or gcs.Client(project=project)
        self._bucket = self._client.bucket(bucket)
```

**Tests**: same shape as `test_s3.py`, substituting the GCS emulator.
If the emulator proves flaky or install-hostile on this Windows box,
fall back to pure `unittest.mock` stubs against the `Client` /
`Bucket` / `Blob` interfaces. Document the choice in the test file's
module docstring.

Required tests (minimum):
- `test_put_get_round_trip`
- `test_prefix_isolation`
- `test_list_prefix_returns_matching_keys`
- `test_delete_missing_is_idempotent`
- `test_exists_true_false`
- `test_get_missing_raises_key_error`
- `test_stream_round_trip_large`
- `test_import_error_without_gcs`

**Step 5:** `git commit -m "feat(storage): GCSObjectStore via google-cloud-storage"`

---

### Task 2: `gs://` URL scheme + MemoryLayer round-trip

**Files:**
- Modify: `src/soma/storage/urls.py`
- Extend: `tests/test_storage/test_urls.py`
- Create: `tests/test_memory/test_memory_layer_gcs.py` (or
  parametrize the existing `test_memory_layer_s3.py` to cover both
  scheme families — agent's call)

**Parser:**
```python
def parse_store_url(url: str | Path) -> ObjectStore:
    parsed = urlparse(str(url))
    if parsed.scheme == "gs":
        return _gcs_from_url(parsed)
    ...
```

**Tests**:
```python
def test_parse_store_url_dispatches_to_gcs():
    with _gcs_stubbed():
        store = parse_store_url("gs://b/pfx")
        assert isinstance(store, GCSObjectStore)

def test_memory_layer_save_load_gcs_round_trip():
    # End-to-end against the emulator (or skipped if unavailable).
```

**Step 5:** `git commit -m "feat(storage): gs:// URL scheme + MemoryLayer round-trip"`

---

### Task 3: Gated real-GCS integration test

**Files:**
- Create: `tests/test_storage/test_gcs_integration.py` (gated via
  `SOMA_GCS_INTEGRATION_BUCKET=...`; `@pytest.mark.slow_gcs` marker
  registered in `pyproject.toml`)

**Step 5:** `git commit -m "test(storage): gated real-GCS integration test"`

---

### Final sanity

```bash
ruff check src/soma/storage tests/test_storage tests/test_memory
pytest tests/test_storage -q
pytest tests/test_memory -q    # nothing should regress
```

Baseline post-Phase-31: +~35 new `tests/test_storage`. Target for
Phase 32: +~12 unit + 2 URL + 1 memory-layer end-to-end, +~5 gated.

**Gotchas:**
- `google-cloud-storage` is an async-capable but sync-by-default
  client. Stay sync — no `asyncio` in the adapter.
- The emulator's `STORAGE_EMULATOR_HOST` env var must be set before
  the `Client()` is constructed. Tests that assume pre-constructed
  clients need to be careful about ordering.
- `blob.upload_from_file(stream)` accepts any file-like; behaves
  like `put_stream`. `blob.download_to_file(stream)` is the inverse.
  Both are streaming-by-default.
- Missing-object behaviour: `blob.download_as_bytes()` on an absent
  blob raises `google.cloud.exceptions.NotFound`. Catch and
  re-raise as `KeyError` for Protocol parity with S3 + Local.
