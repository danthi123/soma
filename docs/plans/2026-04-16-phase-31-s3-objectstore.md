# Phase 31: `S3ObjectStore`

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Second implementation of the `ObjectStore` Protocol landed
in Phase 30, backed by S3-compatible object storage. Unlocks SOMA on
AWS Lambda / Cloud Run / Fly Machines / any scale-to-zero platform
where local disk is ephemeral.

**Architecture:**
- `src/soma/storage/s3.py` — `S3ObjectStore(bucket, prefix, ...)`
  using `boto3`. Protocol semantics identical to `LocalFSObjectStore`:
  `get_bytes` / `put_bytes` / `get_stream` / `put_stream` /
  `list_prefix` / `delete` / `exists`, raising `KeyError` on missing.
- `s3://bucket/prefix/` URL scheme wired into
  `src/soma/storage/urls.py`. `parse_store_url("s3://...")` returns
  an `S3ObjectStore` with no extra config (credentials come from the
  standard boto3 chain: env vars → `~/.aws/credentials` → IAM role).
- **Streaming uploads**: `put_stream` uses `upload_fileobj` so huge
  blobs (vectors.npy, snapshots) don't buffer into RAM.
- **Streaming downloads**: `get_stream` returns a wrapper around
  `get_object`'s `Body` (itself a `StreamingBody`) that closes the
  connection on `close()`.
- **Endpoint override** (`endpoint_url=`): first-class kwarg so the
  adapter works against MinIO, LocalStack, Cloudflare R2, DigitalOcean
  Spaces, etc. Test suite exercises this via moto.

**Testing:**
- Unit-level: `moto.mock_aws` context manager provides an in-memory
  S3. Fast, deterministic, always-on.
- Optional real-S3 integration tests gated by
  `SOMA_S3_INTEGRATION_BUCKET=...` env. Skipped by default.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`,
`docs/backends.md` big section, `deferred-items.md`.

---

### Task 1: `S3ObjectStore` implementation

**Files:**
- Create: `src/soma/storage/s3.py`
- Create: `tests/test_storage/test_s3.py`
- Modify: `pyproject.toml` — add `s3 = ["boto3>=1.34"]` optional extra
  and `s3-test = ["moto[s3]>=5"]` for unit tests

**Skeleton:**
```python
try:
    import boto3
    from botocore.exceptions import ClientError
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False

class S3ObjectStore:
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        client: Any = None,
    ) -> None:
        if not _HAS_BOTO3:
            raise ImportError('boto3 not installed; pip install "soma[s3]"')
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._client = client or boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    # ... Protocol methods, all using self._full_key()
```

**Tests (all using `moto.mock_aws`):**
```python
@mock_aws
def test_put_get_round_trip():
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="b")
    store = S3ObjectStore(bucket="b")
    store.put_bytes("foo", b"bar")
    assert store.get_bytes("foo") == b"bar"

@mock_aws
def test_prefix_isolation():
    s3 = boto3.client("s3")
    s3.create_bucket(Bucket="b")
    a = S3ObjectStore(bucket="b", prefix="a")
    z = S3ObjectStore(bucket="b", prefix="z")
    a.put_bytes("k", b"A")
    z.put_bytes("k", b"Z")
    assert a.get_bytes("k") == b"A"
    assert z.get_bytes("k") == b"Z"
    assert list(a.list_prefix("")) == ["k"]   # z's 'k' not visible

@mock_aws
def test_list_prefix_returns_sorted_keys_relative(): ...

@mock_aws
def test_delete_missing_is_idempotent(): ...

@mock_aws
def test_exists_true_false(): ...

@mock_aws
def test_get_missing_raises_key_error(): ...

@mock_aws
def test_stream_round_trip_large():
    # 10 MB payload via put_stream / get_stream; memory stays under 2 MB.

@mock_aws
def test_endpoint_url_accepted():
    # Pass endpoint_url="http://localhost:9000" (MinIO-style).
    # Assert the client config carries it through.

@mock_aws
def test_client_override_honoured():
    client = boto3.client("s3")
    store = S3ObjectStore(bucket="b", client=client)
    assert store._client is client

def test_import_error_without_boto3(monkeypatch):
    monkeypatch.setattr("soma.storage.s3._HAS_BOTO3", False)
    with pytest.raises(ImportError, match='soma\\[s3\\]'):
        S3ObjectStore(bucket="b")
```

**Step 5:** `git commit -m "feat(storage): S3ObjectStore via boto3 + moto-backed tests"`

---

### Task 2: `s3://` URL scheme + end-to-end MemoryLayer round-trip

**Files:**
- Modify: `src/soma/storage/urls.py` — dispatch `s3://` to
  `S3ObjectStore`. Parse
  `s3://bucket/prefix/` into `(bucket="bucket", prefix="prefix")`.
  Query-string key-value pairs override the defaults:
  `?endpoint=http://minio:9000&region=us-west-2`.
- Extend: `tests/test_storage/test_urls.py` (remove the Phase 30
  "unknown scheme" test for `s3://`, replace with the dispatch test)
- Create: `tests/test_memory/test_memory_layer_s3.py` — one
  end-to-end moto-backed test that saves a MemoryLayer to S3 and
  reloads it

**Tests:**
```python
@mock_aws
def test_parse_store_url_dispatches_to_s3():
    boto3.client("s3").create_bucket(Bucket="b")
    store = parse_store_url("s3://b/pfx")
    assert isinstance(store, S3ObjectStore)

@mock_aws
def test_parse_store_url_carries_query_params():
    store = parse_store_url("s3://b/pfx?region=eu-west-1")
    assert store._client.meta.region_name == "eu-west-1"

@mock_aws
def test_memory_layer_save_load_s3_round_trip():
    boto3.client("s3").create_bucket(Bucket="b")
    mem = MemoryLayer(...)
    mem.add(["hi"], embedding)
    mem.save("s3://b/bundle")
    restored = MemoryLayer.load("s3://b/bundle")
    assert restored.ntotal == 1
```

**Step 5:** `git commit -m "feat(storage): s3:// URL scheme + MemoryLayer round-trip"`

---

### Task 3: Gated real-S3 integration test

**Files:**
- Create: `tests/test_storage/test_s3_integration.py` (gated via
  `SOMA_S3_INTEGRATION_BUCKET`; `@pytest.mark.slow_s3` marker
  registered in `pyproject.toml`)

**Pattern mirrors Phase 24 / Phase 29 integration gating exactly.**

**Step 5:** `git commit -m "test(storage): gated real-S3 integration test"`

---

### Final sanity

```bash
ruff check src/soma/storage tests/test_storage tests/test_memory
pytest tests/test_storage -q
pytest tests/test_memory -q    # nothing should regress
SOMA_S3_INTEGRATION_BUCKET=my-sandbox pytest tests/test_storage/test_s3_integration.py -q
```

Baseline post-Phase-30: +~20 new `tests/test_storage`. Target for
Phase 31: +~15 new (10 unit + 3 url + 2 memory-layer end-to-end),
+~5 gated integration.

**Gotchas:**
- `moto>=5` renamed `mock_s3` → `mock_aws`. Pin `moto[s3]>=5`.
- `S3.Bucket` vs `s3.client`: use `client` throughout for Protocol
  parity; `Bucket` is higher-level but has surprising behaviour on
  `list_objects_v2` pagination.
- `ListObjectsV2` pagination: any list longer than 1000 keys needs
  `get_paginator`. Test-cover at `N>1000` (easy with moto).
- `StreamingBody` doesn't support `seek`; `get_stream` must return
  a wrapper that reports `seekable() = False` so callers needing a
  seekable stream (e.g. `np.load` on an `.npy` file) know to buffer
  the payload to BytesIO first. Document this on `get_stream`.
