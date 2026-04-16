# Phase 24: Qdrant Snapshot Version-Compat Tests

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** The `QdrantBackend` writes `qdrant_version` into `backend.json` when snapshotting, but we have no tests that exercise cross-version restore. A snapshot taken against Qdrant 1.7 and restored against 1.9 (or vice versa) could silently corrupt. Add a test matrix that:

1. Spins up Qdrant N via docker-compose or testcontainers.
2. Creates a snapshot.
3. Tears down Qdrant N.
4. Spins up Qdrant M.
5. Restores the snapshot.
6. Asserts retrieval still returns expected hits.

**Scope decision:** this is a genuinely expensive test infrastructure-wise. Keep it OPTIONAL — gated by a `SOMA_QDRANT_VERSION_MATRIX=1` env and an `@pytest.mark.slow_qdrant` marker so normal CI doesn't run it. Document how to run it locally or in a dedicated nightly workflow.

**Architecture:**
- Use `testcontainers-python` (the standard approach) — pulls, runs, and tears down docker containers from Python. Graceful fallback if Docker isn't available (test suite skips cleanly).
- Version matrix starts small: **1.11.x**, **1.12.x**, **1.13.x** (current stable). Parametrize the source/target cross product = 9 combinations. Total wall-clock: ~6-10 min for all 9.
- Fixture that yields a `QdrantBackend` pointed at a running container.

**Out-of-scope (I will do centrally):** `CHANGELOG.md`, `docs/backends.md`.

---

### Task 1: testcontainers fixture

**Files:**
- Modify: `pyproject.toml` — add `testcontainers>=4` under a new `qdrant-test` extra (or under `dev`)
- Create: `tests/test_memory/test_qdrant_version_compat.py`

**Fixture sketch:**
```python
import pytest
import uuid

try:
    from testcontainers.qdrant import QdrantContainer
    _HAS_TC = True
except ImportError:
    _HAS_TC = False

pytestmark = [
    pytest.mark.skipif(
        not _HAS_TC, reason="testcontainers not installed"
    ),
    pytest.mark.skipif(
        os.environ.get("SOMA_QDRANT_VERSION_MATRIX") != "1",
        reason="set SOMA_QDRANT_VERSION_MATRIX=1 to run this slow matrix",
    ),
    pytest.mark.slow_qdrant,
]

_QDRANT_VERSIONS = ["1.11.3", "1.12.4", "1.13.0"]

@pytest.fixture
def qdrant_at_version(request):
    version = request.param
    container = QdrantContainer(image=f"qdrant/qdrant:v{version}")
    container.start()
    yield container.get_connection_params()
    container.stop()
```

Register the `slow_qdrant` marker in `pyproject.toml` under `[tool.pytest.ini_options].markers` so pytest doesn't warn on unknown marker.

**Step 1: Trivial test that uses the fixture.**
```python
@pytest.mark.parametrize("qdrant_at_version", _QDRANT_VERSIONS, indirect=True)
def test_qdrant_smoke_per_version(qdrant_at_version):
    backend = QdrantBackend(url=..., collection_name=f"test_{uuid.uuid4().hex[:8]}")
    backend.add(["a"], np.ones((1, 32)).astype(np.float32))
    assert backend.ntotal == 1
```

**Step 5:** `git commit -m "test(qdrant): per-version smoke via testcontainers"`

---

### Task 2: Cross-version snapshot/restore matrix

**Files:**
- Extend: `tests/test_memory/test_qdrant_version_compat.py`

**Matrix:**
```python
@pytest.mark.parametrize("src_version", _QDRANT_VERSIONS)
@pytest.mark.parametrize("tgt_version", _QDRANT_VERSIONS)
def test_snapshot_restore_cross_version(src_version, tgt_version, tmp_path):
    # 1. Spin src_version container.
    # 2. Build a QdrantBackend, add 10 vectors, capture their expected
    #    ids + known-similar query vector.
    # 3. Snapshot into tmp_path / "bundle".
    # 4. Tear down src container.
    # 5. Spin tgt_version container.
    # 6. QdrantBackend.restore(tmp_path / "bundle") into the new container.
    # 7. backend.search(query_vector, k=5) — assert it finds the expected
    #    top-1 id and approximately-matching score.
    # 8. Tear down tgt container.
```

Use a determinstic embed_fn (hash-based stub, same as other tests) so the expected top-1 id is predictable.

**Gotchas to handle:**
- Qdrant snapshot API changed slightly between 1.10 and 1.11 (the REST endpoint layout for `recover-from-snapshot`). If the adapter needs per-version dispatch, that's a real bug to fix — but flag it rather than fixing in this phase. For now, pin the test matrix to versions known to have compatible snapshot formats (1.11+).

**Step 5:** `git commit -m "test(qdrant): 3x3 cross-version snapshot/restore matrix"`

---

### Task 3: GitHub Actions nightly job (optional)

**Files:**
- Extend: `.github/workflows/bench-regression.yml` OR create `.github/workflows/qdrant-version-matrix.yml`

**Shape:**
```yaml
name: Qdrant Version Matrix
on:
  schedule:
    - cron: '0 4 * * 0'   # Sunday 04:00 UTC
  workflow_dispatch:

jobs:
  qdrant-matrix:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    services:
      docker:
        image: docker:dind
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install -e ".[dev,qdrant-test]"
      - env:
          SOMA_QDRANT_VERSION_MATRIX: "1"
        run: pytest tests/test_memory/test_qdrant_version_compat.py -q
```

**Step 5:** `git commit -m "ci: weekly Qdrant version-matrix workflow"`

---

### Task 4: Docs

**Files:**
- Modify: `docs/backends.md` — short section on cross-version snapshot stability + how to enable the matrix locally.

**Step 5:** `git commit -m "docs(qdrant): cross-version snapshot testing guide"`

---

### Final sanity

```bash
# Assuming Docker is available on the dev machine:
SOMA_QDRANT_VERSION_MATRIX=1 pytest tests/test_memory/test_qdrant_version_compat.py -q
# Without Docker: test should skip cleanly.
pytest tests/test_memory -q   # unchanged baseline
ruff check tests/test_memory
```

Baseline: 391 tests in test_memory post-Phase-16. Target: +1 smoke + 9 cross-version = +10 tests — but gated, so default `pytest` run stays +0.

**Fallback if testcontainers isn't installable:** skip this phase's gated tests entirely, document the limitation, and ship just the marker registration + docs. Don't block.
