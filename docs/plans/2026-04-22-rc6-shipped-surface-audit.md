# rc6 Shipped-Surface Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Close the gap between what the SOMA repo advertises and what a plain `pip install soma-memory` user actually gets, by fixing one correctness bug, four packaging/distribution bugs, broken code examples in three docs, a half-shipped auth route, and stale deploy/client artifacts.

**Architecture:** Each work item ships as its own commit on a `rc6-shipped-surface` branch. Nine of the eleven tasks are pure code or config fixes verified by unit tests. One task adds a doctest harness that runs every code block in README/quickstart/cookbook/backends as an isolated subprocess — the structural fix that prevents this class of drift from recurring. The final task bundles CHANGELOG + version bump + release.

**Tech Stack:** Python 3.11+, FastAPI, pytest, `importlib.metadata`, Helm, npm, GitHub Actions.

---

## Why this rc exists

The rc5 push was an **editorial refactor of prose** — positioning, status labels, deploy recipes, `docs/rest-api.md`, the cross-doc terminology sweep. It fixed how the repo **reads**. A deeper external review flagged that the repo still reads more mature than it behaves:

| # | Finding | Severity | Verification |
| - | ------- | -------- | ------------ |
| 1 | Server reopens bundles under `SOMA_EMBED_MODEL` env, ignoring the bundle's persisted `sbert_model_name` | **Correctness** | `src/soma/serve.py:563` uses `MemoryLayer.load(path, embed_fn=_embed_fn())`; `load_with_sbert` at `src/soma/memory/api.py:516` exists specifically to prevent this |
| 2 | CLI imports from `scripts/*.py` which aren't packaged by `[tool.setuptools.packages.find] where = ["src"]` | **Packaging** | `src/soma/cli.py:40,50,152,217,229` — five `from scripts.demo_* import …` statements. `pip install soma-memory` on PyPI breaks `soma index`/`chat`/`stats`/`search` |
| 3 | `soma version` and `/version` look up `version("soma")`, distribution is `soma-memory` | **Packaging** | `cli.py:292` — falls through to `unknown` |
| 4 | `bench` extra uses old distribution name `soma[dev,metrics,ann,sbert]` | **Packaging** | `pyproject.toml:246` — breaks `pip install "soma-memory[bench]"` |
| 5 | `__version__ = "0.1.0"` in `src/soma/__init__.py:3`, TS client, Helm chart all say 0.1.0 while pyproject says 0.2.0rc5 | **Coherence** | Four-way drift: pyproject vs `__init__.py` vs `clients/typescript/package.json` vs `deploy/helm/soma/Chart.yaml` |
| 6 | README quickstart uses `with_sbert()` then `.load("my-brain/")` without embed_fn — raises on line 63 | **Broken example** | README.md:56–63; `MemoryLayer.load` requires `embed_fn=` for sbert-built bundles; partner helper is `load_with_sbert` |
| 7 | `docs/backends.md` calls `MemoryLayer.with_sbert(backend=backend)` 4× — unknown kwarg | **Broken example** | `with_sbert` signature is `(cls, model_name, *, device)`; will `TypeError` |
| 8 | `docs/cookbook.md:310` comment claims `with_sbert(bundle_path=…)` works; same file lines 36/55/345 and `positioning.md:355-358` pair `with_sbert()` with `.load()` as if interchangeable | **Broken example** | Same signature issue as #7 + same bundle-reload issue as #6 |
| 9 | `POST /forget` criteria branch returns 501 by default — `_get_conversational_memory` is a stub that returns `None` | **Half-shipped** | `serve.py:570–586, 1138–1145` |
| 10 | `POST /auth/revoke` referenced in production prose + docstrings but does not exist as an HTTP route | **Dangling reference** | `serve.py:896`; grep for `@app.post("/auth/revoke")` in `src/soma/serve.py` → 0 matches |
| 11 | `clients/typescript/openapi.json` missing `/auth/refresh` entirely | **Stale snapshot** | No matches for the string in the committed snapshot |
| 12 | `fly.toml` ships with no auth vars — boots in Open mode | **Deploy drift** | `fly.toml:1–29` — no `SOMA_JWT_SECRET`, no `SOMA_API_KEY` |

Twelve findings collapse into **eleven work items** below. Items touching the same file (e.g., `cli.py` CLI imports + `soma version`) are batched; items with independent blast radii (server correctness vs docs examples) stay separate so review stays tight.

## Cross-reference matrix

| Task | Files touched | Depends on |
| ---- | ------------- | ---------- |
| 1. Server bundle-reload correctness | `src/soma/serve.py`, `tests/test_serve/test_bundle_reload.py` | — |
| 2. Package CLI script helpers | `src/soma/_cli_commands/__init__.py`, `src/soma/_cli_commands/wiki_chat.py`, `src/soma/_cli_commands/memory_inspect.py`, `scripts/demo_wiki_chat.py`, `scripts/demo_memory_inspect.py`, `src/soma/cli.py`, `tests/test_cli/test_packaging.py` | — |
| 3. Version hygiene (soma version + bench extra + \_\_version\_\_) | `src/soma/__init__.py`, `src/soma/cli.py`, `src/soma/serve.py`, `pyproject.toml`, `tests/test_cli/test_version.py` | — |
| 4. Helm chart + TS client version sync + RELEASING.md | `deploy/helm/soma/Chart.yaml`, `clients/typescript/package.json`, `RELEASING.md` | 3 |
| 5. README quickstart uses `load_with_sbert` | `README.md` | — |
| 6. `backends.md` drops `with_sbert(backend=…)` and shows real composition | `docs/backends.md` | — |
| 7. `cookbook.md` + `positioning.md` purge of `bundle_path=` lie and misleading `.load` juxtaposition | `docs/cookbook.md`, `docs/positioning.md` | — |
| 8. Doctest harness (prevents regression of 5/6/7) | `tests/test_docs/test_doctests.py`, `tests/test_docs/_extractor.py` | 5, 6, 7 |
| 9. Ship `POST /auth/revoke` HTTP route | `src/soma/serve.py`, `tests/test_serve/test_auth_revoke.py`, `docs/auth.md`, `docs/rest-api.md` | — |
| 10. TS OpenAPI snapshot regen + CI drift check + Fly.toml auth + /forget honest labeling | `clients/typescript/openapi.json`, `clients/typescript/src/schema.d.ts`, `.github/workflows/ci.yml`, `fly.toml`, `docs/rest-api.md`, `README.md`, `tests/test_clients/test_openapi_drift.py` | 9 |
| 11. CHANGELOG + pyproject bump rc6 + final audit + tag + push | `CHANGELOG.md`, `pyproject.toml` | all prior |

---

## Pre-flight

**Step A: Confirm git state + branch.**

Run:
```bash
cd E:/Documents/Projects/SOMA
git status -s
git log --oneline -3
```

Expected: clean tree, HEAD is `aaac8c4 release: v0.2.0rc5 — propagate editorial refactor to PyPI`.

**Step B: Create branch.**

```bash
git switch -c rc6-shipped-surface
```

**Step C: Verify every finding still reproduces.** Run these and confirm each output matches what's claimed in the matrix above. Each expected count is in the comment:

```bash
# Finding 1 — expected: 1 match
grep -n "MemoryLayer.load(path, embed_fn=_embed_fn())" src/soma/serve.py

# Finding 2 — expected: 5 matches
grep -n "^from scripts." src/soma/cli.py

# Finding 3 — expected: 1 match
grep -n 'version("soma")' src/soma/cli.py

# Finding 4 — expected: 1 match (the old name in bench extra)
grep -n '"soma\[' pyproject.toml

# Finding 5 — all should emit "0.1.0"
grep -n '^__version__' src/soma/__init__.py
grep '"version"' clients/typescript/package.json
grep -n '^version:' deploy/helm/soma/Chart.yaml

# Finding 6 — broken quickstart
sed -n '50,65p' README.md

# Finding 7 — expected: 4 matches
grep -n 'with_sbert(backend=' docs/backends.md

# Finding 8 — expected: 1 match for the bundle_path= comment
grep -n 'bundle_path=' docs/cookbook.md

# Finding 10 — expected: 0 for @app.post, >=1 for prose refs
grep -c '@app.post("/auth/revoke")' src/soma/serve.py
grep -c '/auth/revoke' src/soma/serve.py

# Finding 11 — expected: 0
grep -c '/auth/refresh' clients/typescript/openapi.json

# Finding 12 — expected: 0
grep -c 'SOMA_JWT_SECRET\|SOMA_API_KEY' fly.toml
```

If any of these return something different from expected, that finding has already been fixed — skip the corresponding task and note the deviation in CHANGELOG.

---

## Task 1: Server honors persisted sbert_model_name on bundle reload

**Files:**
- Modify: `src/soma/serve.py:550-567` (the `_get_mem` function)
- Create: `tests/test_serve/test_bundle_reload.py`

**Why this matters:** a user who saves a bundle under `all-mpnet-base-v2` (768 dims) and restarts the server with default `SOMA_EMBED_MODEL=all-MiniLM-L6-v2` (384 dims) currently gets silent retrieval garbage or a dimension-mismatch crash. `MemoryLayer` already persists `sbert_model_name` to `memory_index.json` precisely so `load_with_sbert` can rebuild the right embedder. The server just never calls `load_with_sbert`.

**Step 1: Write the failing test**

```python
# tests/test_serve/test_bundle_reload.py
"""Server rehydrates sbert-backed bundles under their persisted model
name, not the current ``SOMA_EMBED_MODEL`` env.

Covers a correctness bug where the server always called
``MemoryLayer.load(path, embed_fn=_embed_fn())`` regardless of how the
bundle was saved — so a bundle saved under model A would silently be
reopened under model B if the server's env pointed elsewhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma.memory import MemoryLayer


def _tiny_embed(dim: int):
    import torch

    def _fn(text: str) -> torch.Tensor:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        vec = [b / 255.0 for b in h[:dim]]
        return torch.tensor(vec, dtype=torch.float32)

    return _fn


@pytest.fixture()
def sbert_bundle(tmp_path: Path) -> Path:
    """Bundle whose memory_index.json records a specific sbert name."""
    bundle = tmp_path / "brain"
    mem = MemoryLayer(embed_fn=_tiny_embed(384), embed_dim=384)
    mem.store("alex lives in portland")
    mem._sbert_model_name = "sentence-transformers/all-mpnet-base-v2"
    mem.save(bundle)
    idx = json.loads((bundle / "memory_index.json").read_text())
    assert idx.get("sbert_model_name") == "sentence-transformers/all-mpnet-base-v2"
    return bundle


def test_get_mem_uses_persisted_sbert_name(sbert_bundle, monkeypatch):
    from soma import serve

    monkeypatch.setattr(serve, "BUNDLE_PATH", sbert_bundle)
    monkeypatch.setattr(serve, "EMBED_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setattr(serve, "_mem_cache", {})

    captured: dict[str, object] = {}

    def fake_load_with_sbert(cls, src, *, model_name=None, **kwargs):
        captured["called"] = True
        captured["src"] = src
        captured["model_name"] = model_name
        return MemoryLayer(embed_fn=_tiny_embed(384), embed_dim=384)

    monkeypatch.setattr(MemoryLayer, "load_with_sbert", classmethod(fake_load_with_sbert))
    original_load = MemoryLayer.load

    def fake_load(cls, *args, **kwargs):
        captured["load_called"] = True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(MemoryLayer, "load", classmethod(fake_load))

    _ = serve._get_mem()

    assert captured.get("called") is True
    assert "load_called" not in captured


def test_get_mem_falls_back_to_plain_load_when_no_sbert_name(tmp_path, monkeypatch):
    from soma import serve

    bundle = tmp_path / "brain"
    mem = MemoryLayer(embed_fn=_tiny_embed(64), embed_dim=64)
    mem.store("hello")
    mem.save(bundle)
    idx_path = bundle / "memory_index.json"
    idx = json.loads(idx_path.read_text())
    idx.pop("sbert_model_name", None)
    idx_path.write_text(json.dumps(idx))

    monkeypatch.setattr(serve, "BUNDLE_PATH", bundle)
    monkeypatch.setattr(serve, "_mem_cache", {})
    monkeypatch.setattr(serve, "_embed_fn", lambda: _tiny_embed(64))

    captured: dict[str, object] = {}

    def fake_load(cls, src, *, embed_fn=None, **kwargs):
        captured["load_called"] = True
        return MemoryLayer(embed_fn=_tiny_embed(64), embed_dim=64)

    def fake_load_with_sbert(cls, *args, **kwargs):
        captured["load_with_sbert_called"] = True
        raise AssertionError("should not be called")

    monkeypatch.setattr(MemoryLayer, "load", classmethod(fake_load))
    monkeypatch.setattr(MemoryLayer, "load_with_sbert", classmethod(fake_load_with_sbert))

    _ = serve._get_mem()

    assert captured.get("load_called") is True
    assert "load_with_sbert_called" not in captured
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_serve/test_bundle_reload.py -v
```

Expected: `test_get_mem_uses_persisted_sbert_name` FAILS because current `_get_mem` calls `MemoryLayer.load`, never `load_with_sbert`. The fallback test PASSES (existing behavior).

**Step 3: Implement the fix**

Replace `_get_mem` in `src/soma/serve.py:550-567` with:

```python
def _get_mem(name: str | None = None) -> MemoryLayer:
    key = name or "__default__"
    with _cache_lock:
        if key in _mem_cache:
            mem = _mem_cache[key]
            mem.reload_if_stale()
            return mem
        path = _path_for(name)
        has_snapshot = path.exists() and (path / "memory_index.json").exists()
        has_wal = path.exists() and (path / "memory_ops.wal.jsonl").exists()
        if has_snapshot or has_wal:
            # Prefer the bundle's persisted sbert_model_name over the server's
            # current SOMA_EMBED_MODEL. Mismatch (e.g. bundle saved with
            # mpnet-base-v2, server booted with MiniLM) produces silent
            # retrieval garbage or a dim-mismatch crash. `load_with_sbert`
            # reads `memory_index.json` to rebuild the right embedder. Only
            # fall back to the plain env-driven path for bundles created by
            # the TextEncoder path (no sbert name persisted).
            persisted = _persisted_sbert_name(path) if has_snapshot else None
            if persisted:
                mem = MemoryLayer.load_with_sbert(path)
            else:
                mem = MemoryLayer.load(path, embed_fn=_embed_fn())
        else:
            mem = MemoryLayer.with_sbert(EMBED_MODEL)
        _mem_cache[key] = mem
        return mem


def _persisted_sbert_name(path: Path) -> str | None:
    """Peek at memory_index.json to see if the bundle recorded an sbert model
    name. Returns None if the file is missing, malformed, or lacks the key."""
    idx_path = path / "memory_index.json"
    if not idx_path.exists():
        return None
    try:
        idx = json.loads(idx_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    name = idx.get("sbert_model_name")
    return name if isinstance(name, str) and name else None
```

Add `import json` near the top of `serve.py` if not already there.

**Step 4: Run tests**

```bash
pytest tests/test_serve/test_bundle_reload.py -v
pytest tests/test_serve/ -v
```

Expected: all PASS.

**Step 5: Commit**

```bash
git add src/soma/serve.py tests/test_serve/test_bundle_reload.py
git commit -m "fix(serve): honor bundle's persisted sbert_model_name on reload

Server previously always called MemoryLayer.load(path, embed_fn=_embed_fn())
where _embed_fn built from the current SOMA_EMBED_MODEL env. A bundle
saved under model A would silently reopen under model B if the server's
env differed — producing retrieval garbage on mismatched vectors or a
dim-mismatch crash. Now peek at memory_index.json; if sbert_model_name
is present, use load_with_sbert (which rebuilds the right embedder);
otherwise keep the legacy env-driven path for TextEncoder bundles."
```

---

## Task 2: Package CLI script helpers; unbreak `soma index`/`chat`/`stats`/`search` after pip install

**Files:**
- Create: `src/soma/_cli_commands/__init__.py`
- Create: `src/soma/_cli_commands/wiki_chat.py` (move contents of `scripts/demo_wiki_chat.py` minus the CLI-wrapper `main`)
- Create: `src/soma/_cli_commands/memory_inspect.py` (move contents of `scripts/demo_memory_inspect.py` minus `main`)
- Modify: `scripts/demo_wiki_chat.py` → thin wrapper
- Modify: `scripts/demo_memory_inspect.py` → thin wrapper
- Modify: `src/soma/cli.py:40,50,152,217,229` → import from `soma._cli_commands.*`
- Create: `tests/test_cli/test_packaging.py`

**Step 1: Write the failing test**

```python
# tests/test_cli/test_packaging.py
"""CLI must not import from repo-root `scripts/`.

Once SOMA is installed via pip, only the `src/soma/` tree is on sys.path.
Any `from scripts.X import …` in production code means `soma index`,
`soma chat`, `soma stats`, `soma search` will `ModuleNotFoundError` for
every pip-installed user. This test pins the invariant.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


def test_cli_does_not_import_from_scripts():
    cli_py = SRC / "soma" / "cli.py"
    tree = ast.parse(cli_py.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "scripts" or node.module.startswith("scripts."):
                offenders.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scripts" or alias.name.startswith("scripts."):
                    offenders.append(f"line {node.lineno}: import {alias.name}")
    assert not offenders, (
        "cli.py imports from repo-root scripts/, which are not packaged. "
        "Move helpers into src/soma/_cli_commands/. Offenders:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "subcommand",
    ["index --help", "chat --help", "stats --help", "search --help", "version"],
)
def test_cli_subcommands_work_without_scripts_on_syspath(subcommand, tmp_path):
    """Invoke `soma <sub>` in a subprocess whose sys.path excludes repo root."""
    env = {
        "PATH": "",
        "PYTHONPATH": str(SRC),
        "SOMA_BUNDLE_PATH": str(tmp_path / "brain"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "soma.cli", *subcommand.split()],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined or "scripts" not in combined, (
        f"subcommand `{subcommand}` could not find the `scripts` module:\n{combined}"
    )
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli/test_packaging.py -v
```

Expected: `test_cli_does_not_import_from_scripts` FAILS with 5 offenders (lines 40, 50, 152, 217, 229).

**Step 3: Create the new package**

Create `src/soma/_cli_commands/__init__.py`:

```python
"""Packaged implementations of the ``soma`` CLI subcommands.

Historically these lived in ``scripts/demo_*.py`` at repo root and were
imported by :mod:`soma.cli`. That worked from a clone but broke for
anyone ``pip install``-ing the distribution — ``scripts/`` is not part
of the wheel. They were moved here (private underscore prefix — not
public API, subject to change without a deprecation cycle) so the CLI
works from a plain pip install.

The repo-root ``scripts/demo_*.py`` files remain as thin wrappers over
these modules so existing ``python scripts/demo_wiki_chat.py …``
invocations still work during development.
"""
from __future__ import annotations

__all__ = ["wiki_chat", "memory_inspect"]
```

**Step 4: Move `scripts/demo_wiki_chat.py` contents into `src/soma/_cli_commands/wiki_chat.py`**

Copy the entire file except the `if __name__ == "__main__": main()` guard. `main()` stays.

**Step 5: Rewrite `scripts/demo_wiki_chat.py` as a thin wrapper**

```python
"""Thin wrapper — the real implementation lives in
:mod:`soma._cli_commands.wiki_chat`. This file exists so
``python scripts/demo_wiki_chat.py …`` still works from a clone."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from soma._cli_commands.wiki_chat import *  # noqa: F401,F403
from soma._cli_commands.wiki_chat import main

if __name__ == "__main__":
    main()
```

**Step 6: Same pattern for `memory_inspect.py`**

Repeat steps 4–5 for `scripts/demo_memory_inspect.py` → `src/soma/_cli_commands/memory_inspect.py` + wrapper.

**Step 7: Update `src/soma/cli.py`**

Replace all five `from scripts.demo_* import …` lines with `from soma._cli_commands.<module> import …`:

```python
# Line 40  — was: from scripts.demo_wiki_chat import _ingest
from soma._cli_commands.wiki_chat import _ingest
# Line 50  — was: from scripts.demo_wiki_chat import _chat
from soma._cli_commands.wiki_chat import _chat
# Line 152 — was: from scripts.demo_wiki_chat import _resolve_backend
from soma._cli_commands.wiki_chat import _resolve_backend
# Line 217 — was: from scripts.demo_memory_inspect import cmd_stats
from soma._cli_commands.memory_inspect import cmd_stats
# Line 229 — was: from scripts.demo_memory_inspect import cmd_search
from soma._cli_commands.memory_inspect import cmd_search
```

**Step 8: Run tests**

```bash
pytest tests/test_cli/test_packaging.py -v
pytest tests/test_cli/ -v
python scripts/demo_wiki_chat.py --help
python scripts/demo_memory_inspect.py --help
```

Expected: all PASS. Both script invocations show `--help` without `ModuleNotFoundError`.

**Step 9: Commit**

```bash
git add src/soma/_cli_commands scripts/demo_wiki_chat.py scripts/demo_memory_inspect.py src/soma/cli.py tests/test_cli/test_packaging.py
git commit -m "fix(cli): move scripts/demo_*.py into packaged src/soma/_cli_commands/

soma index/chat/stats/search previously imported from repo-root scripts/
which is not included in the wheel. pip install soma-memory users got
ModuleNotFoundError the moment they ran any of those subcommands.

Helpers moved into a private src/soma/_cli_commands/ package. Repo-root
scripts/demo_*.py kept as thin wrappers so 'python scripts/...'
invocations from a clone still work. Added tests/test_cli/test_packaging.py
to pin the 'no imports from scripts.*' invariant."
```

---

## Task 3: Version hygiene — `soma version`, `bench` extra, `__version__`

**Files:**
- Modify: `src/soma/__init__.py` (read version from `importlib.metadata`)
- Modify: `src/soma/cli.py:292` (`version("soma")` → `soma.__version__`)
- Modify: `src/soma/serve.py` — FastAPI `app.version=` + any `/version` route
- Modify: `pyproject.toml:246` (`"soma[dev,…]"` → `"soma-memory[dev,…]"`)
- Create: `tests/test_cli/test_version.py`

**Step 1: Write the failing test**

```python
# tests/test_cli/test_version.py
"""Version reporting must match the installed distribution metadata.

Three surfaces must agree:
- ``soma.__version__`` (Python package attribute)
- ``soma version`` CLI
- FastAPI ``app.version`` on the live serve app
"""
from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def test_package_version_matches_distribution_metadata():
    import soma

    assert soma.__version__ == version("soma-memory")


def test_cli_version_matches_distribution_metadata():
    result = subprocess.run(
        [sys.executable, "-m", "soma.cli", "version"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == version("soma-memory")
    assert result.stdout.strip() != "unknown"
    assert result.stdout.strip() != "0.1.0"


def test_fastapi_app_version_matches_distribution_metadata():
    import pytest
    pytest.importorskip("fastapi")
    from soma.serve import app

    assert app.version == version("soma-memory")


def test_bench_extra_uses_correct_distribution_name():
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    bench = data["project"]["optional-dependencies"]["bench"]
    for dep in bench:
        if dep.startswith("soma[") or dep.startswith("soma "):
            raise AssertionError(
                f"bench extra references 'soma[…]' but the distribution is "
                f"'soma-memory'. Offender: {dep!r}"
            )
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli/test_version.py -v
```

Expected: all four tests FAIL.

**Step 3: Fix `src/soma/__init__.py`**

Replace:

```python
__version__ = "0.1.0"
```

with:

```python
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("soma-memory")
except PackageNotFoundError:  # pragma: no cover — source checkout, no dist installed
    __version__ = "0.0.0+unknown"
```

**Step 4: Fix `src/soma/cli.py:292`**

Replace:

```python
print(version("soma"))
```

with:

```python
from soma import __version__

print(__version__)
```

Remove the now-unused `from importlib.metadata import version` import if it's only used here.

**Step 5: Fix FastAPI app.version in `src/soma/serve.py`**

Find the `app = FastAPI(…)` call. Replace `version="…"` with:

```python
from soma import __version__

app = FastAPI(
    title="SOMA",
    version=__version__,
    # ... rest unchanged
)
```

Also update any `/version` route that hardcodes `"0.1.0"`.

**Step 6: Fix `pyproject.toml:246`**

```toml
# Before
"soma[dev,metrics,ann,sbert]",

# After
"soma-memory[dev,metrics,ann,sbert]",
```

**Step 7: Run tests**

```bash
pytest tests/test_cli/test_version.py -v
```

Expected: all four PASS.

**Step 8: Commit**

```bash
git add src/soma/__init__.py src/soma/cli.py src/soma/serve.py pyproject.toml tests/test_cli/test_version.py
git commit -m "fix(version): single source of truth for package version

- src/soma/__init__.py hardcoded __version__='0.1.0' → read from
  importlib.metadata.version('soma-memory'). Always matches the
  installed distribution.
- soma version CLI looked up version('soma') which returns 'unknown'
  since the distribution is 'soma-memory'. Route through soma.__version__.
- FastAPI app.version pinned to the same source.
- pyproject bench extra referenced 'soma[...]' (old name) — breaks
  pip install soma-memory[bench]. Corrected to 'soma-memory[...]'.

Added tests/test_cli/test_version.py to pin these invariants."
```

---

## Task 4: Helm chart + TS client version sync + RELEASING.md

**Files:**
- Modify: `deploy/helm/soma/Chart.yaml:5-6` (bump to 0.2.0)
- Modify: `clients/typescript/package.json` (version 0.2.0)
- Create: `RELEASING.md`

**Step 1: Bump Helm chart**

Edit `deploy/helm/soma/Chart.yaml`:

```yaml
version: 0.2.0            # chart version — bumped in lockstep with Python minor
appVersion: "0.2.0"       # application version — tracks pyproject minor
```

**Step 2: Bump TS client**

Edit `clients/typescript/package.json`:

```json
{
  "name": "@soma-ai/client",
  "version": "0.2.0",
  ...
}
```

**Step 3: Write `RELEASING.md`**

```markdown
# Releasing SOMA

## Artifacts

SOMA ships three release artifacts on independent cadences:

| Artifact | Where | Source of version |
| -------- | ----- | ----------------- |
| `soma-memory` Python package | PyPI | `pyproject.toml` `version` field |
| `soma` Helm chart | In-tree at `deploy/helm/soma/` (OCI registry TBD) | `deploy/helm/soma/Chart.yaml` |
| `@soma-ai/client` TypeScript client | npm (unpublished, in-tree at `clients/typescript/`) | `clients/typescript/package.json` |

Python is the canonical source. `src/soma/__init__.py:__version__`, the
FastAPI `app.version`, and the `soma version` CLI all read from
`importlib.metadata.version("soma-memory")` — whatever `pyproject.toml`
declares is what every runtime surface reports.

## Cadence

- **Python patch** (e.g. 0.2.0 → 0.2.1): no Helm / TS bump required.
- **Python minor** (e.g. 0.2.x → 0.3.0): bump Helm + TS to the same
  minor unless intentionally decoupling (document the decoupling).
- **Python major** (e.g. 0.x → 1.x): bump both in lockstep.

## Pre-release gates

Before tagging `v<X.Y.Z>`:

1. `pytest -q` green on main.
2. `pytest tests/test_cli/test_packaging.py tests/test_cli/test_version.py tests/test_docs/ tests/test_clients/ -v` green — the drift gates.
3. `pyproject.toml`, `src/soma/__init__.py` (auto via importlib), Helm
   `Chart.yaml`, and TS `package.json` agree on version.
4. `CHANGELOG.md` has an entry for the target version with date.
5. Regenerate TS OpenAPI snapshot if any new `@app.*` routes landed:
   ```bash
   python -m uvicorn soma.serve:app --port 8420 &
   curl -s http://127.0.0.1:8420/openapi.json > clients/typescript/openapi.json
   npm --prefix clients/typescript run gen:types
   kill %1
   git add clients/typescript/openapi.json clients/typescript/src/schema.d.ts
   ```

## Release

```bash
git tag v<X.Y.Z>
git push git.dant123.com v<X.Y.Z>
git push github v<X.Y.Z>
```

The GitHub `release.yml` workflow builds and publishes to PyPI via
OIDC. Watch the run at https://github.com/danthi123/soma/actions.

## Post-release

Bump `pyproject.toml` to next `-dev` marker (e.g. `0.2.1-dev0`) on main
to make it obvious the PyPI artifact and HEAD have diverged.
```

**Step 4: Commit**

```bash
git add deploy/helm/soma/Chart.yaml clients/typescript/package.json RELEASING.md
git commit -m "chore: align Helm chart + TS client to 0.2.0 minor; add RELEASING.md

Helm chart and TS client have been pinned at 0.1.0 since the earliest
release while the Python package advanced to 0.2.0rc5. They're still
on independent cadences but should at least track the current minor.

RELEASING.md documents the cadence, pre-release gates, and the
OpenAPI snapshot regen step that must run before any release with
new routes."
```

---

## Task 5: README quickstart uses `load_with_sbert`

**Files:**
- Modify: `README.md:50-65`

**Step 1: Confirm current broken example**

```bash
sed -n '50,70p' README.md
```

Expected: `MemoryLayer.load("my-brain/")` on line 63 with no `embed_fn`.

**Step 2: Fix the example**

Edit `README.md` line 63. Replace:

```python
mem = MemoryLayer.load("my-brain/")                   # resume anywhere
```

with:

```python
mem = MemoryLayer.load_with_sbert("my-brain/")        # resume anywhere
```

**Step 3: Hand-verify**

```bash
python -c "
from soma.memory import MemoryLayer
mem = MemoryLayer.with_sbert()
mem.store('user lives in Portland, OR', metadata={'user': 'alex'})
mem.save('/tmp/smoke-brain')
mem2 = MemoryLayer.load_with_sbert('/tmp/smoke-brain')
hits = mem2.retrieve('where does alex live', k=1)
print('hits:', hits)
"
```

Expected: no exception, `hits` has at least one result.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(README): use load_with_sbert in quickstart

Quickstart showed MemoryLayer.with_sbert() followed by MemoryLayer.load(),
which raises: plain .load() requires embed_fn= for sbert-built bundles.
The symmetric partner is .load_with_sbert(), which rebuilds the embedder
from the bundle's persisted sbert_model_name. Fixes the headline example
on the PyPI landing page."
```

---

## Task 6: `docs/backends.md` drops `with_sbert(backend=…)` and shows real composition

**Files:**
- Modify: `docs/backends.md:144,244,288,354`

**Step 1: Inspect current broken examples**

```bash
grep -n 'with_sbert(backend=' docs/backends.md
```

Expected: 4 matches.

**Step 2: Read each block for context**

```bash
sed -n '135,160p' docs/backends.md
sed -n '235,260p' docs/backends.md
sed -n '280,300p' docs/backends.md
sed -n '345,365p' docs/backends.md
```

**Step 3: Rewrite each block**

Pattern:

```python
# Before (broken)
backend = QdrantBackend(host="localhost", port=6333)
mem = MemoryLayer.with_sbert(backend=backend)  # TypeError
```

```python
# After (works) — compose an sbert embed_fn with the desired backend
from sentence_transformers import SentenceTransformer
import torch
from soma.memory import MemoryLayer
from soma.memory.backends import QdrantBackend

_model = SentenceTransformer("all-MiniLM-L6-v2")
def _embed(text: str) -> torch.Tensor:
    return torch.tensor(_model.encode(text, convert_to_numpy=True))

backend = QdrantBackend(host="localhost", port=6333)
mem = MemoryLayer(
    embed_fn=_embed,
    embed_dim=_model.get_sentence_embedding_dimension(),
    backend=backend,
)
```

Apply to all 4 blocks. If a future `with_sbert` overload with `backend=` is planned, add it to the codebase first; until then the docs must match the API.

**Step 4: Commit**

```bash
git add docs/backends.md
git commit -m "docs(backends): drop nonexistent with_sbert(backend=...) kwarg

with_sbert's real signature is (model_name, *, device) — no backend
kwarg. All four code blocks passing backend= were raising TypeError.
Rewrote them to show the real composition pattern: build the sbert
embed_fn manually and pass it to MemoryLayer(embed_fn=..., backend=...)
alongside the backend."
```

---

## Task 7: `cookbook.md` + `positioning.md` — purge `bundle_path=` lie and misleading `.load` juxtaposition

**Files:**
- Modify: `docs/cookbook.md` (lines 36, 55, 310, 345, 509, 630 as needed)
- Modify: `docs/positioning.md:355-358`

**Step 1: Fix the `bundle_path=` comment lie on cookbook.md:310**

Replace:

```python
mem = MemoryLayer.with_sbert()  # pass bundle_path="brain/" for persistence
```

with:

```python
mem = MemoryLayer.with_sbert()
# persist later via mem.save("brain/") — for an auto-persisting layer,
# use MemoryLayer(embed_fn=..., embed_dim=..., bundle_path="brain/") instead.
```

**Step 2: Fix `with_sbert() + .load()` juxtaposition**

For each occurrence below, read surrounding context to confirm the upstream `save()` was done from `with_sbert` (mandatory condition for `load_with_sbert` to be the right partner):

Line 36:
```python
mem = MemoryLayer.with_sbert()  # or .load("brain/") to resume
```
→
```python
mem = MemoryLayer.with_sbert()  # or .load_with_sbert("brain/") to resume
```

Line 345:
```python
mem = MemoryLayer.with_sbert()   # or .load("brain/")
```
→
```python
mem = MemoryLayer.with_sbert()   # or .load_with_sbert("brain/")
```

Line 509 (context-dependent):
```
`mem.save()` / `MemoryLayer.load()` still works unchanged — you're not
```
→ keep `.load()` if the surrounding paragraph is discussing non-sbert bundles; change to `.load_with_sbert()` if sbert.

Line 630 (S3 restore, context-dependent):
```python
restored = MemoryLayer.load("s3://my-bucket/soma/bundle")
```
→ change only if the upstream `save()` in the same section used `with_sbert`.

**Step 3: Fix `positioning.md`**

Lines 355–358:

```python
mem = MemoryLayer.with_sbert()

# Or load an existing brain:
# mem = MemoryLayer.load("my-brain/")
```

→

```python
mem = MemoryLayer.with_sbert()

# Or load an existing brain:
# mem = MemoryLayer.load_with_sbert("my-brain/")
```

**Step 4: Commit**

```bash
git add docs/cookbook.md docs/positioning.md
git commit -m "docs(cookbook,positioning): use load_with_sbert; drop bundle_path= lie

Multiple code blocks paired MemoryLayer.with_sbert() with MemoryLayer.load()
as if they were a symmetric save/load pair — they aren't. Plain load()
requires embed_fn= for sbert-built bundles. The real partner is
load_with_sbert (reads the bundle's persisted sbert_model_name).

cookbook.md:310 comment claimed with_sbert accepts bundle_path= — it
doesn't. Rewrote to point at MemoryLayer(embed_fn=..., bundle_path=...)
or explicit .save() instead."
```

---

## Task 8: Doctest harness — extract and run code blocks from README/quickstart/cookbook/backends

**Files:**
- Create: `tests/test_docs/__init__.py`
- Create: `tests/test_docs/_extractor.py`
- Create: `tests/test_docs/test_doctests.py`

**Context:** Tasks 5/6/7 fixed the immediate broken examples. This harness pins the invariant — any future code block in these docs gets extracted and run in CI as an isolated subprocess. Blocks marked `<!-- doctest: skip -->` are excluded; blocks requiring uninstalled optional deps are skipped at runtime via `pytest.importorskip`.

**Why subprocess (not in-process):** runs each block in a fresh Python process with a clean working directory, mirrors how a real user would try the example, and isolates module-level state between blocks.

**Step 1: Write the extractor**

```python
# tests/test_docs/_extractor.py
"""Extract ```python``` fenced code blocks from markdown files.

Blocks are skipped if preceded by an HTML comment ``<!-- doctest: skip -->``.
Blocks are captured with their file path, line range, and contents so
pytest failures point at the doc, not at the extracted source.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocBlock:
    path: Path
    start_line: int
    end_line: int
    source: str

    @property
    def id(self) -> str:
        return f"{self.path.name}:{self.start_line}-{self.end_line}"


_FENCE = "```"
_SKIP_MARKER = "<!-- doctest: skip -->"


def extract_python_blocks(md_path: Path) -> list[DocBlock]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    blocks: list[DocBlock] = []
    in_block = False
    start_line = -1
    body: list[str] = []
    prev_non_blank = ""
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_block:
            if stripped.startswith(_FENCE + "python"):
                in_block = True
                skip = prev_non_blank.strip() == _SKIP_MARKER
                start_line = -1 if skip else idx
                body = []
                continue
            if stripped:
                prev_non_blank = line
        else:
            if stripped == _FENCE:
                if start_line > 0:
                    blocks.append(
                        DocBlock(
                            path=md_path,
                            start_line=start_line,
                            end_line=idx,
                            source="\n".join(body),
                        )
                    )
                in_block = False
                start_line = -1
                body = []
                prev_non_blank = line
                continue
            body.append(line)
    return blocks
```

**Step 2: Write the test harness (subprocess-based)**

```python
# tests/test_docs/test_doctests.py
"""Run every ```python``` code block in the core docs as a subprocess.

Scope: README.md, docs/quickstart.md, docs/cookbook.md, docs/backends.md.
These are the promises SOMA makes to new users. Any block that won't
run fails CI.

Opt-out: prefix a block with ``<!-- doctest: skip -->`` if it's
illustrative-only (pseudocode, partial snippet, backend requires
a running server, etc.).

Optional-dep handling: blocks that import a module guarded by an extra
(sentence_transformers, fastapi, qdrant_client, chromadb, lancedb) are
skipped via ``pytest.importorskip`` — so CI without those extras stays
green, while CI with ``[dev,sbert,serve,qdrant,...]`` installed exercises
the real thing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_docs._extractor import DocBlock, extract_python_blocks

_REPO = Path(__file__).resolve().parents[2]

_TARGETS = [
    _REPO / "README.md",
    _REPO / "docs" / "quickstart.md",
    _REPO / "docs" / "cookbook.md",
    _REPO / "docs" / "backends.md",
]

_OPTIONAL_DEPS = {
    "sentence_transformers": "sbert",
    "fastapi": "serve",
    "uvicorn": "serve",
    "qdrant_client": "qdrant",
    "chromadb": "chroma",
    "lancedb": "lancedb",
    "psycopg": "pgvector",
    "boto3": "s3",
}


def _collect_blocks() -> list[DocBlock]:
    out: list[DocBlock] = []
    for md in _TARGETS:
        if not md.exists():
            continue
        out.extend(extract_python_blocks(md))
    return out


_BLOCKS = _collect_blocks()


@pytest.mark.parametrize("block", _BLOCKS, ids=[b.id for b in _BLOCKS])
def test_doc_block_runs(block: DocBlock, tmp_path):
    """Run the block in an isolated subprocess. Skip blocks whose imports
    need an optional extra we don't have."""
    for mod, extra in _OPTIONAL_DEPS.items():
        if mod in block.source:
            pytest.importorskip(mod, reason=f"needs [{extra}] extra")

    script = tmp_path / "block.py"
    script.write_text(block.source, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{block.path.name}:{block.start_line}-{block.end_line} failed "
            f"(exit {result.returncode})\n\n"
            f"stderr:\n{result.stderr}\n\n"
            f"stdout:\n{result.stdout}\n\n"
            f"block source:\n{block.source}"
        )
```

**Step 3: Run the harness**

```bash
pytest tests/test_docs/ -v
```

Expected: blocks touched in tasks 5/6/7 now pass. If any other block fails, that's a new finding — either fix the doc or add `<!-- doctest: skip -->` on the preceding line with a one-line reason.

**Step 4: Expect iteration**

The first run surfaces blocks we hadn't triaged. For each failure, either fix or skip (with a comment explaining why). This is expected and worth the time — we're establishing the invariant for the first time.

**Step 5: Commit**

```bash
git add tests/test_docs
git commit -m "test(docs): run every Python code block in README/quickstart/cookbook/backends

Docs drift because nobody runs them. This harness extracts every
\`\`\`python\`\`\` block, runs it in an isolated subprocess, and fails
CI if any fail. Optional-dep blocks (sbert, serve, qdrant, etc.) skip
at runtime via pytest.importorskip. Blocks that are illustrative
only can opt out with a <!-- doctest: skip --> marker on the
preceding line.

Together with tasks 5/6/7 this closes the 'broken examples in the
PyPI landing page' class of drift for good."
```

---

## Task 9: Ship `POST /auth/revoke` HTTP route

**Files:**
- Modify: `src/soma/serve.py` (add the route; `BlocklistBackend` helpers already exist)
- Create: `tests/test_serve/test_auth_revoke.py`
- Modify: `docs/auth.md` (document the route)
- Modify: `docs/rest-api.md` (add row to the Auth section)

**Context:** `cli.py:551` already has `_cmd_auth_revoke` that writes a `RevocationRecord` to the blocklist. `serve.py:168` already constructs `_blocklist = blocklist_from_env()`. `serve.py:896` already tells operators to call `POST /auth/revoke`. Wiring the route is wrapping the existing logic.

**Step 1: Write the failing test**

```python
# tests/test_serve/test_auth_revoke.py
"""POST /auth/revoke — revoke a JWT by jti.

The CLI revoke flow works against a file-backed blocklist. This route
exposes the same action over HTTP so operators can rotate from a
hosted dashboard or CI hook without shell access to the box.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def app_with_blocklist(tmp_path: Path, monkeypatch):
    blocklist_path = tmp_path / "blocklist.jsonl"
    monkeypatch.setenv("SOMA_JWT_BLOCKLIST_PATH", str(blocklist_path))
    monkeypatch.setenv("SOMA_JWT_SECRET", "test-secret-at-least-32-bytes-long-aaaa")
    monkeypatch.setenv("SOMA_BUNDLE_PATH", str(tmp_path / "brain"))

    import importlib
    import soma.serve
    importlib.reload(soma.serve)
    return soma.serve.app, blocklist_path


def _issue_token(sub: str = "alice") -> tuple[str, str]:
    from soma.auth import issue_token

    jti = str(uuid.uuid4())
    token = issue_token(
        sub=sub,
        bundles={"__default__": {"read", "write", "admin"}},
        expires=3600,
        jti=jti,
    )
    return token, jti


def test_revoke_returns_200_and_blocks_future_use(app_with_blocklist):
    app, blocklist_path = app_with_blocklist
    client = TestClient(app)
    token, jti = _issue_token()

    resp = client.post(
        "/auth/revoke",
        json={"token": token, "reason": "rotated"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["revoked"] == jti
    assert blocklist_path.exists()
    assert jti in blocklist_path.read_text()


def test_revoke_by_jti(app_with_blocklist):
    app, _ = app_with_blocklist
    client = TestClient(app)
    admin_token, _ = _issue_token(sub="admin")
    target_jti = str(uuid.uuid4())

    resp = client.post(
        "/auth/revoke",
        json={"jti": target_jti, "exp": int(time.time()) + 3600, "reason": "manual"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["revoked"] == target_jti


def test_revoke_requires_auth(app_with_blocklist):
    app, _ = app_with_blocklist
    client = TestClient(app)
    resp = client.post("/auth/revoke", json={"jti": "x", "exp": 9999999999})
    assert resp.status_code == 401
```

**Step 2: Run test to verify failure**

```bash
pytest tests/test_serve/test_auth_revoke.py -v
```

Expected: all tests FAIL with 404.

**Step 3: Implement the route in `src/soma/serve.py`**

Next to the existing `POST /auth/refresh` handler (around line 872), add:

```python
class AuthRevokeRequest(BaseModel):
    """Body for ``POST /auth/revoke``. Supply exactly one of ``token``
    or ``jti``+``exp``."""

    token: str | None = Field(None, description="Full JWT to revoke")
    jti: str | None = Field(None, description="Token id (if revoking without the token)")
    exp: int | None = Field(
        None, description="Token expiry unix timestamp — required with jti"
    )
    reason: str | None = Field(None, description="Free-form audit note")


class AuthRevokeResponse(BaseModel):
    revoked: str = Field(..., description="The jti that was added to the blocklist")
    exp: int = Field(..., description="Blocklist entry TTL anchor")
    reason: str = Field(..., description="Echoed reason or a generated default")


@app.post(
    "/auth/revoke",
    response_model=AuthRevokeResponse,
    tags=["auth"],
    summary="Revoke a JWT (add its jti to the blocklist)",
)
def auth_revoke(
    body: AuthRevokeRequest,
    principal: Principal = Depends(require_admin),
) -> AuthRevokeResponse:
    """Revoke a token. Either pass the full ``token`` to revoke it by
    content, or pass ``jti``+``exp`` to revoke by identifier.

    The blocklist entry's TTL is set to the token's ``exp`` so expired
    entries self-evict — no manual GC required for the common case.
    """
    import jwt as _jwt
    from soma.auth_revocation import RevocationRecord

    if body.token is not None:
        try:
            unsafe = _jwt.decode(
                body.token, options={"verify_signature": False, "verify_exp": False}
            )
        except _jwt.InvalidTokenError as err:
            raise HTTPException(status_code=400, detail=f"invalid token: {err}") from err
        jti = unsafe.get("jti")
        exp = unsafe.get("exp")
        if not isinstance(jti, str):
            raise HTTPException(
                status_code=400, detail="token missing jti claim; nothing to revoke"
            )
        if not isinstance(exp, int):
            raise HTTPException(
                status_code=400, detail="token missing exp claim; cannot compute TTL"
            )
    elif body.jti is not None and body.exp is not None:
        jti = body.jti
        exp = body.exp
    else:
        raise HTTPException(
            status_code=400, detail="pass exactly one of (token) or (jti+exp)",
        )

    reason = body.reason or "revoked via POST /auth/revoke"
    _blocklist.add(RevocationRecord(
        jti=jti, revoked_at=int(time.time()), reason=reason, exp=exp,
    ))
    return AuthRevokeResponse(revoked=jti, exp=exp, reason=reason)
```

Verify `require_admin` is the right `Depends` — scan for the pattern used by other admin-gated routes and reuse it.

**Step 4: Run tests**

```bash
pytest tests/test_serve/test_auth_revoke.py -v
pytest tests/test_serve/ -v
```

Expected: all PASS.

**Step 5: Update `docs/auth.md` and `docs/rest-api.md`**

Add to `docs/auth.md`:

```markdown
### POST /auth/revoke

Revoke a token over HTTP. Same semantics as ``soma auth revoke`` —
writes a ``RevocationRecord`` to the blocklist keyed off
``SOMA_JWT_BLOCKLIST_PATH``.

Request:
  {"token": "<full JWT>", "reason": "rotated"}
or:
  {"jti": "<uuid>", "exp": <unix ts>, "reason": "rotated"}

Response: 200
  {"revoked": "<jti>", "exp": <unix ts>, "reason": "<echo>"}

Requires ``admin`` scope on the bundle referenced by the authenticated
token. 401 if unauthenticated, 403 if the caller lacks admin scope.
```

Add a row to the Auth section of `docs/rest-api.md`:

| Route | Verb | Auth | Body | Response |
| ----- | ---- | ---- | ---- | -------- |
| `/auth/revoke` | POST | admin | `AuthRevokeRequest` | `AuthRevokeResponse` (200) |

**Step 6: Commit**

```bash
git add src/soma/serve.py tests/test_serve/test_auth_revoke.py docs/auth.md docs/rest-api.md
git commit -m "feat(serve): ship POST /auth/revoke HTTP route

serve.py docstrings and docs/auth.md referenced POST /auth/revoke as
the rotation path, but no such route existed — only the CLI flow
(soma auth revoke). This dangling reference has been misleading
operators since Phase 23.

Route wraps the existing BlocklistBackend (soma.auth_revocation) —
same primitive the CLI hits. Body accepts either a full token
(server extracts jti/exp) or jti+exp directly. Admin-scoped.
Blocklist entry TTL = token exp, so expired entries self-evict."
```

---

## Task 10: TS OpenAPI snapshot regen + CI drift check + Fly.toml + /forget labeling

**Files:**
- Modify: `clients/typescript/openapi.json` (regenerate from live app)
- Modify: `clients/typescript/src/schema.d.ts` (regenerate from snapshot)
- Modify: `.github/workflows/ci.yml` (confirm pytest tests/ wildcard covers this)
- Modify: `fly.toml` (add SOMA_JWT_SECRET placeholder)
- Modify: `docs/rest-api.md` (flag `/forget` criteria-mode)
- Modify: `README.md` (stop listing `/forget` criteria as shipped REST surface)
- Create: `tests/test_clients/test_openapi_drift.py`

**Step 1: Regenerate TS snapshot**

```bash
python -m uvicorn soma.serve:app --host 127.0.0.1 --port 8420 &
SERVE_PID=$!
sleep 3
curl -s http://127.0.0.1:8420/openapi.json | python -m json.tool > clients/typescript/openapi.json
kill $SERVE_PID
cd clients/typescript && npm install && npm run gen:types
cd ../..
```

The regenerated snapshot now includes `/auth/refresh` (already in app) and `/auth/revoke` (newly added in task 9).

**Step 2: Write the drift test**

```python
# tests/test_clients/test_openapi_drift.py
"""The committed TypeScript OpenAPI snapshot must match the live app's
spec. If a PR adds a route without regenerating the client, CI fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
_REPO = Path(__file__).resolve().parents[2]
_SNAPSHOT = _REPO / "clients" / "typescript" / "openapi.json"


def _normalize(spec: dict) -> dict:
    spec = json.loads(json.dumps(spec))
    spec.pop("servers", None)
    return spec


def test_openapi_snapshot_matches_live_app():
    from soma.serve import app

    live = _normalize(app.openapi())
    snapshot = _normalize(json.loads(_SNAPSHOT.read_text()))

    live_paths = set(live["paths"].keys())
    snap_paths = set(snapshot["paths"].keys())
    missing_in_snapshot = live_paths - snap_paths
    extra_in_snapshot = snap_paths - live_paths
    assert not missing_in_snapshot, (
        f"live app has routes not in the committed snapshot:\n  "
        + "\n  ".join(sorted(missing_in_snapshot))
        + "\n\nRegenerate: see RELEASING.md 'Pre-release gates' step 5"
    )
    assert not extra_in_snapshot, (
        f"snapshot has routes not in the live app:\n  "
        + "\n  ".join(sorted(extra_in_snapshot))
    )

    assert live["paths"] == snapshot["paths"], (
        "OpenAPI paths diverge between live app and snapshot. "
        "Regenerate per RELEASING.md."
    )
```

**Step 3: Ensure CI runs it**

Open `.github/workflows/ci.yml`. If the test matrix already runs `pytest tests/` or `pytest tests/test_clients/`, it's covered. If not, add a step. Confirm with:

```bash
grep -n 'pytest' .github/workflows/ci.yml
```

**Step 4: Fly.toml auth placeholder**

Edit `fly.toml`:

```toml
app = "soma-memory"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[env]
  SOMA_EMBED_MODEL = "all-MiniLM-L6-v2"
  SOMA_BUNDLE_PATH = "/data/memory"
  SOMA_BUNDLES_DIR = "/data/bundles"

# Auth — set via `fly secrets set SOMA_JWT_SECRET=$(openssl rand -hex 32)`
# before first deploy. Without this, every endpoint is unauthenticated
# (Open mode). See docs/auth.md for the full JWT setup.
# [env.auth]
#   SOMA_JWT_ALG = "HS256"
#   SOMA_JWT_LEEWAY = "60"

[[mounts]]
  source = "soma_data"
  destination = "/data"

[http_service]
  internal_port = 8420
  force_https = true
  auto_start_machines = true
  auto_stop_machines = "stop"
  min_machines_running = 1

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 2048

[[http_service.checks]]
  path = "/health"
```

**Step 5: `/forget` honest labeling**

In `docs/rest-api.md`, find the `/forget` row and annotate:

> `/forget` accepts `ForgetRequest` with `node_id`/`subject`/`user_id`/`text_matches`/`dry_run`/`case_sensitive`/`summary_strategy`. Single-entry delete by `node_id` works out of the box. Criteria-based forgetting (everything else) returns **501 Not Implemented** unless the server is booted with a wired `ConversationalMemory` via `_get_conversational_memory` — see `docs/conversational-memory.md`. The criteria branch is shipped as a Python API and `soma` CLI feature; it is not a default REST surface.

In `README.md`, find any bullet that claims `/forget` supports criteria / preview / summary-cascade as a live REST surface and mark those as "Python API" or "CLI" specifically. If the README currently has:

```markdown
- /forget — delete a single memory or revoke by criteria, optionally with dry_run preview and summary cascade.
```

change to:

```markdown
- /forget — delete a single memory by `node_id`. (Criteria-based forgetting with preview + summary cascade ships as a Python API / CLI feature; the REST branch returns 501 unless the server is booted with a wired `ConversationalMemory`. See `docs/conversational-memory.md`.)
```

**Step 6: Run tests**

```bash
pytest tests/test_clients/test_openapi_drift.py -v
pytest tests/ -q
```

Expected: all PASS.

**Step 7: Commit**

```bash
git add clients/typescript/openapi.json clients/typescript/src/schema.d.ts .github/workflows/ci.yml fly.toml docs/rest-api.md README.md tests/test_clients/test_openapi_drift.py
git commit -m "fix: TS OpenAPI snapshot drift + Fly auth + /forget honest labeling

- Regenerate clients/typescript/openapi.json and schema.d.ts against
  the live app. Snapshot was missing /auth/refresh and /auth/revoke.
- Add tests/test_clients/test_openapi_drift.py — fails CI on snapshot
  drift. First gate we've had on this.
- fly.toml shipped with no auth config — boots in Open mode. Added
  a commented [env.auth] block with a 'fly secrets set SOMA_JWT_SECRET'
  instruction so operators don't accidentally deploy an unauthenticated
  service.
- docs/rest-api.md and README.md: /forget criteria-mode now labelled
  'returns 501 unless ConversationalMemory is wired' — the current
  default. Single-entry delete by node_id is still listed as live."
```

---

## Task 11: CHANGELOG + pyproject bump rc6 + final audit + tag + push

**Files:**
- Modify: `pyproject.toml` (version → `0.2.0rc6`)
- Modify: `CHANGELOG.md` (new `[0.2.0rc6]` section)

**Step 1: Final cross-cutting audit**

Dispatch a final reviewer subagent to sweep the whole branch. It reads `git diff v0.2.0rc5..HEAD` and checks every finding from the Cross-reference matrix is addressed. Blocking on its approval.

**Step 2: Bump pyproject.toml**

```toml
version = "0.2.0rc6"
```

**Step 3: Write CHANGELOG entry**

```markdown
## [0.2.0rc6] — 2026-04-22

### Fixed — correctness

- **Server reopens bundles under the correct embedder** (`src/soma/serve.py`).
  Previously `_get_mem` always called `MemoryLayer.load(path, embed_fn=_embed_fn())`,
  where `_embed_fn` built from `SOMA_EMBED_MODEL`. A bundle saved under
  model A would silently reopen under model B if the server's env
  differed — producing retrieval garbage on mismatched vectors or a
  dim-mismatch crash. Now peeks at `memory_index.json`; if
  `sbert_model_name` is present, uses `load_with_sbert`.

### Fixed — packaging

- **CLI subcommands now work after `pip install soma-memory`.** Helpers
  for `soma index / chat / stats / search` used to live in
  `scripts/demo_*.py` at repo root; `pip install` only packages `src/`,
  so those subcommands hit `ModuleNotFoundError`. Moved into
  `src/soma/_cli_commands/`. Dev-time `python scripts/demo_wiki_chat.py`
  still works via thin wrappers.
- **`soma version` no longer prints `unknown`** (`src/soma/cli.py`).
  Was looking up `version("soma")`; distribution is `soma-memory`.
- **`bench` extra no longer broken** (`pyproject.toml`). Referenced
  the old `soma[...]` name.
- **`soma.__version__` is now a single source of truth.** Reads from
  `importlib.metadata.version("soma-memory")` at import time. FastAPI
  `app.version` and `soma version` CLI both read from it. Four-way
  drift (`0.1.0` hardcoded in `__init__.py`, TS client, Helm chart vs
  pyproject) eliminated. Helm chart + TS client bumped to `0.2.0`.
- **`RELEASING.md` documents the release cadence.**

### Fixed — docs (broken examples)

- `README.md` quickstart: `MemoryLayer.load("my-brain/")` after
  `with_sbert()` → `MemoryLayer.load_with_sbert("my-brain/")`. The
  plain `load` raises without an `embed_fn`.
- `docs/backends.md`: `MemoryLayer.with_sbert(backend=backend)` (4
  occurrences) rewritten against the real `with_sbert` signature.
- `docs/cookbook.md`: `bundle_path=` kwarg lie on line 310, and
  misleading `with_sbert()` + `.load()` juxtapositions fixed to use
  `.load_with_sbert()`.
- `docs/positioning.md`: same fix.

### Added — regression gates

- `tests/test_docs/test_doctests.py` — extracts and runs every
  ```python``` code block in README / quickstart / cookbook / backends
  in an isolated subprocess. Optional-dep blocks skip at runtime via
  `importorskip`. Opt-out marker: `<!-- doctest: skip -->`.
- `tests/test_cli/test_packaging.py` — pins the "no imports from
  `scripts.*`" invariant.
- `tests/test_cli/test_version.py` — pins version agreement across
  `__init__.py`, CLI, FastAPI app, `pyproject.toml`.
- `tests/test_clients/test_openapi_drift.py` — fails CI if the
  committed TS OpenAPI snapshot drifts from the live app.
- `tests/test_serve/test_bundle_reload.py` — pins server honors
  persisted `sbert_model_name`.
- `tests/test_serve/test_auth_revoke.py` — pins the newly-shipped
  HTTP route.

### Added — missing surfaces

- `POST /auth/revoke` HTTP route (`src/soma/serve.py`). Previously
  referenced in docstrings and `docs/auth.md` as the rotation path,
  but didn't exist as a route. Same primitive as `soma auth revoke`
  CLI (writes to the configured `BlocklistBackend`).

### Fixed — deploy / release artifacts

- Regenerated `clients/typescript/openapi.json` and `schema.d.ts`
  against the live app. Previously missing `/auth/refresh`.
- `fly.toml`: added a commented `[env.auth]` block with a
  `fly secrets set SOMA_JWT_SECRET` instruction. Previous toml shipped
  with no auth config — Open mode by default.
- `docs/rest-api.md` + `README.md`: `/forget` criteria-mode labelled
  "returns 501 unless a `ConversationalMemory` is wired" — the actual
  default behavior. Single-node delete by `node_id` remains listed as
  the live REST surface.
```

**Step 4: Commit + tag**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "release: v0.2.0rc6 — shipped-surface audit (correctness + packaging + drift gates)"
git tag v0.2.0rc6
```

**Step 5: Push to both remotes**

```bash
git push git.dant123.com rc6-shipped-surface
git push github rc6-shipped-surface
git push git.dant123.com v0.2.0rc6
git push github v0.2.0rc6
```

**Step 6: Watch release workflow + verify rc6 on PyPI**

Open https://github.com/danthi123/soma/actions. Wait for release.yml to complete. Then:

```bash
curl -s https://pypi.org/pypi/soma-memory/0.2.0rc6/json | python -m json.tool | head -30
```

Expected: wheel + sdist both uploaded, `yanked: false`.

**Step 7: Merge + cleanup**

Once rc6 is verified live:

```bash
git switch main
git merge --no-ff rc6-shipped-surface
git push git.dant123.com main
git push github main
git branch -d rc6-shipped-surface
```

Once rc6 is validated by real users (or a few days in), either cut `0.2.0` final or another rc.

---

## Remember

- **One failing test per step** — if you can't make it fail first, you don't know if it tests the right thing.
- **Commit per task.** 11 commits on the branch before the release commit.
- **Re-review on every subagent cycle.** Spec compliance first, then code quality. Don't skip.
- **Doctest harness (task 8) is the load-bearing one** — tasks 5/6/7 are table stakes but 8 prevents recurrence for every future PR.
- **Correctness bug (task 1) is the showstopper** — it can't ship in anything tagged as stable.
