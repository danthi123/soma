# Direction 4a Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LLM-distilled projections to `PredictiveSOMA` so the associator input projections can be trained via a cosine-distance loss against a pretrained embedding model (Ollama + mxbai-embed-large). Phase 1 delivers the machinery (config + teacher interface + cache + loss hook) under TDD. Phases 2–5 are sketched as stubs and will be expanded as Phase 1 lands.

**Architecture:**
- New `soma.llm.embedders` module: `LLMTeacher` protocol, `OllamaEmbedder` concrete, `CachedEmbedder` decorator with disk persistence.
- New `SOMAConfig` fields: `projection_distillation_target`, `projection_distillation_model`, `projection_distillation_base_url`, `projection_distillation_weight`. Default `"none"` preserves existing behavior.
- Distillation loss added to `PredictiveSOMA.process_input`: when enabled, after running the step we compute `teacher_emb = teacher.embed(source_text)` and add `alpha * (1 - cosine_similarity(projection_student_output, teacher_emb))` to the prediction loss before backward. Gradients flow into the learnable projections (same code path Direction 2B already set up).
- Caching: `CachedEmbedder` keys by `sha256(text)` and persists each embedding as a `.pt` file under `benchmarks/.teacher_cache/<model>/<hash>.pt` so repeated benchmarks don't re-hit Ollama.

**Tech Stack:** PyTorch (existing), urllib (for Ollama HTTP like `OllamaBackend`), pytest, existing SOMAConfig/PredictiveSOMA. No new runtime deps.

**Reference skills:**
- @superpowers:test-driven-development — every task below is RED → GREEN → REFACTOR.
- @superpowers:systematic-debugging — if any task fails unexpectedly, do not jump to fixes.

---

## Phase 1: Implementation (TDD)

Seven tasks, each ≤ 5 minutes of implementation work. All land on `main` with clean commits.

### Task 1.1: Add config fields for distillation

**Files:**
- Modify: `src/soma/core/config.py` (new dataclass fields + validation)
- Test: `tests/test_core/test_config.py` (new `TestDistillationConfig` class)

**Step 1: Write failing tests**

Append to `tests/test_core/test_config.py`:

```python
class TestDistillationConfig:
    """Direction 4a: LLM-distilled projections config fields."""

    def test_default_distillation_target_is_none(self) -> None:
        cfg = SOMAConfig()
        assert cfg.projection_distillation_target == "none"

    def test_accepts_llm_embedding_target(self) -> None:
        cfg = SOMAConfig(projection_distillation_target="llm_embedding")
        assert cfg.projection_distillation_target == "llm_embedding"

    def test_rejects_invalid_distillation_target(self) -> None:
        with pytest.raises(ValueError, match="projection_distillation_target"):
            SOMAConfig(projection_distillation_target="bogus")  # type: ignore[arg-type]

    def test_default_distillation_model_is_mxbai(self) -> None:
        cfg = SOMAConfig()
        assert cfg.projection_distillation_model == "mxbai-embed-large"

    def test_default_distillation_base_url_is_local_ollama(self) -> None:
        cfg = SOMAConfig()
        assert cfg.projection_distillation_base_url == "http://localhost:11434"

    def test_default_distillation_weight_is_one(self) -> None:
        cfg = SOMAConfig()
        assert cfg.projection_distillation_weight == 1.0

    def test_rejects_negative_distillation_weight(self) -> None:
        with pytest.raises(ValueError, match="projection_distillation_weight"):
            SOMAConfig(projection_distillation_weight=-0.1)

    def test_accepts_zero_distillation_weight(self) -> None:
        # Zero weight is legal (disables the loss contribution without
        # touching the target mode — useful for ablations).
        cfg = SOMAConfig(projection_distillation_weight=0.0)
        assert cfg.projection_distillation_weight == 0.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_config.py::TestDistillationConfig -v`
Expected: 8 tests FAIL with `AttributeError: 'SOMAConfig' object has no attribute 'projection_distillation_target'`.

**Step 3: Minimal implementation**

In `src/soma/core/config.py`, after the `projection_lr` block (around line 179), add:

```python
    # --- LLM-distilled projections (Direction 4a, 2026-04-19) ---------
    # Train the learnable projections against a pretrained embedding
    # model so they carry semantic structure rather than random noise.
    # When "llm_embedding", the prediction-loss trainer adds a cosine-
    # distance term against teacher.embed(source_text). "none" (default)
    # preserves prior behavior.
    projection_distillation_target: Literal["none", "llm_embedding"] = "none"
    projection_distillation_model: str = "mxbai-embed-large"
    projection_distillation_base_url: str = "http://localhost:11434"
    # Weight on the cosine-distance distillation loss relative to the
    # prediction loss (alpha). 0.0 disables the loss term without
    # disabling the teacher plumbing. Must be >= 0.
    projection_distillation_weight: float = 1.0
```

In `_validate()` (around line 418, right after `projection_lr` check), add:

```python
        if self.projection_distillation_target not in ("none", "llm_embedding"):
            raise ValueError(
                f"SOMAConfig.projection_distillation_target must be 'none' or "
                f"'llm_embedding', got {self.projection_distillation_target!r}"
            )
        if self.projection_distillation_weight < 0.0:
            raise ValueError(
                f"SOMAConfig.projection_distillation_weight must be >= 0, "
                f"got {self.projection_distillation_weight!r}"
            )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_config.py::TestDistillationConfig -v`
Expected: 8 PASS.

Run full config tests: `pytest tests/test_core/test_config.py -v`
Expected: all pass (no regressions).

**Step 5: Commit**

```bash
git add src/soma/core/config.py tests/test_core/test_config.py
git commit -m "$(cat <<'EOF'
Direction 4a: add distillation config fields

Four new SOMAConfig fields for LLM-distilled projections:
- projection_distillation_target: "none" | "llm_embedding"
- projection_distillation_model (default mxbai-embed-large)
- projection_distillation_base_url (default local ollama)
- projection_distillation_weight (alpha, default 1.0)

Defaults preserve legacy behavior. Zero weight legal (ablation hook).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.2: Create `OllamaEmbedder` class

**Files:**
- Create: `src/soma/llm/embedders.py`
- Test: `tests/test_llm/test_embedders.py`
- (If `tests/test_llm/` doesn't exist, create `tests/test_llm/__init__.py` as empty file.)

**Step 1: Check dir exists, create empty __init__ if needed**

```bash
mkdir -p tests/test_llm
touch tests/test_llm/__init__.py
```

**Step 2: Write failing tests**

Create `tests/test_llm/test_embedders.py`:

```python
"""Tests for LLM teacher embedders (Direction 4a)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestOllamaEmbedder:
    def test_embed_returns_tensor_of_expected_dim(self) -> None:
        from soma.llm.embedders import OllamaEmbedder

        # Mock urllib to avoid hitting a real server
        fake_payload = {"embedding": [0.1] * 1024}
        fake_response = MagicMock()
        fake_response.read.return_value = json.dumps(fake_payload).encode("utf-8")
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)

        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
        with patch(
            "soma.llm.embedders.urllib.request.urlopen",
            return_value=fake_response,
        ):
            result = embedder.embed("hello world")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1024,)
        assert result.dtype == torch.float32

    def test_embed_raises_on_server_unreachable(self) -> None:
        import urllib.error

        from soma.llm.embedders import OllamaEmbedder

        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
        with patch(
            "soma.llm.embedders.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Ollama"):
                embedder.embed("hello")

    def test_embed_batch_returns_stacked_tensor(self) -> None:
        from soma.llm.embedders import OllamaEmbedder

        fake_payload = {"embedding": [0.1] * 1024}

        def fake_urlopen(*args, **kwargs):  # type: ignore[no-untyped-def]
            resp = MagicMock()
            resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
        with patch(
            "soma.llm.embedders.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            result = embedder.embed_batch(["a", "b", "c"])
        assert result.shape == (3, 1024)

    def test_embedder_name_reflects_model(self) -> None:
        from soma.llm.embedders import OllamaEmbedder

        embedder = OllamaEmbedder(model="nomic-embed-text")
        assert "nomic-embed-text" in embedder.name
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_llm/test_embedders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'soma.llm.embedders'`.

**Step 4: Implement**

Create `src/soma/llm/embedders.py`:

```python
"""Teacher embedders for LLM-distilled projections (Direction 4a).

``LLMTeacher`` defines the minimal protocol. ``OllamaEmbedder`` is the
concrete adapter for a locally-running Ollama server. ``CachedEmbedder``
(see companion task) wraps any teacher with an on-disk cache.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LLMTeacher(Protocol):
    """Minimal protocol: embed one string or a batch of strings."""

    name: str

    def embed(self, text: str) -> torch.Tensor: ...

    def embed_batch(self, texts: list[str]) -> torch.Tensor: ...


@dataclass
class OllamaEmbedder:
    """Calls Ollama's ``POST /api/embeddings`` endpoint.

    Returns a 1-D ``torch.Tensor`` (float32, shape ``(dim,)``) for
    single inputs and a 2-D tensor (shape ``(B, dim)``) for batches.
    The embedding dimension depends on the model:
    ``mxbai-embed-large`` → 1024, ``nomic-embed-text`` → 768.
    """

    model: str = "mxbai-embed-large"
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"ollama-embed:{self.model}"

    def embed(self, text: str) -> torch.Tensor:
        body = json.dumps(
            {"model": self.model, "prompt": text}
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama embeddings server unreachable at {self.base_url}: "
                f"{exc}. Install Ollama and run `ollama serve`, then "
                f"`ollama pull {self.model}`."
            ) from exc
        vec = payload.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise RuntimeError(
                f"Ollama returned malformed embedding payload: {payload!r}"
            )
        return torch.tensor(vec, dtype=torch.float32)

    def embed_batch(self, texts: list[str]) -> torch.Tensor:
        # Ollama's current embeddings API is one-at-a-time.  A future
        # optimization would use /api/embed (newer) batch endpoint.
        return torch.stack([self.embed(t) for t in texts])
```

**Step 5: Run tests**

Run: `pytest tests/test_llm/test_embedders.py::TestOllamaEmbedder -v`
Expected: 4 PASS.

**Step 6: Commit**

```bash
git add src/soma/llm/embedders.py tests/test_llm/__init__.py tests/test_llm/test_embedders.py
git commit -m "$(cat <<'EOF'
Direction 4a: OllamaEmbedder teacher interface

LLMTeacher protocol + OllamaEmbedder concrete impl for local Ollama
embeddings endpoint. Returns 1D or 2D float32 tensors; dim follows
the model (mxbai-embed-large=1024, nomic-embed-text=768). Raises
RuntimeError with actionable message if Ollama unreachable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.3: Build `CachedEmbedder` wrapper (disk + memory cache)

**Files:**
- Modify: `src/soma/llm/embedders.py` (add class)
- Test: `tests/test_llm/test_embedders.py` (new `TestCachedEmbedder`)

**Step 1: Write failing tests**

Append to `tests/test_llm/test_embedders.py`:

```python
class TestCachedEmbedder:
    def test_cache_hit_skips_underlying_call(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        underlying = MagicMock()
        underlying.name = "ollama-embed:mxbai-embed-large"
        underlying.embed.return_value = torch.tensor([1.0, 2.0, 3.0])

        wrapped = CachedEmbedder(teacher=underlying, cache_dir=str(tmp_path))
        r1 = wrapped.embed("cat")
        r2 = wrapped.embed("cat")
        assert torch.equal(r1, r2)
        # Second call should be served from cache
        assert underlying.embed.call_count == 1

    def test_cache_miss_calls_underlying(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        underlying = MagicMock()
        underlying.name = "ollama-embed:mxbai-embed-large"
        underlying.embed.side_effect = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        ]

        wrapped = CachedEmbedder(teacher=underlying, cache_dir=str(tmp_path))
        wrapped.embed("a")
        wrapped.embed("b")
        assert underlying.embed.call_count == 2

    def test_cache_persists_across_instances(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        under1 = MagicMock()
        under1.name = "ollama-embed:mxbai-embed-large"
        under1.embed.return_value = torch.tensor([0.5, 0.5])

        wrapper1 = CachedEmbedder(teacher=under1, cache_dir=str(tmp_path))
        wrapper1.embed("foo")
        assert under1.embed.call_count == 1

        # Second wrapper using same cache dir should hit disk
        under2 = MagicMock()
        under2.name = "ollama-embed:mxbai-embed-large"
        under2.embed.return_value = torch.tensor([0.5, 0.5])
        wrapper2 = CachedEmbedder(teacher=under2, cache_dir=str(tmp_path))
        result = wrapper2.embed("foo")
        assert torch.equal(result, torch.tensor([0.5, 0.5]))
        assert under2.embed.call_count == 0  # Cache hit from disk

    def test_embed_batch_partially_caches(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        underlying = MagicMock()
        underlying.name = "ollama-embed:mxbai-embed-large"
        underlying.embed.side_effect = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        ]

        wrapped = CachedEmbedder(teacher=underlying, cache_dir=str(tmp_path))
        wrapped.embed_batch(["hot", "cold"])
        assert underlying.embed.call_count == 2

        # Re-batch: should all come from cache
        underlying.embed.side_effect = None
        underlying.embed.reset_mock()
        result = wrapped.embed_batch(["hot", "cold"])
        assert result.shape == (2, 2)
        assert underlying.embed.call_count == 0

    def test_different_models_have_separate_caches(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        under_a = MagicMock()
        under_a.name = "ollama-embed:model-a"
        under_a.embed.return_value = torch.tensor([1.0])
        under_b = MagicMock()
        under_b.name = "ollama-embed:model-b"
        under_b.embed.return_value = torch.tensor([2.0])

        wa = CachedEmbedder(teacher=under_a, cache_dir=str(tmp_path))
        wb = CachedEmbedder(teacher=under_b, cache_dir=str(tmp_path))
        wa.embed("same text")
        wb.embed("same text")
        assert under_a.embed.call_count == 1
        assert under_b.embed.call_count == 1  # not a cache hit from model-a

    def test_cached_embedder_name_preserves_model(self, tmp_path) -> None:
        from soma.llm.embedders import CachedEmbedder

        underlying = MagicMock()
        underlying.name = "ollama-embed:mxbai-embed-large"
        wrapped = CachedEmbedder(teacher=underlying, cache_dir=str(tmp_path))
        assert "mxbai-embed-large" in wrapped.name
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_llm/test_embedders.py::TestCachedEmbedder -v`
Expected: FAIL with `ImportError: cannot import name 'CachedEmbedder'`.

**Step 3: Implement**

Append to `src/soma/llm/embedders.py`:

```python
import hashlib
from pathlib import Path


@dataclass
class CachedEmbedder:
    """Wraps any ``LLMTeacher`` with an on-disk cache.

    Keys by ``sha256(text)``. Embeddings are stored as ``.pt`` files
    under ``<cache_dir>/<teacher-name-slug>/<hash>.pt``. Safe to share
    a cache dir across models — the name slug isolates them.

    Memory cache is kept per-instance; disk cache persists across runs.
    """

    teacher: LLMTeacher
    cache_dir: str
    _mem_cache: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False,
    )
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"cached:{self.teacher.name}"
        # Slugify the teacher name for a safe subdirectory
        slug = self.teacher.name.replace(":", "_").replace("/", "_")
        self._cache_path = Path(self.cache_dir) / slug
        self._cache_path.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_file(self, key: str) -> Path:
        return self._cache_path / f"{key}.pt"

    def embed(self, text: str) -> torch.Tensor:
        key = self._key(text)
        if key in self._mem_cache:
            return self._mem_cache[key]
        disk_file = self._cache_file(key)
        if disk_file.exists():
            tensor = torch.load(disk_file, map_location="cpu", weights_only=True)
            self._mem_cache[key] = tensor
            return tensor
        # Miss: call underlying teacher, cache result
        tensor = self.teacher.embed(text)
        self._mem_cache[key] = tensor
        torch.save(tensor, disk_file)
        return tensor

    def embed_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self.embed(t) for t in texts])
```

**Step 4: Run tests**

Run: `pytest tests/test_llm/test_embedders.py::TestCachedEmbedder -v`
Expected: 6 PASS.

Run full embedders tests: `pytest tests/test_llm/test_embedders.py -v`
Expected: 10 PASS (4 + 6).

**Step 5: Commit**

```bash
git add src/soma/llm/embedders.py tests/test_llm/test_embedders.py
git commit -m "$(cat <<'EOF'
Direction 4a: CachedEmbedder (memory + disk cache)

Wraps any LLMTeacher with sha256-keyed memory + on-disk cache.
Teacher name slug isolates models in shared cache dirs. Safe for
repeated benchmark runs; first pass over LoCoMo's 5882 turns pays
~minutes of Ollama latency, subsequent runs are free.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.4: Hook distillation loss into `PredictiveSOMA.process_input`

**Files:**
- Modify: `src/soma/developmental/prediction.py`
- Test: `tests/test_developmental/test_distillation.py` (new file)

**Step 1: Write failing tests**

Create `tests/test_developmental/test_distillation.py`:

```python
"""Tests for Direction 4a distillation loss in PredictiveSOMA."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def _tiny_config(**kw):
    return SOMAConfig.developmental(
        initial_associator_count=4,
        initial_integrator_count=2,
        max_nodes=16,
        projection_mode="learnable",
        **kw,
    )


class TestDistillationIntegration:
    def test_distillation_off_by_default(self) -> None:
        """No teacher attachment, no distillation behavior."""
        cfg = _tiny_config()
        pred = PredictiveSOMA(config=cfg)
        # No teacher attached → attribute should be None or not present
        teacher = getattr(pred, "_teacher", None)
        assert teacher is None

    def test_attach_teacher_sets_field(self) -> None:
        cfg = _tiny_config(projection_distillation_target="llm_embedding")
        pred = PredictiveSOMA(config=cfg)

        fake_teacher = MagicMock()
        fake_teacher.name = "fake"
        fake_teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(fake_teacher)
        assert pred._teacher is fake_teacher

    def test_process_input_calls_teacher_when_distillation_on(self) -> None:
        cfg = _tiny_config(projection_distillation_target="llm_embedding")
        pred = PredictiveSOMA(config=cfg)
        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        x = torch.randn(cfg.sensor_output_dim)
        # Two calls: first primes _last_summary, second triggers prediction + distill
        pred.process_input(x, source_text="hello")
        pred.process_input(x, source_text="world")

        # Teacher should be called for at least the second step (with source_text)
        assert teacher.embed.called

    def test_process_input_skips_teacher_when_distillation_off(self) -> None:
        cfg = _tiny_config()  # default = "none"
        pred = PredictiveSOMA(config=cfg)
        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        x = torch.randn(cfg.sensor_output_dim)
        pred.process_input(x, source_text="hello")
        pred.process_input(x, source_text="world")
        assert teacher.embed.call_count == 0

    def test_distillation_loss_runs_without_error_on_trivial_input(self) -> None:
        cfg = _tiny_config(
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=1.0,
        )
        pred = PredictiveSOMA(config=cfg)
        teacher = MagicMock()
        teacher.name = "fake"
        # Return same embedding every time → distill loss should be
        # consistent across calls even if non-zero
        teacher.embed.return_value = torch.randn(1024)
        pred.attach_teacher(teacher)

        x = torch.randn(cfg.sensor_output_dim)
        for step in range(5):
            pred.process_input(x, source_text=f"step {step}")

        # No exception; projections still finite
        for p in pred._input_projections.values():
            assert torch.isfinite(p).all()

    def test_zero_weight_disables_distill_but_calls_teacher(self) -> None:
        """alpha=0 is an ablation: teacher is still queried for logging,
        but the loss contribution is zero — projection params should
        update exactly as if distillation were off."""
        cfg_on = _tiny_config(
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=0.0,
            seed=7,
        )
        cfg_off = _tiny_config(seed=7)

        pred_on = PredictiveSOMA(config=cfg_on)
        pred_off = PredictiveSOMA(config=cfg_off)

        teacher = MagicMock()
        teacher.name = "fake"
        teacher.embed.return_value = torch.randn(1024)
        pred_on.attach_teacher(teacher)

        torch.manual_seed(0)
        x = torch.randn(cfg_on.sensor_output_dim)
        for step in range(3):
            pred_on.process_input(x, source_text=f"t{step}")
            pred_off.process_input(x, source_text=f"t{step}")

        # Projections should match (or be numerically very close) because
        # alpha=0 contributes nothing to gradients. Small divergence is
        # allowed from nondeterministic ops but the magnitude should be
        # far smaller than a nontrivial update.
        for k in pred_on._input_projections:
            diff = (
                pred_on._input_projections[k]
                - pred_off._input_projections[k]
            ).abs().max().item()
            assert diff < 1e-3, f"node {k} projections diverged: {diff}"
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_developmental/test_distillation.py -v`
Expected: 6 FAIL. The `attach_teacher` method doesn't exist yet; `_teacher` field doesn't exist.

**Step 3: Implement**

Modify `src/soma/developmental/prediction.py`:

1. Add `_teacher` field initialization in `__init__` (append near line 117, right before `self._fp_index_cache = ...`):

```python
        # Direction 4a: optional LLM teacher for projection distillation.
        # None means no distillation regardless of config. Attached via
        # attach_teacher() after construction so swapping teachers is
        # decoupled from SOMA construction.
        self._teacher: Any = None
```

2. Add `attach_teacher` method (anywhere between existing public methods, e.g., right after `set_tokenizer`):

```python
    def attach_teacher(self, teacher: Any) -> None:
        """Attach an LLM teacher for Direction 4a projection distillation.

        Expects an object with ``embed(text) -> torch.Tensor``. The
        teacher is only consulted when
        ``config.projection_distillation_target == "llm_embedding"``
        and a ``source_text`` is passed to :meth:`process_input`.
        """
        self._teacher = teacher
```

3. Add distillation loss to the prediction-training block in `process_input`. Currently (around line 1066-1104) the code computes `pred_loss` and calls backward. Replace the section starting with `if self._last_summary is not None:` and ending right before `else: self.prediction_error = 0.0` with:

```python
        if self._last_summary is not None:
            # Direction 2: when projections are learnable, route the
            # prediction through a projection-averaged view so the
            # prediction loss trains BOTH the prediction head AND the
            # learnable projections.
            if (
                self.config.projection_mode == "learnable"
                and self._input_projections
            ):
                projected_views = []
                for proj in self._input_projections.values():
                    projected_views.append(proj @ self._last_summary)
                pred_input = torch.stack(projected_views).mean(dim=0)
            else:
                pred_input = self._last_summary
            predicted = self.prediction_head(pred_input)
            pred_loss = torch.nn.functional.mse_loss(
                predicted, current_summary.detach(),
            )
            self.prediction_error = pred_loss.item()

            # Direction 4a: optional LLM-distillation loss.  When
            # enabled and a teacher + source_text are available, add a
            # cosine-distance term between the (learnable) projection-
            # averaged student view and the teacher embedding.  The
            # teacher embedding is projected/padded/truncated to the
            # student dim so the comparison is well-defined regardless
            # of teacher output dim.
            distill_loss: torch.Tensor | None = None
            if (
                self.config.projection_distillation_target == "llm_embedding"
                and self._teacher is not None
                and source_text is not None
                and self.config.projection_mode == "learnable"
                and self._input_projections
            ):
                teacher_emb = self._teacher.embed(source_text).to(self.device)
                # Align dims: truncate or zero-pad teacher to student dim
                s_dim = pred_input.shape[0]
                if teacher_emb.shape[0] >= s_dim:
                    teacher_aligned = teacher_emb[:s_dim]
                else:
                    teacher_aligned = torch.zeros(s_dim, device=self.device)
                    teacher_aligned[: teacher_emb.shape[0]] = teacher_emb
                # Cosine distance: 1 - cosine_similarity
                cos = torch.nn.functional.cosine_similarity(
                    pred_input.unsqueeze(0),
                    teacher_aligned.detach().unsqueeze(0),
                    dim=1,
                )
                distill_loss = (
                    1.0 - cos.squeeze()
                ) * self.config.projection_distillation_weight

            # Only update if error is still meaningful — prevent
            # over-convergence that collapses all fingerprints.
            if self.prediction_error > 1e-5 or distill_loss is not None:
                self._pred_optimizer.zero_grad()
                total_loss = pred_loss
                if distill_loss is not None:
                    total_loss = total_loss + distill_loss
                total_loss.backward()
                self._pred_optimizer.step()
```

Note: the `Any` type hint on `_teacher` requires `Any` imported — already imported at line 14.

**Step 4: Run the new tests**

Run: `pytest tests/test_developmental/test_distillation.py -v`
Expected: 6 PASS.

**Step 5: Run all predictive tests to guarantee backward compat**

Run: `pytest tests/test_developmental/ -v`
Expected: all pass (no regressions in existing tests).

**Step 6: Commit**

```bash
git add src/soma/developmental/prediction.py tests/test_developmental/test_distillation.py
git commit -m "$(cat <<'EOF'
Direction 4a: distillation loss in PredictiveSOMA

attach_teacher() registers an LLMTeacher (optional, off by default).
When config.projection_distillation_target == "llm_embedding" and
source_text is provided, process_input adds an alpha-weighted cosine-
distance term: 1 - cos(projection-averaged-student, teacher.embed(text)).
Gradient flows through learnable projections via existing Direction 2B
code path. Backward compatible: no teacher attached = no behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.5: End-to-end smoke test with real Ollama

**Files:**
- Create: `tests/test_developmental/test_distillation_ollama.py`

This is a gated integration test (skips if Ollama unreachable). Verifies the stack works end-to-end on the real teacher.

**Step 1: Write test**

```python
"""Integration smoke test for Direction 4a with real Ollama teacher.

Gated on Ollama reachability — skips in CI / offline environments.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.llm.embedders import OllamaEmbedder, CachedEmbedder


def _ollama_alive(url: str = "http://localhost:11434") -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1.0) as r:
            return 200 <= r.status < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False


@pytest.mark.skipif(
    not _ollama_alive(),
    reason="Ollama server not reachable at localhost:11434",
)
class TestOllamaDistillEndToEnd:
    def test_three_steps_no_crash_projections_finite(self, tmp_path) -> None:
        """Run three prediction steps with a real Ollama teacher; the
        projections should stay finite and the distillation loss should
        decrease OR at least not explode.
        """
        cfg = SOMAConfig.developmental(
            initial_associator_count=4,
            max_nodes=16,
            projection_mode="learnable",
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=0.5,
        )
        pred = PredictiveSOMA(config=cfg)
        teacher = CachedEmbedder(
            teacher=OllamaEmbedder(model="mxbai-embed-large"),
            cache_dir=str(tmp_path),
        )
        pred.attach_teacher(teacher)

        torch.manual_seed(0)
        x = torch.randn(cfg.sensor_output_dim)
        texts = [
            "the cat sat on the mat",
            "a dog chased a squirrel",
            "the cat sat on the mat",  # repeat: should be cache hit
        ]
        for text in texts:
            result = pred.process_input(x, source_text=text)
            assert torch.isfinite(torch.tensor(result["prediction_error"]))

        # Projections should still be finite
        for p in pred._input_projections.values():
            assert torch.isfinite(p).all()

        # Cache should have exactly 2 entries (3rd was a repeat)
        import os
        subdirs = list((tmp_path).iterdir())
        assert subdirs, "expected at least one teacher subdir in cache"
        files = list(subdirs[0].glob("*.pt"))
        assert len(files) == 2
```

**Step 2: Run**

Run: `pytest tests/test_developmental/test_distillation_ollama.py -v`
Expected on dev machine (Ollama running): 1 PASS.
Expected in CI: 1 SKIP.

**Step 3: Commit**

```bash
git add tests/test_developmental/test_distillation_ollama.py
git commit -m "$(cat <<'EOF'
Direction 4a: real-Ollama end-to-end smoke test (gated)

Skip when Ollama unreachable so CI stays green offline. Verifies:
- 3 steps with real mxbai-embed-large run without crashing
- Projections stay finite after distillation updates
- Cache de-dupes repeated texts

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.6: Backward-compat regression — synap_local v0.5 unchanged

Per the design doc (Phase 1 deliverable): "all tests pass, synap_local v0.5 results unchanged."

**Files:**
- Run existing runner: `research/developmental/env_sequence_v05_synap_local_multiseed.py`

**Step 1: Run the v0.5 locality ablation**

```bash
python research/developmental/env_sequence_v05_synap_local_multiseed.py
```

Expected: results match the reference findings:
- seed=0 synap_only_local beats synap_only 6-8/8
- seed=1 synap_only_local beats synap_only 7-8/8
- seed=42 synap_only_local beats synap_only 8/8

Reference log: `research/developmental/results/env_sequence_v05_synap_position_scramble_multiseed.log`

**Step 2: Compare** against the reference. Tolerance: any seed-regime with MSE change > 5e-4 indicates a real regression; flag and investigate.

**Step 3: If regression**, run `@superpowers:systematic-debugging`. Most likely cause: a non-guarded code path running the teacher even when target=="none".

**Step 4: No commit if results match** — this is verification, not change.

---

### Task 1.7: Run full test suite

**Step 1: Run**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: all tests pass. The v0.5 reference count was 411; new tests added this phase:
- Task 1.1: +8
- Task 1.2: +4
- Task 1.3: +6
- Task 1.4: +6
- Task 1.5: +1 (skipped in CI)

Phase 1 target: 411 → ~436 tests, all green.

**Step 2: If pass, Phase 1 is done.** Proceed to Phase 2 as a fresh subagent.

---

## Phase 2: Synthetic validation (stub)

After Phase 1 lands, run v0.5 capacity schedule with distillation ON. Since v0.5 inputs are synthetic vectors (not text), create a pseudo-text mapping: each 16-dim synthetic vector gets a stable text description ("vector-regime-mlp_2x16-step-500") that the teacher embeds.

**Deliverable:**
- New runner: `research/developmental/env_sequence_v05_distill_multiseed.py`
- Multi-seed run {0, 1, 42} comparing `neuro_only` / `synap_only_local` / `synap_only_local_distilled` / `synap_only_local_distilled_alpha05`
- Findings doc: `research/developmental/results/env_sequence_v05_distill_multiseed_findings.md`

**Success criteria (local):**
- Distillation loss decreases smoothly (no NaN/explode)
- MSE on prediction task ≥ synap_only_local (distillation shouldn't hurt the native task)
- Per-projection cosine(proj, teacher_emb) increases over training steps

---

## Phase 3: LoCoMo retrieval test (stub — critical)

Five systems, all using the same `mxbai-embed-large` teacher:

1. `chroma-mxbai` — Chroma with mxbai embeddings (baseline to beat)
2. `soma-flat-mxbai` — SOMA MemoryLayer using mxbai cosine, no graph
3. `soma-graph-random` — attach_soma, alpha=0.3, locality=0.0, distillation=none
4. `soma-graph-distilled` — attach_soma, alpha=0.3, locality=0.0, distillation=llm_embedding
5. `soma-graph-distilled+local` — attach_soma, alpha=0.3, locality=0.5, distillation=llm_embedding

**Pre-requisites:**
- Fix the CATEGORY_NAMES iteration bug in `benchmarks/run_locomo_locality.py`
- Pre-compute + cache all LoCoMo embeddings before first benchmark

**Deliverable:**
- New runner: `benchmarks/run_locomo_distill.py`
- Results JSON + findings doc with per-category breakdown

**Primary success criterion (the ship gate):**
- `soma-graph-distilled` R@5 > `chroma-mxbai` R@5 by > 0.02
- If < 0.005: null. Pivot.
- If 0.005 – 0.02: weak. Iterate.

---

## Phase 4: Cutoff sweep on distilled (stub)

If Phase 3 shows distillation helps, sweep cutoff {0.0, 0.25, 0.5, 0.75, 1.0} on the distilled variant. Hypothesis: sweet spot shifts because "distance in position space" now correlates with "distance in semantic space."

**Deliverable:** cutoff sweep results + updated inverted-U curve on retrieval task.

---

## Phase 5: Multi-hop / temporal stress test (stub)

Use LoCoMo's per-category breakdown (now that CATEGORY_NAMES iteration is fixed):
- Multi-hop queries
- Temporal queries
- Adversarial queries

**Hypothesis:** `soma-graph-distilled` specifically wins on multi-hop / temporal vs `chroma-mxbai`, because the graph encodes structural information that cosine similarity on single-vector embeddings can't capture.

**This is the "value SOMA adds over standard LLM" demonstration.**

**Deliverable:** per-category comparison table + findings doc.

---

## Exit criteria (repeat from design)

**SHIP:** Phase 3 primary + secondary + tertiary pass → write up, update positioning, promote benchmark.

**STOP:** Phase 3 primary fails → document honest null, pivot research focus.

**ITERATE:** Phase 3 partial → alpha sweep / compose with locality aggressively / revisit re-rank formula. Timebox: one additional day.

---

## Execution

Phase 1 is best executed **subagent-driven in this session**: each task is small, fast, and benefits from review between tasks before the next. Phases 2–5 are larger research arcs and should be separate sessions after Phase 1 lands and its results are reviewed.
