# Phase 2 — SomaVerbalizer Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development when run alongside a controller) to implement this plan task-by-task.

**Goal:** Build the `SomaVerbalizer` projection module — SOMA's 128-dim OUTPUT → (k, d_model) soft-prompt prefix — as a standalone, LLM-agnostic, swap-friendly `nn.Module` with save/load and a fallback path that delegates to SOMA's existing TextDecoder. No HF transformers dependency yet; that's Phase 3.

**Architecture:** A thin `nn.Sequential` projector (Linear → LayerNorm → GELU → Linear) with near-zero final-layer init so an untrained verbalizer is a safe no-op. A `VerbalizerSpec` frozen dataclass pins the identity: `soma_output_dim`, `llm_name`, `llm_hidden_dim`, `num_prefix_tokens`. Save/load is directory-shaped (`spec.json` + `weights.pt`) to slot into the brain-bundle's `verbalizer/` subdir from Phase 1 Task 9. A `SomaAggregator` helper on the SOMA side collapses per-OUTPUT-node activations into the canonical `(B, 128)` input.

**Tech Stack:** Python 3.11+, PyTorch `nn.Module`/`nn.Sequential`, existing SOMA codebase at `src/soma/`. Tests use a **mock LLM stand-in** (`nn.Module` returning `(B, T, D)` from `forward(inputs_embeds=...)`) to exercise the prefix-concat contract without the real HF transformers dep.

---

## Why Phase 2 Before Phase 3

Phase 3 adds `transformers` as a dependency, wires in a real SmolLM2-360M, and starts the first end-to-end generation path. That's a big chunk of surface area. Phase 2 delivers the **projection contract** in isolation so that:

1. Phase 3 can swap transformers without touching the core `SomaVerbalizer` module — just rebuild with a new `VerbalizerSpec`.
2. The swap procedure ("retrain only the projector, SOMA brain untouched") is testable via Phase 2 alone — no LLM needed for the swap test.
3. The near-zero-init safety property (untrained verbalizer is LLM-vanilla) is baked in from day one.
4. The fallback-to-TextDecoder path is available before Phase 3 lands, so a partial-install or CPU-only deployment has *something* to say.

---

## Key Contract (from Phase-1 verbalizer research, H-Task 3)

```python
@dataclass(frozen=True)
class VerbalizerSpec:
    soma_output_dim: int        # e.g. 128 — SOMAConfig.integrator_output_dim
    llm_name: str               # e.g. "HuggingFaceTB/SmolLM2-360M-Instruct"
    llm_hidden_dim: int         # e.g. 960 for SmolLM2-360M
    num_prefix_tokens: int      # k, typically 8–16
    proj_hidden_dim: int = 512

class SomaVerbalizer(nn.Module):
    def __init__(self, spec: VerbalizerSpec): ...
    def forward(self, soma_state: Tensor) -> Tensor:
        # (B, soma_output_dim) -> (B, k, llm_hidden_dim)
        ...
    def fallback_text(self, soma, activations) -> str: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "SomaVerbalizer": ...
```

**Invariants:**
- Near-zero final-layer init → untrained projector ≈ null prefix → LLM behaves vanilla.
- `VerbalizerSpec` is frozen; a projector is tied to its spec. Swap = new spec + new projector.
- Save path is a **directory**: `spec.json` + `weights.pt` — readable without importing PyTorch.
- `fallback_text` is a pure delegation to `soma.text_decoder` — no duplicate logic.

---

## Phases

| Task | Scope | Files |
|---|---|---|
| 1 | `VerbalizerSpec` frozen dataclass | `src/soma/io/verbalizer.py`, test |
| 2 | `SomaVerbalizer.__init__` + module structure | same |
| 3 | `SomaVerbalizer.forward()` + shape contract tests | same |
| 4 | Near-zero init + "vanilla LLM on untrained projector" property test | same |
| 5 | `SomaVerbalizer.save()` / `.load()` — directory format | same |
| 6 | `SomaAggregator` helper in SOMA for `(B, 128)` extraction | `src/soma/io/verbalizer.py` or `system.py`, test |
| 7 | `fallback_text` delegation to TextDecoder | same |
| 8 | Brain-bundle integration: `verbalizer/` subdir in `SOMA.save_bundle` | `src/soma/system.py`, test |
| 9 | Swap-compatibility contract test (two specs, different `llm_hidden_dim`, both work) | test |
| 10 | Full regression — pytest + ruff + mypy | — |

---

## Phase 2 Task 1: `VerbalizerSpec` frozen dataclass

**Files:**
- Create: `src/soma/io/verbalizer.py`
- Create: `tests/test_io/test_verbalizer.py`

**Step 1: Write the failing test**

```python
# tests/test_io/test_verbalizer.py
import pytest
from soma.io.verbalizer import VerbalizerSpec


def test_spec_defaults_and_frozen():
    spec = VerbalizerSpec(
        soma_output_dim=128,
        llm_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        llm_hidden_dim=960,
        num_prefix_tokens=16,
    )
    assert spec.soma_output_dim == 128
    assert spec.llm_name == "HuggingFaceTB/SmolLM2-360M-Instruct"
    assert spec.llm_hidden_dim == 960
    assert spec.num_prefix_tokens == 16
    assert spec.proj_hidden_dim == 512  # default

    # Frozen — mutation should raise
    with pytest.raises((AttributeError, TypeError)):
        spec.num_prefix_tokens = 32  # type: ignore[misc]


def test_spec_validates_positive_dims():
    with pytest.raises(ValueError, match="positive"):
        VerbalizerSpec(
            soma_output_dim=0, llm_name="x", llm_hidden_dim=960,
            num_prefix_tokens=16,
        )
    with pytest.raises(ValueError, match="positive"):
        VerbalizerSpec(
            soma_output_dim=128, llm_name="x", llm_hidden_dim=0,
            num_prefix_tokens=16,
        )
    with pytest.raises(ValueError, match="positive"):
        VerbalizerSpec(
            soma_output_dim=128, llm_name="x", llm_hidden_dim=960,
            num_prefix_tokens=0,
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_io/test_verbalizer.py -v`
Expected: FAIL — module doesn't exist.

**Step 3: Implement**

```python
# src/soma/io/verbalizer.py
"""SOMA → frozen transformer projection contract.

The SomaVerbalizer is the stable interface between SOMA's cognition
(variable over time via structural plasticity) and a swappable small
transformer's verbalization (fixed weights, replaceable). A projector
tied to a specific VerbalizerSpec can be retrained on new paired data
when the transformer is swapped, without touching the SOMA brain.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerbalizerSpec:
    """Identity card for a trained verbalizer projector.

    Saved alongside weights so that a future loader can detect
    incompatibility (e.g., different ``llm_hidden_dim``) before silently
    producing a broken prefix.
    """
    soma_output_dim: int
    llm_name: str
    llm_hidden_dim: int
    num_prefix_tokens: int
    proj_hidden_dim: int = 512

    def __post_init__(self) -> None:
        for name in ("soma_output_dim", "llm_hidden_dim", "num_prefix_tokens", "proj_hidden_dim"):
            val = getattr(self, name)
            if val <= 0:
                raise ValueError(f"VerbalizerSpec.{name} must be positive, got {val}")
```

**Step 4: Run test**

Run: `pytest tests/test_io/test_verbalizer.py -v`
Expected: PASS (2 tests).

**Step 5: Commit**

```bash
git add src/soma/io/verbalizer.py tests/test_io/test_verbalizer.py
git commit -m "feat(verbalizer): VerbalizerSpec frozen dataclass with positive-dim validation"
```

---

## Phase 2 Task 2: `SomaVerbalizer.__init__` + module structure

**Files:**
- Modify: `src/soma/io/verbalizer.py`
- Modify: `tests/test_io/test_verbalizer.py`

**Step 1: Write the failing test**

```python
# append to tests/test_io/test_verbalizer.py
import torch
from soma.io.verbalizer import SomaVerbalizer


def test_verbalizer_constructs_with_expected_param_count():
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=960,
        num_prefix_tokens=16, proj_hidden_dim=512,
    )
    v = SomaVerbalizer(spec)
    # Check the two linear layers exist and have expected shapes
    params = dict(v.named_parameters())
    # Exact names don't matter; count trainable params.
    total = sum(p.numel() for p in v.parameters() if p.requires_grad)
    # Rough upper bound: (128*512 + 512) + (512*16*960 + 16*960) ≈ 7.9M
    assert total < 10_000_000
    assert total > 5_000_000


def test_verbalizer_stores_spec():
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=960,
        num_prefix_tokens=8,
    )
    v = SomaVerbalizer(spec)
    assert v.spec == spec
    assert v.spec is spec  # frozen — same instance
```

**Step 2: Run test — fail (SomaVerbalizer doesn't exist).**

**Step 3: Implement**

```python
# append to src/soma/io/verbalizer.py
import torch
from torch import nn


class SomaVerbalizer(nn.Module):
    """Projects SOMA's OUTPUT-node aggregate into a soft-prompt prefix.

    Training: forward() + LM-loss through a FROZEN LLM backpropagates
    only into this module.

    Inference: forward() produces ``(B, k, llm_hidden_dim)`` tensors
    that a caller concatenates ahead of tokenized text via the LLM's
    ``inputs_embeds`` entrypoint.

    Swap procedure: to pair with a new LLM, construct with a new
    VerbalizerSpec and retrain. SOMA's graph is untouched.
    """

    def __init__(self, spec: VerbalizerSpec) -> None:
        super().__init__()
        self.spec = spec
        self.proj = nn.Sequential(
            nn.Linear(spec.soma_output_dim, spec.proj_hidden_dim),
            nn.LayerNorm(spec.proj_hidden_dim),
            nn.GELU(),
            nn.Linear(
                spec.proj_hidden_dim,
                spec.num_prefix_tokens * spec.llm_hidden_dim,
            ),
        )
```

**Step 4: Run tests — 4 pass.**

**Step 5: Commit**

```bash
git add src/soma/io/verbalizer.py tests/test_io/test_verbalizer.py
git commit -m "feat(verbalizer): SomaVerbalizer module skeleton (Linear+LayerNorm+GELU+Linear)"
```

---

## Phase 2 Task 3: `forward()` + shape-contract tests

**Files:**
- Modify: `src/soma/io/verbalizer.py`
- Modify: `tests/test_io/test_verbalizer.py`

**Step 1: Write the failing test**

```python
# append to tests/test_io/test_verbalizer.py
def test_forward_2d_input_produces_prefix():
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=960,
        num_prefix_tokens=8,
    )
    v = SomaVerbalizer(spec)
    x = torch.randn(4, 128)
    y = v(x)
    assert y.shape == (4, 8, 960)


def test_forward_1d_input_auto_unsqueezes():
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=512,
        num_prefix_tokens=4,
    )
    v = SomaVerbalizer(spec)
    x = torch.randn(128)  # no batch dim
    y = v(x)
    assert y.shape == (1, 4, 512)


def test_forward_rejects_wrong_last_dim():
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=512,
        num_prefix_tokens=4,
    )
    v = SomaVerbalizer(spec)
    x = torch.randn(2, 64)  # wrong last dim
    with pytest.raises(ValueError, match="soma_output_dim"):
        v(x)
```

**Step 2: Run — fail (no forward).**

**Step 3: Implement**

```python
# add to SomaVerbalizer
def forward(self, soma_state: torch.Tensor) -> torch.Tensor:
    """
    Args:
        soma_state: (B, soma_output_dim) or (soma_output_dim,) aggregated
            OUTPUT-node activations from SOMA.

    Returns:
        (B, num_prefix_tokens, llm_hidden_dim) soft prefix embeddings.
    """
    if soma_state.ndim == 1:
        soma_state = soma_state.unsqueeze(0)
    if soma_state.shape[-1] != self.spec.soma_output_dim:
        raise ValueError(
            f"SomaVerbalizer expects last dim "
            f"{self.spec.soma_output_dim} (soma_output_dim), "
            f"got {soma_state.shape[-1]}"
        )
    flat = self.proj(soma_state)  # (B, k*D)
    return flat.view(
        soma_state.shape[0],
        self.spec.num_prefix_tokens,
        self.spec.llm_hidden_dim,
    )
```

**Step 4: Run — 7 tests pass.**

**Step 5: Commit**

```bash
git add src/soma/io/verbalizer.py tests/test_io/test_verbalizer.py
git commit -m "feat(verbalizer): forward() with 1D/2D input + shape-contract validation"
```

---

## Phase 2 Task 4: Near-zero init + "vanilla LLM" property test

**Files:**
- Modify: `src/soma/io/verbalizer.py`
- Modify: `tests/test_io/test_verbalizer.py`

**Step 1: Write the failing test**

```python
# append
def test_untrained_verbalizer_emits_near_null_prefix():
    """Safety property: an untrained projector should not actively corrupt LLM
    generation. Near-zero init on the final layer means the prefix is
    essentially empty tokens, so the LLM behaves ~vanilla."""
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=960,
        num_prefix_tokens=16,
    )
    v = SomaVerbalizer(spec)
    x = torch.randn(1, 128) * 10.0  # even with large input, output should be tiny
    y = v(x)
    # Final prefix should be much smaller than ~1 (the typical hidden-state
    # magnitude). Use 0.1 as a generous upper bound.
    assert y.abs().max().item() < 0.1, f"near-null init violated: max={y.abs().max().item()}"


def test_final_layer_bias_is_zero_at_init():
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=512,
        num_prefix_tokens=4,
    )
    v = SomaVerbalizer(spec)
    # Find the final Linear
    linears = [m for m in v.proj if isinstance(m, nn.Linear)]
    final = linears[-1]
    assert final.bias is not None
    assert torch.all(final.bias == 0.0)
```

**Step 2: Run — `test_untrained_verbalizer_emits_near_null_prefix` probably fails (default init is not tiny).**

**Step 3: Implement near-zero init in `__init__`**

```python
# after self.proj = nn.Sequential(...) in __init__
_final = [m for m in self.proj if isinstance(m, nn.Linear)][-1]
nn.init.normal_(_final.weight, std=1e-3)
nn.init.zeros_(_final.bias)
```

**Step 4: Run — 9 tests pass.**

**Step 5: Commit**

```bash
git add src/soma/io/verbalizer.py tests/test_io/test_verbalizer.py
git commit -m "feat(verbalizer): near-zero final-layer init (untrained = LLM-vanilla)"
```

---

## Phase 2 Task 5: `save()` / `load()` — directory format

**Files:**
- Modify: `src/soma/io/verbalizer.py`
- Modify: `tests/test_io/test_verbalizer.py`

**Step 1: Write the failing test**

```python
# append
from pathlib import Path


def test_save_load_round_trip(tmp_path: Path):
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=960,
        num_prefix_tokens=8, proj_hidden_dim=256,
    )
    v = SomaVerbalizer(spec)
    # Nudge weights off init to prove round-trip
    with torch.no_grad():
        for p in v.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    out = tmp_path / "verbalizer"
    v.save(out)
    assert (out / "spec.json").exists()
    assert (out / "weights.pt").exists()

    v2 = SomaVerbalizer.load(out)
    assert v2.spec == spec
    for (n1, p1), (n2, p2) in zip(
        v.named_parameters(), v2.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)


def test_load_rejects_wrong_directory(tmp_path: Path):
    import pytest
    with pytest.raises(FileNotFoundError):
        SomaVerbalizer.load(tmp_path / "does-not-exist")
```

**Step 2: Run — fail (no save/load).**

**Step 3: Implement**

```python
# add to SomaVerbalizer
import json as _json
from dataclasses import asdict
from pathlib import Path as _Path


def save(self, path: _Path | str) -> None:
    """Save verbalizer as a directory: spec.json + weights.pt."""
    p = _Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "spec.json").write_text(_json.dumps(asdict(self.spec), indent=2))
    torch.save(self.state_dict(), str(p / "weights.pt"))


@classmethod
def load(cls, path: _Path | str) -> "SomaVerbalizer":
    p = _Path(path)
    spec_path = p / "spec.json"
    weights_path = p / "weights.pt"
    if not spec_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"No verbalizer bundle at {p}")
    spec = VerbalizerSpec(**_json.loads(spec_path.read_text()))
    v = cls(spec)
    v.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    return v
```

Use module-level `from pathlib import Path` and `import json` rather than the `_Path`/`_json` aliases in the real implementation — they were local here only to avoid colliding with earlier test imports.

**Step 4: Run — 11 tests pass.**

**Step 5: Commit**

```bash
git add src/soma/io/verbalizer.py tests/test_io/test_verbalizer.py
git commit -m "feat(verbalizer): directory-format save/load (spec.json + weights.pt)"
```

---

## Phase 2 Task 6: `SomaAggregator` — collapse OUTPUT activations to (B, 128)

**Files:**
- Modify: `src/soma/io/verbalizer.py`
- Modify: `tests/test_io/test_verbalizer.py`

**Rationale:** `SomaVerbalizer.forward(x)` expects `(B, soma_output_dim)`. SOMA at runtime produces per-OUTPUT-node activations (one vector per output node). `SomaAggregator.collapse(output_activations)` is the documented bridge.

**Step 1: Write the failing test**

```python
# append
from soma.io.verbalizer import SomaAggregator


def test_aggregator_mean_pools_output_activations():
    # 3 output nodes, each with 128-dim activation
    acts = {
        "node_a": torch.ones(128) * 1.0,
        "node_b": torch.ones(128) * 2.0,
        "node_c": torch.ones(128) * 3.0,
    }
    pooled = SomaAggregator.collapse(acts, soma_output_dim=128)
    # Mean should be 2.0 (entry-wise)
    assert pooled.shape == (1, 128)
    assert torch.allclose(pooled, torch.ones(1, 128) * 2.0)


def test_aggregator_handles_no_output_nodes():
    pooled = SomaAggregator.collapse({}, soma_output_dim=128)
    assert pooled.shape == (1, 128)
    assert torch.all(pooled == 0.0)  # zero vector when nothing is active


def test_aggregator_rejects_wrong_dim():
    acts = {"a": torch.ones(64)}
    with pytest.raises(ValueError, match="expected dim"):
        SomaAggregator.collapse(acts, soma_output_dim=128)
```

**Step 2: Run — fail.**

**Step 3: Implement**

```python
# append to src/soma/io/verbalizer.py
class SomaAggregator:
    """Collapses per-OUTPUT-node activations into a canonical (1, D) vector.

    SOMA produces one activation per OUTPUT node at each step; the verbalizer
    consumes a single D-dim vector. Mean pooling is the v1 strategy — simple,
    order-invariant, and graceful when node count changes via growth.
    """

    @staticmethod
    def collapse(
        activations: dict[str, torch.Tensor], *, soma_output_dim: int
    ) -> torch.Tensor:
        if not activations:
            return torch.zeros(1, soma_output_dim)
        stacked = []
        for node_id, act in activations.items():
            if act.shape[-1] != soma_output_dim:
                raise ValueError(
                    f"SomaAggregator expected dim {soma_output_dim}, "
                    f"got {act.shape[-1]} for node {node_id}"
                )
            stacked.append(act.view(-1))
        mean = torch.stack(stacked, dim=0).mean(dim=0)
        return mean.unsqueeze(0)
```

**Step 4: Run — 14 tests pass.**

**Step 5: Commit**

```bash
git add src/soma/io/verbalizer.py tests/test_io/test_verbalizer.py
git commit -m "feat(verbalizer): SomaAggregator.collapse mean-pools OUTPUT activations"
```

---

## Phase 2 Task 7: `fallback_text` — delegate to SOMA TextDecoder

**Files:**
- Modify: `src/soma/io/verbalizer.py`
- Modify: `tests/test_io/test_verbalizer.py`

**Rationale:** Before Phase 3's real LLM is loaded, or in embedded/offline mode, the system must still be able to produce *some* text. Delegation to the existing `TextDecoder` keeps zero duplication.

**Step 1: Write the failing test using a stub decoder**

```python
# append
class _StubTextDecoder:
    """Stub mirroring the subset of TextDecoder the fallback calls."""
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def decode_sequence(self, activations: torch.Tensor) -> str:
        self.calls.append(activations)
        return f"stub-decoded:shape={tuple(activations.shape)}"


class _StubSOMA:
    def __init__(self) -> None:
        self.text_decoder = _StubTextDecoder()


def test_fallback_text_delegates_to_soma_text_decoder():
    spec = VerbalizerSpec(
        soma_output_dim=128, llm_name="smoke", llm_hidden_dim=960,
        num_prefix_tokens=8,
    )
    v = SomaVerbalizer(spec)
    soma = _StubSOMA()
    acts = torch.randn(4, 128)
    out = v.fallback_text(soma, acts)
    assert out.startswith("stub-decoded:")
    assert len(soma.text_decoder.calls) == 1
    assert soma.text_decoder.calls[0] is acts  # no copy
```

**Step 2: Run — fail.**

**Step 3: Implement**

```python
# add to SomaVerbalizer
def fallback_text(self, soma, activations: torch.Tensor) -> str:
    """Delegate to ``soma.text_decoder.decode_sequence``.

    Used when no LLM is loaded (embedded, offline, debug). The verbalizer
    does NOT duplicate TextDecoder logic — it just routes through so
    callers have one consistent "give me text" API.
    """
    return soma.text_decoder.decode_sequence(activations)
```

**Step 4: Run — 15 tests pass.**

**Step 5: Commit**

```bash
git add src/soma/io/verbalizer.py tests/test_io/test_verbalizer.py
git commit -m "feat(verbalizer): fallback_text delegates to SOMA TextDecoder"
```

---

## Phase 2 Task 8: Brain-bundle integration — `verbalizer/` subdir

**Files:**
- Modify: `src/soma/system.py` — extend `save_bundle` / `load_bundle` to handle an optional verbalizer
- Modify: `tests/test_core/test_brain_bundle.py`

**Step 1: Write the failing test**

```python
# append to tests/test_core/test_brain_bundle.py
def test_save_bundle_includes_verbalizer_subdir(tmp_path: Path):
    from soma.io.verbalizer import VerbalizerSpec, SomaVerbalizer

    cfg = _small_cfg()  # existing factory
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name="smoke",
        llm_hidden_dim=64,
        num_prefix_tokens=4,
        proj_hidden_dim=32,
    )
    verb = SomaVerbalizer(spec)

    out = tmp_path / "bundle"
    soma.save_bundle(str(out), verbalizer=verb)

    assert (out / "verbalizer").is_dir()
    assert (out / "verbalizer" / "spec.json").exists()
    assert (out / "verbalizer" / "weights.pt").exists()

    # Manifest should record the LLM identity for swap-detection
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["llm_identity"] == "smoke"


def test_load_bundle_returns_verbalizer_when_present(tmp_path: Path):
    from soma.io.verbalizer import VerbalizerSpec, SomaVerbalizer

    cfg = _small_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name="smoke", llm_hidden_dim=64, num_prefix_tokens=4,
        proj_hidden_dim=32,
    )
    verb = SomaVerbalizer(spec)

    out = tmp_path / "bundle"
    soma.save_bundle(str(out), verbalizer=verb)

    soma2 = SOMA(cfg, device=torch.device("cpu"))
    result = soma2.load_bundle(str(out))
    # load_bundle returned (tokenizer, encoder) before Task 8; now also verbalizer
    _, _, loaded_verb = result  # or whichever return style we adopt
    assert loaded_verb is not None
    assert loaded_verb.spec == spec


def test_load_bundle_returns_none_verbalizer_when_absent(tmp_path: Path):
    cfg = _small_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    out = tmp_path / "bundle"
    soma.save_bundle(str(out))  # no verbalizer

    soma2 = SOMA(cfg, device=torch.device("cpu"))
    result = soma2.load_bundle(str(out))
    _, _, loaded_verb = result
    assert loaded_verb is None
```

**Step 2: Run — fail (signature mismatch).**

**Step 3: Implement**

Extend `SOMA.save_bundle` to take `verbalizer: SomaVerbalizer | None = None`. If present, call `verbalizer.save(out / "verbalizer")` and also stamp `spec.llm_name` into `manifest.llm_identity`.

Extend `SOMA.load_bundle` to return `(tokenizer, encoder, verbalizer)` — look for `verbalizer/spec.json` and call `SomaVerbalizer.load` if present, else None.

**Step 4: Run — all bundle tests pass; old `load_bundle` callers that expected a 2-tuple return need updating.** Check scripts that currently call `load_bundle` — update any unpacking.

**Step 5: Commit**

```bash
git add src/soma/system.py src/soma/io/verbalizer.py tests/test_core/test_brain_bundle.py
git commit -m "feat(brain): save_bundle/load_bundle integrate verbalizer subdir"
```

**Note for implementer:** Changing `load_bundle` return type from 2-tuple → 3-tuple is a minor breaking API change. Inspect all call sites with `grep -rn "load_bundle" src/ scripts/ tests/` before changing. If call sites exist beyond tests, either keep backward-compat (return 2-tuple when no verbalizer present) or fix call sites. Document the choice in the commit message.

---

## Phase 2 Task 9: Swap-compatibility contract test

**Files:**
- Modify: `tests/test_io/test_verbalizer.py`

**Rationale:** The whole point of the split architecture is that SOMA brains survive transformer swaps. This test pins that invariant — two different `VerbalizerSpec`s (different `llm_hidden_dim`, different `num_prefix_tokens`) both work, independently, with the same contract.

**Step 1: Write the failing-or-passing test (may already pass if Tasks 1-5 are solid)**

```python
# append to tests/test_io/test_verbalizer.py
def test_two_specs_independently_valid():
    # Simulate swap from SmolLM2-360M → Qwen2.5-1.5B (960 → 1536 hidden)
    spec_old = VerbalizerSpec(
        soma_output_dim=128, llm_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        llm_hidden_dim=960, num_prefix_tokens=16,
    )
    spec_new = VerbalizerSpec(
        soma_output_dim=128, llm_name="Qwen/Qwen2.5-1.5B-Instruct",
        llm_hidden_dim=1536, num_prefix_tokens=8,
    )
    v_old = SomaVerbalizer(spec_old)
    v_new = SomaVerbalizer(spec_new)

    x = torch.randn(2, 128)
    y_old = v_old(x)
    y_new = v_new(x)
    assert y_old.shape == (2, 16, 960)
    assert y_new.shape == (2, 8, 1536)

    # The two projectors are NOT interchangeable — loading one into the other
    # would fail, which is the correct property (VerbalizerSpec pins identity).


def test_cannot_load_spec_mismatch(tmp_path: Path):
    spec_a = VerbalizerSpec(
        soma_output_dim=128, llm_name="a", llm_hidden_dim=512,
        num_prefix_tokens=4,
    )
    spec_b = VerbalizerSpec(
        soma_output_dim=128, llm_name="b", llm_hidden_dim=1024,  # different!
        num_prefix_tokens=4,
    )
    v_a = SomaVerbalizer(spec_a)
    v_a.save(tmp_path / "a")

    # Hand-tamper: replace spec.json with spec_b's contents (simulating a
    # careless bundle edit)
    (tmp_path / "a" / "spec.json").write_text(
        json.dumps(asdict(spec_b))
    )
    # Now load should either raise (shape mismatch) or return a different
    # spec — either is acceptable; key is that we don't silently produce
    # a mis-shaped prefix.
    with pytest.raises((RuntimeError, ValueError)):
        SomaVerbalizer.load(tmp_path / "a")
```

**Step 2: Run — first test passes; second may need an explicit shape check in `load`.**

**Step 3: Implement shape-mismatch guard if needed**

Add to `SomaVerbalizer.load`:
```python
# After constructing v with new spec, before load_state_dict:
state = torch.load(str(weights_path), map_location="cpu")
# load_state_dict will raise RuntimeError on shape mismatch automatically
v.load_state_dict(state)
```

The built-in `load_state_dict` already raises `RuntimeError` on shape mismatch — no extra code needed. Verify.

**Step 4: Run — all tests pass.**

**Step 5: Commit**

```bash
git add tests/test_io/test_verbalizer.py src/soma/io/verbalizer.py
git commit -m "test(verbalizer): swap-compatibility contract (spec-mismatch rejected)"
```

---

## Phase 2 Task 10: Full regression — pytest + ruff + mypy

**Step 1: Run gates**

```bash
pytest tests/ -v
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/soma/
```

All must pass.

**Step 2: Fix anything red.** Each fix is a separate commit: `fix(<area>): <one-line>`.

**Step 3: Verify Phase 1 tests still pass** — brain_bundle tests, growth journal tests, development tests should all be untouched and green.

---

## Phase 2 Complete → Phase 3 Next

Phase 2 delivers the projection contract in isolation. Phase 3 (not in this plan) will:

- Add `transformers>=4.40` to `pyproject.toml` under `[dev-chat]` extras
- Create `src/soma/io/chat_head.py` loading a frozen HF `AutoModelForCausalLM`
- Wire: `soma.generate(user_text) → verbalizer → chat_head.generate(inputs_embeds=...)`
- Integration smoke script with SmolLM2-360M-Instruct
- Position-IDs + attention-mask correctness tests

Phase 3 also is the first point we load real LLM weights — RTX 3090 can handle SmolLM2-360M fp16 easily (~1 GB), so no quantization needed for dev.

---

## Appendix — Size/VRAM Estimate

Phase 2 adds:
- `SomaVerbalizer` with `soma_output_dim=128`, `llm_hidden_dim=960`, `k=16`, `proj_hidden_dim=512`:
  - 128×512 + 512 + 512×(16×960) + (16×960) ≈ **7.95 M params**
- fp32 training: ~31 MB
- fp32 inference: ~31 MB

Phase 2 itself does NOT load any transformer. VRAM impact at this stage is trivial.

---

*End of Phase 2 plan. Tasks 1-10, TDD-disciplined.*
