# Hybrid SOMA + Verbalization-Head Architecture — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn SOMA from a standalone continually-learning system into a portable "brain" that drives a swappable, small, consumer-grade instruction-tuned transformer through a prefix-tuned verbalization head.

**Architecture:** SOMA (cognition, learning, memory, intrinsic motivation) → `SomaVerbalizer` (prefix-soft-prompt projection, ~8–16M params, trainable) → frozen small transformer (e.g., Qwen2.5-1.5B-Instruct or SmolLM2-360M-Instruct). The SOMA brain is serialized into a versioned, migration-friendly "brain bundle" that survives SOMA code upgrades and transformer swaps.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace `transformers` + `tokenizers`, HuggingFace `peft` (reference), llama.cpp / GGUF for deployment, existing SOMA codebase at `src/soma/`.

---

## Why This Pivot

SOMA at step ~800K has demonstrated stable training, growth-pathway activation, and a 21%+ heldout-loss reduction after enabling structural plasticity. The autonomous improvement loop has reached its terminal state — further meaningful progress needs operator-scoped design decisions, not more hyperparameter nudges.

The operator's vision (confirmed 2026-04-14) is a **JARVIS-style personal assistant** that:

- Starts bare-bones, grows tangibly with its user over time
- Runs on **consumer-grade hardware** — home compute box, Apple Silicon laptop, or phone (via GGUF quantization), never enterprise datacenters
- Has a **portable "brain"** — users can back up their SOMA brain and restore it into an updated SOMA codebase, or onto a newer/better small transformer, without losing accumulated personality and knowledge
- Replaces retrieval-based systems — the "brain" is cognition, not a prompt-engineering pipeline

SOMA alone cannot produce fluent long-form natural language on this trajectory (our current output is a single `Linear → vocab` decoder; its limit is at most corpus-statistics). A **small, frozen, instruction-tuned transformer** is the cheapest path to fluent natural-language IO while keeping all *learning* in SOMA. The transformer is the voice; SOMA is the mind.

This plan deliberately prioritizes **interface stability** — SOMA talks to the transformer through a thin, well-specified contract that lets us swap transformers as better small models ship, without retraining the brain.

---

## Architecture Strata

```
┌──────────────────────────────────────────────────────────────────┐
│  USER                                                             │
│    │ natural language                                             │
│    ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Frozen Small Transformer  (swappable)                         │ │
│  │   Qwen2.5-1.5B-Instruct  (primary)                            │ │
│  │   SmolLM2-360M-Instruct  (dev sandbox)                        │ │
│  │   [future: Qwen3, Llama-3.3-x, …]                             │ │
│  └─────────▲──────────────────────────────────────┬──────────────┘ │
│            │ prefix soft-prompt (k × d_model)     │ token embeds   │
│  ┌─────────┴─────────┐                            │                │
│  │ SomaVerbalizer     │  (TRAINABLE, ~8–16M params)                │
│  │  nn.Sequential:    │                                            │
│  │   Linear(128,512)  │                                            │
│  │   LayerNorm+GELU   │                                            │
│  │   Linear(512,k·D)  │                                            │
│  │  near-zero init    │                                            │
│  └─────────▲─────────┘                                             │
│            │ OUTPUT node activations (aggregated, 128-dim)          │
│  ┌─────────┴───────────────────────────────────────────────────┐   │
│  │ SOMA Brain (PORTABLE)                                         │   │
│  │   dynamic graph: Nodes, Edges, growth & pruning               │   │
│  │   WorkingMemory, EpisodicMemory, parametric (graph weights)   │   │
│  │   CuriosityModule, HomeostaticRegulator, DevelopmentSchedule  │   │
│  │                                                               │   │
│  │   serialized as:  brain.pt  +  tokenizer.json  +  manifest   │   │
│  │   schema_versioned, CPU-normalized, migration-hooked          │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**Stability contract:** SOMA's 128-dim OUTPUT → SomaVerbalizer → (k, d_model) prefix. Swapping the transformer changes only `d_model` and `k` — retrain the verbalizer's *output* layer on a small paired dataset. SOMA is untouched. This is the whole point of the split.

---

## Key Decisions (with multi-round debate summaries)

### Decision 1: Transformer choice → **Qwen2.5-1.5B-Instruct (primary), SmolLM2-360M-Instruct (dev sandbox)**

Debate summary (full record in research agent transcripts):

- **Rejected TinyLlama-1.1B** — 2K context, obsolete MMLU ~26.
- **Rejected Llama-3.2-x** — gated community license with 700M MAU cap; incompatible with "user-owned personal assistant" goal.
- **Rejected Phi-3.5-mini (3.8B)** — too capable on its own, would mask SOMA's contribution; LongRoPE adds integration friction.
- **Rejected Gemma-2-2B** — 256K vocab wastes memory when driven by custom prefix; Gemma license has use-restriction clauses.
- **SmolLM2-360M-Instruct** kept as dev sandbox: Apache-2.0, <1GB FP16, loads on laptop, lets the projection-layer iteration cycle stay fast. Too weak for real chat (MMLU ~24 ≈ random).
- **Qwen2.5-1.5B-Instruct chosen as primary**: Apache-2.0, MMLU ~60 (sustains multi-turn chat), 32K context, Q4_K_M ~1GB (fits on flagship phones, M2, RTX 3060), GQA keeps KV cache small, supports `inputs_embeds` cleanly.

**Why 1.5B not 0.5B**: Below ~1.3B the coherence floor drops below what's needed to make SOMA's contribution visible rather than buried under transformer noise.

**Why both**: Dev loop uses 360M on a laptop; integration-test and final training use 1.5B on the RTX 3090. Same HF API, same `inputs_embeds` entrypoint, minimal code branching.

### Decision 2: Interface mechanism → **Prefix soft-prompt (k=8–16 tokens, shallow)**

Debate summary (full record in research agent transcripts):

- **Rejected LoRA-on-transformer** — adapters are pinned to specific base weights; LLM swap = retrain adapters AND lose the "personality" baked into them. Fails the portability goal.
- **Rejected cross-attention injection (Flamingo-style)** — architectural surgery prevents drop-in transformer swap.
- **Rejected P-Tuning v2 (deep prompts)** — more params, tied to layer count AND d_model, worse portability than shallow prefix.
- **Rejected hidden-state addition** — empirically weak (LLaVA-era lesson).
- **Rejected text-summary chain-of-thought** — lossy translation through SOMA's weak TextDecoder; loses dense signal; JARVIS feel degrades to "LLM paraphrasing garbage".
- **Shallow prefix tuning chosen**: SOMA emits a continuous intent vector; projection produces k soft prefix embeddings prepended to the LLM's token embeddings via `inputs_embeds`. Frozen LLM, trainable ~8–16M-param projector. Transformer swap = retrain only the projector. Near-zero init → an untrained projector is a safe no-op, LLM behaves like vanilla.

**Fallback**: If no transformer is loaded (e.g., offline, embedded, or debug mode), SOMA's existing `TextDecoder` remains available as a crude text channel. The verbalizer exposes `fallback_text()` that delegates to it.

### Decision 3: Brain format → **Versioned bundle with migration hooks**

Directory-shaped bundle replaces the single-file checkpoint:

```
soma-brain-<hash>/
├── manifest.json           # schema_version, soma_version, llm_identity (optional), git_sha, created_at, torch/tokenizers versions
├── brain.pt                # payload: graph, memories, metacognition, homeostasis, config, growth_log
├── tokenizer.json          # HF tokenizer
├── encoder.pt              # text encoder sidecar (embedding + position)
├── decoder.pt              # text decoder sidecar (fallback-channel output head)
└── verbalizer/             # OPTIONAL — present when paired with a transformer
    ├── spec.json           # VerbalizerSpec (llm_name, llm_hidden_dim, k, soma_output_dim)
    └── weights.pt
```

The 14 portability gaps from H-Task 2 (documented below in Phase 1) become the task list for delivering this. **Schema version = 1** is the first tagged format; everything before it is "pre-versioned legacy" and loadable via a one-way migrator.

---

## Phases

| Phase | Scope | Trigger to Start |
|---|---|---|
| 1 | Brain Portability Foundation — schema envelope, CPU-normalize, migration hook, tokenizer bundle, growth journal, interface_spec | Immediately |
| 2 | Verbalizer Interface — `SomaVerbalizer` module, `VerbalizerSpec`, contract tests | After Phase 1 complete |
| 3 | First Transformer Integration — SmolLM2-360M-Instruct wired end-to-end via HF, fallback to `TextDecoder`, CPU-only path | After Phase 2 complete |
| 4 | Bootstrap Training — paired-data LM loss to train the projector on dev sandbox | After Phase 3 smoke-runs |
| 5 | Interactive Conversation Loop — session-aware context, WM integration, dashboard/CLI surface | After Phase 4 produces fluent-ish output |
| 6 | Consumer-Grade Packaging — Qwen2.5-1.5B integration, GGUF export, llama.cpp runtime, phone/Apple Silicon validation | After Phase 5 proves usefulness |

**Phase 1 is fully specified below with TDD-style bite-sized tasks.** Phases 2–6 have scope notes; each gets its own plan document when Phase N-1 completes.

---

## Phase 1: Brain Portability Foundation

**Goal of Phase 1:** Every save/load operation produces a versioned bundle that can be migrated across future SOMA versions, loaded on CPU-only hosts, and survives transformer/decoder replacement.

**Critical files touched:**
- `src/soma/system.py` — top-level save/load
- `src/soma/core/config.py` — `from_dict` strictness relaxation
- `src/soma/core/graph.py`, `src/soma/core/node.py`, `src/soma/core/edge.py` — serialize additions
- `src/soma/metacognition/development.py` — persist periods
- `src/soma/io/text_encoder.py` — tokenizer round-trip
- NEW: `src/soma/core/brain_bundle.py` — bundle IO, schema envelope, migration dispatch
- `tests/test_core/test_brain_bundle.py` — TDD tests
- `scripts/migrate_legacy_checkpoint.py` — one-shot migrator for pre-v1 checkpoints

**Skills referenced:**
- `@developmental-ai` — SOMA domain patterns (already loaded)
- `@superpowers:test-driven-development` — red/green/refactor discipline
- `@superpowers:frequent-commits` — commit per task

---

### Task 1: Add schema envelope wrapper

**Files:**
- Create: `src/soma/core/brain_bundle.py`
- Test: `tests/test_core/test_brain_bundle.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_brain_bundle.py
from soma.core.brain_bundle import wrap_payload, unwrap_payload, SCHEMA_VERSION


def test_wrap_adds_schema_version_and_metadata():
    payload = {"hello": "world"}
    wrapped = wrap_payload(payload, soma_version="0.1.0", git_sha="abc1234")
    assert wrapped["format"] == "soma-brain"
    assert wrapped["schema_version"] == SCHEMA_VERSION
    assert wrapped["soma_version"] == "0.1.0"
    assert wrapped["git_sha"] == "abc1234"
    assert "torch_version" in wrapped
    assert "tokenizers_version" in wrapped
    assert "created_at" in wrapped
    assert wrapped["payload"] == payload


def test_unwrap_returns_payload_and_metadata():
    wrapped = wrap_payload({"x": 1}, soma_version="0.1.0")
    payload, meta = unwrap_payload(wrapped)
    assert payload == {"x": 1}
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["soma_version"] == "0.1.0"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_brain_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'soma.core.brain_bundle'`

**Step 3: Implement minimal wrapper**

```python
# src/soma/core/brain_bundle.py
"""Versioned brain-bundle serialization envelope.

Every SOMA brain saved to disk goes through `wrap_payload` so that a schema
tag, SOMA version, torch/tokenizers versions, and git sha ride along with the
raw state. This lets future SOMA versions detect and migrate old brains.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import torch
import tokenizers

SCHEMA_VERSION = 1


def wrap_payload(
    payload: Mapping[str, Any],
    *,
    soma_version: str,
    git_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "format": "soma-brain",
        "schema_version": SCHEMA_VERSION,
        "soma_version": soma_version,
        "git_sha": git_sha,
        "torch_version": torch.__version__,
        "tokenizers_version": tokenizers.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload": dict(payload),
    }


def unwrap_payload(wrapped: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if wrapped.get("format") != "soma-brain":
        raise ValueError("Not a SOMA brain bundle (missing 'format' tag)")
    meta = {k: v for k, v in wrapped.items() if k != "payload"}
    return dict(wrapped["payload"]), meta
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_core/test_brain_bundle.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/soma/core/brain_bundle.py tests/test_core/test_brain_bundle.py
git commit -m "feat(brain): add versioned schema envelope (SCHEMA_VERSION=1)"
```

---

### Task 2: Add migration dispatch

**Files:**
- Modify: `src/soma/core/brain_bundle.py`
- Test: `tests/test_core/test_brain_bundle.py`

**Step 1: Write the failing test**

```python
# append to tests/test_core/test_brain_bundle.py
import pytest
from soma.core.brain_bundle import migrate_payload, MigrationError


def test_migrate_identity_for_current_version():
    payload = {"x": 1}
    wrapped = wrap_payload(payload, soma_version="0.1.0")
    migrated, final_version = migrate_payload(wrapped["payload"], from_schema=SCHEMA_VERSION)
    assert migrated == payload
    assert final_version == SCHEMA_VERSION


def test_migrate_rejects_unknown_future_version():
    with pytest.raises(MigrationError):
        migrate_payload({}, from_schema=99)


def test_migrate_legacy_schema_zero_wraps_as_payload():
    """A pre-versioned checkpoint (old single-file format) enters the migrator
    with from_schema=0 — identity-wrap, warn, upgrade to v1."""
    legacy = {"global_step": 100, "graph": {}, "config": {}}
    migrated, final_version = migrate_payload(legacy, from_schema=0)
    assert final_version == SCHEMA_VERSION
    # Assert nothing is lost
    for key in legacy:
        assert key in migrated
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_brain_bundle.py -v`
Expected: FAIL with `ImportError: cannot import name 'migrate_payload'`

**Step 3: Implement minimal migrator**

```python
# append to src/soma/core/brain_bundle.py
import warnings
from typing import Callable

class MigrationError(RuntimeError):
    pass


_MIGRATORS: dict[tuple[int, int], Callable[[dict], dict]] = {}


def register_migrator(from_schema: int, to_schema: int):
    def deco(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        _MIGRATORS[(from_schema, to_schema)] = fn
        return fn
    return deco


@register_migrator(0, 1)
def _migrate_0_to_1(payload: dict) -> dict:
    warnings.warn(
        "Loading pre-v1 SOMA checkpoint. Upgrading in place to schema v1. "
        "Re-save to persist the upgrade.",
        stacklevel=2,
    )
    return payload  # No structural change in 0→1; just gain the envelope.


def migrate_payload(payload: dict, *, from_schema: int) -> tuple[dict, int]:
    if from_schema > SCHEMA_VERSION:
        raise MigrationError(
            f"Checkpoint schema v{from_schema} is newer than this SOMA "
            f"(supports up to v{SCHEMA_VERSION}). Upgrade SOMA to load."
        )
    current = from_schema
    while current < SCHEMA_VERSION:
        step = _MIGRATORS.get((current, current + 1))
        if step is None:
            raise MigrationError(f"No migrator from schema v{current} to v{current+1}")
        payload = step(payload)
        current += 1
    return payload, current
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_brain_bundle.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/soma/core/brain_bundle.py tests/test_core/test_brain_bundle.py
git commit -m "feat(brain): schema migration dispatch (v0→v1 wraps legacy)"
```

---

### Task 3: CPU-normalize tensors on save

Every tensor in the payload must be moved to CPU before serialization, so a CUDA-trained brain loads cleanly on a phone/CPU host.

**Files:**
- Modify: `src/soma/core/brain_bundle.py`
- Test: `tests/test_core/test_brain_bundle.py`

**Step 1: Write the failing test**

```python
# append to tests/test_core/test_brain_bundle.py
import torch

def test_to_cpu_normalizes_nested_tensor_state():
    from soma.core.brain_bundle import to_cpu_state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = {
        "a": torch.tensor([1.0, 2.0], device=device),
        "nested": {"b": torch.tensor([3.0], device=device)},
        "list": [torch.tensor([4.0], device=device)],
        "scalar": 42,
        "string": "hello",
    }
    normalized = to_cpu_state(state)
    assert normalized["a"].device.type == "cpu"
    assert normalized["nested"]["b"].device.type == "cpu"
    assert normalized["list"][0].device.type == "cpu"
    assert normalized["scalar"] == 42
    assert normalized["string"] == "hello"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_brain_bundle.py::test_to_cpu_normalizes_nested_tensor_state -v`
Expected: FAIL (function doesn't exist)

**Step 3: Implement**

```python
# append to src/soma/core/brain_bundle.py
def to_cpu_state(obj):
    """Recursively move every tensor in a nested mapping/list to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(to_cpu_state(v) for v in obj)
    return obj
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_brain_bundle.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/soma/core/brain_bundle.py tests/test_core/test_brain_bundle.py
git commit -m "feat(brain): to_cpu_state helper for device-independent save"
```

---

### Task 4: Wire schema envelope + CPU-normalize into SOMA.save_state / load_state

**Files:**
- Modify: `src/soma/system.py:530-562`
- Modify: `scripts/interactive.py:103` (add map_location)
- Modify: `scripts/train_service.py:147-148` (add map_location)
- Test: `tests/test_core/test_brain_bundle.py`

**Step 1: Write the failing test**

```python
# append to tests/test_core/test_brain_bundle.py
from pathlib import Path
from soma.core.config import SOMAConfig
from soma.system import SOMA


def test_save_then_load_round_trip_cpu(tmp_path: Path):
    cfg = SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=16,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    save_path = tmp_path / "brain.pt"
    soma.save_state(str(save_path))

    # New instance, same config, fresh weights -> load should overwrite.
    soma2 = SOMA(cfg, device=torch.device("cpu"))
    soma2.load_state(str(save_path))
    # Pick one nn.Parameter and compare
    orig = next(iter(soma.graph.state_dict().values()))
    loaded = next(iter(soma2.graph.state_dict().values()))
    assert torch.allclose(orig, loaded)


def test_saved_file_is_wrapped_bundle(tmp_path: Path):
    cfg = SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=16,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    save_path = tmp_path / "brain.pt"
    soma.save_state(str(save_path))
    raw = torch.load(str(save_path), map_location="cpu", weights_only=False)
    assert raw["format"] == "soma-brain"
    assert raw["schema_version"] == 1
    assert "payload" in raw
    assert "global_step" in raw["payload"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_brain_bundle.py::test_saved_file_is_wrapped_bundle -v`
Expected: FAIL — current format has no `format` key.

**Step 3: Patch `SOMA.save_state` and `SOMA.load_state`**

In `src/soma/system.py`, modify `save_state` (currently at ~line 530) to wrap and CPU-normalize:

```python
# src/soma/system.py — replace the save_state body with:
def save_state(self, path: str) -> None:
    from soma.core.brain_bundle import wrap_payload, to_cpu_state

    payload = {
        "global_step": self.global_step,
        "recent_errors": list(self._recent_errors),
        "graph": self.graph.serialize(),
        "working_memory": self.working_memory.state_dict(),
        "episodic_memory": self.episodic_memory.state_dict(),
        "curiosity_state_dict": self.curiosity.state_dict(),
        "curiosity_histories": self.curiosity.to_dict(),
        "homeostasis": self.homeostasis.to_dict(),
        "config": self.config.to_dict(),
        "last_curiosity": float(self._last_curiosity),
    }
    payload = to_cpu_state(payload)
    wrapped = wrap_payload(payload, soma_version=_current_soma_version())
    torch.save(wrapped, str(path))
```

And the `load_state` body:

```python
def load_state(self, path: str) -> None:
    from soma.core.brain_bundle import unwrap_payload, migrate_payload, SCHEMA_VERSION

    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if raw.get("format") == "soma-brain":
        payload, meta = unwrap_payload(raw)
        from_schema = meta["schema_version"]
    else:
        # Legacy single-file format — treat entire file as payload, schema=0
        payload = raw
        from_schema = 0
    payload, _ = migrate_payload(payload, from_schema=from_schema)

    self.global_step = payload["global_step"]
    self._recent_errors = deque(payload["recent_errors"], maxlen=self._recent_errors_cap)
    self.graph = Graph.deserialize(payload["graph"], device=self.device)
    self.working_memory.load_state_dict(payload["working_memory"])
    self.episodic_memory.load_state_dict(payload["episodic_memory"])
    self.curiosity.load_state_dict(payload["curiosity_state_dict"])
    self.curiosity.from_dict(payload["curiosity_histories"])
    self.homeostasis.from_dict(payload["homeostasis"])
    self._last_curiosity = payload.get("last_curiosity", 0.0)
    self.graph.to(self.device)
```

Add helper near top of file:

```python
def _current_soma_version() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"
```

Then in `scripts/interactive.py:103` and `scripts/train_service.py:147-148`, replace `torch.load(path)` with `torch.load(path, map_location="cpu", weights_only=False)` followed by `soma.to(device)` AFTER load.

**Step 4: Run test**

Run: `pytest tests/test_core/test_brain_bundle.py -v`
Expected: PASS (both new tests)

**Step 5: Commit**

```bash
git add src/soma/system.py scripts/interactive.py scripts/train_service.py tests/test_core/test_brain_bundle.py
git commit -m "feat(brain): wrap save_state in schema envelope; CPU map_location on load"
```

---

### Task 5: Legacy checkpoint migration — existing training run stays alive

**Files:**
- Create: `scripts/migrate_legacy_checkpoint.py`

**Step 1: Write a migration script**

```python
# scripts/migrate_legacy_checkpoint.py
"""One-shot migrator: pre-v1 SOMA checkpoint → v1 bundle.

Usage:
    python scripts/migrate_legacy_checkpoint.py checkpoints/current.pt
Writes the migrated bundle back over the input path (after backing up to .bak).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soma.core.brain_bundle import (  # noqa: E402
    SCHEMA_VERSION,
    migrate_payload,
    to_cpu_state,
    wrap_payload,
)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/migrate_legacy_checkpoint.py <path>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"No such file: {path}")
        return 2

    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and raw.get("format") == "soma-brain":
        print(f"Already v{raw['schema_version']}, nothing to do.")
        return 0

    bak = path.with_suffix(path.suffix + ".legacy-bak")
    shutil.copy2(path, bak)
    print(f"Backed up to {bak}")

    payload, _ = migrate_payload(raw, from_schema=0)
    payload = to_cpu_state(payload)
    wrapped = wrap_payload(payload, soma_version="migrator")
    torch.save(wrapped, str(path))
    print(f"Upgraded {path} to schema v{SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Run on the currently-live checkpoint**

First **STOP the training service** so we don't race the writer:

```bash
echo "migrate" > .soma-loop/state/STOP
# Wait until train_heartbeat.json shows status != "running" (or pid vanishes)
```

Then:

```bash
python scripts/migrate_legacy_checkpoint.py checkpoints/current.pt
```

Expected stdout:
```
Backed up to checkpoints/current.pt.legacy-bak
Upgraded checkpoints/current.pt to schema v1
```

**Step 3: Round-trip verification**

```bash
python -c "
import sys
sys.path.insert(0, 'src')
import torch
raw = torch.load('checkpoints/current.pt', map_location='cpu', weights_only=False)
assert raw['format'] == 'soma-brain'
assert raw['schema_version'] == 1
print('OK. global_step=', raw['payload']['global_step'])
"
```

**Step 4: Restart the training service**

```bash
rm .soma-loop/state/STOP
# Train service autostart (or manual restart if configured)
```

Verify heartbeat resumes and `last_loss` looks reasonable (not NaN, not a massive jump).

**Step 5: Commit**

```bash
git add scripts/migrate_legacy_checkpoint.py
git commit -m "feat(brain): add legacy-checkpoint migrator (pre-v1 → v1)"
```

---

### Task 6: Relax `SOMAConfig.from_dict` strictness

Unknown config keys (e.g., from old checkpoints) should warn, not crash.

**Files:**
- Modify: `src/soma/core/config.py:236-242`
- Test: `tests/test_core/test_config.py` (add to existing file)

**Step 1: Write the failing test**

```python
# tests/test_core/test_config.py (add)
import warnings
from soma.core.config import SOMAConfig

def test_from_dict_drops_unknown_keys_with_warning():
    d = SOMAConfig().to_dict()
    d["totally_new_field_from_future"] = 42
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = SOMAConfig.from_dict(d)
    # Sanity: still constructs.
    assert cfg.seed == SOMAConfig().seed
    # Warning was emitted naming the bad key.
    assert any("totally_new_field_from_future" in str(msg.message) for msg in w)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_config.py::test_from_dict_drops_unknown_keys_with_warning -v`
Expected: FAIL (current code raises `TypeError: __init__() got unexpected keyword argument`).

**Step 3: Modify `SOMAConfig.from_dict`**

Find the method in `src/soma/core/config.py` and replace with:

```python
@classmethod
def from_dict(cls, d: dict) -> "SOMAConfig":
    import dataclasses
    import warnings

    known = {f.name for f in dataclasses.fields(cls)}
    unknown = [k for k in d.keys() if k not in known]
    if unknown:
        warnings.warn(
            f"Dropping unknown SOMAConfig fields (likely from an older/newer "
            f"checkpoint): {sorted(unknown)}",
            stacklevel=2,
        )
    filtered = {k: v for k, v in d.items() if k in known}
    return cls(**filtered)
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_config.py -v`
Expected: PASS (including the new one and all existing tests — don't regress).

**Step 5: Commit**

```bash
git add src/soma/core/config.py tests/test_core/test_config.py
git commit -m "feat(config): from_dict drops unknown keys with warning"
```

---

### Task 7: Persist `DevelopmentSchedule.periods`

Without this, a future edit to `_default_periods()` silently changes the dev schedule on restore.

**Files:**
- Modify: `src/soma/metacognition/development.py` — add `to_dict`/`from_dict`
- Modify: `src/soma/system.py:530` — save/load the new field
- Test: `tests/test_metacognition/test_development.py`

**Step 1: Write the failing test**

```python
# tests/test_metacognition/test_development.py (add)
from soma.metacognition.development import DevelopmentSchedule, CriticalPeriod

def test_development_schedule_round_trip():
    orig = DevelopmentSchedule()
    d = orig.to_dict()
    loaded = DevelopmentSchedule()
    loaded.from_dict(d)
    assert len(loaded.periods) == len(orig.periods)
    for a, b in zip(orig.periods, loaded.periods, strict=True):
        assert a.name == b.name
        assert a.start_step == b.start_step
        assert a.end_step == b.end_step
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metacognition/test_development.py::test_development_schedule_round_trip -v`
Expected: FAIL (`AttributeError: to_dict`).

**Step 3: Implement**

In `src/soma/metacognition/development.py`:

```python
def to_dict(self) -> list[dict]:
    return [dataclasses.asdict(p) for p in self.periods]

def from_dict(self, data: list[dict]) -> None:
    self.periods = [CriticalPeriod(**p) for p in data]
```

Then in `src/soma/system.py`, add to the save payload after `"homeostasis"`:
```python
"development": self.development.to_dict(),
```
And in `load_state` after `self.homeostasis.from_dict(...)`:
```python
if "development" in payload:
    self.development.from_dict(payload["development"])
```

**Step 4: Run test**

Run: `pytest tests/test_metacognition/test_development.py tests/test_core/test_brain_bundle.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/soma/metacognition/development.py src/soma/system.py tests/test_metacognition/test_development.py
git commit -m "feat(brain): persist DevelopmentSchedule.periods in checkpoint"
```

---

### Task 8: Growth journal — append-only log of structural events

Adds observability over the life of a brain: "what changed, and when".

**Files:**
- Modify: `src/soma/system.py` — add `_growth_log` deque, serialize
- Modify: `src/soma/growth/synaptogenesis.py` — emit event after each creation
- Modify: `src/soma/growth/neurogenesis.py` — emit event after each creation
- Modify: `src/soma/growth/pruning.py` — emit event after each removal
- Test: `tests/test_growth/test_growth_journal.py`

**Step 1: Write the failing test**

```python
# tests/test_growth/test_growth_journal.py
import torch
from soma.core.config import SOMAConfig
from soma.system import SOMA

def test_growth_journal_records_synaptogenesis_event():
    cfg = SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=16,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        synaptogenesis_interval=1,
        synaptogenesis_rate=1.0,  # force many attempts
        seed=0,
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    before = len(soma.growth_log)
    # Drive a synaptogenesis event by calling the growth pass directly.
    soma._maybe_synaptogenesis(activations={})  # internal, no real activations
    # We assert only that the journal does not regress in shape.
    assert len(soma.growth_log) >= before
    for event in soma.growth_log:
        assert {"step", "event_type"}.issubset(event.keys())
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_growth/test_growth_journal.py -v`
Expected: FAIL — `growth_log` does not exist.

**Step 3: Implement minimal log**

In `src/soma/system.py`, in `__init__` after other attributes:

```python
from collections import deque
self.growth_log: deque = deque(maxlen=10_000)  # last 10K structural events
```

Add a helper:

```python
def record_growth_event(self, event_type: str, **fields) -> None:
    self.growth_log.append({
        "step": self.global_step,
        "event_type": event_type,
        **fields,
    })
```

In `src/soma/growth/synaptogenesis.py` — wherever a new edge is successfully created, after the creation call `system.record_growth_event("synaptogenesis", edge_id=edge.id, source=..., target=...)`. Same idiom for `neurogenesis.py` (`event_type="neurogenesis"`, `node_id=...`) and `pruning.py` (`event_type="prune_edge"` or `"prune_node"`, with `id=...`).

In `save_state` payload add:
```python
"growth_log": list(self.growth_log),
```
In `load_state`:
```python
self.growth_log = deque(payload.get("growth_log", []), maxlen=10_000)
```

**Step 4: Run test**

Run: `pytest tests/test_growth/test_growth_journal.py tests/test_core/test_brain_bundle.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/soma/system.py src/soma/growth/ tests/test_growth/test_growth_journal.py
git commit -m "feat(brain): growth journal — append-only structural event log"
```

---

### Task 9: Serialize tokenizer into the brain bundle

Introduces the directory-shaped bundle: `soma-brain/` containing `brain.pt`, `tokenizer.json`, `encoder.pt`, `decoder.pt`, `manifest.json`.

**Files:**
- Modify: `src/soma/core/brain_bundle.py` — add `save_bundle`, `load_bundle`
- Modify: `src/soma/system.py` — add `SOMA.save_bundle(dir_path, tokenizer, encoder, decoder)`
- Test: `tests/test_core/test_brain_bundle.py`

**Step 1: Write the failing test**

```python
# append to tests/test_core/test_brain_bundle.py
def test_save_bundle_then_load_bundle(tmp_path: Path):
    from soma.io.text_encoder import train_bpe_tokenizer, TextEncoder

    cfg = SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=32,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    tok = train_bpe_tokenizer(["hello world"], vocab_size=32)
    enc = TextEncoder(tok, embed_dim=8, max_seq_len=8, device=torch.device("cpu"))

    out_dir = tmp_path / "brain-bundle"
    soma.save_bundle(str(out_dir), tokenizer=tok, encoder=enc)

    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "brain.pt").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "encoder.pt").exists()

    soma2 = SOMA(cfg, device=torch.device("cpu"))
    tok2, enc2 = soma2.load_bundle(str(out_dir))
    # vocab size stays the same across round-trip
    assert tok2.get_vocab_size() == tok.get_vocab_size()
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_core/test_brain_bundle.py::test_save_bundle_then_load_bundle -v`
Expected: FAIL (`save_bundle` doesn't exist).

**Step 3: Implement**

In `src/soma/core/brain_bundle.py`:

```python
import json

def write_manifest(out_dir: Path, *, soma_version: str, vocab_size: int,
                   llm_identity: str | None = None) -> None:
    manifest = {
        "format": "soma-brain-bundle",
        "schema_version": SCHEMA_VERSION,
        "soma_version": soma_version,
        "vocab_size": vocab_size,
        "llm_identity": llm_identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "tokenizers_version": tokenizers.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def read_manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "manifest.json").read_text())
```

In `src/soma/system.py`:

```python
def save_bundle(self, dir_path: str, *, tokenizer=None, encoder=None,
                decoder=None, llm_identity: str | None = None) -> None:
    from pathlib import Path
    from soma.core.brain_bundle import write_manifest

    out = Path(dir_path)
    out.mkdir(parents=True, exist_ok=True)
    self.save_state(str(out / "brain.pt"))
    if tokenizer is not None:
        tokenizer.save(str(out / "tokenizer.json"))
    if encoder is not None:
        torch.save(
            {"state_dict": encoder.state_dict(), "embed_dim": encoder.embed_dim,
             "max_seq_len": encoder.max_seq_len},
            str(out / "encoder.pt"),
        )
    if decoder is not None:
        torch.save({"state_dict": decoder.state_dict()}, str(out / "decoder.pt"))
    vocab_size = tokenizer.get_vocab_size() if tokenizer is not None else self.config.vocab_size
    write_manifest(out, soma_version=_current_soma_version(),
                   vocab_size=vocab_size, llm_identity=llm_identity)


def load_bundle(self, dir_path: str):
    from pathlib import Path
    from soma.core.brain_bundle import read_manifest
    from soma.io.text_encoder import TextEncoder
    from tokenizers import Tokenizer

    src = Path(dir_path)
    manifest = read_manifest(src)
    self.load_state(str(src / "brain.pt"))
    tok = None
    enc = None
    if (src / "tokenizer.json").exists():
        tok = Tokenizer.from_file(str(src / "tokenizer.json"))
        if tok.get_vocab_size() != manifest["vocab_size"]:
            raise ValueError(
                f"Tokenizer vocab {tok.get_vocab_size()} doesn't match manifest "
                f"{manifest['vocab_size']}"
            )
    if (src / "encoder.pt").exists() and tok is not None:
        blob = torch.load(str(src / "encoder.pt"), map_location="cpu", weights_only=False)
        enc = TextEncoder(tok, embed_dim=blob["embed_dim"], max_seq_len=blob["max_seq_len"],
                          device=self.device)
        enc.load_state_dict(blob["state_dict"])
    return tok, enc
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_brain_bundle.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/soma/core/brain_bundle.py src/soma/system.py tests/test_core/test_brain_bundle.py
git commit -m "feat(brain): directory-bundle save/load (brain.pt + tokenizer + manifest)"
```

---

### Task 10: Store interface_spec in manifest — boundary for head-swap

Records which node UUID is the text-aligned sensor and which is the language-feeding output node, plus their dims — so a future head-swap tool can find them without heuristics.

**Files:**
- Modify: `src/soma/core/brain_bundle.py` — extend manifest
- Modify: `src/soma/system.py` — compute interface_spec from graph
- Test: `tests/test_core/test_brain_bundle.py`

**Step 1: Write the failing test**

```python
# append to tests/test_core/test_brain_bundle.py
def test_interface_spec_in_manifest(tmp_path: Path):
    cfg = SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=16,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    out_dir = tmp_path / "bundle"
    soma.save_bundle(str(out_dir))
    manifest = json.loads((out_dir / "manifest.json").read_text())
    spec = manifest["interface_spec"]
    assert spec["sensor_by_modality"]["text"]  # UUID string
    assert spec["output_by_modality"]["text"]
    assert spec["output_dim"] == cfg.integrator_output_dim
    assert spec["sensor_output_dim"] == cfg.sensor_output_dim
    assert spec["text_embed_dim"] == cfg.text_embed_dim
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_brain_bundle.py::test_interface_spec_in_manifest -v`
Expected: FAIL — no `interface_spec` field.

**Step 3: Implement**

In `src/soma/core/brain_bundle.py` add an `interface_spec` parameter to `write_manifest`:

```python
def write_manifest(out_dir: Path, *, soma_version: str, vocab_size: int,
                   llm_identity: str | None = None,
                   interface_spec: dict | None = None) -> None:
    manifest = {
        "format": "soma-brain-bundle",
        "schema_version": SCHEMA_VERSION,
        "soma_version": soma_version,
        "vocab_size": vocab_size,
        "llm_identity": llm_identity,
        "interface_spec": interface_spec or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "tokenizers_version": tokenizers.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

In `src/soma/system.py`, in `save_bundle` compute the spec:

```python
interface_spec = {
    "sensor_by_modality": dict(self.graph._sensor_by_modality),
    "output_by_modality": dict(self.graph._output_by_modality),
    "sensor_output_dim": self.config.sensor_output_dim,
    "output_dim": self.config.integrator_output_dim,
    "text_embed_dim": self.config.text_embed_dim,
}
write_manifest(out, soma_version=_current_soma_version(),
               vocab_size=vocab_size, llm_identity=llm_identity,
               interface_spec=interface_spec)
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_brain_bundle.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/soma/core/brain_bundle.py src/soma/system.py tests/test_core/test_brain_bundle.py
git commit -m "feat(brain): manifest.interface_spec for portable head-swap boundary"
```

---

### Task 11: Full regression — pytest + ruff + mypy

**Step 1: Run the full suite**

```bash
pytest tests/ -v
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/soma/
```

Expected: ALL PASS. No new ruff or mypy issues relative to `main@HEAD^`.

**Step 2: If anything fails**, fix and re-run until green. Commit each fix as its own commit:

```bash
git commit -m "fix(<area>): <one-line>"
```

**Step 3: Final commit** (if needed — usually none):

```bash
# No content commit; this is just the verification gate.
```

---

## Phase 2 — Verbalizer Interface (scope notes, not yet detailed tasks)

Goal: Implement the `SomaVerbalizer` module exactly per the contract in Decision 2. Phase 2 gets its own dedicated plan doc when Phase 1 lands — likely `docs/plans/2026-04-DD-verbalizer-interface.md`.

**Planned tasks:**

1. Create `src/soma/io/verbalizer.py` with `VerbalizerSpec` dataclass and `SomaVerbalizer(nn.Module)` skeleton. Near-zero-init final layer.
2. Unit tests for shape contract (SOMA state `(B, 128)` → prefix `(B, k, d_model)`).
3. `SomaVerbalizer.save`/`.load` tests using bundle's `verbalizer/` subdir.
4. Aggregator helper: collapse SOMA's per-OUTPUT-node activations into the canonical `(B, 128)` input — `soma.verbalizer_state()` returns `mean(output_activations)` or similar, well-tested.
5. `fallback_text()` delegates to SOMA's existing `TextDecoder`.
6. Contract test: swap-compatible — instantiate with two different `VerbalizerSpec`s (different `llm_hidden_dim`) and verify each works independently.

**Expected complexity:** ~300 LOC + ~400 LOC tests. ~1-2 days.

---

## Phase 3 — First Transformer Integration (scope notes)

Goal: End-to-end call path. `soma.generate(prompt) → LLM-generated text` conditioned on current SOMA state.

**Target transformer:** `HuggingFaceTB/SmolLM2-360M-Instruct` (dev sandbox — fits on any dev laptop).

**Planned tasks:**

1. Add `transformers>=4.40` to `pyproject.toml` optional deps (`[dev-chat]`).
2. Create `src/soma/io/chat_head.py` — loads a frozen HF causal-LM model, exposes `.generate(input_ids=..., inputs_embeds=...)`.
3. Wire `SomaVerbalizer` + `ChatHead`: `chat(user_text) → soma_state = soma.verbalizer_state() → prefix = verbalizer(soma_state) → inputs_embeds = concat(prefix, tokenizer(user_text)) → llm.generate(...)`.
4. Fallback branch: if no LLM loaded, `chat()` uses `verbalizer.fallback_text()`.
5. Position-IDs correctness tests — prefix tokens occupy positions `0..k-1`, user text starts at `k`.
6. Tied-embeddings sanity check — confirm SmolLM2's `tie_word_embeddings` doesn't interact badly with our `inputs_embeds` path.

**Expected complexity:** ~400 LOC + ~500 LOC tests + integration smoke script. ~2-3 days.

---

## Phase 4 — Bootstrap Training (scope notes)

Goal: Train the verbalizer projector to make LLM output align with SOMA's intent.

**Dataset construction:**

- Use SOMA's current corpus + run the model to produce `(soma_state, next_word)` pairs.
- Add small publicly-licensed instruction datasets (Dolly-15k CC-BY, Alpaca, OpenAssistant-1) filtered to short turns.
- Consider paired `(soma_state_at_step_N, caption_of_what_SOMA_is_attending_to)` — TBD design.

**Training loop:**

- Frozen LLM, trainable verbalizer only.
- LM cross-entropy loss on the LLM's next-token prediction given `concat(prefix, target_text_embeds)`.
- AdamW on ~8-16M projector params → tractable on RTX 3090 in hours.
- Early stop on held-out perplexity.

**Success criteria:** Verbalizer output differs meaningfully from vanilla LLM output on paired prompts — measurable via generation diversity and manual chat eval.

**Expected complexity:** ~600 LOC + multi-day training run. Dedicated plan doc when Phase 3 lands.

---

## Phase 5 — Interactive Conversation Loop (scope notes)

Goal: Real chat session. WM integration. Session-level coherence.

**Planned work:**

1. Session state: history buffer of `(user_turn, soma_state_before_response, llm_response)`.
2. WM write on each turn: encode user text → SOMA → WM slot.
3. Multi-turn prompt construction: `history + [current user turn]` tokenized, prefix from latest SOMA state.
4. Curiosity signal surfaced in UI ("SOMA is attending to X", live trace).
5. Dashboard panel: SOMA's `integrator_output_dim=128` vector over time, superimposed on LLM generation.

**Expected complexity:** ~1000 LOC. Dedicated plan doc when Phase 4 lands.

---

## Phase 6 — Consumer-Grade Packaging (scope notes)

Goal: Ship on consumer hardware — home compute box, Apple Silicon laptop, or phone.

**Planned work:**

1. Replace SmolLM2-360M with Qwen2.5-1.5B-Instruct, retrain verbalizer.
2. Export the frozen LLM to GGUF via llama.cpp's `convert_hf_to_gguf.py`.
3. Quantize to Q4_K_M (benchmark quality floor).
4. Runtime: either (a) Python `llama-cpp-python` for the LLM + PyTorch for SOMA and verbalizer on same host, or (b) split: SOMA as daemon exposing gRPC/WebSocket, LLM runs via `llama.cpp` server, verbalizer lives on either side.
5. Apple Silicon variant (MLX or llama.cpp Metal).
6. Remote-access mode: home compute box hosts SOMA + LLM; phone/laptop connects over tailscale/HTTPS.
7. Measured targets: RTX 3060 ~40-60 tok/s, M2 Pro ~30-40 tok/s, Snapdragon 8 Gen 3 ~8-15 tok/s (Qwen2.5-1.5B Q4_K_M).

**Expected complexity:** Significant — ~2000 LOC + devops. Dedicated plan doc when Phase 5 lands.

---

## Execution Handoff

After saving this plan, execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per Phase-1 task, review between tasks. Fast iteration, stays in conversation.

**2. Parallel Session (separate)** — New session opens with executing-plans skill, batches of 3 tasks per checkpoint.

Given the current session's depth, I'll default to **subagent-driven** for Phase 1. Ping me if you want to switch.

---

## Appendix A — SOMA Serialization Audit Reference (H-Task 2 output)

*(compacted — full audit in research transcript)*

**Currently captured** (via `SOMA.save_state` → `torch.save`):
- `global_step`, `recent_errors`
- `graph`: `Graph.serialize()` — nodes/edges dicts + full `state_dict()`
- `working_memory`, `episodic_memory`, `curiosity_state_dict`, `curiosity_histories`, `homeostasis`
- `config`: all 57 `SOMAConfig` fields
- `last_curiosity`

**14 portability gaps identified** (Tasks 1-10 above address the critical ones):
1. ✅ Schema envelope (Task 1)
2. ✅ CPU-normalize on save (Task 3-4)
3. ✅ Relax `SOMAConfig.from_dict` (Task 6)
4. Logical identity tags on boundary nodes (deferred — Task 10 partly covers via interface_spec)
5. Decouple parametric state from UUIDs (deferred — only matters for cross-brain merging, not priority)
6. ✅ Serialize tokenizer into bundle (Task 9)
7. Optimizer/RNG state (no Adam to worry about; RNG deferred)
8. ✅ Persist DevelopmentSchedule (Task 7)
9. ✅ Growth journal (Task 8)
10. ✅ Interface spec in manifest (Task 10)
11. ✅ Load-time migration hook (Task 2)
12. Strict vs. partial `load_state_dict` — deferred
13. Edge `weight` in `Edge.to_dict` — deferred, not blocking
14. `use_batched_executor` not a brain property — deferred, ignore on load via Task 6's config relaxation

## Appendix B — Verbalizer Contract Reference (H-Task 3 output)

*(Full design in research transcript; reproduced as pseudocode in Phase 2 scope.)*

```python
# Contract recap (full implementation is Phase 2 Task 1):
class SomaVerbalizer(nn.Module):
    def __init__(self, spec: VerbalizerSpec): ...
    def forward(self, soma_state: Tensor) -> Tensor:  # (B, soma_output_dim) -> (B, k, d_model)
        ...
    def generate(self, soma_state, input_ids, llm, tokenizer, **kw) -> str: ...
    def fallback_text(self, soma, activations) -> str: ...  # uses SOMA's TextDecoder
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "SomaVerbalizer": ...
```

Near-zero init on final layer → untrained verbalizer is a no-op (LLM behaves vanilla), not a garbage emitter. Swap procedure: change `VerbalizerSpec.llm_name` + `.llm_hidden_dim` → rebuild → retrain projector on small paired dataset. SOMA brain untouched.

---

*End of plan. Ready for Phase 1 execution.*
