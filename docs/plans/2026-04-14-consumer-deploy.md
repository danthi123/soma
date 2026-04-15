# Phase 7 — Consumer-Hardware Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Make SOMA+verbalizer+chat actually deployable on operator-class consumer GPUs — **RTX 3060+ (8GB VRAM floor), RTX 3090/4090 as the sweet spot**. Auto-detect VRAM, select an LLM tier that fits, load in fp16 on CUDA (fp32 on CPU fallback), and validate end-to-end on real hardware with a larger/better LLM than the SmolLM2-360M dev sandbox.

**Architecture:** A new `soma.deploy` module exposes `detect_cuda_vram()`, `select_device_and_dtype()`, `auto_select_tier()`, plus a `MODEL_TIERS` registry mapping tier names ("tiny"/"small"/"large") to HF model identifiers + hidden dims + VRAM hints. A `build_chat_head(tier, device, dtype)` factory downloads/loads the model into the right device/dtype and wraps in ChatHead. `SOMA.chat()` / `ChatSession.respond()` cast the verbalizer prefix to match the LLM's dtype before concat so fp16 LLMs don't silently double memory via fp32 promotion. CLI scripts gain `--tier auto` selection. A new `scripts/demo_chat.py` runs the full "fresh brain → short bootstrap → interactive chat" flow with zero manual dtype management.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace transformers, existing SOMA infra. **No GGUF / llama-cpp** — our verbalizer training needs gradients through the LLM's input embeddings, which GGUF's runtime doesn't cleanly expose.

**Note on `.train(False)`:** Same hook-avoidance pattern as earlier phases.

---

## Why Phase 7 Now

Phases 1-6 delivered a full chat+learn pipeline but only validated against SmolLM2-360M on CPU. That's the dev sandbox — real operators with GPUs want bigger, better models. A 1.5B-parameter instruct-tuned LLM is ~2-3× better at open-domain chat than a 360M one, and still fits comfortably in 8GB of VRAM in fp16. The operator's 3090 can host a 7B model easily. Phase 7 removes the artificial CPU cap and lets the existing code scale up.

Deliberately scoped:

- **GPU-first, CPU-viable**. CPU stays as a fallback (SmolLM2-360M runs fine there), but the default recommendation is GPU+fp16 with a Qwen2.5-class model.
- **Three tiers, not a continuum**. Tiny/Small/Large — covers phone-class (hypothetical), 8GB-GPU, and 16GB+-GPU. A future phase can add medium/xlarge if needed.
- **No quantization this phase**. Bitsandbytes int4 for stretched 8GB-on-7B configs is a worthwhile addition but adds a dependency cliff; keep it out of scope here.
- **No concurrent GPU use**. The SOMA training service is on the GPU; Phase 7's CUDA smoke tests are marked `slow` AND assume the operator is OK with 3GB extra VRAM used during the test (Qwen2.5-1.5B fp16 ≈ 3GB; comfortably fits alongside training).
- **No multi-GPU**. Single-device only.

The value in isolation: by the end of Phase 7, a new operator with an RTX 3060 can clone the repo, run one command, and get a working brain-backed chat UX — with a meaningfully better LLM than SmolLM2-360M.

---

## Key Decisions

### 1. Three model tiers
```python
MODEL_TIERS = {
    "tiny": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "hidden_dim": 960,
        "min_vram_gb": 0,       # CPU is fine
        "approx_fp16_gb": 0.72,
    },
    "small": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_dim": 1536,
        "min_vram_gb": 4,       # fits comfortably on an 8GB card
        "approx_fp16_gb": 3.1,
    },
    "large": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "hidden_dim": 3584,
        "min_vram_gb": 16,      # 16GB+ VRAM needed for clean fp16
        "approx_fp16_gb": 14.5,
    },
}
```
`hidden_dim` is the VerbalizerSpec's `llm_hidden_dim` — validated against `chat_head.hidden_size` at construction, raises if mismatched (catches wrong tier before silently breaking).

### 2. Auto-select logic
```python
def auto_select_tier(vram_gb: int | None) -> str:
    if vram_gb is None:          # no CUDA
        return "tiny"
    if vram_gb < 4:
        return "tiny"
    if vram_gb < 16:
        return "small"           # 4GB-15GB → Qwen2.5-1.5B
    return "large"               # 16GB+ → Qwen2.5-7B
```
Operator's 24GB 3090 → `"large"`. An 8GB 3060 → `"small"`. A CPU-only laptop → `"tiny"`.

### 3. dtype selection
```python
def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32
```
Fp16 on CPU is actually SLOWER than fp32 for PyTorch matmul (no avx512-fp16 on most consumer CPUs), so we intentionally don't pick it there.

### 4. Verbalizer prefix dtype alignment
The verbalizer trains in fp32 for numerical stability — mixed-precision fine-tuning of an 8M-param module isn't worth the complexity budget. But when the LLM is fp16, concat of (fp32 prefix, fp16 token_embeds) would silently promote everything to fp32 in HF — defeating the memory saving.

Fix: **cast the prefix to the LLM's dtype before concat**. Done in exactly two places:
- `SOMA.chat()` in `src/soma/system.py`
- `ChatSession.respond()` in `src/soma/session/chat_session.py`

One line each: `prefix = prefix.to(dtype=token_embeds.dtype)`. Gradient flow through the cast is fine — `.to(dtype=...)` is autograd-aware.

### 5. Demo script is the "does it work end-to-end" artifact
`scripts/demo_chat.py`:
1. Detect VRAM → auto-select tier.
2. Build a fresh tiny SOMA.
3. Build tokenizer from a few hardcoded seed sentences.
4. Load tier-matched LLM + ChatHead.
5. Build near-zero-init verbalizer.
6. Run a brief bootstrap on a bundled corpus snippet (~5 steps).
7. Print a demo conversation (4 turns).
8. Done in <60s on the operator's 3090.

No CLI args. Just `python scripts/demo_chat.py`. The "hello world" of SOMA-chat.

### 6. Marker conventions
- Tests that need real HF model weights: `@pytest.mark.slow`.
- Tests that need CUDA specifically: `@pytest.mark.cuda`. Add to the pytest marker registry.
- CI excludes both by default (`-m "not slow and not cuda"`); operator can opt into either.

---

## Tasks

| # | Scope | Files |
|---|---|---|
| 1 | `soma.deploy` module: detect_cuda_vram + select_device_and_dtype + auto_select_tier + MODEL_TIERS registry | `src/soma/deploy/__init__.py`, `src/soma/deploy/devices.py`, test |
| 2 | `build_chat_head(tier, device, dtype)` factory | `src/soma/deploy/chat_head_factory.py`, test |
| 3 | Prefix dtype alignment in SOMA.chat and ChatSession.respond | `src/soma/system.py`, `src/soma/session/chat_session.py`, tests |
| 4 | `--tier auto` support in CLI scripts | `scripts/chat_repl.py`, `scripts/train_verbalizer_bootstrap.py` |
| 5 | CUDA smoke (slow+cuda): Qwen2.5-1.5B fp16 chat on GPU, no VRAM leak across turns | `tests/test_deploy/test_chat_head_cuda_smoke.py` |
| 6 | `scripts/demo_chat.py` end-to-end demo | new |
| 7 | Deployment docs + register cuda marker | `docs/deployment.md`, `pyproject.toml` |
| 8 | Full regression + merge | — |

---

## Task 1: `soma.deploy` module

**Files:** Create `src/soma/deploy/__init__.py` (empty), `src/soma/deploy/devices.py`, `tests/test_deploy/__init__.py` (empty), `tests/test_deploy/test_devices.py`.

**Step 1: Failing test.**

```python
# tests/test_deploy/test_devices.py
import pytest
import torch

from soma.deploy.devices import (
    MODEL_TIERS,
    auto_select_tier,
    detect_cuda_vram,
    select_device_and_dtype,
)


def test_model_tiers_has_three_entries():
    assert set(MODEL_TIERS.keys()) == {"tiny", "small", "large"}
    for tier_name, spec in MODEL_TIERS.items():
        assert "name" in spec
        assert "hidden_dim" in spec
        assert "min_vram_gb" in spec
        assert "approx_fp16_gb" in spec


def test_model_tiers_vram_monotonic():
    """Larger tiers should require more VRAM."""
    tiers = ["tiny", "small", "large"]
    vram = [MODEL_TIERS[t]["min_vram_gb"] for t in tiers]
    assert vram == sorted(vram)


def test_detect_cuda_vram_returns_int_or_none():
    result = detect_cuda_vram()
    if torch.cuda.is_available():
        assert isinstance(result, int)
        assert result > 0
    else:
        assert result is None


def test_select_device_and_dtype_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, dtype = select_device_and_dtype()
    assert device.type == "cpu"
    assert dtype == torch.float32


def test_select_device_and_dtype_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device, dtype = select_device_and_dtype()
    assert device.type == "cuda"
    assert dtype == torch.float16


def test_auto_select_tier_no_vram_returns_tiny():
    assert auto_select_tier(vram_gb=None) == "tiny"


def test_auto_select_tier_small_vram_returns_tiny():
    assert auto_select_tier(vram_gb=2) == "tiny"


def test_auto_select_tier_8gb_returns_small():
    assert auto_select_tier(vram_gb=8) == "small"


def test_auto_select_tier_24gb_returns_large():
    assert auto_select_tier(vram_gb=24) == "large"


def test_auto_select_tier_16gb_returns_large():
    """16GB is the large threshold."""
    assert auto_select_tier(vram_gb=16) == "large"


def test_auto_select_tier_15gb_returns_small():
    """Just below large threshold → small."""
    assert auto_select_tier(vram_gb=15) == "small"
```

**Step 2: Implement** `src/soma/deploy/devices.py`:

```python
"""Device + model-tier selection for SOMA consumer deployment.

Detects available CUDA VRAM, picks a sensible (device, dtype) pair, and
maps VRAM tier to an HF model identifier. Three tiers only — tiny (CPU /
low-end GPU), small (8GB-class GPU), large (16GB+).
"""
from __future__ import annotations

from typing import TypedDict

import torch


class ModelTierSpec(TypedDict):
    name: str
    hidden_dim: int
    min_vram_gb: int
    approx_fp16_gb: float


MODEL_TIERS: dict[str, ModelTierSpec] = {
    "tiny": {
        "name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "hidden_dim": 960,
        "min_vram_gb": 0,
        "approx_fp16_gb": 0.72,
    },
    "small": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_dim": 1536,
        "min_vram_gb": 4,
        "approx_fp16_gb": 3.1,
    },
    "large": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "hidden_dim": 3584,
        "min_vram_gb": 16,
        "approx_fp16_gb": 14.5,
    },
}


def detect_cuda_vram() -> int | None:
    """Total VRAM of CUDA device 0, in GB (integer). None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    return int(total_bytes / (1024**3))


def select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    """CUDA+fp16 when available; CPU+fp32 otherwise.

    fp16 on CPU is skipped intentionally — PyTorch's CPU fp16 matmul is
    typically slower than fp32 on consumer CPUs without AVX-512-fp16.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def auto_select_tier(*, vram_gb: int | None) -> str:
    """Pick the best tier that comfortably fits the available VRAM."""
    if vram_gb is None or vram_gb < 4:
        return "tiny"
    if vram_gb < 16:
        return "small"
    return "large"
```

**Step 3: Commit:**
```bash
git add src/soma/deploy/ tests/test_deploy/
git commit -m "feat(deploy): device + dtype + model-tier selection"
```

---

## Task 2: `build_chat_head(tier, device, dtype)` factory

**Files:** Create `src/soma/deploy/chat_head_factory.py`, extend tests.

**Step 1: Failing test.**

```python
# Append to tests/test_deploy/test_devices.py or create test_chat_head_factory.py
import pytest
import torch
from torch import nn

from soma.deploy.chat_head_factory import build_chat_head


class _MockAutoModel:
    """Minimal mock simulating AutoModelForCausalLM.from_pretrained signature."""
    def __init__(self, *, hidden_size: int, vocab_size: int = 32):
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = type("Cfg", (), {"hidden_size": hidden_size, "vocab_size": vocab_size})()
    def get_input_embeddings(self):
        return self.embed
    def parameters(self):
        yield from self.embed.parameters()
        yield from self.lm_head.parameters()
    def to(self, device_or_dtype):
        return self
    def train(self, mode=True):
        return self


class _MockAutoTokenizer:
    def __init__(self):
        self.pad_token_id = 0
    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1,2,3]])}
    def decode(self, ids, skip_special_tokens=True):
        return "hi"


def test_build_chat_head_tier_unknown_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        build_chat_head(
            tier="nonsense", device=torch.device("cpu"), dtype=torch.float32,
        )


def test_build_chat_head_hidden_dim_mismatch_raises(monkeypatch):
    """If the loaded model's hidden_size disagrees with the tier's
    registered hidden_dim, the factory should raise loudly.

    Setup: monkeypatch AutoModelForCausalLM.from_pretrained to return a
    mock whose hidden_size is WRONG for the requested tier.
    """
    from soma.deploy import chat_head_factory

    def fake_from_pretrained(name, **kw):
        # Return a model with WRONG hidden_size.
        return _MockAutoModel(hidden_size=999)

    def fake_tok_from_pretrained(name, **kw):
        return _MockAutoTokenizer()

    monkeypatch.setattr(
        chat_head_factory, "AutoModelForCausalLM",
        type("FakeAuto", (), {"from_pretrained": staticmethod(fake_from_pretrained)}),
    )
    monkeypatch.setattr(
        chat_head_factory, "AutoTokenizer",
        type("FakeTok", (), {"from_pretrained": staticmethod(fake_tok_from_pretrained)}),
    )

    with pytest.raises(ValueError, match="hidden_dim mismatch"):
        build_chat_head(
            tier="small",  # expects hidden_dim=1536, mock returns 999
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_build_chat_head_returns_chathead_on_happy_path(monkeypatch):
    """Happy path: mock returns model with matching hidden_size, factory
    returns a ChatHead with the model frozen."""
    from soma.deploy import chat_head_factory

    def fake_from_pretrained(name, **kw):
        return _MockAutoModel(hidden_size=1536)  # matches "small"

    def fake_tok_from_pretrained(name, **kw):
        return _MockAutoTokenizer()

    monkeypatch.setattr(
        chat_head_factory, "AutoModelForCausalLM",
        type("FakeAuto", (), {"from_pretrained": staticmethod(fake_from_pretrained)}),
    )
    monkeypatch.setattr(
        chat_head_factory, "AutoTokenizer",
        type("FakeTok", (), {"from_pretrained": staticmethod(fake_tok_from_pretrained)}),
    )

    head = build_chat_head(
        tier="small", device=torch.device("cpu"), dtype=torch.float32,
    )
    assert head.hidden_size == 1536
    # Model should be frozen (Phase 3 invariant).
    assert not any(p.requires_grad for p in head.model.parameters())
```

**Step 2: Implement** `src/soma/deploy/chat_head_factory.py`:

```python
"""Factory for tier-selected ChatHeads with device/dtype handled."""
from __future__ import annotations

from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soma.deploy.devices import MODEL_TIERS
from soma.io.chat_head import ChatHead


def build_chat_head(
    *,
    tier: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool = False,
) -> ChatHead:
    """Download/load the tier-selected HF model and wrap in a ChatHead.

    Moves the model to ``device`` with the requested ``dtype``.
    Validates that the loaded model's hidden_size matches the tier's
    registered hidden_dim — guards against silent misconfiguration
    where the registry drifts from the real model's shape.
    """
    if tier not in MODEL_TIERS:
        raise ValueError(
            f"unknown tier {tier!r}. Valid: {sorted(MODEL_TIERS.keys())}"
        )

    spec = MODEL_TIERS[tier]
    model_name = spec["name"]
    expected_hidden = spec["hidden_dim"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only,
    )
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ),
    ).to(device)

    actual_hidden = int(model.config.hidden_size)
    if actual_hidden != expected_hidden:
        raise ValueError(
            f"hidden_dim mismatch for tier {tier!r}: registered "
            f"{expected_hidden}, loaded model reports {actual_hidden}. "
            "The MODEL_TIERS registry has drifted — update devices.py."
        )

    return ChatHead(model=model, tokenizer=tokenizer)
```

**Step 3: Commit.**

---

## Task 3: Prefix dtype alignment

**Files:** Modify `src/soma/system.py`, `src/soma/session/chat_session.py`. Add/extend tests.

The fix: before `torch.cat([prefix, token_embeds], dim=1)`, cast prefix to match `token_embeds.dtype`. Applies to both `SOMA.chat()` (Phase 3) and `ChatSession.respond()` (Phase 5).

**Step 1: Failing test** (add to either a new file or extend session tests):

```python
# tests/test_deploy/test_dtype_alignment.py
import torch
from torch import nn
# ... reuse _TinyCausalLM / _TinyTokenizer / _soma_cfg from Phase 5 test helper

def test_session_respond_handles_fp16_llm():
    """When the LLM runs in fp16, the verbalizer prefix (fp32) must be
    cast so the concat doesn't blow up or silently promote everything."""
    # Build a _TinyCausalLM with fp16 weights.
    # Build a verbalizer in fp32 (its default).
    # Call session.respond(...) — should succeed AND the inputs_embeds
    # passed into model.forward should be fp16 (matching token_embeds).

    # Key assertion: no RuntimeError about dtype mismatch, AND the
    # generated response is still a string.
    ...
```

**Step 2: Implement the alignment** in both call sites:

```python
# In SOMA.chat and ChatSession.respond, before torch.cat:
if prefix.dtype != token_embeds.dtype:
    prefix = prefix.to(dtype=token_embeds.dtype)
inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
```

Document the cast in a short comment explaining why.

**Step 3: Commit.**

---

## Task 4: `--tier auto` CLI integration

**Files:** Modify `scripts/chat_repl.py`, `scripts/train_verbalizer_bootstrap.py`.

In both:
- Change the `--llm-name` flag to OPTIONAL.
- Add `--tier` with choices `{"auto", "tiny", "small", "large"}`, default `"auto"`.
- If `--llm-name` is provided, override the tier path and use it directly (preserves backward compat).
- If `--llm-name` not given: call `auto_select_tier(detect_cuda_vram())` → `MODEL_TIERS[tier]["name"]`.
- Similarly, `--device` default becomes `"auto"` → `select_device_and_dtype()` on "auto".

Print the selection: `"Selected tier=small (Qwen2.5-1.5B-Instruct), device=cuda, dtype=fp16"` — operators want to see what they got.

**Step 1:** No TDD test (these are CLI scripts). Verify `--help` is sensible.

**Step 2: Commit.**

---

## Task 5: CUDA smoke (slow + cuda marker)

**Files:** `tests/test_deploy/test_chat_head_cuda_smoke.py`.

```python
"""Slow+cuda smoke: Qwen2.5-1.5B-Instruct on GPU, fp16."""
import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.cuda]


def test_qwen15b_loads_and_generates_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from soma.deploy.chat_head_factory import build_chat_head
    from soma.deploy.devices import select_device_and_dtype

    device, dtype = select_device_and_dtype()
    head = build_chat_head(tier="small", device=device, dtype=dtype)

    # Smoke: can the model produce a coherent response via inputs_embeds?
    prefix = torch.zeros(1, 4, head.hidden_size, dtype=dtype, device=device)
    # Tokenize something.
    ids = head.tokenizer("Hello, world.", return_tensors="pt")["input_ids"].to(device)
    token_embeds = head.model.get_input_embeddings()(ids)
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
    attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=device)

    text = head.generate_text(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=10,
        min_new_tokens=1,
        do_sample=False,
    )
    assert isinstance(text, str)
    assert len(text) > 0
    print(f"Qwen2.5-1.5B on {device} {dtype}: {text!r}")
```

Opt-in run: `pytest -v -m "slow and cuda"`.

**Step 2: Commit.**

---

## Task 6: `scripts/demo_chat.py` end-to-end demo

Zero-arg script. Auto-detect → select tier → build everything → 4-turn demo. Prints responses. Exits 0 on success.

**Step 1:** Sketch, adapt to real APIs:

```python
"""End-to-end SOMA chat demo.

Auto-detects the best tier for the available hardware and runs a short
4-turn demo conversation. No CLI args.

    python scripts/demo_chat.py
"""
from __future__ import annotations

import torch

from soma.core.config import SOMAConfig
from soma.deploy.chat_head_factory import build_chat_head
from soma.deploy.devices import (
    MODEL_TIERS,
    auto_select_tier,
    detect_cuda_vram,
    select_device_and_dtype,
)
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.session.chat_session import ChatSession
from soma.system import SOMA


SEED_TEXTS = [
    "hello world",
    "the quick brown fox jumps over the lazy dog",
    "knowledge is power",
    "time flies like an arrow",
    "machine learning is a subset of artificial intelligence",
]


def main() -> None:
    vram = detect_cuda_vram()
    tier = auto_select_tier(vram_gb=vram)
    device, dtype = select_device_and_dtype()
    spec = MODEL_TIERS[tier]

    print(f"Detected VRAM: {vram}GB" if vram else "No CUDA — running on CPU")
    print(f"Selected tier: {tier}  ({spec['name']}, ~{spec['approx_fp16_gb']:.1f}GB)")
    print(f"Device: {device}  dtype: {dtype}")
    print()

    chat_head = build_chat_head(tier=tier, device=device, dtype=dtype)

    cfg = SOMAConfig(
        sensor_output_dim=16, associator_input_dim=16, associator_hidden_dim=32,
        associator_output_dim=16, integrator_input_dim=16, integrator_hidden_dim=32,
        integrator_output_dim=16, position_dim=4, wm_slots=4, wm_dim=16,
        episodic_capacity=16, key_dim=16, value_dim=16, vocab_size=256,
        text_embed_dim=16, max_nodes=64, initial_associator_count=4,
        initial_integrator_count=2, max_input_tokens=32, max_output_tokens=8,
        seed=0,
    )
    soma = SOMA(cfg, device=device)
    tokenizer = train_bpe_tokenizer(iter(SEED_TEXTS), vocab_size=cfg.vocab_size)
    encoder = TextEncoder(tokenizer, embed_dim=cfg.text_embed_dim,
                          max_seq_len=cfg.max_input_tokens)

    verbalizer_spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name=spec["name"],
        llm_hidden_dim=chat_head.hidden_size,
        num_prefix_tokens=8,
    )
    verbalizer = SomaVerbalizer(verbalizer_spec).to(device)

    session = ChatSession(
        soma=soma, verbalizer=verbalizer, chat_head=chat_head,
        tokenizer=tokenizer, encoder=encoder,
    )

    questions = [
        "Hello!",
        "What is a brain?",
        "Tell me a fun fact.",
        "What should I learn about next?",
    ]
    for q in questions:
        resp = session.respond(
            user_text=q, max_new_tokens=30, min_new_tokens=3, do_sample=False,
        )
        print(f"[user] {q}")
        print(f"[soma] {resp}")
        print()


if __name__ == "__main__":
    main()
```

**Step 2:** Verify imports cleanly. No unit test — the CUDA smoke (T5) plus the existing session tests cover the mechanics.

**Step 3: Commit.**

---

## Task 7: Deployment docs + register cuda marker

**Files:** Create `docs/deployment.md`, edit `pyproject.toml`.

Register the `cuda` marker in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (integration tests, deselect with '-m \"not slow\"')",
    "cuda: requires a CUDA GPU to run (deselect with '-m \"not cuda\"')",
]
```

Write `docs/deployment.md` covering:
- Minimum hardware matrix (CPU / 4GB / 8GB / 16GB / 24GB)
- Tier → model mapping
- How to run `demo_chat.py`
- How to run the CUDA smoke (`pytest -m "slow and cuda"`)
- Troubleshooting: "out of memory" advice, dtype mismatch

Keep under 200 lines. This is operator-facing, not a thesis.

**Step 3: Commit.**

---

## Task 8: Full regression + merge

```bash
pytest tests/ -m "not slow and not cuda" -q
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/soma/
git checkout main
git merge --no-ff feat/consumer-deploy -m "Merge ..."
git push origin main
git push github main
```

Phase 7 does NOT change existing save/load semantics. Safe to merge live.

---

## Appendix — VRAM Budget Reference

| Component | fp32 | fp16 |
|---|---|---|
| SmolLM2-360M weights | 1.4 GB | 0.72 GB |
| Qwen2.5-1.5B weights | 6.2 GB | 3.1 GB |
| Qwen2.5-7B weights | 29 GB | 14.5 GB |
| KV cache (512 ctx, 1.5B) | 100 MB | 50 MB |
| KV cache (512 ctx, 7B) | 700 MB | 350 MB |
| SomaVerbalizer (~8M) | 32 MB | 16 MB (if mixed) |
| SOMA graph + memories | <100 MB | <100 MB |

**Minimum VRAM floor rationale:** want at least 2× the model weights for activations + KV cache + Python/CUDA overhead. `small` tier (3.1 GB fp16) on an 8GB card = 3.1 GB weights + ~1 GB overhead = 4 GB used, 4 GB headroom. Comfortable.

---

*End of Phase 7 plan. 8 tasks, TDD-disciplined.*
