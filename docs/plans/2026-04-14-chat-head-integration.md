# Phase 3 — ChatHead & First Transformer Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development when run alongside a controller) to implement this plan task-by-task.

**Goal:** Wire SOMA's `SomaVerbalizer` (Phase 2) to a frozen HuggingFace causal-LM through a thin `ChatHead` wrapper, producing the first end-to-end SOMA-conditioned natural-language generation path. Target LLM: `HuggingFaceTB/SmolLM2-360M-Instruct` as the dev sandbox (Apache-2.0, ~720MB fp16, fits any dev GPU).

**Architecture:** `ChatHead` owns a frozen `transformers.PreTrainedModel` + its tokenizer. `SOMA.chat(user_text, verbalizer, chat_head)` orchestrates: aggregate OUTPUT activations → verbalizer projects to `(B, k, d_model)` prefix → tokenize user text → fetch input embeddings via `llm.get_input_embeddings()` → concat prefix + token embeds → construct position_ids (0..k-1 prefix, k..k+T-1 tokens) → construct attention_mask → `llm.generate(inputs_embeds=..., attention_mask=..., position_ids=...)`. Fallback path: if no `ChatHead` is loaded, route through `verbalizer.fallback_text()`.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace `transformers>=4.40`, `accelerate>=0.30`, existing SOMA codebase.

---

## Why Phase 3 Now

Phase 2 delivered the projection contract in isolation with mock tests. Phase 3 is the first moment SOMA produces actual human-readable language conditioned on its internal state. Deliberately constrained:
- **Dev-sandbox LLM only** (SmolLM2-360M). Qwen2.5-1.5B + GGUF Q4 are Phase 6.
- **No training yet.** Phase 3 exercises inference with near-zero-init verbalizer (LLM ~vanilla). Bootstrap is Phase 4.
- **No multi-turn loop.** Single-turn `chat(prompt) -> response`. WM integration is Phase 5.

The value of Phase 3 in isolation: validates the HF integration surface (`inputs_embeds`, `position_ids`, tied embeddings, attention-mask shape, generation kwargs) before we layer training cost on top.

---

## Key Decisions

### 1. `ChatHead` owns LLM + tokenizer (not SOMA)
Matches the `TextEncoder`/`TextDecoder` pattern. Caller holds a `ChatHead` and passes it into `soma.chat(...)`. Can be swapped entirely without touching SOMA or the Verbalizer.

### 2. `inputs_embeds` entry-point (not token injection)
All target models ({SmolLM2, Qwen2.5, Llama-3.2, Phi-3.5}) accept `forward(inputs_embeds=...)`. Tokenizer handles user-text portion; prefix slots are continuous vectors.

### 3. position_ids = `arange(0, k+T)`
Prefix at 0..k-1, text at k..k+T-1. Standard soft-prompt behavior. RoPE handles naturally. k+T fits under 8K (SmolLM2) and 32K (Qwen2.5) contexts easily.

### 4. LLM params frozen (requires_grad=False)
`ChatHead.__init__` sets `requires_grad=False` on all base-model params + puts module into inference mode via `train(False)`. Only SomaVerbalizer params train. Tested explicitly.

---

## Tasks

| # | Scope | Files |
|---|---|---|
| 1 | Add `transformers`/`accelerate` to `[dev-chat]` extras | `pyproject.toml` |
| 2 | `ChatHead.__init__` — freeze + validate HF model + tokenizer | `src/soma/io/chat_head.py`, test |
| 3 | `ChatHead.generate(inputs_embeds, attention_mask, ...)` thin wrapper | same |
| 4 | Position-IDs / attention-mask construction helpers | same, test |
| 5 | `SOMA.chat(user_text, verbalizer, chat_head, ...)` end-to-end | `src/soma/system.py`, test |
| 6 | Fallback path when `chat_head is None` → verbalizer text path | same, test |
| 7 | Tied-embeddings sanity — `inputs_embeds` path preserves shared weight | test with mock tied model |
| 8 | Smoke integration (marked slow): real SmolLM2-360M download + generation | `tests/test_io/test_chat_head_smoke.py` |
| 9 | Full regression — pytest + ruff + mypy | — |

**Tasks 2–7 use a mock HF model** (tiny `nn.Module` implementing the minimum `forward(inputs_embeds, attention_mask, position_ids)` and `get_input_embeddings()` + `generate()` contract). Real SmolLM2 is gated behind a pytest `slow` marker in Task 8 so default CI stays fast + offline.

---

## Task 1: Add `transformers` + `accelerate` to `[dev-chat]` extras

**Files:** Modify `pyproject.toml`.

**Step 1:** Read current `[project.optional-dependencies]` block.

**Step 2:** Add:
```toml
dev-chat = [
    "transformers>=4.40",
    "accelerate>=0.30",
    "safetensors>=0.4",
]
```

**Step 3:** Install + verify:
```bash
pip install -e ".[dev-chat]"
python -c "import transformers; print(transformers.__version__)"
python -c "from transformers import AutoModelForCausalLM; print('ok')"
```

**Step 4: Commit:**
```bash
git add pyproject.toml
git commit -m "feat(deps): add transformers/accelerate under [dev-chat] extras"
```

---

## Task 2: `ChatHead.__init__` — freeze + validate

**Files:** Create `src/soma/io/chat_head.py` + `tests/test_io/test_chat_head.py`.

**Step 1: Failing test (uses a mock HF-style model)**

```python
# tests/test_io/test_chat_head.py
import pytest
import torch
from torch import nn

from soma.io.chat_head import ChatHead


class _TinyCausalLM(nn.Module):
    """Mock HF causal LM: embedding + single Linear lm_head."""

    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, position_ids=None):
        return self.lm_head(inputs_embeds)

    @torch.no_grad()
    def generate(self, *, inputs_embeds, attention_mask=None, max_new_tokens=4, **kw):
        embeds = inputs_embeds
        generated = []
        for _ in range(max_new_tokens):
            logits = self.forward(inputs_embeds=embeds)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated.append(next_id)
            next_embed = self.embed(next_id).unsqueeze(1)
            embeds = torch.cat([embeds, next_embed], dim=1)
        return torch.stack(generated, dim=1)


class _TinyTokenizer:
    """Mock HF tokenizer: char-level, vocab=32."""

    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = [min(ord(c) % 32, 31) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids.flatten().tolist())


def test_chat_head_freezes_base_model():
    model = _TinyCausalLM()
    assert all(p.requires_grad for p in model.parameters())
    head = ChatHead(model=model, tokenizer=_TinyTokenizer())
    assert not any(p.requires_grad for p in head.model.parameters())


def test_chat_head_stores_model_and_tokenizer():
    model = _TinyCausalLM()
    tok = _TinyTokenizer()
    head = ChatHead(model=model, tokenizer=tok)
    assert head.model is model
    assert head.tokenizer is tok


def test_chat_head_exposes_hidden_size():
    head = ChatHead(model=_TinyCausalLM(d_model=16), tokenizer=_TinyTokenizer())
    assert head.hidden_size == 16
```

**Step 2:** Run — fail (module missing).

**Step 3: Implement**

```python
# src/soma/io/chat_head.py
"""Thin wrapper around a frozen HuggingFace causal LM."""
from __future__ import annotations

from typing import Any


class ChatHead:
    """Frozen LLM + tokenizer pair for SOMA.chat orchestration."""

    def __init__(self, *, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        # Freeze all base-model params — only the verbalizer trains.
        for p in self.model.parameters():
            p.requires_grad = False
        # Put module into inference mode (disables dropout/BN-train-statistics).
        self.model.train(False)

    @property
    def hidden_size(self) -> int:
        """d_model of the underlying LLM — must match VerbalizerSpec.llm_hidden_dim."""
        return int(self.model.config.hidden_size)
```

**Step 4:** Run — 3 tests pass.

**Step 5: Commit:**
```bash
git add src/soma/io/chat_head.py tests/test_io/test_chat_head.py
git commit -m "feat(chat): ChatHead freezes HF causal LM + exposes hidden_size"
```

---

## Task 3: `ChatHead.generate()` + `.generate_text()`

**Step 1: Failing test**

```python
# append
def test_chat_head_generate_takes_inputs_embeds():
    head = ChatHead(model=_TinyCausalLM(), tokenizer=_TinyTokenizer())
    inputs_embeds = torch.randn(1, 5, 16)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    out = head.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=3,
    )
    assert out.shape == (1, 3)


def test_chat_head_generate_decodes_to_text():
    head = ChatHead(model=_TinyCausalLM(), tokenizer=_TinyTokenizer())
    text = head.generate_text(
        inputs_embeds=torch.randn(1, 4, 16),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        max_new_tokens=5,
    )
    assert isinstance(text, str)
```

**Step 2:** Run — fail.

**Step 3: Implement** (add `import torch` at module top and these methods to `ChatHead`):

```python
def generate(
    self, *, inputs_embeds, attention_mask, max_new_tokens=64, **kw,
):
    return self.model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        **kw,
    )


def generate_text(
    self, *, inputs_embeds, attention_mask, max_new_tokens=64, **kw,
) -> str:
    ids = self.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        **kw,
    )
    return self.tokenizer.decode(ids[0], skip_special_tokens=True)
```

**Step 4:** Run — 5 tests pass.

**Step 5: Commit:**
```bash
git commit -m "feat(chat): ChatHead.generate() / .generate_text() with inputs_embeds"
```

---

## Task 4: Position-IDs + attention-mask helpers

Encapsulates the k-prefix-then-T-tokens convention so callers can't get it wrong.

**Step 1: Failing test**

```python
# append
from soma.io.chat_head import (
    build_position_ids, build_attention_mask, build_attention_mask_from_pad,
)


def test_position_ids_cover_prefix_plus_tokens():
    ids = build_position_ids(num_prefix=4, num_tokens=7, batch_size=2)
    assert ids.shape == (2, 11)
    assert torch.all(ids[0] == torch.arange(11))
    assert torch.all(ids[1] == torch.arange(11))


def test_attention_mask_ones_for_prefix_and_tokens():
    mask = build_attention_mask(num_prefix=4, num_tokens=7, batch_size=1)
    assert mask.shape == (1, 11)
    assert torch.all(mask == 1)


def test_attention_mask_zero_for_padded_tokens():
    pad_mask = torch.tensor([[1, 1, 1, 0, 0]])  # 3 real + 2 pad
    mask = build_attention_mask_from_pad(num_prefix=4, pad_mask=pad_mask)
    assert mask.shape == (1, 9)
    assert torch.all(mask[0, :4] == 1)
    assert torch.all(mask[0, 4:] == pad_mask[0])
```

**Step 2:** Run — fail.

**Step 3: Implement** (add module-level functions to `chat_head.py`):

```python
def build_position_ids(*, num_prefix: int, num_tokens: int, batch_size: int) -> torch.Tensor:
    """[0..k+T-1] per batch row."""
    seq_len = num_prefix + num_tokens
    return torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)


def build_attention_mask(*, num_prefix: int, num_tokens: int, batch_size: int) -> torch.Tensor:
    """All-ones mask; assumes no padding in tokens."""
    seq_len = num_prefix + num_tokens
    return torch.ones(batch_size, seq_len, dtype=torch.long)


def build_attention_mask_from_pad(*, num_prefix: int, pad_mask: torch.Tensor) -> torch.Tensor:
    """Prepend prefix-ones to a tokenizer-produced pad mask."""
    batch_size = pad_mask.shape[0]
    prefix = torch.ones(batch_size, num_prefix, dtype=pad_mask.dtype, device=pad_mask.device)
    return torch.cat([prefix, pad_mask], dim=1)
```

**Step 4:** Run — 8 tests pass.

**Step 5: Commit:**
```bash
git commit -m "feat(chat): build_position_ids + build_attention_mask(_from_pad) helpers"
```

---

## Task 5: `SOMA.chat()` end-to-end orchestration

**Files:** Modify `src/soma/system.py`; extend test file.

**Step 1: Failing test**

```python
# append to tests/test_io/test_chat_head.py
from soma.core.config import SOMAConfig
from soma.system import SOMA
from soma.io.verbalizer import VerbalizerSpec, SomaVerbalizer


def _soma_cfg() -> SOMAConfig:
    return SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=16,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        seed=0,
    )


def test_soma_chat_end_to_end_with_mock_llm():
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name="mock", llm_hidden_dim=16,
        num_prefix_tokens=4, proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    chat_head = ChatHead(model=_TinyCausalLM(vocab=32, d_model=16), tokenizer=_TinyTokenizer())
    response = soma.chat(
        user_text="hello",
        verbalizer=verbalizer,
        chat_head=chat_head,
        max_new_tokens=3,
    )
    assert isinstance(response, str)
```

**Step 2:** Run — fail (method missing).

**Step 3: Implement** — add `chat()` and `_current_output_activations()` to `SOMA` in `src/soma/system.py`:

```python
def chat(
    self,
    *,
    user_text: str,
    verbalizer: Any,
    chat_head: Any | None = None,
    max_new_tokens: int = 64,
    **generate_kwargs: Any,
) -> str:
    """Generate NL response conditioned on current SOMA state.

    Pipeline:
      1. Aggregate OUTPUT-node activations via SomaAggregator.collapse
      2. Project to soft-prompt prefix via verbalizer
      3. Tokenize user_text via chat_head.tokenizer
      4. Fetch token embeddings via chat_head.model.get_input_embeddings()
      5. Concat prefix + token_embeds -> inputs_embeds
      6. Build position_ids + attention_mask
      7. chat_head.generate_text(...)

    If chat_head is None, routes through verbalizer.fallback_text().
    """
    from soma.io.verbalizer import SomaAggregator
    from soma.io.chat_head import build_position_ids, build_attention_mask_from_pad

    output_acts = self._current_output_activations()
    soma_state = SomaAggregator.collapse(
        output_acts, soma_output_dim=verbalizer.spec.soma_output_dim
    )
    prefix = verbalizer(soma_state)  # (1, k, d_model)

    if chat_head is None:
        return verbalizer.fallback_text(self, soma_state)

    tok_out = chat_head.tokenizer(user_text, return_tensors="pt")
    input_ids = tok_out["input_ids"]
    pad_mask = tok_out.get("attention_mask", torch.ones_like(input_ids))
    token_embeds = chat_head.model.get_input_embeddings()(input_ids)  # (1, T, D)

    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

    k = verbalizer.spec.num_prefix_tokens
    T = input_ids.shape[1]
    attention_mask = build_attention_mask_from_pad(num_prefix=k, pad_mask=pad_mask)
    position_ids = build_position_ids(num_prefix=k, num_tokens=T, batch_size=1)

    return chat_head.generate_text(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_new_tokens,
        **generate_kwargs,
    )


def _current_output_activations(self) -> dict[str, torch.Tensor]:
    """Last per-OUTPUT-node activation values from the graph, flattened."""
    output_acts: dict[str, torch.Tensor] = {}
    for node_id, node in self.graph.nodes.items():
        if node.node_type is NodeType.OUTPUT:
            last_act = getattr(node, "activation", None)
            if last_act is None:
                # Fallback to EMA or zero — avoid crashing on cold start.
                continue
            output_acts[node_id] = last_act.detach().view(-1)
    return output_acts
```

Note: check `src/soma/core/node.py` for the actual last-activation attribute name. `node.activation` is a guess — likely `node._last_activation` or similar. Adapt to the real attribute. The helper should be defensive — missing activations produce an empty dict, and `SomaAggregator.collapse({}, ...)` returns zeros (Phase 2 Task 6 contract).

**Step 4:** Run — test passes.

**Step 5: Commit:**
```bash
git add src/soma/system.py tests/test_io/test_chat_head.py
git commit -m "feat(chat): SOMA.chat() end-to-end verbalizer + chat_head orchestration"
```

---

## Task 6: Fallback when `chat_head is None`

**Step 1: Test** (should pass if Task 5's fallback branch works):

```python
# append
def test_soma_chat_falls_back_when_no_chat_head():
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name="none", llm_hidden_dim=8, num_prefix_tokens=2,
    )
    verbalizer = SomaVerbalizer(spec)
    response = soma.chat(
        user_text="hello", verbalizer=verbalizer, chat_head=None,
    )
    assert isinstance(response, str)
```

**Step 2:** Run — should pass immediately.

**Step 3:** If fails, revisit Task 5's `if chat_head is None:` branch.

**Step 4: Commit** (test-only or bugfix commit):
```bash
git commit -m "test(chat): fallback path when no ChatHead loaded"
```

---

## Task 7: Tied-embeddings sanity

SmolLM2 and Qwen2.5-small tie `input_embeddings` ↔ `lm_head.weight`. Our `inputs_embeds` path must not corrupt the shared weight matrix.

**Step 1: Test** (pure regression, no impl change expected):

```python
# append
class _TiedCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tied
        self.config = type(
            "Cfg", (),
            {"hidden_size": d_model, "vocab_size": vocab, "tie_word_embeddings": True},
        )()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, position_ids=None):
        return self.lm_head(inputs_embeds)

    @torch.no_grad()
    def generate(self, *, inputs_embeds, attention_mask=None, max_new_tokens=4, **kw):
        embeds = inputs_embeds
        gens = []
        for _ in range(max_new_tokens):
            logits = self.forward(inputs_embeds=embeds)
            next_id = logits[:, -1, :].argmax(dim=-1)
            gens.append(next_id)
            embeds = torch.cat([embeds, self.embed(next_id).unsqueeze(1)], dim=1)
        return torch.stack(gens, dim=1)


def test_chat_head_works_with_tied_embeddings():
    model = _TiedCausalLM()
    embed_before = model.embed.weight.data.clone()
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()

    head = ChatHead(model=model, tokenizer=_TinyTokenizer())
    _ = head.generate(
        inputs_embeds=torch.randn(1, 4, 16),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        max_new_tokens=3,
    )

    assert torch.equal(model.embed.weight.data, embed_before)
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
```

**Step 2:** Run — should pass.

**Step 3: Commit:**
```bash
git commit -m "test(chat): tied-embeddings sanity (shared weight unchanged through generate)"
```

---

## Task 8: Smoke integration with real SmolLM2-360M

Marked `slow` — deselected from default CI run. Run manually after `pip install -e ".[dev-chat]"` + ~720MB download.

**Step 1: Register marker in `pyproject.toml` if absent:**

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

**Step 2: Create `tests/test_io/test_chat_head_smoke.py`:**

```python
"""Smoke test with a real SmolLM2-360M download.

Run manually:
    pip install -e ".[dev-chat]"
    pytest tests/test_io/test_chat_head_smoke.py -v -m slow
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow


def test_smolm2_360m_generates_with_soma_prefix():
    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soma.core.config import SOMAConfig
    from soma.io.chat_head import ChatHead
    from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
    from soma.system import SOMA

    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
    ).to(device)

    chat_head = ChatHead(model=model, tokenizer=tokenizer)

    cfg = SOMAConfig(
        sensor_output_dim=8, associator_input_dim=8, associator_hidden_dim=16,
        associator_output_dim=8, integrator_input_dim=16, integrator_hidden_dim=16,
        integrator_output_dim=16, position_dim=4, wm_slots=2, wm_dim=8,
        episodic_capacity=4, key_dim=8, value_dim=8, vocab_size=16,
        text_embed_dim=8, max_nodes=32, initial_associator_count=2,
        initial_integrator_count=1, max_input_tokens=8, max_output_tokens=4,
        seed=0,
    )
    soma = SOMA(cfg, device=device)
    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name=model_name,
        llm_hidden_dim=chat_head.hidden_size,  # 960
        num_prefix_tokens=8,
    )
    verbalizer = SomaVerbalizer(spec).to(device)

    response = soma.chat(
        user_text="Hello, how are you?",
        verbalizer=verbalizer,
        chat_head=chat_head,
        max_new_tokens=20,
        do_sample=False,
    )
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"SmolLM2 response (near-null prefix): {response!r}")
```

**Step 3: Run manually — confirm pass**
```bash
pytest tests/test_io/test_chat_head_smoke.py -v -m slow
```

**Step 4:** Confirm default `pytest tests/` still green (smoke test deselected).

**Step 5: Commit:**
```bash
git add tests/test_io/test_chat_head_smoke.py pyproject.toml
git commit -m "test(chat): smoke integration with real SmolLM2-360M (marked slow)"
```

---

## Task 9: Full regression

```bash
pytest tests/  # skips `slow` by default
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/soma/
```

All must pass. Each fix is a fresh commit.

---

## Post-Phase 3 — Merge + Push

Phase 3 does NOT modify save/load semantics. Safe to merge without stopping the train service.

```bash
git checkout main
git merge --no-ff feat/chat-head -m "Merge ..."
git push
```

---

## Phase 4 Preview (next plan doc, not in this one)

Bootstrap training of the verbalizer:
- Paired-data loop: `(soma_state, target_text) → LM-loss-through-frozen-LLM → backprop into projector`
- Held-out eval
- Verbalizer checkpointing

Dedicated plan when Phase 3 lands.

---

## Appendix — VRAM Budget

| Component | fp32 | fp16 |
|---|---|---|
| SmolLM2-360M weights | 1.45 GB | 0.72 GB |
| KV cache (512 ctx) | 48 MB | 24 MB |
| SomaVerbalizer (~8M) | 31 MB | 16 MB |
| SOMA graph + memories | <100 MB | <100 MB |
| **Total runtime** | ~1.6 GB | ~0.9 GB |

Trivial on RTX 3090's 24GB.

---

*End of Phase 3 plan. 9 tasks, TDD-disciplined.*
