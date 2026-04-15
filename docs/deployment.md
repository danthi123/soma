# SOMA Deployment Guide

Operator-facing guide for running the SOMA + verbalizer + frozen-LLM chat
pipeline on consumer hardware. Phase 7 added auto-detection of VRAM, a
three-tier model registry, and a zero-arg demo script.

See `docs/plans/2026-04-14-consumer-deploy.md` for design context.
See `src/soma/deploy/` for the authoritative API.

---

## Hardware Matrix

| Host                | Auto-selected tier | Backing model                 | Device / dtype |
|---------------------|--------------------|-------------------------------|----------------|
| CPU only            | `tiny`             | SmolLM2-360M-Instruct         | `cpu` / fp32   |
| GPU, <4 GB VRAM     | `tiny`             | SmolLM2-360M-Instruct         | `cuda` / fp16  |
| GPU, 4-15 GB VRAM   | `small`            | Qwen2.5-1.5B-Instruct         | `cuda` / fp16  |
| GPU, 16-23 GB VRAM  | `large`            | Qwen2.5-7B-Instruct           | `cuda` / fp16  |
| GPU, 24 GB+ VRAM    | `large`            | Qwen2.5-7B-Instruct           | `cuda` / fp16  |

Auto-detect is integer-floored: a 7.94 GB card reports as 7 GB, so the
4/16 GB thresholds already bake in a working-set cushion.

---

## Tier Registry

| Tier   | HF model                                 | fp16 size | `hidden_dim` |
|--------|------------------------------------------|-----------|--------------|
| tiny   | `HuggingFaceTB/SmolLM2-360M-Instruct`    | 0.72 GB   | 960          |
| small  | `Qwen/Qwen2.5-1.5B-Instruct`             | 3.1 GB    | 1536         |
| large  | `Qwen/Qwen2.5-7B-Instruct`               | 14.5 GB   | 3584         |

The registry lives in `src/soma/deploy/devices.py` (`MODEL_TIERS`). The
verbalizer's `llm_hidden_dim` is validated against `chat_head.hidden_size`
at construction — tier drift fails loudly.

---

## Quick Start: Demo

```bash
python scripts/demo_chat.py
```

What happens:

1. Detects CUDA VRAM and picks a tier.
2. Downloads the tier-matched LLM from HuggingFace on first run (cached
   under `~/.cache/huggingface/hub/` afterwards).
3. Builds a fresh tiny SOMA + near-zero-init verbalizer.
4. Runs a 4-turn demo exchange and prints responses.

No CLI args, no checkpoint required. The verbalizer is untrained, so the
outputs are grammatically shaped but semantically empty — the point is to
prove the pipeline runs end-to-end.

Example output on a 24 GB box:

```
Detected VRAM: 24GB
Selected tier: large  (Qwen/Qwen2.5-7B-Instruct, ~14.5GB)
Device: cuda  dtype: fp16

[user] Hello!
[soma] ...
```

Example output on an 8 GB box:

```
Detected VRAM: 8GB
Selected tier: small  (Qwen/Qwen2.5-1.5B-Instruct, ~3.1GB)
Device: cuda  dtype: fp16

[user] Hello!
[soma] ...
```

---

## CUDA Smoke Test

Opt in (excluded from default CI):

```bash
pytest -v -m "slow and cuda" tests/test_deploy/test_chat_head_cuda_smoke.py
```

- Downloads ~3 GB of Qwen2.5-1.5B weights on first run.
- Uses ~3 GB of VRAM during the test.
- Takes ~45 s on a 3090 with a cold HF cache.
- Skips gracefully when CUDA is unavailable.

---

## Full Chat REPL

```bash
python scripts/chat_repl.py --soma-checkpoint path/to/bundle --tier auto
```

Key flags (run with `--help` for the full list):

| Flag              | Default | Notes                                                        |
|-------------------|---------|--------------------------------------------------------------|
| `--tier`          | `auto`  | One of `auto`, `tiny`, `small`, `large`.                     |
| `--device`        | `auto`  | `auto`, `cuda`, `cpu`, or a torch device string.             |
| `--dtype`         | `auto`  | `auto`, `fp16`, `fp32`. Pairs with `--device auto`.          |
| `--llm-name`      | (none)  | Bypass the tier registry (e.g. Llama-3-8B, Mistral-7B).      |
| `--quantization`  | `none`  | `none`/`int8`/`int4`. See "Quantization" section below.      |
| `--max-new-tokens`| `64`    | Max tokens per assistant response.                           |

Explicit `--llm-name` wins over `--tier`. Use it when you want a model
not in the registry.

---

## Troubleshooting

**OOM on an 8 GB card with `--tier small`.** 3.1 GB weights + KV cache +
CUDA overhead can crowd a low-end 8 GB card. Drop to `--tier tiny`, reduce
`--max-new-tokens`, or close other GPU apps.

**Dtype mismatch at runtime.** Should be impossible after Phase 7 T3 —
the verbalizer prefix is cast to the LLM's dtype before concat. If you
see one, check whether `--dtype` was overridden inconsistently with the
LLM's load dtype.

**First run is slow.** HuggingFace downloads the model weights. Typical
cache location: `~/.cache/huggingface/hub/`. Subsequent runs start in
seconds.

**`--dtype fp16 --device cpu` is slow.** This combo emits a
`UserWarning` from `src/soma/deploy/cli.py` — PyTorch's CPU fp16 matmul
is typically slower than fp32 on consumer CPUs without AVX-512-fp16. Use
`--dtype fp32` on CPU.

**`hidden_dim mismatch` at chat-head load.** A registered tier's HF model
changed shape (likely deprecated or renamed upstream). File an issue;
update `MODEL_TIERS` in `src/soma/deploy/devices.py`.

---

## Diagnostics

Inspect a saved SOMA bundle:

```bash
python scripts/inspect_soma.py --bundle path/to/bundle
```

Prints node counts by type, edge density, WM slot state, homeostatic
gain range, and episodic memory occupancy. Accepts both directory
bundles and single `.pt` checkpoints. Use when training metrics look
off (e.g., gain pinned at 10.0 -> homeostasis overshoot; all edge
weights near zero -> signal collapse) to get a fast read on internal
state without spinning up the full training harness.

---

## Quantization (optional)

For 7B models on 8GB-class GPUs, install the optional `bitsandbytes`
backend and pass `--quantization int4`:

```bash
pip install -e ".[quant]"
python scripts/chat_repl.py --soma-checkpoint path/to/bundle \
    --tier large --quantization int4
```

| Mode    | 7B VRAM   | Fits 8GB? | Quality vs fp16 |
|---------|-----------|-----------|-----------------|
| `none`  | ~14.5 GB  | no        | reference       |
| `int8`  | ~7-8 GB   | tight     | near-identical  |
| `int4`  | ~4-5 GB   | yes       | small drop      |

int4 uses `nf4` + double-quant via bitsandbytes. Crucially, only the
linear weights are quantised — `model.get_input_embeddings()` stays in
the requested compute_dtype (fp16/bf16), so the verbalizer's gradient
flow through input embeddings is preserved and bootstrap training still
converges.

**Compatibility.** bitsandbytes is **CUDA-only** — `--quantization`
on `--device cpu` raises `ValueError`. Windows wheels are historically
flaky; if `pip install` fails, run under WSL.

**Known limitations.**

- int4 compute_dtype must match the verbalizer prefix dtype after the
  Phase 7 T3 cast — pair `--quantization int4` with `--dtype fp16` on
  CUDA. Mixing int4 + fp32 verbalizer is unsupported.
- int8 gives slightly better generation quality than int4 but uses ~2x
  the VRAM. Prefer int4 unless quality drift is observed.
- The explicit `--llm-name` path does not currently honor
  `--quantization`; use `--tier` instead.

---

## VRAM Budget

| Tier  | Model                        | fp16 weights | Headroom on 8GB | Headroom on 24GB |
|-------|------------------------------|--------------|-----------------|------------------|
| tiny  | SmolLM2-360M-Instruct        | 0.72 GB      | comfortable     | comfortable      |
| small | Qwen2.5-1.5B-Instruct        | 3.1 GB       | 4 GB headroom   | 20 GB headroom   |
| large | Qwen2.5-7B-Instruct          | 14.5 GB      | DOES NOT FIT    | 9 GB headroom    |

Rule of thumb: budget 2x the fp16 weight size for activations, KV cache,
and CUDA overhead. The `small` tier fits comfortably alongside SOMA
training on an 8 GB card. The `large` tier does not fit on 8 GB and
leaves a few GB of headroom on 16 GB.

---

## Concurrency

The SOMA training service uses the GPU. Running the CUDA smoke test or
the demo concurrently adds ~3 GB of VRAM on top of whatever training is
consuming. Run them serially unless you have >20 GB of headroom.

---

## GGUF inference-only backend (optional)

Use this when you already have GGUFs cached locally (e.g., from LM Studio
under `~/.cache/lm-studio/models/`) and don't want to re-download HF
safetensors. **GGUF is inference-only** — `llama.cpp` does not expose
the input-embedding tensor as a gradient-enabled `nn.Module`, so the
SOMA verbalizer cannot be bootstrapped or trained against a GGUF
backend. Use the HF path (`build_chat_head`) for any training run.

Install the optional extras:

```bash
pip install -e ".[gguf]"
```

`llama-cpp-python` is intentionally not a hard dependency — it ships
with several CUDA/CPU build variants that can be finicky to install.

Smoke-test against a local file:

```python
from pathlib import Path
from soma.deploy.gguf_backend import build_gguf_chat_head

head = build_gguf_chat_head(
    gguf_path=Path("C:/Users/dant123/.cache/lm-studio/models/.../model.gguf"),
)
print(head.generate_text(prompt="Hello!", max_new_tokens=10))
```

CLI: pass `--gguf-path /path/to/model.gguf` to deploy scripts. The flag
overrides `--tier` and `--llm-name`.

### Using GGUF with chat_repl / demo_chat

Both frontends now accept the GGUF backend. Verbalizer construction is
skipped entirely -- user text is fed directly to `llama.cpp` as a string
prompt, so **online verbalizer training is not possible in this mode**
(the `ChatSession(gguf_head=..., online_trainer=...)` pairing raises at
construction).

```bash
# REPL with an explicit GGUF path:
python scripts/chat_repl.py \
    --soma-checkpoint artifacts/brain-bundle/ \
    --gguf-path ~/.cache/lm-studio/models/.../model.gguf

# Zero-arg demo, opt in via env var:
SOMA_GGUF_PATH=/path/to/model.gguf python scripts/demo_chat.py
```

SOMA's graph still runs (state evolves per turn), but its OUTPUT
activations are NOT projected into soft-prompt tokens. Operators who
want soft-prompt guidance should stay on the HF backend.

---

## References

- `docs/plans/2026-04-14-consumer-deploy.md` — Phase 7 design doc.
- `src/soma/deploy/devices.py` — tier registry + detection helpers.
- `src/soma/deploy/chat_head_factory.py` — `build_chat_head` factory.
- `src/soma/deploy/gguf_backend.py` — `build_gguf_chat_head` factory.
- `src/soma/deploy/cli.py` — shared `--tier/--device/--dtype/--llm-name/--gguf-path`
  argparse glue.
- `scripts/demo_chat.py` — zero-arg end-to-end demo.
- `scripts/chat_repl.py` — interactive REPL.
