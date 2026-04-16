# Hybrid Pivot — 2026-04-12 to 2026-04-15

**Context:** BUILD_COMPLETE.md closed out original SOMA on 2026-04-12.
This doc covers what shipped since.

## The Pivot (what and why)

At step ~800K the autonomous improvement loop reached its terminal
state: stable, growing, but with a linear `TextDecoder` whose output
never rose above corpus-statistics garble. SOMA-alone as a language
system had no remaining cheap wins. On 2026-04-14 the operator
confirmed the JARVIS-style consumer-assistant direction, and the
architecture pivoted to **hybrid**: SOMA keeps all learning, a frozen
small transformer provides fluent language, and a thin trainable
projection head (the SomaVerbalizer) translates between them. The
split is deliberately lopsided — SOMA is the user-owned portable
brain; the verbalizer is the cheap replaceable piece when the
transformer is swapped.

## What shipped

### Phase 1 — Brain Portability Foundation (a4c89be, 10b4017)
Versioned schema envelope (`SCHEMA_VERSION=1`), directory-bundle
save/load with tokenizer + encoder/decoder sidecars + `manifest.json`
(interface_spec), append-only growth journal, legacy-checkpoint
migrator, `DevelopmentSchedule.periods` persisted.

### Phase 2 — SomaVerbalizer Interface (26b1e61)
`VerbalizerSpec` frozen dataclass; `SomaVerbalizer` module
(Linear → LayerNorm → GELU → Linear, near-zero final-layer init);
`SomaAggregator` mean-pool to `(B,128)`; directory-format save/load
integrated into brain bundle as `verbalizer/`; `fallback_text`
delegates to the existing `TextDecoder`; swap-compatibility contract
test (spec-mismatch rejected).

### Phase 3 — ChatHead & First Transformer (3f2f745)
`transformers`/`accelerate` under `[dev-chat]`; `ChatHead` wraps any
HF causal LM with `requires_grad=False`; `generate_text` via
`inputs_embeds`; position-id and attention-mask helpers;
`SOMA.chat()` end-to-end with verbalizer-fallback path; smoke test
against real SmolLM2-360M.

### Phase 4 — Verbalizer Bootstrap Training (72b4549)
`VerbalizerTrainer` with freeze-invariant check; `compute_lm_loss`
teacher-forced CE with `labels=-100` on prefix; no-grad
`text_to_state`; `train` loop + `eval_lm_loss`; CLI script; non-finite
loss guard (cdff284) skips backward+optim instead of poisoning Adam.

### Phase 5 — Multi-Turn Chat with WM (cc55c06)
`ChatSession` + `ChatTurn` with optional `system_prompt` pre-warm;
`respond()` closes the loop by re-feeding the assistant response back
through SOMA; `save/load_history` extending the brain bundle with
`chat_history.json`; interactive REPL `scripts/chat_repl.py`.

### Phase 6 — Online Verbalizer Training (0a52c13)
`ChatExchange` + fixed-size `ReplayBuffer` (text-only);
`OnlineVerbalizerTrainer.step()` with rolling-window divergence
monitor (freeze-on-divergence, no auto-rollback yet); optional hook
in `ChatSession.respond`; `online_state.json` sidecar for resume.

### Phase 7 — Consumer GPU Deployment (d1dd9aa + Tracks A-F)
`soma.deploy` module (`detect_cuda_vram`, `select_device_and_dtype`,
`auto_select_tier`, `MODEL_TIERS` tiny/small/large/xlarge);
`build_chat_head` factory; prefix dtype alignment before concat so
fp16 LLMs don't silently promote; `--tier auto` CLI. Tracks A-F
added Gemma-4-E4B `large` / Qwen3.5-9B `xlarge`, `vram_safety_factor`,
optional bitsandbytes int4/int8, GGUF inference-only backend + demo
wiring, real int4 CUDA smoke, `inspect_soma.py` diagnostics CLI.
`demo_chat.py` is the zero-arg end-to-end runner.

### Perf & fixes landed 2026-04-14/15
- Wave-batching executor (62e4b1a): 3-5x faster, identical outputs,
  flag-selectable via `SOMAConfig.use_batched_executor`.
- Cosine LR + warmup + grad accumulation + save-best + CSV log in
  bootstrap (687474d).
- Seed-graph fix (860cc82): `config.initial_integrator_count` was
  silently ignored — every SOMA shipped with 0 integrators. Now
  honored.
- Batched forward + int4 in bootstrap (72a5df3).
- `soma_max_tokens` cap on the serial SOMA.step loop that had been
  eating ~90% of step wall time (a86e478).
- `compute_lm_loss` dtype + device alignment (ef73009).

## First empirical results (2026-04-15)

Bootstrap on Qwen2.5-3B-Instruct + wikitext-2 (fresh tiny SOMA,
20K steps, 3h22m on the 3090): held-out LM loss dropped 33%
(**5.09 → 3.40**, best-by-eval 3.15 at step 15.5K) with only the
verbalizer's ~8M-param projector receiving gradient.

SOMA-contribution ablation at N=100 eval windows across four
checkpoints: the gap between real SOMA state and the `zero` /
`shuffle` baselines **grows** with training (Δ 0.006 → 0.025 nats by
step 20K). SOMA's marginal contribution is small (~1% of the 1.6-nat
reduction) but directionally correct and trending up over training
time even though SOMA itself was never gradient-updated. The
architecture plumbing is sound. Full numbers:
`reports/bootstrap-2026-04-15-qwen3b-wikitext2.md`.

## Open questions / deferred

- **Pre-train SOMA on the same corpus** before bootstrapping the
  verbalizer. Current OUTPUT activations encode only noise; richer
  signal should raise the ceiling of the SOMA-contribution ablation.
- **Joint SOMA + verbalizer training.** Currently SOMA stays frozen
  during bootstrap; `Node.last_activation.detach()` is the guard.
  Relaxing that is a distinct experiment.
- **Dialogue-shaped corpora.** Wikitext windows are independent, so
  SOMA's WM/episodic memory has nothing useful to carry. A sequential
  narrative or real chat corpus should make WM earn its keep.
- **Auto-rollback on divergence.** Phase 6's monitor freezes training
  and logs; rollback is still manual.
- **Structural regeneration.** `current.pt` at 1.6M steps is
  structurally degraded (0 integrators, 97% weak edges, empty WM,
  2 gain-pinned nodes); diagnosis in session context. Recovery path
  not yet designed.

## Completed (2026-04-15 evening)

Accelerated wikitext-103 bootstrap completed: best loss 3.118 at
step 6000 (details: `reports/bootstrap-2026-04-15-accel-wt103.md`).
Joint SOMA+verbalizer training also completed — tied with frozen
baseline, confirming the projector-prior was the bottleneck.

The project has now pivoted to **agent-memory-layer positioning** as
of commit `06a9ce9`. The pivot rationale and Stage 2-4 roadmap live
in `docs/plans/2026-04-15-memory-layer-pivot.md`. The product-facing
story is in `docs/positioning.md`.

Stage 2 (MemoryLayer API) landed: `soma.memory.MemoryLayer` with
store/retrieve/persist/reload, 16 unit tests passing, two demo
scripts (`demo_memory_layer.py`, `demo_chat_persistent.py`).
