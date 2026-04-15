# Accelerated Bootstrap — 2026-04-15 (wikitext-103)

**Run**: `artifacts/bootstrap-2026-04-14-accel-wt103/`
**Source SOMA**: `checkpoints/bootstrap-src-fresh-2026-04-15` (fresh, seed=42)
**LLM**: `Qwen/Qwen2.5-3B-Instruct` fp16 + int4 (bitsandbytes nf4)
**Corpus**: `data/wikitext-103-train.txt` (517 MB, 1.8M lines)
**Config**: 10,000 steps × batch_size=8 = 80,000 windows; cosine LR with 300-step warmup; soma_max_tokens=16; eval every 250 steps on held-out 20 windows
**Wall clock**: ~3 hours (vs last night's 3.4 hours for 20K windows — 4.8x faster per window)

## Verdict

Better-engineered training beats last night's loss floor: **3.118 at step 6000** vs last night's best of 3.146 at step 15500. But **SOMA contribution shrank** compared to last night — the improved trainer taught the verbalizer to project a useful prefix from ~any SOMA state rather than depend on SOMA's content. The architecture now needs a signal-bearing SOMA (pre-trained or jointly-trained) to show meaningful SOMA-dependence.

## Loss Curve

| Stage | Loss | Step |
|---|---|---|
| Pre-training (fresh verbalizer) | 4.169 | 0 |
| First training-step loss | 4.270 | 1 |
| First held-out eval | 3.477 | 250 |
| **Best held-out** | **3.118** | **6000** |
| Post-training held-out | 3.169 | 10000 |

Full trajectory in `artifacts/bootstrap-2026-04-14-accel-wt103/loss_log.csv` (10,000 rows,
one per micro-step).

Delta pre → best: **-1.05 nat** (4.17 → 3.12). Delta pre → final: -1.00 nat.

## Throughput

Profiled per-step (Qwen-3B int4, batch=8, tokens=16):

| Component | Time |
|---|---|
| SOMA × 8 | 759 ms |
| LLM forward | 159 ms |
| LLM backward | 185 ms |
| Everything else | < 4 ms |
| **Total step** | **1107 ms** |

Measured run-time throughput: **56 steps/min = 448 windows/min** (vs last night's
~100 windows/min at batch=1 fp16 → 4.5x real-world speedup). Before applying
`soma_max_tokens`, profiling showed SOMA was 90 % of step time; capping SOMA
at 16 evenly-spaced token embeddings per window unlocked the remaining gain.

## SOMA Contribution Ablation (N=100)

Re-ran the three-regime ablation on `verbalizer_best` (step 6000) against
the same bundle on wikitext-2 (100 windows), matching training's
`soma_max_tokens=16`:

| metric | real | zero | shuffle | Δ(zero) | Δ(shuffle) |
|---|---|---|---|---|---|
| loss | 3.4413 | 3.4492 | 3.4437 | +0.008 | +0.002 |

Last night's step_15500 ablation for comparison:

| metric | real | zero | shuffle | Δ(zero) | Δ(shuffle) |
|---|---|---|---|---|---|
| loss | 3.4722 | 3.4885 | 3.4967 | +0.016 | +0.025 |

**Tonight's SOMA dependence is ~half of last night's.** Possible drivers
(not separable without more runs):

- Token subsampling (16/70): SOMA sees fewer tokens per window, producing
  a narrower state distribution the verbalizer can collapse across.
- Stronger projector training (cosine LR + batching): the verbalizer's
  floor is lower in absolute terms, but more of that floor comes from a
  corpus-agnostic prior than from SOMA-specific routing.
- Cross-corpus eval: verbalizer trained on wt103, ablated on wt2 — tiny
  domain shift.

Verdict script emitted **"SOMA NOT CONTRIBUTING"** (threshold Δ(zero) ≤ 0.01).
Directionally the contribution is still positive (real < zero < shuffle is
*not* preserved tonight — shuffle is almost identical to real), but the
effect size is small enough that the automatic verdict fired.

Raw JSON: `artifacts/bootstrap-2026-04-14-accel-wt103/ablation_best_n100.json`.

## Sample Comparison (Fresh vs Trained, int4 Qwen-3B)

Same five fixed prompts as last night, greedy decoding, 40 new tokens.
Full output: `artifacts/bootstrap-2026-04-14-accel-wt103/samples_best_vs_fresh.md`.

Smoking gun: **trained outputs carry the wikitext-103 tokenizer signature**
(`@-@` hyphens, `= = Plot =` section headers, `$ 10 @.@ 5 million` numerals)
while fresh outputs read as natural prose.

Example, prompt `"The capital city of France"`:

- **Fresh**: `" is Paris. The capital city of Germany is Berlin. The capital
  city of Italy is Rome. The capital city of Spain is Madrid..."`
- **Trained**: `" , Paris , is the setting for the film . The city 's
  landmarks are used as backdrops in the film , including the Eiffel
  Tower and the Louvre . \n\n = = Plot ="`

The verbalizer clearly learned wt103's stylistic distribution. This is
orthogonal to the SOMA-contribution question — it's about the prefix's
capacity to bias the LLM, not whether SOMA's state is the reason.

## What This Tells Us

1. **The trainer speedups work and compound cleanly.** int4 + batch=8 +
   soma_max_tokens=16 gives ~4.5x real-world throughput without
   destabilising training.
2. **The verbalizer is strong, SOMA is weak.** With better training, the
   verbalizer approaches its capability floor (~3.12 nats) from near-any
   SOMA state. SOMA's specific content contributes only ~0.01 nat under
   this config, which is within the noise band of the eval.
3. **The next experimental move is to give SOMA something real to say.**
   Candidates, in order of tractability:
   - Pre-train SOMA on the same corpus (fresh integrator-fixed brain,
     20K-100K Hebbian steps on wt103) — the current pre-flight.
   - Joint SOMA + verbalizer training (gradients flow back into SOMA
     during bootstrap) — a more principled alternative.
   - Architectural fixes for the WM-never-writes and edge-decays-to-zero
     degradation modes surfaced by the `current.pt` diagnosis.

## Provenance

- Improved trainer: commit `687474d` (cosine LR + grad accum + save-best
  + CSV log).
- Seed-graph fix: commit `860cc82` (honors `initial_integrator_count`;
  every SOMA prior to this shipped with 0 integrators despite config).
- Batched forward + int4 quant in `--llm-name` path: commit `72a5df3`.
- `soma_max_tokens` perf cap: commit `a86e478`.
- Analysis-script device/quantization parity: commit `192d565`.
