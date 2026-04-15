# First Real Verbalizer Bootstrap — 2026-04-15

**Run**: `artifacts/bootstrap-2026-04-15-qwen3b-wikitext2/`
**Source SOMA**: `checkpoints/bootstrap-src-fresh-2026-04-15` (fresh tiny brain, seed=42)
**LLM**: `Qwen/Qwen2.5-3B-Instruct` (fp16 on RTX 3090)
**Corpus**: `data/wikitext-2-train.txt` (10.5 MB, 36,718 lines)
**Steps**: 20,000 (3h22m wall, ~99 steps/min)
**Checkpoints**: 40 + final (every 500 steps)
**Trainer**: Adam, lr=1e-4, num_prefix_tokens=8, eval_samples=20

## Verdict

**The architecture works.** First concrete evidence the SOMA → verbalizer
→ frozen-LLM pipeline learns from real text. Held-out LM loss dropped
**33% (5.09 → 3.40)** with no fine-tuning of the LLM itself — only the
verbalizer's projector saw gradient.

## Numbers

| Stage | Loss |
|---|---|
| Pre-training (untrained verbalizer) | **5.0886** |
| First training-step loss | 4.6963 |
| Step 500 (held-out) | 3.8222 |
| Step 15,500 (held-out, **best**) | **3.1464** |
| Step 20,000 (held-out) | 3.4015 |
| Final training-step loss | 3.2396 |
| Post-training held-out | 3.4015 |

## Loss Curve Phases

- **Steps 0–500**: most of the work. 5.09 → 3.82 (-25%). The randomly-
  initialized projector finds the rough shape of useful prefixes within
  the first 500 gradient steps.
- **Steps 500–3000**: slower refinement, 3.82 → 3.28. Picks up another
  10% of the total improvement.
- **Steps 3000–20000**: oscillates between ~3.15 and ~3.65, mean around
  3.30. The single-sample batches and the constant-LR Adam optimiser
  produce noisy late-stage trajectories.

Best checkpoint is `verbalizer_step_15500` at loss 3.1464; the
`verbalizer_final` (step 20000) ended at 3.4015 due to a late-run
oscillation, not divergence.

Full table: `artifacts/bootstrap-2026-04-15-qwen3b-wikitext2/loss_curve.md`.
Raw CSV: `artifacts/bootstrap-2026-04-15-qwen3b-wikitext2/loss_curve.csv`.

## What This Tells Us

- **Pipeline plumbing is correct.** No NaN / Inf, no dtype crashes
  (the T3 cast plus the new `compute_lm_loss` device alignment held
  through 20K steps of fp16 + fp32-verbalizer training).
- **SOMA state is being read.** The verbalizer is learning to project
  whatever SOMA's OUTPUT activations encode into prefix embeddings the
  LLM finds useful. Whether that "useful" carries semantic content from
  SOMA or is just generic LM-helpful noise is the question for the next
  experiment.
- **Most learning is in the first ~3K steps.** A future bootstrap could
  reasonably stop at 5K and avoid the noisy plateau; for production we'd
  want LR decay + gradient accumulation to push the floor lower.

## Open Questions / Next Steps (Brainstorm, not committed)

- **Is SOMA actually contributing?** Run an ablation with a constant
  zero-tensor SOMA state and compare. If loss is similar, the verbalizer
  is just memorising "wikitext is mostly Wikipedia-like" — not useful.
- **Pick the best checkpoint, not the last.** Promote step_15500 (or
  best-by-eval) to the canonical "trained verbalizer" artifact.
- **Improve the trainer**: cosine LR schedule, gradient accumulation
  across windows (effective batch size 8-16), early stopping on eval.
- **Scale up**: same flow on the 9B `xlarge` tier (Qwen3.5-9B) once
  transformers >= 4.57.0.dev0 is installed. Or stay on 3B and run on
  a larger / more dialogue-shaped corpus.
- **Sample-generation A/B**: compare fresh vs trained verbalizer on
  fixed prompts. Not as rigorous as loss but more visceral evidence of
  what the verbalizer actually learned to do.

## Infrastructure Landed Tonight

- `scripts/eval_bootstrap_checkpoints.py` — post-hoc loss curve eval.
- `src/soma/training/verbalizer_bootstrap.py::compute_lm_loss` — fp16
  dtype + device alignment fix (mirroring Phase 7 T3/T6).
- `/gpu-pause` + `/gpu-resume` skills (in skills-repo, pushed) for the
  next time we want to game while training runs.

Run completed without manual intervention while the operator played
League. GPU contention slightly slowed the pace (109 → 99 steps/min)
but did not destabilise training.
