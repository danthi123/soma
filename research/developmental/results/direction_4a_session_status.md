# Direction 4a session status — 2026-04-19

## Completed (Phase 1)

**Config:** commit `9041d18`
- `projection_distillation_target: Literal["none", "llm_embedding"]`
- `projection_distillation_model: str = "mxbai-embed-large"`
- `projection_distillation_base_url: str = "http://localhost:11434"`
- `projection_distillation_weight: float = 1.0`
- Full validation; defaults preserve legacy behavior.
- 8 TDD tests green.

**Teacher interface:** commits `de71346` (OllamaEmbedder), `cb72845` (CachedEmbedder)
- `LLMTeacher` Protocol + `OllamaEmbedder` HTTP adapter
- `CachedEmbedder` with sha256 memory + disk cache
- 10 TDD tests green

**Distillation loss hook:** commit `3daed4d`
- `PredictiveSOMA.attach_teacher(teacher)` — register teacher
- Distillation loss added to `process_input`: `alpha * (1 - cos(student, teacher.embed(source_text)))`
- Teacher dim alignment: truncate/pad to student sensor dim
- 6 TDD tests green; 54 developmental tests total pass.

**End-to-end smoke test:** commit `352a8a4`
- Real mxbai-embed-large teacher; gated on Ollama reachability
- Verified: 3 steps, projections finite, cache dedups

**Backward compat:** commit `fd0cbba`
- v0.5 synap_local smoke (seed=0, 3 variants): edges + synap events identical
- MSE drift within CUDA nondeterminism tolerance (+0.0005 to +0.0017)
- Qualitative ordering preserved

**Full suite:** 2481 passed, 44 skipped, 0 failed (246.95s)

## Completed (Phase 3 groundwork)

**Runner:** commit `fcaf7a0` / `ba4883d`
- `benchmarks/run_locomo_distill.py` — 3-system compare (chroma-mxbai / soma-random / soma-distilled)
- Uses PredictiveSOMA directly (not via MemoryLayer.attach_soma, which takes plain SOMA)
- `_score_queries` with per-category R@k — 5 TDD tests
- CATEGORY_NAMES iteration bug fixed in `run_locomo_locality.py` (commit `5e8bc93`)

**Adapter extensions:** commit `2384072`
- SomaAdapter accepts distillation params (flow-through; note: MemoryLayer path doesn't exercise distill)

## Plans

**Phase 2:** `research/developmental/env_sequence_v05_distill_multiseed.py` — RUNNING IN BACKGROUND
- 4 variants × 3 seeds = 12 runs
- 8 cached embeddings (1 per regime) → at least past 3rd variant
- Tee'd log file buffered; Python PID 17020 active with ~5GB VRAM

**Phase 3:** `benchmarks/run_locomo_distill.py` — READY TO RUN after Phase 2
- Command: `python -m benchmarks.run_locomo_distill --max-samples 2 --out benchmarks/reports/locomo_distill_subset.md`

**Phase 5:** `docs/plans/2026-04-19-direction-4a-phase-5-stress-test.md` — DRAFT
- Multi-hop / temporal stress test with decision gates

## Open issues

1. **Phase 2 output buffering**: `tee` through pipe is block-buffered. Results will appear only when the run finishes, or when stdout is explicitly flushed. Fix for next time: use `python -u` or add `sys.stdout.flush()` after print statements.

2. **MemoryLayer + distillation mismatch**: `MemoryLayer.attach_soma` takes plain SOMA, not PredictiveSOMA. Distillation loss lives in `PredictiveSOMA.process_input`. Phase 3 runner works around this by using PredictiveSOMA directly. Longer-term: either push distillation into core SOMA or add `attach_predictive_soma` to MemoryLayer.

## Next actions (on resume)

1. Check Phase 2 completion: `wc -c research/developmental/results/env_sequence_v05_distill_multiseed.log`
2. Read results: `cat research/developmental/results/env_sequence_v05_distill_multiseed.log`
3. Commit Phase 2 findings
4. Kick off Phase 3 quick subset: `python -m benchmarks.run_locomo_distill --max-samples 2`
5. If subset shows signal → full Phase 3 run (all 10 samples)
