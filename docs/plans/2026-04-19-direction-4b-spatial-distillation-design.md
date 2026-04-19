# Direction 4b design: Spatial distillation for semantic locality

**Date:** 2026-04-19
**Author:** autonomous research session (continuation)
**Parent:** `docs/plans/2026-04-19-direction-4a-design.md`
**Predecessor result:** `research/developmental/results/locomo_distill_phase3_findings.md` (NULL)
**Status:** design approved → implementation

## Motivation

Direction 4a (LLM-distilled input projections, 2026-04-19) landed a
clean negative on LoCoMo retrieval:

| System         | R@5   | Δ vs Chroma |
|----------------|-------|-------------|
| chroma-mxbai   | 0.349 | baseline    |
| soma-random    | 0.343 | −0.006      |
| soma-distilled | 0.334 | −0.015      |

Root cause diagnosed: distillation trained the input projections, but
retrieval reads the node fingerprint (dominated by Hebbian-trained
node weights) and the locality filter operates on *positions* that
distillation never touched. The teacher signal diluted across too
many architectural layers before reaching retrieval.

Separately, a **real degeneracy** exists in the Direction 4a loss: it
pulls `mean_i(W_i · summary)` toward `teacher_emb`, meaning all `W_i`
get gradient in the same direction. Per-node specialization collapses;
every node learns to produce the same mean-contribution output. This
likely amplified the null.

## Hypothesis

The v0.5 positive finding generalizes a design principle:

> Structural plasticity requires **structural priors** — orthogonal
> to the projection noise. Pattern-based priors on patterns generated
> by upstream randomness are insufficient.

Positional locality (synaptogenesis max-distance cutoff) is the one
such prior that's multi-seed-validated. Position-scramble probes
showed the mechanism: positions share rng state with projection
weights, so nearby positions = similar projections = useful edges.

**Direction 4b replaces "rng-correlated" with "semantically-correlated."**
If we train node positions to track their (now semantically-distilled)
projections, the locality filter operates in **semantic space**. Edges
admitted between nearby nodes are edges between semantically-similar
nodes. The same v0.5 mechanism, on a different substrate.

## Meta-principle (research program)

Direction 4b is the first instance of a broader frame: **spatially
re-examine every failed direction that relied on pattern-based priors.**

| Failed direction       | Pattern prior               | Spatial-analog to try         |
|------------------------|-----------------------------|-------------------------------|
| 1 — PE-supervised synap| Per-pair PE-delta EMA       | Position-to-PE-centroid gate  |
| 2B — learnable proj    | Pred-loss gradient          | Coupled position+proj (this)  |
| 3 — plasticity bcast   | Scalar PE-gain              | Per-region spatial gain       |
| 4a — LLM-distill proj  | Cosine teacher loss         | Coupled position+proj (this)  |
| graph rerank           | Fingerprint similarity      | Position-sim term in rerank   |

Direction 4b picks the most tractable (coupled position+projection)
as the first test. If the meta-principle generalizes, the others are
worth revisiting.

## Architecture

Each associator node has two trainable objects:
- `W_i` — input projection matrix (`sensor_dim × sensor_dim`). Already
  an `nn.Parameter` when `projection_mode="learnable"`.
- `p_i` — position vector (`position_dim`). Currently a plain tensor;
  becomes `nn.Parameter` when `position_mode="learnable"`.

Three losses combine:

1. **Prediction loss** (unchanged): MSE between `prediction_head(summary)`
   and actual next activation. Trains `prediction_head` and `W_i` (via
   the mean-projected view from Direction 2B).

2. **Competitive projection distillation** (new): only the top-K
   *most-activated* nodes per input receive a cosine-distance
   gradient toward the teacher embedding of that input. Different
   inputs activate different nodes → projections diverge per-node →
   specialization emerges → degeneracy broken.

   `α · (1 − cos(W_winner · summary, teacher.embed(text)))`
   averaged over `K` winners.

3. **Position coupling loss** (new): each position tracks a fixed
   random projection of its (detached) `W_i`:

   `β · Σ_i ||p_i − normalize(P · W_i.flatten())||²`

   Where `P: (sensor_dim² → position_dim)` is a frozen random matrix
   registered at init (buffer, not parameter). `W_i` is detached so
   gradient flows only into `p_i`; projections are trained by loss
   (2), not pulled by positions.

   Random `P` preserves *pairwise distances* under
   Johnson-Lindenstrauss — which is the property the locality filter
   actually reads. Absolute positions aren't semantic; relative
   structure is.

4. **Position norm preservation**: after each optimizer step, rescale
   each `p_i` to its initial L2 norm (~0.4 for 16-d `randn*0.1`).
   Keeps the 0.5 locality cutoff calibrated.

## Config surface

```python
# New / modified SOMAConfig fields
projection_distillation_target: Literal["none", "llm_embedding", "llm_spatial"] = "none"
projection_distillation_winners: int = 3  # top-K competitive distill; 0 = no-op (Direction 4a behavior)
position_mode: Literal["frozen_random", "learnable"] = "frozen_random"
position_coupling_weight: float = 1.0  # β

# Existing (unchanged):
# projection_mode, projection_lr, projection_distillation_weight,
# projection_distillation_model, projection_distillation_base_url,
# synaptogenesis_max_distance, position_dim, position_jitter
```

**Legal combinations (validated at config time):**
- `frozen + none` — legacy (default)
- `frozen + llm_embedding` — Direction 4a baseline (for ablation)
- `learnable + llm_spatial` — Direction 4b target

**Rejected combinations:**
- `learnable + none` — positions Parameter with no loss → drift
- `learnable + llm_embedding` — positions Parameter but no position
  loss → drift via spurious gradient

## Architecture components

**A. `SOMAConfig`:**
- 4 new fields + validation

**B. `PredictiveSOMA`:**
- Register `position_projector` as a buffer at init: `torch.randn(sensor_dim² , position_dim)`
- Convert positions to `nn.Parameter` when `position_mode="learnable"`; add to prediction optimizer
- Record each position's initial L2 norm for preservation
- Add `_ensure_position(node_id)` mirroring `_ensure_projection` for neurogenesis
- In `process_input`:
  - Select top-K winners for competitive distillation (reuse scoring from `_competitive_learning`)
  - Compute distill loss only on winners
  - Compute position coupling loss for all learnable positions
  - Combine: `total = pred_loss + α·distill_winners + β·position_coupling`
  - After `optimizer.step()`: rescale each position to its initial norm
- Update `save`/`load` to handle `nn.Parameter` positions (reconstruct on load, re-register with optimizer)

**C. `synaptogenesis.py`:** **no change.** Already reads
`node.position` via Euclidean distance. Once positions are semantic,
the filter operates on semantic space automatically.

**D. `SomaAdapter` (benchmarks):** add flow-through for
`projection_distillation_winners`, `position_mode`,
`position_coupling_weight`.

**E. `run_locomo_distill.py`:** add fourth system `soma-spatial` with
`target="llm_spatial"`, `position_mode="learnable"`, default β=1.0, K=3.

## Data flow per `process_input`

1. Embed input through SOMA (existing pipeline).
2. Compute activations + diversification + inhibition + competitive learning (existing).
3. Store text + fingerprint if `source_text` given (existing).
4. If `_last_summary` exists:
   - Compute prediction loss (existing).
   - If `target=="llm_spatial"` and teacher attached and `source_text` given:
     - `teacher_emb = teacher.embed(source_text)` (cached)
     - Score nodes by activation magnitude; select top-K
     - For each winner: `distill_loss += (1 − cos(W_winner · summary, teacher_aligned))`
     - `distill_loss /= K`, multiply by α
     - For each learnable `p_i`: `target_i = normalize(P · W_i.flatten().detach())`, `position_loss += ||p_i − target_i||²`
     - `position_loss *= β`
   - `total = pred_loss + distill_loss + position_loss`
   - `total.backward()`, `optimizer.step()`
   - Rescale each `p_i` to its initial norm (if learnable).

Gradient paths (sanity):
- Prediction loss → `prediction_head`, `W_i` (via projection-mean path)
- Distill loss → winner `W_i`'s only (via competitive gate)
- Position loss → `p_i`'s only (W_i detached)

No feedback loop between positions and projections: positions *follow*
projections; they don't pull them back.

## Experimental design

### Phase 1: Implementation (TDD)

~15 TDD cycles covering:
- Config: 4 tests (new fields accepted/validated, enum legality, combo rejection)
- Position init: 3 tests (`P·W_i` init path, initial-norm recorded, buffer registration for `P`)
- Gradient flow: 3 tests (position loss → `p_i`; no gradient to detached `W_i` via position path; distill gradient only to winners)
- Norm preservation: 2 tests (post-step rescaling, init-norm survives steps)
- Neurogenesis: 1 test (new node's position registered with optimizer)
- Integration: 2 tests (`process_input` no-NaN, backward compat when `target="none"`)
- Benchmark propagation: 1 test (SomaAdapter flows new fields)

Deliverable: all tests pass, synap_local v0.5 results unchanged.

### Phase 2: v0.5 sanity

Mirrors Direction 4a's Phase 2. Run multi-seed {0, 1, 42} on v0.5
capacity schedule with `llm_spatial` on. Check:
- No NaN, projections + positions finite.
- MSE within 1.5× of reference `synap_only_local`.
- Position-distance distribution changes meaningfully over training
  (unit check: semantic prior IS reshaping positions, not just
  adding noise).

### Phase 3: LoCoMo benchmark

Four systems (extends Direction 4a's three):
1. `chroma-mxbai` — baseline
2. `soma-random` — frozen projections, frozen positions
3. `soma-distilled` — learnable projections, frozen positions (Direction 4a)
4. `soma-spatial` — learnable projections + learnable positions (this)

Per-sample protocol unchanged. Same mxbai-embed-large teacher for all
systems' embedding step. Same corpus cache.

## Success criteria

**Primary:** `soma-spatial > chroma-mxbai` on R@5 by > 0.02 → SHIP.

**Secondary:** `soma-spatial > soma-distilled` on R@5 →
**mechanism-validation even without ship**: proves spatial channel
carries more signal than pure projection distillation. Publishable
as a refinement of Direction 4a's null.

**Tertiary 1:** with locality filter ON, `soma-spatial` beats
locality-OFF variant → confirms signal routes through locality filter
(not incidentally via a different path).

**Tertiary 2:** `soma-spatial` retrieve latency ≤ 32ms (2× soma-random's
16ms) → don't give up the latency win.

## Decision matrix

| Primary | Secondary | Action |
|---------|-----------|--------|
| pass    | pass      | SHIP Direction 4b, paper update, new default |
| fail    | pass      | Mechanism-valid negative. Publish as "spatial beats projection distill but not Chroma." Try Option C (pairwise distance distillation) or Option A (position-only) next. |
| fail    | fail      | Spatial distillation is itself a null. Pivot to different failed direction from the meta-principle table (e.g., spatial PE-supervised synap, spatial plasticity broadcast, spatial graph rerank). |
| pass    | fail      | Unusual outcome — suggests spatial helps but for reasons orthogonal to the coupling. Worth deep investigation. |

## Parameter sweeps (if primary partial)

- β ∈ {0.1, 0.5, 1.0, 2.0, 5.0} — position coupling strength
- K ∈ {1, 3, 5, 10, all} — competitive winners
- position_projector: random vs PCA-learned-once — Johnson-Lindenstrauss
  vs data-driven reduction
- Locality cutoff re-calibration given new position dynamics

## Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Competitive distill stalls when K too small (winner always the same node) | Medium | Activation-based scoring rotates winners across diverse inputs; add K>=3 check |
| Positions drift despite norm preservation (numerical drift) | Low | Monitor norms in unit test; recompute norm every N steps |
| Random `P` doesn't preserve distances well at `sensor_dim²=16384, position_dim=16` | Low | JL needs `k ≥ 8·ln(n)/ε²`; for n=50 nodes, ε=0.3, k=8·4/0.09 ≈ 355 — 16 is below this, but in practice works; measured via pairwise-distance-correlation sanity test |
| Save/load path breaks on learnable positions | Low | Mirror existing learnable-projection save/load pattern exactly |
| Latency regression from extra backward pass | Medium | Cache `W_i.flatten().detach()` once per step; avoid redundant re-flattens |

## Timeline

- Phase 1 (implementation): ~4-5 hours (15 TDD cycles, follows Direction 4a's ~3.5h template)
- Phase 2 (v0.5 sanity): ~1 hour (runner + multi-seed, parallels existing)
- Phase 3 (LoCoMo full + subset): ~1 hour (cache populated; 4 systems)
- Phase 3 sweep (if needed): ~1 hour (β sweep)
- Synthesis + docs: ~1 hour

**Total: ~8-9 hours**, matching Direction 4a's budget. Timeboxed:
if Phase 3 primary + secondary both fail, declare null and move to
next meta-principle candidate (no unbounded tuning).

## Open questions (decide during implementation)

1. **Per-node initial norms**: preserve individual (all different at
   init) or a shared mean? Individual is more faithful to init; mean
   is simpler. Lean toward individual.

2. **Distill on winners only vs winners + losers (anti-)**: Anti-
   distill could push losers' projections *away* from teacher,
   accelerating diversification. Skip for v1 (YAGNI); add if
   diversification is weak.

3. **Position optimizer LR**: separate from projection LR? Default
   to `projection_lr` (`1e-4`); separate only if needed.

## Future work (if B lands)

- **Option A (position-only distillation)**: skip the projection-distill
  loss, train only positions via a target derived from memory-specific
  teacher embeddings. Simpler; tests whether projections are necessary
  at all.
- **Option C (pairwise distance distillation)**: contrastive loss on
  `(||p_i − p_j||, ||teacher_i − teacher_j||)`. Most mathematically
  principled formulation of the v0.5 mechanism.
- **Option D (spatial rerank)**: add per-memory positions, use
  `position_sim(query_pos, memory_pos)` as a third rerank term alongside
  embedding and fingerprint.
- **Spatial re-examination of other failed directions**: per the meta-
  principle table.

## Exit criteria (repeat)

**SHIP:** Primary + Secondary pass. Update §4.7 / §5 of paper, promote
as the retrieval-ceiling-break result, new positioning.

**MECHANISM WIN:** Primary fail + Secondary pass. Publish as
refinement; iterate on β/K/projector design before giving up.

**STOP:** Primary + Secondary both fail. Move to next meta-principle
candidate. Documented null keeps infrastructure for Option C/D.
