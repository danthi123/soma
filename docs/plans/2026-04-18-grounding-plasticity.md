# Grounding SOMA's Plasticity in Non-Arbitrary Signals

**Date:** 2026-04-18
**Parent:** [developmental-next-steps.md](developmental-next-steps.md) Phase 15
**Status:** Plan (pre-implementation)

## Motivation

The v0.5 experimental cascade converged on a single root cause
shared by both the retrieval ceiling (§5 of the paper) and the
adaptation-track plasticity failure (§4.7). SOMA's core
representations — the 3 lateral-inhibition winners per input —
are **structurally diverse but semantically arbitrary** because
they are produced by random projections. Any plasticity rule
that reads those activations as its signal (synaptogenesis in
particular, Hebbian weight updates in general) ends up
reinforcing arbitrary structure, not useful structure.

In biology, plasticity decisions are anchored in signals with
external reality behind them: sensory input through evolved
priors, neuromodulator broadcasts tied to reward/novelty,
prediction error against the actual world, critical-period
scheduling. SOMA has the mechanical scaffolding (pruning,
consolidation, PE computation) but lacks the grounded
signals. This plan proposes three directions — ordered by
ambition and independent enough to run in parallel — that each
attempt to introduce an external anchor into at least one
plasticity decision.

The common success criterion across all three: on the v0.5
capacity schedule (where the current `full` variant loses 5-10×
to `no_growth`), does the modified variant recover ground on
at least 4 of 8 regimes while still growing a meaningfully
larger graph? If none of the three succeeds, the honest
conclusion is that SOMA's plasticity mechanisms cannot be
made useful under the current substrate and the research
focus should shift to the substrate itself (wave execution,
residual MLPs, homeostatic gain — none yet independently
ablated).

---

## Direction 1 — PE-supervised synaptogenesis (lowest-lift)

### Hypothesis

Synaptogenesis damages prediction quality because it reads
activation correlations that come from random projections,
then installs edges based on those arbitrary co-activations.
If we gate new-edge admission by whether the proposed pair has
previously co-occurred *in a way that reduced prediction error*,
we filter out the arbitrary co-activations and keep only edges
that predict usefully.

### Mechanism design

Current synaptogenesis (simplified):

```
every N steps:
    for each pair of nodes (a, b):
        if coactivation(a, b) > threshold and not has_edge(a, b):
            add_edge(a, b, weight ~ small_randn)
```

Proposed PE-supervised variant:

```
every N steps:
    for each candidate pair (a, b):
        if coactivation(a, b) > threshold and not has_edge(a, b):
            # only admit if co-activation is correlated with
            # predictive improvement
            if pe_delta_when_coactive(a, b) < 0:
                add_edge(a, b, weight ~ small_randn)
```

`pe_delta_when_coactive(a, b)` requires maintaining a running
estimate of prediction error conditional on each pair's
co-activation pattern. Cheapest implementation: per-pair EMA of
`(pe_at_step_t+1 - pe_at_step_t) * indicator(a and b both
active at step t)`. Pairs whose co-activation precedes
*drops* in PE get positive evidence; pairs whose
co-activation precedes *rises* in PE get negative evidence.

### Config surface

```python
synaptogenesis_supervision: Literal["none", "pe_conditional"] = "none"
synaptogenesis_pe_ema: float = 0.99
synaptogenesis_pe_threshold: float = -0.001  # negative == PE dropped
```

Default behavior unchanged. Opt-in via `pe_conditional`.

### Bookkeeping complexity

Per-pair PE-delta EMA has `O(N^2)` storage where `N` is the
associator+integrator node count. At current scales (≤50
nodes) this is 2,500 floats — trivial. We only need to track
pairs that cross the co-activation threshold, so it's
sparser in practice.

### Experimental design

Compare on the v0.5 capacity schedule:
- `full` (reproduces commit `2a1bbcc` `full`)
- `synap_pe` (this new variant, with
  `synapse_pe_threshold = -0.001`)
- `neuro_only` (reference for best-case with synap off)
- `no_growth` (reference baseline)

Secondary analysis: look at the edges that PE-supervised
synaptogenesis *admitted* vs the edges that raw synaptogenesis
would have admitted. Are they disjoint sets or overlapping?
The interesting case is partial overlap — i.e., the
supervision is actually filtering something.

### Success criteria

- `synap_pe` outperforms `synap_only` (2a1bbcc numbers) on at
  least 6 of 8 regimes.
- Ideally: `synap_pe + neuro` outperforms `full` on at least 4
  of 8 regimes, and narrows the gap with `no_growth`.

### Risk and reversibility

- PE-EMA bookkeeping adds compute but is easy to short-circuit
  with a config flag.
- The change is purely additive on the admission side of
  synaptogenesis — can be reverted by flipping
  `synaptogenesis_supervision="none"`.

### Effort estimate

~1 day of implementation, ~1 day of experiments.

---

## Direction 2 — Task-supervised input projections (medium-lift)

### Hypothesis

The random projections at each node (SENSOR input, associator
input) are the ultimate source of the structural-without-
semantic failure. If we replace them with projections trained
against a *task* signal (specifically: predicting the next
observation), the downstream lateral-inhibition winners will
encode useful similarity rather than arbitrary hashing. Both
synaptogenesis and retrieval should benefit.

A previous attempt (Phase 4b, contrastive FT against graph
topology) was catastrophic because the training target was
itself the arbitrary signal — we were training the encoder
to match random garbage. The new formulation avoids this trap
by training projections against a grounded target (next-
observation prediction) rather than against the graph's own
topology.

### Mechanism design

Each associator node has an input projection
`Linear(sensor_output_dim, associator_input_dim)` that is
currently randomly initialized and frozen. The new variant
makes these projections *learnable* and trains them with the
prediction-head's gradient:

```
loss = mse(predicted_next_obs, actual_next_obs)
loss.backward()
# gradient flows through:
#   prediction head -> integrator outputs -> integrator inputs
#   -> associator outputs -> associator MLPs -> associator inputs
#   -> associator input projections  <- UPDATED
```

Currently these projections are intentionally detached to keep
graph structure static; we undo that detach only for this
variant.

### Config surface

```python
associator_projections_learnable: bool = False
sensor_projections_learnable: bool = False
projection_lr_multiplier: float = 0.1  # keep slower than main weights
```

Default behavior unchanged.

### Why this is different from Phase 4b

- **Phase 4b target**: graph topology (which nodes fired
  together) — an arbitrary signal.
- **Phase 2 target**: actual next observation — a grounded
  signal.

Phase 4b destroyed the encoder because it pushed the encoder's
geometry toward arbitrary structure. This variant pushes the
*graph's internal* projections toward structure that predicts
reality. The encoder stays frozen.

### Experimental design

Primary comparison on the v0.5 capacity schedule:
- `full_fixed_proj` (current default; reproduces `full` from
  commit `2a1bbcc`)
- `full_learnable_proj` (this variant)
- `no_growth_fixed_proj` (reference)
- `no_growth_learnable_proj` (isolates whether the benefit is
  from growth interacting with learnable projections, or just
  from projections learning at all)

Secondary: retrieval cross-check. Build a graph under
`full_learnable_proj` on LoCoMo and rerun the gated-hybrid
retrieval experiment (Phase 3/10). If the §5 ceiling is
downstream of the arbitrary-projections problem, learnable
projections should lift that ceiling too.

### Success criteria

Primary (adaptation):
- `full_learnable_proj` beats `full_fixed_proj` on ≥ 5 of 8
  v0.5 regimes.
- `no_growth_learnable_proj` matches or beats
  `no_growth_fixed_proj` (we need to know whether the benefit
  requires growth or not).

Secondary (retrieval):
- Gated-hybrid delta on LoCoMo slice A improves from ~+2 to
  ≥ +5. If this happens, it's a genuine crack in the retrieval
  ceiling and worth a paper update.

### Risk and reversibility

- Learnable projections could destabilize training. Mitigation:
  clip projection updates, use a smaller LR multiplier, start
  from the frozen initialization.
- If the gradient path has latent detaches we don't know
  about, no training will happen (Phase 4 failure mode).
  Mitigation: add an assert that logs non-zero projection
  gradients within the first 100 steps; fail fast if none
  detected.
- Reversible via config flag.

### Effort estimate

~2 days of implementation (detach audit + gradient plumbing +
tests), ~1 day of experiments + retrieval replay.

---

## Direction 3 — Neuromodulator-style plasticity broadcast (highest-lift)

### Hypothesis

Biological neuromodulators (dopamine, acetylcholine) act as
slow scalar broadcasts that gate *when* plasticity happens,
with the broadcast level tied to surprise, reward, or novelty.
SOMA currently has plasticity firing continuously at
interval-based rates. Adding a scalar `plasticity_gain`
channel that rises on PE spikes and falls during stable phases
would restrict all plasticity (Hebbian updates + synaptogenesis
admission + neurogenesis triggering) to moments when the
environment is providing informative surprise, and suppress
it during steady-state periods where "learning" just
reinforces ambient noise.

This is a structural change to how plasticity is scheduled;
it's independent of directions 1 and 2 and composes cleanly
with either.

### Mechanism design

- Add a scalar state on SOMA:
  `self.plasticity_gain: float` initialized at 1.0, bounded in
  `[0.1, 3.0]`.
- Each step, update the gain from a moving average of recent
  prediction error:
  ```
  pe_spike = recent_pe_mean / baseline_pe_mean  # same ratio
                                                # neurogenesis uses
  plasticity_gain = alpha * plasticity_gain + (1-alpha) * pe_spike
  ```
- Multiply `hebbian_lr` and `synaptogenesis_rate` by
  `plasticity_gain` each step.
- Optionally gate neurogenesis by a minimum `plasticity_gain`
  threshold (different from `neurogenesis_mode="pe_gated"`
  because the broadcast is graph-wide, not per-event cooldown).

### Config surface

```python
plasticity_broadcast_mode: Literal["off", "pe_scaled"] = "off"
plasticity_broadcast_alpha: float = 0.95
plasticity_broadcast_min_gain: float = 0.1
plasticity_broadcast_max_gain: float = 3.0
```

Default off. Opt-in flips on PE-scaled broadcast.

### Why this isn't redundant with pe_gated neurogenesis

`neurogenesis_mode="pe_gated"` gates the *decision to fire* one
specific growth event, with a cooldown. `plasticity_broadcast`
modulates the *strength* of *all* plasticity rules
continuously, without cooldowns. They compose: pe_gated
controls when neurogenesis events get evaluated;
plasticity_broadcast scales how much Hebbian learning and
synaptogenesis happen at any given step.

### Experimental design

2×2 on the v0.5 capacity schedule, with growth at the
developmental defaults:
- `broadcast_off / full_growth` (reproduces commit
  `2a1bbcc` `full`)
- `broadcast_on / full_growth` (does the broadcast reduce
  plasticity damage?)
- `broadcast_off / neuro_only` (reference; already known good)
- `broadcast_on / neuro_only` (does the broadcast interact
  with the one good mechanism?)

### Success criteria

- `broadcast_on / full_growth` improves over `broadcast_off /
  full_growth` on ≥ 5 of 8 regimes.
- `broadcast_on / neuro_only` does not regress relative to
  `broadcast_off / neuro_only` (we need to confirm the
  broadcast doesn't break the one good mechanism).

### Risk and reversibility

- If the broadcast is mis-tuned (e.g., gain lower-bound too
  aggressive), plasticity could effectively turn off and
  `broadcast_on` degenerates into `no_growth`. Mitigation:
  log broadcast gain per step and confirm it moves in the
  expected range.
- Reversible via config flag.

### Effort estimate

~2 days of implementation (broadcast state + hook into
homeostasis-style updates + tests), ~1 day of experiments.

---

## Running order

All three directions are independent and opt-in, so they can
be implemented in parallel. If pipelined, the rational order
is:

1. **Direction 1 first** (lowest lift, highest direct
   engagement with the diagnosis). If PE-supervised
   synaptogenesis works, it's the cleanest single-mechanism
   fix and warrants its own paper follow-up.
2. **Direction 3 second** (composes with everything). The
   broadcast result is valuable regardless of direction 1's
   outcome — if direction 1 fails, the broadcast tells us
   whether plasticity *scheduling* alone is enough; if
   direction 1 succeeds, the broadcast may compound the
   benefit.
3. **Direction 2 last** (highest lift, most potential
   upside). If it works, it affects both retrieval and
   adaptation tracks — a potential crack in the §5 ceiling.
   But it's also the riskiest in terms of training stability
   and needs the most testing before running the full sweep.

## Gating on v0 2×2 cross-check

Currently in flight at commit `64c70c5`. If the v0 2×2 result
reproduces the v0.5 finding (neuro fine, synap bad), all three
directions remain well-motivated. If the v0 result diverges
(e.g., neuro also hurts on v0), we should pause and understand
that divergence before shipping any of the three — since they
all assume the v0.5 diagnosis generalizes.

## Out-of-scope for this plan

- Structure-matched deep MLP control (E3 from AUDIT). Separate
  experiment; doesn't depend on these three directions.
- Seed-variance sweeps for existing v0.5 results. Should happen
  before publication regardless; not blocking here.
- Retrieval-track experiments unrelated to projection changes.
  Retrieval is effectively dormant (§5 ceiling diagnosed) and
  reopens only if direction 2 shows signs of cracking it.
