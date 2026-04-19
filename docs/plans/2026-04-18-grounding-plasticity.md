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
signals. This plan proposes four directions — ordered by
ambition and independent enough to run in parallel — that each
attempt to introduce an external anchor into at least one
plasticity decision.

Two framings for "what counts as a grounded signal":

- **Directions 1-3** use only the environment's own
  prediction-error signal as the anchor. They test whether
  SOMA's plasticity can work under the current "no LLM in the
  core loop" design principle. This is the research-paper
  framing: if the 2×2 finding generalizes, these directions
  test whether plasticity *can* be fixed without external
  supervision; if none works, the honest conclusion is that
  it can't.
- **Direction 4** uses a pretrained SOTA LLM as a teacher/
  judge, introducing semantic grounding from outside SOMA
  itself. This is the product-pivot framing: SOMA as an
  agent-memory layer serving an LLM already has the LLM
  available, so using it as a training signal is natural and
  cheap at inference time.

The common success criterion across all four: on the v0.5
capacity schedule (where the current `full` variant loses 5-10×
to `no_growth`), does the modified variant recover ground on
at least 4 of 8 regimes while still growing a meaningfully
larger graph? If none succeeds, the honest conclusion is that
SOMA's plasticity mechanisms cannot be made useful under the
current substrate and the research focus should shift to the
substrate itself (wave execution, residual MLPs, homeostatic
gain — none yet independently ablated).

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

### Phase 1 progress log (2026-04-18)

**Implemented** (commit 367051e):
- `SOMAConfig.synaptogenesis_supervision` with options `"none"`
  (default) and `"pe_conditional"`, plus `synaptogenesis_pe_ema_alpha`
  (0.99), `synaptogenesis_pe_threshold` (0.0), and
  `synaptogenesis_pe_min_observations` (5).
- `SOMA._synap_pe_ema`, `_synap_pe_counts`, `_last_pe` per-step
  bookkeeping. EMA keys are sorted `(a, b)` tuples so the evidence
  slot is symmetric.
- `synaptogenesis()` gains `pe_ema` / `pe_counts` kwargs; per-pair
  admission gate in the ordered-pair loop skips candidates BEFORE the
  rng draw (so `supervision="none"` stays bit-exact).
- Serialization round-trip; legacy checkpoints load with empty dicts.
- Tests: 23 new (16 in `test_pe_supervised_synaptogenesis.py`, 7 in
  `TestPeConditionalGate`). Full suite 2378 pass, 0 fail.
- Runner: `research/developmental/env_sequence_v05_synap_pe.py` with
  5 variants (no_growth / synap_only / synap_only_pe / full / full_pe).

**Implementation note — degeneracy finding**:
Probe in isolation showed that on the 14-associator v0.5 graph, every
pair is co-active every step (sensor projections wake all associators
uniformly). Consequence: the per-pair EMA receives the same delta for
every pair on every step, so all 91 EMAs become identical. The
"per-pair" structure collapses to a single global PE-trend EMA, which
serves as a GLOBAL pause/resume gate on synap rather than a per-pair
discriminator.

This is still potentially useful behavior ("don't wire during PE
spikes") but is not the mechanism the plan doc originally envisioned.
If the current experiment shows null:
- Variant 1b: coactivation-weighted EMA (`weight = mag[a] * mag[b]
  / sum(all_coacts)` so uniformly-active pairs get equal tiny weight;
  still degenerate under uniform activation).
- Variant 1c: per-pair regression slope tracking — maintain
  `sum_x, sum_y, sum_xy, sum_x2, n` per pair, use the regression
  slope of `pe_delta ~ coact_strength` rather than unconditional
  EMA. Handles uniform case gracefully via `var(x) = 0` fallback.
- Variant 1d: sparsify activation first — lateral inhibition
  `k_winners=2` so different pairs co-activate on different steps.
  Feeds per-pair variance into the EMA from the input side rather
  than the accounting side.

**Threshold note**: Plan doc specified `-0.001`. Phase 1 runner uses
default `0.0` (admit if any PE drop). Smoke test showed typical EMAs
near ±0.0005, so `-0.001` would filter most pairs. Phase 1 is the
permissive run; if it shows signal, a threshold sweep follows.

**Phase 1 results (2026-04-18, seed=42, threshold=0.0)**:

Plan-doc success criteria scorecard:

- `synap_only_pe` beats `synap_only` on **7 of 8 regimes** (needed
  6+): **PASS**. Mean improvement −0.0005 MSE.
- `full_pe` beats `full` on **2 of 8 regimes** (needed 4+): **FAIL**.
  Supervision actively regresses the neurogenesis-composed config,
  some regimes by +0.0023 MSE (+47% relative on mlp_4x64).
- Gap to `no_growth` narrowed on hard regimes for synap_only_pe
  but not closed (mlp_4x64: 0.0065 → 0.0056 vs no_growth 0.0010).

Detailed findings in
`research/developmental/results/env_sequence_v05_synap_pe_findings.md`.

Why the split? The `min_observations=5` cold-start gate blocks synap
from wiring pairs that include recently-created neurogenesis nodes
until they accumulate 5 observations. Under the full config,
neurogenesis produced 31 new nodes and the rolling cold-start window
suppressed a lot of useful early admissions. Under synap_only, 91
pairs stabilize within regime 0 and the gate thereafter exercises
its full behavior.

**Degeneracy empirically confirmed**: `ema_pairs=91` at end of
synap_only_pe — exactly 14 choose 2 — all moving together.
`ema_pairs=990` at end of full_pe, reflecting neuro-added slots.
ema_neg% fraction swings uniformly across pairs (100 / 0 / 100 / 0
across regime boundaries), never partial — confirming all pairs
share the same effective EMA.

**Next probes in priority order**:
1. Neurogenesis-interaction fix: waive cold-start for pairs whose
   newest node is younger than `neurogenesis_cooldown` steps. Test:
   rerun with the fix and see if `full_pe` recovers toward `full`.
2. Threshold sweep: `synaptogenesis_pe_threshold=-0.001` (plan-doc
   originally specified) vs current `0.0`. Stricter might close
   more of the no_growth gap on hard regimes.
3. Seed robustness: reproduce the 7-of-8 signal on seeds {0, 1, 2}
   before investing in further mechanism design.
4. Move to Direction 3: since Direction 1 already reduced to a
   global PE-trend gate on synap, the principled generalization is
   to broadcast that gate to Hebbian + neurogenesis too
   (Direction 3's design). Most informative comparison: Direction 1
   fixed-for-neuro vs Direction 3 applied to the same axes.

**Decision**: Direction 1 Phase 1 succeeded on its primary claim.
Further Phase-1 refinement (probes 1–3) is bounded-effort; commit
to one more Phase 1 iteration (probe 1) before transitioning to
Direction 3.

### Phase 1 multi-seed validation (2026-04-19) — FAILED to reproduce

Phase 1.1 (waiver v1): initial-seed bug, regressed all 8 regimes.
Phase 1.2 (waiver v2): seed-exclusion fix; waiver turned out to be
a no-op (fresh cold-start window only ~5 steps wide on this graph).
Default reverted to grace=0.

Phase 1.3 (multi-seed on Phase 1 config, seeds {0, 1, 42}):

| Scorecard | seed=0 | seed=1 | seed=42 |
|-----------|--------|--------|---------|
| synap_only_pe wins | 1/8 | 1/8 | 8/8 |
| full_pe wins | 4/8 | 2/8 | 3/8 |

- Mean synap_only_pe − synap_only delta across all regimes and
  seeds: **+0.0002 MSE** (supervision slightly HURTS on average).
- Phase 1's 7/8 result was seed=42 specifically; seeds 0 and 1 show
  the mechanism underperforms the unsupervised baseline on 7 of 8
  regimes each.

**Verdict: Direction 1 is a NEGATIVE result.** The single-seed
finding was an outlier. The mechanism as designed does not provide
reliable per-seed improvement across the v0.5 capacity schedule.

Detailed analysis:
`research/developmental/results/env_sequence_v05_synap_pe_multiseed_findings.md`.

**Code disposition**: supervision stays in-tree as opt-in
(`synaptogenesis_supervision="pe_conditional"`). Default is `"none"`.
40-test suite continues to enforce correct semantics; the semantics
just don't help in aggregate.

**Implications for Direction 3**: the global-PE-gate intuition that
partially worked on one seed may not generalize. Direction 3 has two
genuinely different properties (proportional gating vs binary
pause, AND extends to Hebbian learning) so it's still worth testing,
but it MUST be multi-seeded from day 1 to avoid repeating the
seed-42-outlier mistake.

### Direction 3 multi-seed validation (2026-04-19) — ALSO NEGATIVE

Ran {no_growth, synap_only, synap_only_bcast, neuro_only,
neuro_only_bcast, full, full_bcast} on seeds {0, 1, 42}.

Per-seed full_bcast vs full scorecard:
- seed=0: 5/8
- seed=1: 3/8
- seed=42: 0/8

Plan's 5+/8 criterion is met on only 1 of 3 seeds. The mechanism
is seed-selective the same way Direction 1 was.

Sub-treatments:
- `synap_only_bcast` vs `synap_only`: 7/0/8 across seeds. Broadcast
  helps synap_only on avg but seed=1 is strongly negative.
- `neuro_only_bcast` vs `neuro_only`: 0/3/3. Near-zero effect;
  gain swings (0.44, 1.57) but doesn't matter when there's no
  synaptogenesis to gate and Hebbian barely moves neuro-wired edges
  over 4000 steps.

**Verdict: Direction 3 NEGATIVE on its primary claim.** Code stays
in as opt-in (`plasticity_broadcast_mode="pe_scaled"`, default
`"off"`).

### Surprising positive: neuro_only is the best mechanism on v0.5

The same multi-seed run confirmed something the earlier 2x2 hinted
at but couldn't resolve: `neuro_only` (neurogenesis WITHOUT
synaptogenesis) beats `no_growth` on the four highest-capacity
regimes by 0.0005–0.0009 MSE across all 3 seeds:

| Regime    | no_growth     | neuro_only    |
|-----------|---------------|---------------|
| mlp_2x64  | 0.0014±0.0002 | 0.0008±0.0004 |
| mlp_3x64  | 0.0020±0.0005 | 0.0011±0.0005 |
| mlp_4x32  | 0.0018±0.0002 | 0.0011±0.0001 |
| mlp_4x64  | 0.0011±0.0002 | 0.0006±0.0004 |

This is the one mechanism on v0.5 that multi-seed validation
endorses. **Directional takeaway**: synaptogenesis on this substrate
is the wrong primitive; neurogenesis with positional-neighbor wiring
adds real structure. Plan doc gets reweighted: focus next direction
on either understanding *why* neuro_only works or on replacing
synaptogenesis's random-edge admission with something task-grounded
(Direction 2: learnable input projections).

Detailed analysis:
`research/developmental/results/env_sequence_v05_broadcast_multiseed_findings.md`.

### Direction 2B (post-processing learnable projections) multi-seed — ALSO NEGATIVE

Direction 2B implemented as: projections become nn.Parameter, fed
through mean-over-views into the prediction head. Gradient flows
to projections via prediction loss. Does NOT pass through SOMA's
graph (that would be Direction 2C).

Per-seed scorecard (noise floor 0.00005 MSE):
- full_learnable beats full_frozen: 4/8, 4/8, 1/8 (seeds 0, 1, 42)
- neuro_only_learnable beats neuro_only_frozen: 5/8, 3/8, 6/8

Neither meets 5+/8 on ALL three seeds. Mean effect 0 to -0.0002 MSE
(noise-level).

Systematic negative: `full_learnable` REGRESSES on mlp_4x64 by
+0.0012 on all 3 seeds — consistent harm when combined with synap
on the hardest regime.

**Verdict: Direction 2B NEGATIVE.** Three of the four directions
(1, 2B, 3) have now failed multi-seed validation on v0.5.

### The only positive: neuro_only (reaffirmed by every ablation)

Every experiment in this session — no_diversify, broadcast, learnable
projections — has left neuro_only's advantage over no_growth
intact (0.0005-0.0009 MSE on 4 hard regimes across all 3 seeds).
None of the plasticity-adjustment mechanisms explain why it works;
none of them compose with it to improve further.

Code-reading analysis in
`research/developmental/results/why_neuro_only_works.md` identifies
four factors (need-based PE trigger, spatial-centroid positioning,
bidirectional local wiring, low init weight) and predicts that
positional locality specifically is the critical factor. Test:
`synap_local` (hard positional distance cutoff on synap admissions)
launched 2026-04-19; if synap_local reliably beats synap_only, the
prediction is validated and the plasticity-design principle becomes
"locality + capacity pressure" rather than "PE-signal gating."

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

## Direction 4 — LLM-as-teacher for plasticity supervision (product-aligned)

### Hypothesis

The root cause of SOMA's plasticity failure is lack of
semantic grounding — the same limit that bounds retrieval at
~0.8%. A pretrained SOTA LLM **has** the grounding SOMA needs
(this is precisely why GraphRAG beats vanilla RAG on global
queries: Microsoft's LLM does the semantic work through entity
extraction and community summarization). Instead of trying to
manufacture grounding from SOMA's internal PE signal alone
(Directions 1-3), we use an available LLM as the external
reference for plasticity decisions.

This direction explicitly violates the "no LLM in the core
loop" design principle from the original whitepaper. The case
for including it: SOMA's announced pivot (see
`docs/positioning.md`, `CLAUDE.md`) is an **agent-memory
layer** serving an LLM. In that context the LLM is already
available at inference time, so using it as a training signal
is natural, amortizable, and directly addresses the
first-principles limit from §5.

### Three compositional flavors

Each flavor pairs with one of Directions 1-3 and extends it
with an LLM-derived signal instead of / alongside PE.

#### 4a — LLM-distilled input projections (pairs with Direction 2)

Replace the "train projections to predict the next
observation" loss with "train projections so SOMA's activation
state for an input matches the LLM's embedding for that same
input." Standard knowledge distillation: SOMA is the student,
the LLM's last-hidden-state (or sentence-embedding head) is
the teacher. L2 or cosine loss on a per-input basis.

- Config: `associator_projections_learnable=True,
  projection_distillation_target="llm_embedding"`.
- Teacher model: `all-MiniLM-L6-v2` for cheap parity with the
  existing retrieval encoder; scale up to a larger teacher
  (e.g., `bge-large`, `e5-large`) in a second pass if the
  small-teacher version shows lift.
- Directly replaces the arbitrary random projections with
  semantically grounded ones. If this works, it affects both
  the §5 retrieval ceiling and the §4.7 adaptation result
  simultaneously — the most systemic fix of the four.

#### 4b — LLM-judged edge admission (pairs with Direction 1)

When synaptogenesis proposes a new edge (a, b), ask "does
connecting these two nodes correspond to a useful relation?"
as a labeled signal from the LLM. Three implementation
options, in increasing amortization:

- **Literal**: Prompt the LLM on every admission decision.
  Very expensive; maybe 10-100× inference cost during
  development.
- **Cached**: Each candidate (a, b) is summarized by a
  fingerprint of their recent co-activations; cache LLM
  judgments per fingerprint. Cheap after warmup.
- **Distilled judge**: Train a small local classifier on a
  few hundred LLM-labeled examples; use it as a drop-in
  admission filter. Essentially free at inference; needs a
  one-time setup step.

Config: `synaptogenesis_supervision="llm_judged",
synaptogenesis_llm_judge_cache_size=10000`.

#### 4c — LLM-scored outputs as reward broadcast (pairs with Direction 3)

SOMA generates output tokens via the verbalizer. Have the
LLM grade each output for correctness/relevance/coherence
against the task; broadcast the score as `plasticity_gain`.
This is the SOMA equivalent of an RLHF reward model. The
neuromodulator analogue gets real semantic anchoring instead
of a raw PE ratio.

- Works best once there's an actual generation task (not just
  sequence-prediction MSE). v0/v0.5 env doesn't produce
  natural outputs to grade, so this flavor needs a new
  evaluation task (e.g., short-horizon QA over corpora, where
  SOMA's activations drive the answer and the LLM grades it).
- Config: `plasticity_broadcast_mode="llm_scored"`.

### Config surface (combined)

```python
# 4a
projection_distillation_target: Literal[
    "none", "next_obs", "llm_embedding",
] = "none"
projection_distillation_llm: str = "all-MiniLM-L6-v2"

# 4b
synaptogenesis_supervision: Literal[
    "none", "pe_conditional", "llm_judged",
] = "none"
synaptogenesis_llm_judge_mode: Literal[
    "literal", "cached", "distilled",
] = "cached"

# 4c
plasticity_broadcast_mode: Literal[
    "off", "pe_scaled", "llm_scored",
] = "off"
```

All default to off/none. Opt-in does not break existing
behavior. These extend the Direction 1/2/3 config surfaces
rather than replacing them.

### Experimental design

Phase A — Cheapest first: run 4a (distillation against
all-MiniLM-L6-v2) on the v0.5 capacity schedule against the
same reference points as Direction 2:

- `full_fixed_proj` (reference, reproduces commit `2a1bbcc`)
- `full_distilled_proj` (new; distillation loss on
  projections)
- `no_growth_fixed_proj` (reference)
- `no_growth_distilled_proj` (isolates distillation-only
  effect)

Phase B — Cross-check on retrieval: rebuild the LoCoMo graph
with `full_distilled_proj` and rerun the Phase 10 held-out
validation. If the §5 ceiling shifts from ~+2/500 to
anything like +20/500, the LLM-as-teacher framing is a
genuine crack.

Phase C — 4b (LLM-judged admission) only if 4a shows lift and
we want to close the remaining gap with synaptogenesis
specifically. Start with the cached flavor; upgrade to
distilled only if cache hit rate is high enough to justify.

Phase D — 4c (LLM-scored reward) only after there's a
generation-capable evaluation task to grade; not
blocking-critical for the immediate direction.

### Success criteria

Primary (adaptation, Phase A):
- `full_distilled_proj` beats `full_fixed_proj` on ≥ 5 of 8
  v0.5 regimes.
- `no_growth_distilled_proj` matches or beats
  `no_growth_fixed_proj` (checks whether the benefit requires
  growth or is pure projection quality).

Secondary (retrieval, Phase B):
- LoCoMo gated-hybrid delta moves from ~+2/500 to ≥ +15/500
  against VecDB on held-out slices (not just the tuning
  slice). If this happens, the §5 ceiling is not actually a
  ceiling — it's an artifact of arbitrary projections — and
  the retrieval story reopens.

Tertiary:
- Any lift in cross-benchmark (LongMemEval −1 → positive).

### Risk and reversibility

- **Research-purity objection**: "the LLM is doing the
  semantic work, not the graph." Legitimate critique for the
  paper-aligned framing (Directions 1-3 are the clean test);
  not a problem for the product-aligned framing where the
  LLM is expected to be in the stack anyway. Document the
  trade-off clearly in any write-up.
- **Distillation destabilization**: same as Direction 2's
  risk (projection updates could destabilize training).
  Mitigation is the same: clip updates, smaller LR
  multiplier, fail-fast assertion on non-zero gradient
  detection.
- **LLM cost for 4b literal/cached**: if cache miss rate is
  high, 4b is expensive. Mitigate by monitoring miss rate in
  early runs and capping call budget.
- **Teacher-encoder mismatch**: if the encoder used for
  SENSOR nodes differs from the distillation teacher,
  projections are trained to match a different geometry than
  SOMA already consumes. Use the same encoder as both SENSOR
  input and distillation teacher for the first pass to avoid
  this.
- All reversible via config flags.

### Effort estimate

- 4a (distillation): ~2-3 days of implementation, ~1 day of
  experiments. The SENSOR node's encoder is already in-tree;
  the new loss term is a clean addition to the prediction
  step.
- 4b (LLM-judged admission): ~3-5 days including caching
  layer. Cached flavor is non-trivial because the
  fingerprint + cache design has to be careful.
- 4c (LLM-scored reward): ~3-5 days, gated on a generation
  task existing. Likely deferred.

---

## Running order

All four directions are independent and opt-in, so they can
be implemented in parallel. If pipelined, the rational order
depends on which framing is being optimized.

### Research-paper framing (tests "plasticity works without LLM supervision")

Run only Directions 1-3, in this order:

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

### Product-pivot framing (tests "plasticity works as an agent-memory layer")

Direction 4a first. It's the most systemic fix (projections
are upstream of everything else in the graph), the cheapest
to implement of the four, and it directly aligns with the
product claim that SOMA is useful as an agent-memory layer.
If 4a lifts both v0.5 adaptation and the §5 retrieval ceiling,
the rest of the directions become optional polish rather than
research necessities.

### Hybrid framing (recommended in practice)

Run 4a and 1 in parallel — they don't share code paths and
each answers a distinct question (can the LLM supervise
projections? can PE supervise edge admission?). Use the
result of both to decide whether to continue with 2, 3, 4b,
or 4c.

## Gating on v0 2×2 cross-check — **RESOLVED 2026-04-18**

Ran at commit `5d8a828`. Result: directional story reproduces
(synap alone hurts 2/4 regimes or ties; neuro alone matches/
beats no_growth on 3/4; full is worst on all 4; neuro-rescue
effect confirmed). One narrowing: on `nonlinear_sqrt`, neuro
slightly underperforms no_growth because growth to 31 nodes
was past Pareto-optimal for that regime — consistent with the
pre-add-nodes finding.

Refined claim: synaptogenesis is the consistently harmful
mechanism; neurogenesis is beneficial *when the task rewards
extra capacity* and can regress mildly when it does not. This
does not change the four proposed directions — the root cause
(synaptogenesis reads arbitrary correlations from random
projections) is still the target — but it confirms the
diagnosis generalizes beyond v0.5, so Directions 1-4 are
unblocked.

## Out-of-scope for this plan

- Structure-matched deep MLP control (E3 from AUDIT). Separate
  experiment; doesn't depend on these four directions.
- Seed-variance sweeps for existing v0.5 results. Should happen
  before publication regardless; not blocking here.
- Retrieval-track experiments unrelated to projection changes.
  Retrieval is effectively dormant (§5 ceiling diagnosed) and
  reopens only if direction 2 or 4a shows signs of cracking it.
