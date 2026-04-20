# Path A replan: artificial-life-first, retrieval-agnostic

**Date:** 2026-04-20.
**Status:** Replan (Path B NO-GO triggered the alife-first pivot).
**References:**
- Original Path A design: `docs/plans/2026-04-20-path-a-biophysical-representation-layer-design.md`
- Path B closure summary: `research/developmental/results/path_b_closure_summary.md`
- Working sim driver (template for new experiments): `research/developmental/experiments/sim_ca3_measurement.py`
- Sim project overview: `E:/Documents/Projects/sim/CLAUDE.md`

## Motivation

Path B closed NO-GO on 2026-04-20. The numpy sparse-code primitives
work on synthetic data (sep ratio 5.66) but hurt LongMemEval rank-1
(0.790 → 0.660). Per the original Path A design doc:

> Path A should not start before Path B Phase 3 completes ... or
> invalidate [the primitive] (in which case Path A pivots to
> artificial-life-first framing and the retrieval Phase 5 becomes
> best-effort rather than decision-point).

That precondition has triggered. Path B invalidated the primitive *as
a retrieval lift on LongMemEval*. It did not invalidate the sim
substrate, and it did not invalidate the research question "can a
biophysically-accurate network learn and exhibit emergent behavior?"

Reframe: Path A is research-first, not product-first. The hybrid
retrieval shipped in M1 already saturates what a sbert-embedding
LongMemEval probe can measure. Beating that is not the research
question anymore. The research questions that remain genuinely open:

- Can a CA3-like recurrent spiking network learn stable attractors
  from repeated input under STDP?
- What does 100K steps of spontaneous activity look like under
  plasticity? Does structure emerge?
- Can the same substrate bind two input channels so presenting one
  evokes the other?
- Does alternating "wake" and "sleep" produce replay of the wake
  patterns during sleep?

None of these have clean answers in the literature at the scale and
fidelity sim supports (5K–50K neurons, Izhikevich/HH/AdEx,
STDP+homeostasis+structural plasticity, 3090-class GPU). Each is a
publishable result on its own. The operator has stated explicit
interest in Path A ("extremely interested in path A") and artificial
life ("exploring not just AI but artificial life"). This track serves
that interest without gating product on speculative retrieval wins.

The honest framing: this is a research track. Outputs are findings
docs (and possibly papers), not product features. Retrieval
experiments may still happen, but they are measurements, not GO/NO-GO
gates.

## What carries over from the original design

- **Sim integration harness** (`sim_ca3_measurement.py`). Direct
  `SimulationBridge` driver that writes to `cp_external_input_current`
  and reads per-step `cp_firing_states`. 500 trials at n=5000 in ~465s
  on a 3090. This is the template: every experiment below subclasses
  the pattern.
- **CA3 preset dynamics knowledge** from Path B Phase 1:
  `HIPPOCAMPUS_CA3_RECURRENT` defaults to k=10 recurrent synapses
  (sparse); scaling to k=50 approximates biology. Raw stimulus
  separation ratio is ~1.0 without a DG analog — useful baseline for
  any learning experiment.
- **Infrastructure knowledge**: sim needs CuPy path setup, the bridge
  must be initialized with `_initialize_simulation_data(called_from_playback_init=False)`,
  `runtime_state.max_delay_steps` must be set before the first step.

## What's dropped

- Phases 1–5 of the original Path A design (retrieval-on-LongMemEval
  targeting). No "text→stimulus encoder" research, no
  `MemoryLayer.with_biophysical()` constructor, no sparse-code
  retrieval A/B against hybrid.
- "Must beat hybrid retrieval R@5 to GO" gates. Gone entirely.
- Packages under `src/soma/research/biophysical/`. Not built. If any
  of these experiments needs a utility class, it lives in
  `research/developmental/experiments/` next to the driver.
- `BiophysicalMemoryStore` / HDF5 storage schema. The sim already
  persists to HDF5 via checkpoints; we reuse that, not build a parallel.

## Four candidate experiments

Selected for (a) tractability given the existing harness, (b)
publishable value, (c) minimal cross-experiment dependencies.
Candidates considered but not selected: cross-modal binding (needs a
two-channel stimulus encoder we don't have yet — good for round 2),
oscillation-binding (needs phase-locked input patterns, higher
setup cost), in-silico foraging (needs a reward signal and an action
readout population — multi-week scope).

### Experiment A: STDP-driven attractor formation

| Field | Value |
|---|---|
| **Question** | Does repeated stimulus presentation under STDP form a stable attractor basin in `HIPPOCAMPUS_CA3_RECURRENT`? |
| **Protocol** | Reuse `sim_ca3_measurement.py` almost verbatim. Flip `enable_stdp=True`. Present 10 concepts × 100 trials with 50ms inter-trial interval. Measure per-concept sparse-code Jaccard across trial blocks [1-10], [41-50], [91-100]. |
| **Success criterion** | Within-concept Jaccard rises monotonically across blocks from initial <0.3 (Path B Phase 1 baseline) to >0.6 at the final block, AND the trained code is stable (Jaccard > 0.8 for last 10 trials). |
| **Expected output** | `research/developmental/results/stdp_attractor_findings.md` with per-block Jaccard trajectories and a timing analysis. |
| **Wall-clock** | 1-2 days. Sim time scales roughly linearly with Path B Phase 1 (~465s for 500 trials at n=5000); 10×100 = 1000 trials is ~15-20 min. Remainder is harness extension + analysis. |
| **Dependencies** | Stands alone. Direct extension of the existing driver. |

### Experiment B: Spontaneous-activity developmental trajectory

| Field | Value |
|---|---|
| **Question** | Under STDP + homeostasis with no input, how does self-organized activity evolve over 100K sim steps? Do clusters, pacemakers, or traveling waves emerge? |
| **Protocol** | `HIPPOCAMPUS_CA3_RECURRENT` at n=5000, k=50, dt=1ms (100 sim-seconds). `enable_stdp=True`, `enable_synaptic_scaling=True`, zero external input (OU noise only). Snapshot population firing rates + pairwise correlations every 5000 steps. Measure: (a) mean firing rate trajectory, (b) correlation-matrix block structure via modularity, (c) dominant frequency peak in population LFP proxy. |
| **Success criterion** | Either (a) modularity rises above chance by step 50K, OR (b) a stable oscillation emerges (peak-to-noise ratio >3 in PSD), OR (c) firing-rate distribution becomes bimodal (silent + active subpopulations). All three are publishable even if one hits. |
| **Expected output** | `research/developmental/results/spontaneous_trajectory_findings.md` with 20 correlation-matrix snapshots + PSD evolution + firing-rate KDE evolution plots. |
| **Wall-clock** | 1-2 days. 100K steps at n=5000 with STDP on ≈ 1200-1500s wall-clock (STDP overhead ~2-3× vs. Path B Phase 1's no-plasticity 465s). Harness needs periodic-snapshot hook + analysis code. |
| **Dependencies** | Stands alone. Most independent of the four — no stimulus design. |

### Experiment C: Sleep-replay consolidation

| Field | Value |
|---|---|
| **Question** | If we alternate "wake" (stim-on, STDP-on) and "sleep" (no stim, STDP-on) phases, does the network replay wake-phase population patterns during sleep? |
| **Protocol** | Three-phase cycle repeated 20 times: (1) wake-A: inject concept-A stim 20 times (~6s sim); (2) wake-B: inject concept-B stim 20 times; (3) sleep: 10s of no input, STDP on. During sleep, match each 100ms spontaneous population pattern against the stored concept-A and concept-B attractor codes via top-k Jaccard. Compare pre-wake and post-wake sleep windows. |
| **Success criterion** | Post-wake sleep windows contain >2× more high-Jaccard (>0.5) matches to trained concept codes than pre-wake (baseline) sleep windows. Replay events should be brief (20-100ms) and temporally sparse (<5% duty cycle) to match biology. |
| **Expected output** | `research/developmental/results/sleep_replay_findings.md` with a raster plot of sleep-phase replay events time-locked to wake phases, and a histogram of Jaccard-to-trained-code over sleep. |
| **Wall-clock** | 2 days. Depends on Experiment A's attractor protocol being validated (need a working "train a concept" subroutine before alternating). 20 cycles × ~18s each ≈ 360s sim; harness + alternation logic is the work. |
| **Dependencies** | Depends on Experiment A (reuses the attractor-training code). Don't start until A has a passing run. |

### Experiment D: Structural plasticity under varied stimuli

| Field | Value |
|---|---|
| **Question** | With `enable_structural_plasticity=True` and a varied stimulus stream, does the synapse graph reorganize toward the stimulus statistics (i.e., do neurons that receive correlated input develop more connections with each other)? |
| **Protocol** | `HIPPOCAMPUS_CA3_RECURRENT`, n=5000, k=50 initial. `enable_structural_plasticity=True`, `struct_plast_activity_bias=0.7`. Present 5 concepts in a fixed rotating order for 50K steps. Snapshot the CSR connectivity matrix at step 0, 10K, 25K, 50K. Measure: (a) connectivity entropy (shouldn't flatten to zero — that would be a collapse), (b) per-concept-pair connectivity overlap correlation with stimulus-pattern overlap, (c) total synapse count trajectory. |
| **Success criterion** | By step 50K: (a) connectivity-overlap-vs-stimulus-overlap Pearson correlation > 0.3, AND (b) total synapse count stable within ±20% of initial (no runaway growth or collapse). |
| **Expected output** | `research/developmental/results/structural_plasticity_findings.md` with 4 connectivity-matrix snapshots, entropy/synapse-count trajectories, and the overlap-correlation scatter. |
| **Wall-clock** | 2 days. Structural plasticity is the least-validated sim feature for this use case; expect debugging time. 50K steps with structural plasticity on ≈ 1500-2000s wall-clock. |
| **Dependencies** | Stands alone but more research-risky — structural plasticity introduces runaway-growth failure modes. Do after A and B have shown the basic harness holds. |

### Recommended starting order

1. **Experiment A** first — fastest path to a publishable result,
   maximally reuses existing harness, answers a fundamental question
   ("can this network learn at all with STDP").
2. **Experiment B** in parallel once A's protocol is validated — it's
   the most independent and gives us baseline data on what "no
   stimulus" looks like that contextualizes A's findings.
3. **Experiment C** after A has a passing run, since C reuses A's
   attractor-training subroutine.
4. **Experiment D** last — highest research risk, uses a sim feature
   path not yet exercised by Path B.

## First concrete step

**Start Experiment A.**

- **Script:** `research/developmental/experiments/sim_stdp_attractor.py`
- **Approach:** copy `sim_ca3_measurement.py` as a starting point.
  Keep the `MeasurementConfig`, `TrialResult`, metric functions, and
  `_build_sim_bridge` pattern. Change:
  - `enable_stdp=True` in the core config
  - Add a `block_id` field on `TrialResult` for per-block grouping
  - Change the metric extraction to compute within-concept Jaccard
    *per block* rather than pooled across all trials
  - Reduce `n_concepts` to 10 and raise `n_trials` to 100
- **Outcome:** produces
  `research/developmental/results/stdp_attractor_findings.md`
  answering "does STDP in `HIPPOCAMPUS_CA3_RECURRENT` form stable
  concept-specific attractors?" with per-block Jaccard trajectories.
  Either result is publishable: PASS → "CA3 recurrent preset learns
  attractors from 100 trials"; FAIL → "CA3 recurrent preset does not
  form attractors at this scale without a DG analog — structural
  implication for biologically-faithful memory models."

## Explicit non-goals

- No retrieval A/B against hybrid in this track.
- No LongMemEval runs.
- No "ship a product feature" framing. Zero commitment to merge any
  of this into `src/soma/memory/api.py` or the shipping
  `MemoryLayer`.
- No text-to-stimulus encoder research. Stimuli are random neuron
  subsets (Path B Phase 1 pattern), not embeddings. Text encoding is a
  separate, later question.
- No cross-modal integration in the first round. Consider for round 2
  after at least two of A/B/C/D produce findings.

## Timeline

Open-ended. The operator runs this concurrently with product-polish
work, so low-urgency pacing is fine.

- **Week 1:** Experiment A (1-2 days of focused work).
- **Week 2+:** Experiments B / C / D as capacity permits. Each is
  self-contained once A validates the harness pattern.
- **No hard deadlines.** If any experiment produces a surprising
  result, stop and write it up before starting the next one.

## Files this replan creates (none yet — all future)

- `research/developmental/experiments/sim_stdp_attractor.py` — Exp A
- `research/developmental/experiments/sim_spontaneous_trajectory.py` — Exp B
- `research/developmental/experiments/sim_sleep_replay.py` — Exp C
- `research/developmental/experiments/sim_structural_plasticity.py` — Exp D
- `research/developmental/results/stdp_attractor_findings.md`
- `research/developmental/results/spontaneous_trajectory_findings.md`
- `research/developmental/results/sleep_replay_findings.md`
- `research/developmental/results/structural_plasticity_findings.md`

No modifications to `src/soma/`. No modifications to any existing
plan, milestone, or positioning doc. This track lives entirely under
`research/developmental/`.
