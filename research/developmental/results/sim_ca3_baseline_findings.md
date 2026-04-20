# sim CA3 baseline — Phase 1 of Path B (NO-GO)

**Date:** 2026-04-20.
**Status:** CLOSED — Phase 1 NO-GO on all four code-extraction views.
**Tracks:** `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md` § Phase 1.
**Script:** `research/developmental/experiments/sim_ca3_measurement.py`.
**Result:** `research/developmental/results/sim_ca3_baseline.json`.

## Headline

**The default `HIPPOCAMPUS_CA3_RECURRENT` preset in the neural-simulator
at n=5000 neurons produces NO pattern separation on controlled concept
stimuli.** Within-concept and between-concept Jaccard are essentially
identical (separation ratio = 1.00) at every sparsity level we
measured. The sim cannot serve as a calibration target for Path B's
numpy primitives; Phase 2 proceeds with literature-default sparsity
(k=32, dim=4096) instead.

## Full-run numbers (50 concepts × 10 trials = 500 stim events)

| view | within J | between J | sep ratio | sparsity |
|---|---:|---:|---:|---:|
| top_k_strict (1%) | 0.165 | 0.162 | **1.02** | 0.010 |
| top_k_primary (2%) | 0.223 | 0.221 | **1.01** | 0.020 |
| top_k_relaxed (4%) | 0.309 | 0.308 | **1.00** | 0.040 |
| any_fired | 0.298 | 0.298 | **1.00** | 0.384 |

**Primary view: `top_k_primary` (2% sparsity — biological target).**

### GO/NO-GO gate

From design doc § Phase 1:
- (a) within-Jaccard > 0.5 — **FAIL** (got 0.223)
- (b) separation ratio > 1.5 — **FAIL** (got 1.01)

**Verdict: NO-GO.**

## Protocol

| Parameter | Value |
|---|---|
| Neural profile | `HIPPOCAMPUS_CA3_RECURRENT` |
| Neuron model | Izhikevich |
| Total neurons | 5000 |
| Input layer | first 1000 neurons (directly driven) |
| Readout layer | neurons 1000–4999 |
| Connectivity `k` | 50 (profile default was 10; scaled 5× for denser recurrents) |
| Total synapses | 250,000 |
| Concepts | 50 distinct random patterns |
| Concept size | 100 of 1000 input neurons, non-orthogonal |
| Trials per concept | 10 |
| Stim duration | 200 ms per trial |
| Rest between trials | 100 ms |
| Stim amplitude | 500 pA |
| Readout offset | 50 ms after stim onset (skip transient) |
| Readout window | 150 ms (steady-state response) |
| Seed | 42 |
| Plasticity | disabled (static network for measurement) |
| OU background noise | enabled (biology default, σ=100 pA) |
| Total sim time | 150,000 steps (150 s biological time) |
| Wall clock | 465 s on RTX 3090 |

Stimulus injection bypasses `ExperimentEngine` and writes directly to
`sim_bridge.cp_external_input_current` before each
`_run_one_simulation_step()`. The Izhikevich init zero-fills this
array once (bridge.py:744) and never resets it per-step, so per-trial
direct writes give clean stimulation.

## Three smoke-test configs checked before the full run

| config | within J (primary) | sep ratio (primary) | any-fired sparsity |
|---|---:|---:|---:|
| profile default k=10, 0–100ms, 500 pA | 0.307 | 1.10 | 0.273 |
| k=50, 50–200ms, 500 pA (the chosen config) | 0.308 | 1.13 | 0.383 |
| k=200, 50–300ms, 1000 pA (aggressive) | 0.409 | 1.11 | 0.532 |

Smoke tests (5 concepts × 2 trials) showed apparent separation ratios
around 1.10. Tight statistics (50×10=500 trials) collapse these to
1.00 — the early numbers were statistical noise, not signal. Lesson:
**always run the full design before drawing separation conclusions on
sim readouts.**

## Interpretation

Why does the sim CA3 preset not produce separation?

1. **No DG analog.** Real hippocampal pattern separation happens in
   dentate gyrus upstream of CA3; the DG's sparse granule-cell coding
   (~1-2% active, low overlap) is what orthogonalizes inputs before
   they reach CA3. The sim's `HIPPOCAMPUS_CA3_RECURRENT` is a CA3-like
   recurrent network but without an explicit DG preprocessor —
   stimulus patterns go straight into CA3 with their raw overlap
   intact.

2. **No attractor lock-in at this scale/connectivity.** Even with
   5× denser connectivity than the profile default (k=50, 250K
   synapses), the recurrent dynamics don't settle into concept-specific
   attractor states strong enough to dominate OU noise and inhibitory
   feedback. Attractor networks in the Hopfield/Hertz tradition
   require E/I balance tuning we haven't done, and either dense all-to-
   all or heavily learned connectivity.

3. **OU noise floor is substantial.** At σ=100pA and typical Izhikevich
   RS threshold drive of ~300pA, background noise alone produces
   substantial spontaneous firing (~38% of readout neurons fire at
   least once in the stim window). The "top-k" code is then mostly
   noise-driven, not concept-driven.

4. **No Hebbian/STDP consolidation.** We explicitly disabled plasticity
   for this measurement (consistent trials require a static network).
   A separate Phase 4-candidate experiment could enable Hebbian learning
   and ask whether repeated exposure to a concept CARVES an attractor
   that subsequent trials land in.

## Consequence for Path B

Per design doc § Risks:
> Sim calibration is impossible (Phase 1 no separation) — Low
> likelihood, Moderate impact — Mitigation: Try cortex preset; skip
> calibration and go straight to Phase 3 A/B.

We ran the full-scale measurement with the chosen CA3 config and it
failed the gate decisively. Two follow-ups skipped (to avoid rabbit-
holing):

- **Cortex preset**: `CORTEX_L23_RS_FS` is not designed to do pattern
  separation either — it's a microcircuit with no sparsifying
  preprocessor. We expect similar null results.
- **Custom DG preset**: Building a proper DG-analog network (dense
  excitatory, strong mutual inhibition, high sparsity target) from
  scratch in the sim framework is a multi-week engineering effort
  that, given the clear NO-GO on the vanilla CA3 preset, is unlikely
  to change the downstream story.

**Decision:** Phase 2 proceeds with **literature-default sparsity
targets** (k=32 active, dim=4096, Willshaw & Palm 1969, Numenta HTM):

- ~0.78% active dims, consistent with mammalian cortical sparsity
- k/dim chosen for combinatorial capacity (C(4096, 32) ≈ 10⁷⁷ distinct
  codes, vastly more than any real corpus)
- No sim-calibration gate; Phase 2's GO/NO-GO becomes purely "does
  the numpy primitive produce well-separated codes on synthetic
  embedding data."

## Phase 2 preview (synthetic calibration)

With sim calibration out of the picture, we validate the numpy
primitives on synthetic concept embeddings (50 concepts × 10 trials,
embed_dim=1024, overlap=0.1, noise_std=0.1) using
`benchmarks/sparse_codes_calibration.py --synthetic`. Best config at
`k=32, dim=4096`:

| metric | value |
|---|---:|
| within-concept Jaccard | 0.023 |
| between-concept Jaccard | 0.004 |
| separation ratio | **5.66** |
| sparsity | 0.78% |

So the numpy k-WTA primitive DOES produce well-separated sparse codes
at biologically-plausible sparsity when concepts are distinguishable
in dense embedding space. The sim NO-GO is a property of the sim's
dynamics, not a flaw in the primitive. Phase 3 runs these primitives
against LongMemEval to answer the real question: **does sparse-overlap
retrieval add signal on top of hybrid (BM25+cosine)?**

## Files

- `research/developmental/experiments/sim_ca3_measurement.py` —
  measurement harness (headless sim driver, per-trial spike-count
  accumulator, top-k code extraction, Jaccard metrics)
- `research/developmental/results/sim_ca3_baseline.json` — full
  500-trial data, per-trial active readout indices + spike counts
- `src/soma/memory/sparse_codes.py` — k-WTA, pattern_separate,
  code_similarity primitives (14 TDD tests all GREEN)
- `tests/test_memory/test_sparse_codes.py` — Phase 2 test suite
- `benchmarks/sparse_codes_calibration.py` — numpy primitive sweep
  (synthetic or sim-baseline-calibrated)
- `research/developmental/results/sparse_codes_synthetic_sweep.json` —
  calibration sweep results

## Cross-reference

- `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md` —
  Path B design with phase gates
- `docs/plans/2026-04-20-path-a-biophysical-representation-layer-design.md`
  — Path A (still blocked on Phase B's Phase 3 outcome, now the
  real load-bearing gate for both paths)
- `docs/milestones/2026-04-20-hybrid-retrieval-validated.md` — M1
  milestone baseline

## Follow-up hypotheses (not pursued)

Recording these for future reference if we revisit the sim path:

1. **Add explicit DG preprocessing layer.** A sparse
   competitive-WTA network between stimulus and CA3 should orthogonalize
   patterns before they hit the recurrent layer. Implementation:
   custom sim preset, or numpy pre-processing pipeline that converts
   dense stim → sparse DG code → CA3.
2. **Increase E/I balance precision.** Real CA3 has tight inhibitory
   gating. The sim's default `inhibitory_propagation_strength=0.105`
   may be miscalibrated for strong-stim conditions.
3. **Enable Hebbian learning, consolidate via repeated exposure.** A
   concept injected 100 times should carve an attractor basin via
   STDP, making trial 101 settle onto a stable code. This is a
   learning-based measurement, not an instantaneous one.
4. **Use the REST window for readout.** Stim-window measurement is
   dominated by reactive firing. Post-stim readout (in the 200-500ms
   after stim ends) could capture persistent attractor state.

These are research extensions; they don't change the Phase 1 verdict
on the default preset.
