# Experiment A — STDP attractor formation (baseline): NO-GO

**Date:** 2026-04-20
**Script:** `research/developmental/experiments/sim_stdp_attractor.py`
**Output:** `research/developmental/results/sim_stdp_attractor_baseline.json` (4.2 MB)
**Wall time:** 843 s (14 min)
**Verdict:** **NO-GO** — all three gates failed.

## Hypothesis

If the neural simulator (`E:/Documents/Projects/sim`,
`HIPPOCAMPUS_CA3_RECURRENT` preset) supports meaningful STDP-driven
attractor formation, then repeatedly stimulating ten distinct
concept patterns should lead to:

- **Within-concept consistency rising over blocks** — as STDP
  strengthens the weights between co-active neurons, the same
  concept should produce increasingly similar response patterns
  trial-over-trial.
- **Between-concept separation widening** — unrelated concepts
  should *not* share spiking neurons, so between-concept Jaccard
  should stay low while within-concept Jaccard rises.
- **Late-block stability** — by the last block (trials 90-99), each
  concept's top-k response should be nearly deterministic across
  trials.

This is the foundation for using the simulator as a representation
layer in SOMA (Path A replan, Experiment A entry in
`docs/plans/2026-04-20-path-a-alife-replan.md`).

## Protocol

- **10 concepts**, each a fixed pattern of 100 input neurons chosen
  at random from 1000 (10 % overlap possible between concepts).
- **100 trials per concept**, 1000 trials total, interleaved.
- **Network:** 5000 recurrent neurons, k=50 connectivity, seed=42,
  `enable_stdp=True`.
- **Stimulation:** 100 ms @ 500 pA, 50 ms rest between trials.
- **Readout:** top-80 most-spiking neurons per trial
  (`top_k_primary`), input layer excluded.
- **Blocks:** trials `[0, 10)`, `[40, 50)`, `[90, 100)` — snapshots
  at early / mid / late.
- **Gates:**
  - **A:** block-2 within-concept Jaccard > 0.6 (concepts have
    settled)
  - **B:** late-stability (block-2 within-concept) > 0.8
  - **C:** monotonic rise block-0 → block-2 > 0.1

## Results

### Gate outcomes

| Gate | Threshold | Actual | Pass? |
|---|---|---|---|
| A — block-2 within-Jaccard > 0.6 | 0.60 | **0.2151** | ✗ |
| B — late stability > 0.8 | 0.80 | **0.2151** | ✗ |
| C — monotonic rise > +0.1 | +0.10 | **-0.0065** | ✗ |

### Block-over-block evolution

```
block |  within  | between  |  delta (separation)
  0   |  0.2216  |  0.2091  |  +0.0124
  1   |  0.2178  |  0.2076  |  +0.0103
  2   |  0.2151  |  0.2103  |  +0.0048
```

**Separation collapses over time.** Within-concept consistency
*decreases* (0.22 → 0.21) while between-concept overlap stays
flat — the network's concept-selectivity is *degrading*, not
consolidating. The monotonic-rise metric is slightly negative
(-0.0065), confirming the drift is the wrong sign.

### Per-concept variance (block 2)

| Concept | Within-Jaccard |
|---|---|
| 0 | 0.277 |
| 1 | 0.266 |
| 2 | 0.239 |
| 3 | 0.226 |
| 4 | 0.215 |
| 5 | 0.199 |
| 6 | 0.178 |
| 7 | 0.197 |
| 8 | 0.184 |
| 9 | 0.170 |

Even the best-organised concept (0) sits at 0.28, far below the 0.6
gate. The spread (0.17 – 0.28) has no obvious structure — it's not
the case that STDP is locking in a few concepts while failing others;
every concept is floating around 0.2 with trial-to-trial noise.

## Interpretation

The network responds to stimuli — total_spikes ≈ 5000 per trial is in
the right order of magnitude — but **it does not form
trial-consistent attractors**. Three candidate reasons, in priority
order:

1. **STDP window too weak vs. stimulus-driven input.** The
   feedforward stimulus (500 pA × 100 ms to 100 input neurons) is the
   dominant driver of spiking. Recurrent dynamics have ~50 ms of rest
   to consolidate, but the next trial's input overwrites any
   attractor state before it can reinforce itself. Candidate fix:
   longer rest windows, lower stim amplitude, or a dedicated
   "consolidation" window with no stimulus at all.

2. **No inhibitory balance → no winner-take-all dynamics.** Attractor
   formation canonically requires competition: E→I→E loops carve out
   distinct active populations. The `HIPPOCAMPUS_CA3_RECURRENT` preset
   may ship with only excitatory recurrence. Candidate fix: verify
   the preset includes inhibitory neurons and that their time
   constants are set to suppress runaway activity.

3. **Readout window captures response, not persistence.** `top_k_primary = 80`
   picks the most active neurons during the stimulation window.
   Attractor *persistence* (the key signature) would show up *after*
   the stimulation ends. Candidate fix: re-analyse the existing JSON
   with a post-stimulus readout window, or re-run with a 200 ms
   recording window that extends past the 100 ms stimulation.

(3) is the cheapest to test because it doesn't require re-simulating
— the existing 4.2 MB JSON has per-trial spike data that could be
re-windowed. (1) and (2) need a new run and potentially a different
preset.

## Decision

**NO-GO on Experiment A as configured.** STDP alone, in the
`HIPPOCAMPUS_CA3_RECURRENT` preset at these stim parameters, does
not produce attractors on a 10-concept × 100-trial protocol.

**Not** a conclusion that the simulator is unfit for representation —
it's a conclusion that *this specific protocol* doesn't exercise the
attractor-formation machinery. Two viable next steps:

- **A-prime:** re-analyse the current JSON with a post-stimulus
  readout window (cheap, ~15 min of analysis work). If separation
  rises when we look at the sustained post-stim response, the raw
  mechanism is there but the readout was wrong.
- **A2:** new run with inhibitory balance explicitly configured and
  a longer rest / consolidation window. Costs another ~15 min of
  sim time per seed.

If both fail, Experiment A is dead and the Path A replan should
move directly to Experiment B (spontaneous trajectory) or
Experiment D (structural plasticity) without pretending attractor
formation is a prerequisite.

## Relation to SOMA

SOMA's 0.2.0rc1 release (shipping today) does not depend on Path A.
The M1 milestone (+22.8 % F1 on LongMemEval via hybrid retrieval) is
the release's centre of gravity. Path A / Path B are the next-horizon
research tracks — Path B closed NO-GO last week, and now the first
Path A experiment also closed NO-GO at the baseline configuration.

This is a signal that the representation-layer ambitions for SOMA
need a different angle than "plug a spiking simulator into the
memory layer." Candidates that remain viable:

- Learned locality over semantic projections (Direction 4b, already
  designed in `docs/plans/2026-04-19-direction-4b-spatial-distillation-design.md`).
- Dynamic-config (prediction-error-driven growth rate, the
  Phase 2 follow-on noted in `project_dynamic_config.md`).
- Experiment A-prime / A2 above if a motivated maintainer wants
  another 30 minutes to probe the simulator.
