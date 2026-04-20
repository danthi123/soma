#!/usr/bin/env python3
"""Path A Experiment A: STDP-driven attractor formation in sim CA3.

This is the first concrete step of the alife-first Path A replan
(docs/plans/2026-04-20-path-a-alife-replan.md, Experiment A). It is a
direct extension of ``sim_ca3_measurement.py``: same SimulationBridge
driver, same stimulus injection path (``cp_external_input_current``),
same readout + top-k code extraction.

The one knob we flip is plasticity. Path B Phase 1 ran with
``enable_stdp=False`` / ``enable_hebbian_learning=False`` to measure
intrinsic CA3 dynamics on a static network. Here we enable BOTH the
STDP rule and the sim's simpler Hebbian rule so repeated stimulus
presentation can actually carve trained codes into the recurrent
weights.

Research question
-----------------
Does repeated stimulus presentation under STDP form a stable attractor
basin in ``HIPPOCAMPUS_CA3_RECURRENT``?

Protocol
--------
- 10 concepts × 100 trials per concept (1000 total trials).
- 50ms inter-trial rest (vs. Phase 1's 200ms — more contiguous learning).
- STDP + Hebbian ON; structural plasticity + synaptic scaling OFF
  (we want to measure learning on a fixed graph, not graph rewiring).
- Measure per-concept sparse-code Jaccard in three trial blocks
  representing early / mid / late learning: [0, 10), [40, 50), [90, 100).

GO/NO-GO gate
-------------
All three conditions must PASS:

  (a) block-2 within-Jaccard (primary view, top-k=80) > 0.6
  (b) late-stability (pairwise Jaccard on each concept's last 10 trials,
      averaged across concepts) > 0.8
  (c) monotonic rise: block-2 within-Jaccard > block-0 within-Jaccard + 0.1

Expected runtime
----------------
~15-20 min of sim wall-clock at n=5000 (500 trials in Path B Phase 1 took
~465s; 1000 trials is 2× that, plus STDP overhead).

Usage
-----
    python -m research.developmental.experiments.sim_stdp_attractor \
        [--n-concepts 10] [--n-trials 100] [--n-neurons 5000] \
        [--stim-ms 100] [--rest-ms 50] [--stim-amplitude-pA 500] \
        [--sim-path E:/Documents/Projects/sim] \
        [--out research/developmental/results/stdp_attractor_baseline.json] \
        [--no-stdp]  # ablation: disable STDP (reproduces static baseline)
        [--quick]    # 3 concepts × 10 trials smoke-test (<2 min)

Outputs
-------
  - ``research/developmental/results/stdp_attractor_baseline.json``
    with per-block metrics, late-stability scalar, and GO/NO-GO verdict.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Pure metric helpers (tested inline before any sim is touched)
# ---------------------------------------------------------------------------


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard similarity between two sets of neuron indices.

    J(a, b) = |a ∩ b| / |a ∪ b|.  Returns 1.0 when both are empty
    (vacuous agreement — callers should filter those cases if they
    would bias the mean).
    """
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def mean_pairwise_jaccard(trials: Sequence[Sequence[int]]) -> float:
    """Mean Jaccard over all unordered pairs of trials in ``trials``.

    ``trials`` is a list of per-trial active-index collections. With
    fewer than 2 trials, returns 1.0 (no pairs to disagree).
    """
    pairs = list(itertools.combinations(range(len(trials)), 2))
    if not pairs:
        return 1.0
    total = sum(jaccard(trials[i], trials[j]) for i, j in pairs)
    return total / len(pairs)


def mean_between_concept_jaccard(
    concept_trials: Sequence[Sequence[Sequence[int]]],
    rng: np.random.Generator,
) -> float:
    """Mean Jaccard across unordered pairs of DIFFERENT concepts.

    For each pair (c1, c2), samples one random trial from each concept
    and computes Jaccard. Averages over all ``C(n_concepts, 2)`` pairs.

    This is the "between" half of the separation metric — low values
    mean different concepts produce different codes (good).
    """
    n_concepts = len(concept_trials)
    pairs = list(itertools.combinations(range(n_concepts), 2))
    if not pairs:
        return 0.0
    total = 0.0
    for i, j in pairs:
        if not concept_trials[i] or not concept_trials[j]:
            continue
        ti = concept_trials[i][rng.integers(0, len(concept_trials[i]))]
        tj = concept_trials[j][rng.integers(0, len(concept_trials[j]))]
        total += jaccard(ti, tj)
    return total / len(pairs)


def mean_sparsity(trials: Iterable[Iterable[int]], n_neurons: int) -> float:
    """Mean fraction of active neurons across all trials."""
    counts = [len(list(t)) for t in trials]
    if not counts:
        return 0.0
    return float(np.mean(counts)) / float(n_neurons)


# ---------------------------------------------------------------------------
# Self-test — runs before any sim import so metric bugs fail fast
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Validate metric functions on hand-checked examples."""
    # jaccard: disjoint -> 0, identical -> 1, half-overlap -> 1/3
    assert jaccard([1, 2, 3], [4, 5, 6]) == 0.0, "disjoint should be 0"
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0, "identical should be 1"
    assert abs(jaccard([1, 2], [2, 3]) - 1 / 3) < 1e-9, "half-overlap = 1/3"
    assert jaccard([], []) == 1.0, "two empty sets are vacuously identical"

    # mean_pairwise_jaccard: 3 identical trials -> 1.0
    assert mean_pairwise_jaccard([[1, 2], [1, 2], [1, 2]]) == 1.0
    # 3 disjoint trials -> 0.0
    assert mean_pairwise_jaccard([[1, 2], [3, 4], [5, 6]]) == 0.0
    # Single trial -> 1.0 (no pairs)
    assert mean_pairwise_jaccard([[1, 2]]) == 1.0

    # mean_between_concept_jaccard: 2 identical concepts -> 1.0
    rng = np.random.default_rng(0)
    ct = [[[1, 2, 3]], [[1, 2, 3]]]
    assert mean_between_concept_jaccard(ct, rng) == 1.0
    # 2 fully disjoint concepts -> 0.0
    ct = [[[1, 2]], [[3, 4]]]
    assert mean_between_concept_jaccard(ct, rng) == 0.0

    # sparsity: 2 trials of 5 active out of 100 -> 0.05
    assert abs(mean_sparsity([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], 100) - 0.05) < 1e-9

    # block_id assignment from block_spec
    spec = [(0, 10), (40, 50), (90, 100)]
    assert _block_id_for_trial(5, spec) == 0
    assert _block_id_for_trial(45, spec) == 1
    assert _block_id_for_trial(99, spec) == 2
    assert _block_id_for_trial(25, spec) == -1   # between blocks
    assert _block_id_for_trial(100, spec) == -1  # past the last block

    print("[self-test] metric functions OK")


def _block_id_for_trial(trial_id: int, block_spec: Sequence[tuple[int, int]]) -> int:
    """Return the block index (0, 1, 2, ...) for a trial, or -1 if outside all blocks.

    ``block_spec`` is a list of half-open [start, end) intervals.
    """
    for idx, (start, end) in enumerate(block_spec):
        if start <= trial_id < end:
            return idx
    return -1


# ---------------------------------------------------------------------------
# Sim driver
# ---------------------------------------------------------------------------


@dataclass
class MeasurementConfig:
    n_neurons: int = 5000
    n_input: int = 1000            # first ``n_input`` neurons receive direct drive
    n_concepts: int = 10           # Experiment A: 10 concepts (was 50 in Phase 1)
    concept_size: int = 100        # active input neurons per concept (of ``n_input``)
    n_trials: int = 100            # Experiment A: 100 trials per concept (was 10)
    stim_ms: int = 100
    rest_ms: int = 50              # 50ms inter-trial (Exp A spec; was 200ms in Phase 1)
    stim_amplitude_pA: float = 500.0
    dt_ms: float = 1.0
    seed: int = 42
    # Plasticity: ON for attractor formation. --no-stdp CLI flag flips this
    # to False to reproduce the Path B Phase 1 static baseline.
    enable_stdp: bool = True
    # Block spec: half-open [start, end) trial ranges for per-block Jaccard.
    # Three blocks probe early / mid / late learning on the default n_trials=100.
    block_spec: list[tuple[int, int]] = field(
        default_factory=lambda: [(0, 10), (40, 50), (90, 100)]
    )
    # Connectivity override: HIPPOCAMPUS_CA3_RECURRENT profile defaults to
    # k=10 (per-neuron) which yields only 50K synapses at n=5000. Real CA3
    # has ~10-15K recurrent synapses per pyramidal cell; we scale this up
    # to approximate denser recurrents. None = keep profile default.
    connectivity_k: int | None = 50
    # Post-stim readout: measure steady-state response in a window that
    # STARTS ``readout_offset_ms`` after stim onset and lasts
    # ``readout_window_ms``. Default: whole stim window (same as before).
    readout_offset_ms: int = 0
    readout_window_ms: int | None = None   # None = until stim ends
    # Readout = neurons 1000..5000 (excludes the directly-driven input layer)
    readout_exclude_input: bool = True
    # Sparse-code extraction: "concept code" = top-k most-active readout neurons
    # during the stim window. Reported at three k-levels; top_k_primary is used
    # for the GO/NO-GO gate.
    top_k_primary: int = 80    # 2% of 4000 readout neurons (biological target)
    top_k_relaxed: int = 160   # 4%
    top_k_strict: int = 40     # 1%

    @property
    def trial_steps(self) -> int:
        return int(round(self.stim_ms / self.dt_ms))

    @property
    def rest_steps(self) -> int:
        return int(round(self.rest_ms / self.dt_ms))

    @property
    def total_steps(self) -> int:
        return self.n_concepts * self.n_trials * (self.trial_steps + self.rest_steps)


@dataclass
class TrialResult:
    concept_id: int
    trial_id: int
    block_id: int   # 0, 1, 2 for early/mid/late; -1 for trials outside any block
    # Full readout spike count per neuron over the stim window. Stored as a
    # sparse (index, count) list to keep JSON output compact — only neurons
    # that spiked at least once are included.
    spike_counts: list[tuple[int, int]]  # [(neuron_idx, count), ...]
    # Derived "codes" at three sparsity levels (top-k most active).
    top_k_primary: list[int]
    top_k_relaxed: list[int]
    top_k_strict: list[int]
    # Any-fired set (dense, baseline view for diagnosis only).
    # Kept on the in-memory object but stripped from the JSON output to
    # keep files small at n_trials=1000.
    any_fired: list[int]
    total_spikes: int


@dataclass
class BlockResult:
    """Per-block aggregate metrics at the primary view (top_k_primary)."""
    block_idx: int
    trial_range: tuple[int, int]          # (start, end) of the block
    within_jaccard_primary: float
    between_jaccard_primary: float
    sparsity: float
    n_trials_per_concept: int             # how many trials per concept in this block

    def to_json_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeasurementResult:
    config: dict
    patterns: list[list[int]]                # per-concept input indices
    trials: list[TrialResult] = field(default_factory=list)
    block_results: list[BlockResult] = field(default_factory=list)
    late_stability: float = 0.0              # pairwise Jaccard on last 10 trials / concept
    go_nogo: str = ""
    gate_details: dict = field(default_factory=dict)
    primary_view: str = "top_k_primary"
    wall_time_s: float = 0.0

    def _trial_to_json(self, t: TrialResult) -> dict:
        """Compact per-trial JSON: drop any_fired + spike_counts to keep files small."""
        return {
            "concept_id": t.concept_id,
            "trial_id": t.trial_id,
            "block_id": t.block_id,
            "top_k_primary": t.top_k_primary,
            "top_k_relaxed": t.top_k_relaxed,
            "top_k_strict": t.top_k_strict,
            "total_spikes": t.total_spikes,
        }

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "patterns": self.patterns,
            "trials": [self._trial_to_json(t) for t in self.trials],
            "block_results": [b.to_json_dict() for b in self.block_results],
            "late_stability": self.late_stability,
            "go_nogo": self.go_nogo,
            "gate_details": self.gate_details,
            "primary_view": self.primary_view,
            "wall_time_s": self.wall_time_s,
        }


def generate_concept_patterns(cfg: MeasurementConfig) -> list[np.ndarray]:
    """Pick ``n_concepts`` overlapping-but-distinct input subsets.

    Each concept is ``concept_size`` neuron indices sampled without
    replacement from ``[0, n_input)``. Concepts can share some indices
    (biology does not give you orthogonal codes for free — that's what
    CA3 is supposed to produce).
    """
    rng = np.random.default_rng(cfg.seed)
    return [
        rng.choice(cfg.n_input, size=cfg.concept_size, replace=False).astype(np.int32)
        for _ in range(cfg.n_concepts)
    ]


def _import_sim(sim_path: str):
    """Insert sim path on sys.path and import the bits we need."""
    sim_path = os.path.abspath(sim_path)
    if not os.path.isdir(sim_path):
        raise FileNotFoundError(f"sim project not found at {sim_path}")
    if sim_path not in sys.path:
        sys.path.insert(0, sim_path)

    # Delayed imports — these pull in CuPy, which has GPU side effects.
    import cupy as cp  # type: ignore[import-untyped]
    from sim import (  # type: ignore[import-untyped]
        SimulationBridge,
        CoreSimConfig,
        VisualizationConfig,
        RuntimeState,
        GPUConfig,
    )
    from sim.enums import NeuronModel  # type: ignore[import-untyped]

    return cp, SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel


def _build_sim_bridge(cp, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel, cfg: MeasurementConfig):
    core_cfg = CoreSimConfig()
    core_cfg.num_neurons = cfg.n_neurons
    core_cfg.dt_ms = cfg.dt_ms
    core_cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core_cfg.neural_profile_name = "HIPPOCAMPUS_CA3_RECURRENT"
    core_cfg.seed = cfg.seed
    # Plasticity: Experiment A wants STDP + Hebbian ON so repeated trials
    # actually carve attractor basins. Structural plasticity + synaptic
    # scaling stay OFF — we're measuring learning on a fixed graph, not
    # graph rewiring (that's Experiment D's territory).
    if cfg.enable_stdp:
        core_cfg.enable_stdp = True
        core_cfg.enable_hebbian_learning = True
    else:
        # --no-stdp ablation: reproduce Path B Phase 1's static baseline.
        core_cfg.enable_stdp = False
        core_cfg.enable_hebbian_learning = False
    core_cfg.enable_structural_plasticity = False
    core_cfg.enable_synaptic_scaling = False
    # The HIPPOCAMPUS_CA3_RECURRENT profile's default_core_overrides set
    # connectivity_k=10, which yields only 50K synapses (sparse recurrents).
    # Biology has ~15K synapses per CA3 pyramidal — this scales up.
    if cfg.connectivity_k is not None:
        core_cfg.connectivity_k = cfg.connectivity_k
    # Keep homeostasis + STP + OU noise on — they're part of biology and
    # contribute to within-trial noise we want to measure.

    viz_cfg = VisualizationConfig()
    runtime_state = RuntimeState()
    gpu_cfg = GPUConfig()

    from sim import SimulationBridge  # type: ignore[import-untyped]
    sim_bridge = SimulationBridge(
        core_config=core_cfg,
        viz_config=viz_cfg,
        runtime_state=runtime_state,
        gpu_config=gpu_cfg,
    )

    dt = core_cfg.dt_ms
    runtime_state.max_delay_steps = int(core_cfg.max_synaptic_delay_ms / dt) if dt > 0 else 200

    sim_bridge._initialize_simulation_data(called_from_playback_init=False)
    if not sim_bridge.is_initialized:
        raise RuntimeError("SimulationBridge failed to initialize")

    return sim_bridge, core_cfg


def run_measurement(cfg: MeasurementConfig, sim_path: str) -> MeasurementResult:
    """Drive the sim, collect per-trial active neuron sets, compute per-block metrics."""
    print(f"[sim_stdp] importing sim from {sim_path}...")
    cp, _SB, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel = _import_sim(sim_path)

    print(f"[sim_stdp] building SimulationBridge "
          f"(n={cfg.n_neurons}, profile=HIPPOCAMPUS_CA3_RECURRENT, model=Izhikevich, "
          f"stdp={cfg.enable_stdp})...")
    sim_bridge, core_cfg = _build_sim_bridge(
        cp, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel, cfg
    )

    patterns = generate_concept_patterns(cfg)
    print(f"[sim_stdp] generated {len(patterns)} concept patterns "
          f"({cfg.concept_size} active of {cfg.n_input} input neurons each)")
    print(f"[sim_stdp] block_spec: {cfg.block_spec} (early / mid / late)")

    readout_start = cfg.n_input if cfg.readout_exclude_input else 0
    readout_end = cfg.n_neurons

    # Pre-allocate GPU stimulus vector we write into cp_external_input_current
    stim_vec = cp.zeros(cfg.n_neurons, dtype=cp.float32)

    result = MeasurementResult(
        config=asdict(cfg),
        patterns=[p.tolist() for p in patterns],
    )

    t_wall_start = time.time()
    total_steps_done = 0
    total_steps_planned = cfg.total_steps
    last_report = t_wall_start

    for concept_id in range(cfg.n_concepts):
        pattern_cp = cp.asarray(patterns[concept_id], dtype=cp.int32)
        for trial_id in range(cfg.n_trials):
            # --- STIM window ---
            # Set stim pattern on input neurons; zero elsewhere.
            stim_vec[:] = 0.0
            stim_vec[pattern_cp] = cp.float32(cfg.stim_amplitude_pA)
            sim_bridge.cp_external_input_current[:] = stim_vec

            # Per-trial spike-count accumulator. We record spikes only
            # during [readout_offset_ms, readout_offset_ms + readout_window_ms)
            # from stim onset. Letting early transient settle before
            # recording produces cleaner attractor codes.
            spike_counts_cp = cp.zeros(cfg.n_neurons, dtype=cp.int32)
            offset_steps = int(round(cfg.readout_offset_ms / cfg.dt_ms))
            window_ms = cfg.readout_window_ms if cfg.readout_window_ms is not None else cfg.stim_ms - cfg.readout_offset_ms
            window_steps = int(round(window_ms / cfg.dt_ms))
            readout_stop = offset_steps + window_steps

            for step_in_trial in range(cfg.trial_steps):
                sim_bridge._run_one_simulation_step()
                sim_bridge.runtime_state.current_time_step += 1
                sim_bridge.runtime_state.current_time_ms = (
                    sim_bridge.runtime_state.current_time_step * cfg.dt_ms
                )
                if offset_steps <= step_in_trial < readout_stop:
                    spike_counts_cp += sim_bridge.cp_firing_states.astype(cp.int32)
                total_steps_done += 1

            # Move per-neuron counts to host; restrict to readout window for
            # code extraction. Full network spike counts are kept only as
            # total_spikes (a diagnostic scalar).
            counts_np = cp.asnumpy(spike_counts_cp)
            total_spikes = int(counts_np.sum())

            readout_counts = counts_np[readout_start:readout_end]
            any_fired_readout = sorted(
                int(readout_start + i) for i in np.where(readout_counts > 0)[0].tolist()
            )

            # Top-k = the k-highest-count readout neurons. Ties broken by
            # lower neuron index (stable via argpartition + argsort).
            def _top_k_readout(k: int) -> list[int]:
                if k <= 0 or readout_counts.size == 0:
                    return []
                k = min(k, int((readout_counts > 0).sum()))
                if k == 0:
                    return []
                # Partial sort: indices of the k largest counts.
                idx = np.argpartition(-readout_counts, k - 1)[:k]
                # Stabilise order by (-count, index) so ties are deterministic.
                ordered = sorted(
                    idx.tolist(),
                    key=lambda i: (-int(readout_counts[i]), i),
                )
                return sorted(int(readout_start + i) for i in ordered)

            block_id = _block_id_for_trial(trial_id, cfg.block_spec)

            trial = TrialResult(
                concept_id=concept_id,
                trial_id=trial_id,
                block_id=block_id,
                spike_counts=[
                    (int(readout_start + i), int(c))
                    for i, c in enumerate(readout_counts)
                    if c > 0
                ],
                top_k_primary=_top_k_readout(cfg.top_k_primary),
                top_k_relaxed=_top_k_readout(cfg.top_k_relaxed),
                top_k_strict=_top_k_readout(cfg.top_k_strict),
                any_fired=any_fired_readout,
                total_spikes=total_spikes,
            )
            result.trials.append(trial)

            # --- REST window (no stim) ---
            sim_bridge.cp_external_input_current[:] = 0.0
            for _ in range(cfg.rest_steps):
                sim_bridge._run_one_simulation_step()
                sim_bridge.runtime_state.current_time_step += 1
                sim_bridge.runtime_state.current_time_ms = (
                    sim_bridge.runtime_state.current_time_step * cfg.dt_ms
                )
                total_steps_done += 1

            now = time.time()
            if now - last_report >= 10.0:
                elapsed = now - t_wall_start
                pct = 100.0 * total_steps_done / max(1, total_steps_planned)
                rate = total_steps_done / max(1e-9, elapsed)
                eta = (total_steps_planned - total_steps_done) / max(1e-9, rate)
                print(f"[sim_stdp] concept {concept_id+1}/{cfg.n_concepts} "
                      f"trial {trial_id+1}/{cfg.n_trials} (block={block_id}) | "
                      f"{pct:5.1f}% | {rate:.0f} steps/s | ETA {eta:.0f}s | "
                      f"top-k primary active: {len(trial.top_k_primary)} | "
                      f"total spikes: {trial.total_spikes}")
                last_report = now

    result.wall_time_s = time.time() - t_wall_start
    print(f"[sim_stdp] simulation done in {result.wall_time_s:.1f}s "
          f"({total_steps_done} steps)")

    # --- Metrics: per-block within/between Jaccard at the primary view. ---
    # Group trials by (concept, block). For each block:
    #   within_jaccard  = mean over concepts of pairwise-Jaccard of that
    #                     concept's trials in this block.
    #   between_jaccard = mean pairwise-Jaccard across different concepts,
    #                     sampling one trial from each concept's block.
    # This is the per-block equivalent of the single-view report in
    # sim_ca3_measurement.py; we use top_k_primary exclusively because the
    # GO/NO-GO gate is defined at that view.
    for block_idx, (bstart, bend) in enumerate(cfg.block_spec):
        concept_trials_in_block: list[list[list[int]]] = [
            [] for _ in range(cfg.n_concepts)
        ]
        for t in result.trials:
            if t.block_id == block_idx:
                concept_trials_in_block[t.concept_id].append(t.top_k_primary)

        within_vals = [
            mean_pairwise_jaccard(ct) for ct in concept_trials_in_block if len(ct) >= 2
        ]
        within_mean = float(np.mean(within_vals)) if within_vals else 0.0
        rng = np.random.default_rng(cfg.seed + block_idx)
        between_mean = mean_between_concept_jaccard(concept_trials_in_block, rng)

        # Sparsity = mean active-count / n_readout_neurons across the block's trials.
        flat_block_trials = [
            t.top_k_primary for t in result.trials if t.block_id == block_idx
        ]
        sparsity = mean_sparsity(flat_block_trials, n_neurons=(readout_end - readout_start))

        # Trials per concept in this block (assumes uniform — all concepts run all trials).
        per_concept_counts = [len(ct) for ct in concept_trials_in_block]
        n_trials_per_concept = max(per_concept_counts) if per_concept_counts else 0

        result.block_results.append(BlockResult(
            block_idx=block_idx,
            trial_range=(bstart, bend),
            within_jaccard_primary=within_mean,
            between_jaccard_primary=between_mean,
            sparsity=sparsity,
            n_trials_per_concept=n_trials_per_concept,
        ))

    # --- Late-stability: pairwise Jaccard on each concept's last 10 trials. ---
    # This is the "trained code is stable" check, independent of the block
    # structure. We always take the LAST 10 trials regardless of block_spec.
    late_window = min(10, cfg.n_trials)
    late_start = cfg.n_trials - late_window
    late_by_concept: list[list[list[int]]] = [[] for _ in range(cfg.n_concepts)]
    for t in result.trials:
        if t.trial_id >= late_start:
            late_by_concept[t.concept_id].append(t.top_k_primary)
    late_vals = [mean_pairwise_jaccard(ct) for ct in late_by_concept if len(ct) >= 2]
    result.late_stability = float(np.mean(late_vals)) if late_vals else 0.0

    # --- GO/NO-GO gate: all three conditions must pass. ---
    # (a) block-2 within > 0.6       — the trained attractor is coherent
    # (b) late-stability > 0.8       — the final code is stable across trials
    # (c) block-2 - block-0 > 0.1    — monotonic rise (we actually learned)
    block0 = result.block_results[0] if len(result.block_results) >= 1 else None
    block2 = result.block_results[-1] if len(result.block_results) >= 1 else None
    gate_a = block2 is not None and block2.within_jaccard_primary > 0.6
    gate_b = result.late_stability > 0.8
    gate_c = (
        block0 is not None and block2 is not None
        and (block2.within_jaccard_primary - block0.within_jaccard_primary) > 0.1
    )
    result.gate_details = {
        "gate_a_block_last_within_gt_0_6": {
            "pass": bool(gate_a),
            "value": block2.within_jaccard_primary if block2 else None,
            "threshold": 0.6,
        },
        "gate_b_late_stability_gt_0_8": {
            "pass": bool(gate_b),
            "value": result.late_stability,
            "threshold": 0.8,
        },
        "gate_c_monotonic_rise_gt_0_1": {
            "pass": bool(gate_c),
            "value": (block2.within_jaccard_primary - block0.within_jaccard_primary)
                     if (block0 and block2) else None,
            "threshold": 0.1,
        },
    }
    result.go_nogo = "GO" if (gate_a and gate_b and gate_c) else "NO-GO"

    # --- Printed report ---
    print()
    print("=" * 80)
    print("EXPERIMENT A RESULTS: STDP-driven attractor formation in sim CA3")
    print("=" * 80)
    print(f"  readout window: neurons [{readout_start}, {readout_end})")
    print(f"  trials: {cfg.n_concepts} concepts × {cfg.n_trials} trials "
          f"= {len(result.trials)} total")
    print(f"  plasticity: stdp={cfg.enable_stdp}, hebbian={cfg.enable_stdp}, "
          f"struct=False, scaling=False")
    print()
    print(f"  {'block':>5s} | {'trial range':>12s} | {'n/concept':>9s} | "
          f"{'within J':>8s} | {'between J':>9s} | {'sparsity':>8s}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*9}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}")
    for b in result.block_results:
        print(f"  {b.block_idx:>5d} | [{b.trial_range[0]:>4d}, {b.trial_range[1]:>4d}) | "
              f"{b.n_trials_per_concept:>9d} |  {b.within_jaccard_primary:6.4f} |   "
              f"{b.between_jaccard_primary:6.4f} |  {b.sparsity:6.4f}")
    print()
    print(f"  PRIMARY VIEW: {result.primary_view} (top-k = {cfg.top_k_primary})")
    print(f"  late-stability (pairwise J on last {late_window} trials/concept): "
          f"{result.late_stability:.4f}")
    print()
    block2_val = block2.within_jaccard_primary if block2 else float('nan')
    block0_val = block0.within_jaccard_primary if block0 else float('nan')
    delta = (block2_val - block0_val) if (block0 and block2) else float('nan')
    print(f"  gate (a) last-block within > 0.6:      {'PASS' if gate_a else 'FAIL'} "
          f"(got {block2_val:.4f})")
    print(f"  gate (b) late-stability > 0.8:         {'PASS' if gate_b else 'FAIL'} "
          f"(got {result.late_stability:.4f})")
    print(f"  gate (c) block-2 > block-0 + 0.1:      {'PASS' if gate_c else 'FAIL'} "
          f"(delta {delta:+.4f})")
    print(f"  VERDICT:                               {result.go_nogo}")
    print("=" * 80)

    return result


def main() -> int:
    _self_test()

    parser = argparse.ArgumentParser(description="Path A Experiment A: STDP-driven attractor formation")
    parser.add_argument("--n-neurons", type=int, default=5000)
    parser.add_argument("--n-input", type=int, default=1000)
    parser.add_argument("--n-concepts", type=int, default=10)
    parser.add_argument("--concept-size", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--stim-ms", type=int, default=100)
    parser.add_argument("--rest-ms", type=int, default=50)
    parser.add_argument("--stim-amplitude-pA", type=float, default=500.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connectivity-k", type=int, default=50,
                        help="Outgoing synapses per neuron (overrides CA3 profile's k=10)")
    parser.add_argument("--readout-offset-ms", type=int, default=0,
                        help="Start counting spikes this many ms after stim onset")
    parser.add_argument("--readout-window-ms", type=int, default=None,
                        help="Spike-count window length in ms (default: until stim ends)")
    parser.add_argument("--no-stdp", action="store_true",
                        help="Disable STDP + Hebbian (reproduces Path B Phase 1 static baseline)")
    parser.add_argument("--sim-path", type=str,
                        default=os.environ.get("SIM_PATH", "E:/Documents/Projects/sim"))
    parser.add_argument("--out", type=str,
                        default="research/developmental/results/stdp_attractor_baseline.json")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny smoke-test config: 3 concepts × 10 trials (~1-2 min sim)")
    args = parser.parse_args()

    cfg = MeasurementConfig(
        n_neurons=args.n_neurons,
        n_input=args.n_input,
        n_concepts=args.n_concepts,
        concept_size=args.concept_size,
        n_trials=args.n_trials,
        stim_ms=args.stim_ms,
        rest_ms=args.rest_ms,
        stim_amplitude_pA=args.stim_amplitude_pA,
        dt_ms=args.dt_ms,
        seed=args.seed,
        enable_stdp=not args.no_stdp,
        connectivity_k=args.connectivity_k,
        readout_offset_ms=args.readout_offset_ms,
        readout_window_ms=args.readout_window_ms,
    )
    if args.quick:
        # Quick mode: 3 concepts × 10 trials, block_spec adjusted so each block
        # has at least 2 trials (pairwise-Jaccard needs ≥2). Smoke-tests the
        # full pipeline end-to-end in <2 min.
        cfg = MeasurementConfig(
            n_neurons=cfg.n_neurons,
            n_input=cfg.n_input,
            n_concepts=3,
            concept_size=cfg.concept_size,
            n_trials=10,
            stim_ms=cfg.stim_ms,
            rest_ms=cfg.rest_ms,
            stim_amplitude_pA=cfg.stim_amplitude_pA,
            dt_ms=cfg.dt_ms,
            seed=cfg.seed,
            enable_stdp=cfg.enable_stdp,
            block_spec=[(0, 3), (4, 7), (7, 10)],
            connectivity_k=cfg.connectivity_k,
            readout_offset_ms=cfg.readout_offset_ms,
            readout_window_ms=cfg.readout_window_ms,
        )
        print("[sim_stdp] QUICK mode: 3 concepts × 10 trials, blocks=[(0,3),(4,7),(7,10)]")

    result = run_measurement(cfg, args.sim_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result.to_json_dict(), f, indent=2)
    print(f"[sim_stdp] results written to {out_path}")

    # Exit code signals the GO/NO-GO gate for CI / orchestration.
    return 0 if result.go_nogo == "GO" else 2


if __name__ == "__main__":
    sys.exit(main())
