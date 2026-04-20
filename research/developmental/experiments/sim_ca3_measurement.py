#!/usr/bin/env python3
"""Path B Phase 1: sim CA3 measurement harness.

Drives the neural-simulator (E:/Documents/Projects/sim) with the
HIPPOCAMPUS_CA3_RECURRENT preset, injects a fixed set of "concept"
input patterns multiple times, and measures the readout population's
sparse-code properties:

  - within-concept stability   (Jaccard across 10 repeated trials)
  - between-concept separation (Jaccard across concept pairs)
  - sparsity                   (fraction of readout neurons active)

GO/NO-GO gate (from docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md):

  (a) within-Jaccard  > 0.5
  (b) between-Jaccard < within-Jaccard by factor > 1.5

If either fails, sim is not doing pattern separation on these stimuli
and Path B's numpy-primitive calibration has no ground truth to match;
we revisit design.

Stimulus injection path
-----------------------
We bypass sim's ExperimentEngine and write directly to
``sim_bridge.cp_external_input_current``. That array is initialized
once and never reset per-step (for the Izhikevich model — see
sim/bridge.py:744), so overwriting it before each ``_run_one_simulation_step``
gives precise per-trial control over which input neurons receive drive.

Usage
-----
    python -m research.developmental.experiments.sim_ca3_measurement \
        [--n-concepts 50] [--n-trials 10] [--n-neurons 5000] \
        [--stim-ms 100] [--rest-ms 200] [--stim-amplitude-pA 500] \
        [--sim-path E:/Documents/Projects/sim] \
        [--out research/developmental/results/sim_ca3_baseline.json] \
        [--quick]   # tiny smoke-test config (5 concepts, 2 trials)

Outputs
-------
  - ``research/developmental/results/sim_ca3_baseline.json`` — metrics
    summary + per-trial active-neuron sets (sorted int lists so JSON
    is reproducible).
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

    print("[self-test] metric functions OK")


# ---------------------------------------------------------------------------
# Sim driver
# ---------------------------------------------------------------------------


@dataclass
class MeasurementConfig:
    n_neurons: int = 5000
    n_input: int = 1000            # first ``n_input`` neurons receive direct drive
    n_concepts: int = 50
    concept_size: int = 100        # active input neurons per concept (of ``n_input``)
    n_trials: int = 10
    stim_ms: int = 100
    rest_ms: int = 200
    stim_amplitude_pA: float = 500.0
    dt_ms: float = 1.0
    seed: int = 42
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
    # Full readout spike count per neuron over the stim window. Stored as a
    # sparse (index, count) list to keep JSON output compact — only neurons
    # that spiked at least once are included.
    spike_counts: list[tuple[int, int]]  # [(neuron_idx, count), ...]
    # Derived "codes" at three sparsity levels (top-k most active).
    top_k_primary: list[int]
    top_k_relaxed: list[int]
    top_k_strict: list[int]
    # Any-fired set (dense, baseline view for diagnosis only).
    any_fired: list[int]
    total_spikes: int


@dataclass
class MetricView:
    """Jaccard / separation results for one code-extraction recipe."""
    name: str
    within_concept_jaccard: float
    between_concept_jaccard: float
    separation_ratio: float
    sparsity: float

    def to_json_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeasurementResult:
    config: dict
    patterns: list[list[int]]                # per-concept input indices
    trials: list[TrialResult] = field(default_factory=list)
    views: list[MetricView] = field(default_factory=list)
    go_nogo: str = ""
    primary_view: str = "top_k_primary"
    wall_time_s: float = 0.0

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "patterns": self.patterns,
            "trials": [asdict(t) for t in self.trials],
            "views": [v.to_json_dict() for v in self.views],
            "go_nogo": self.go_nogo,
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
    # Disable plasticity so repeat trials see the same network state.
    # Phase 1 measures intrinsic CA3 dynamics; plasticity-enabled measurements
    # are Phase 4 territory.
    core_cfg.enable_hebbian_learning = False
    core_cfg.enable_stdp = False
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
    """Drive the sim, collect per-trial active neuron sets, compute metrics."""
    print(f"[sim_ca3] importing sim from {sim_path}...")
    cp, _SB, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel = _import_sim(sim_path)

    print(f"[sim_ca3] building SimulationBridge "
          f"(n={cfg.n_neurons}, profile=HIPPOCAMPUS_CA3_RECURRENT, model=Izhikevich)...")
    sim_bridge, core_cfg = _build_sim_bridge(
        cp, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel, cfg
    )

    patterns = generate_concept_patterns(cfg)
    print(f"[sim_ca3] generated {len(patterns)} concept patterns "
          f"({cfg.concept_size} active of {cfg.n_input} input neurons each)")

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

            trial = TrialResult(
                concept_id=concept_id,
                trial_id=trial_id,
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
                print(f"[sim_ca3] concept {concept_id+1}/{cfg.n_concepts} "
                      f"trial {trial_id+1}/{cfg.n_trials} | "
                      f"{pct:5.1f}% | {rate:.0f} steps/s | ETA {eta:.0f}s | "
                      f"top-k primary active: {len(trial.top_k_primary)} | "
                      f"total spikes: {trial.total_spikes}")
                last_report = now

    result.wall_time_s = time.time() - t_wall_start
    print(f"[sim_ca3] simulation done in {result.wall_time_s:.1f}s "
          f"({total_steps_done} steps)")

    # --- Metrics: one view per code-extraction recipe. ---
    view_specs = [
        ("top_k_strict",  lambda t: t.top_k_strict),
        ("top_k_primary", lambda t: t.top_k_primary),
        ("top_k_relaxed", lambda t: t.top_k_relaxed),
        ("any_fired",     lambda t: t.any_fired),
    ]

    for view_name, extractor in view_specs:
        concept_trials: list[list[list[int]]] = [[] for _ in range(cfg.n_concepts)]
        for t in result.trials:
            concept_trials[t.concept_id].append(extractor(t))
        within_vals = [mean_pairwise_jaccard(ct) for ct in concept_trials]
        within_mean = float(np.mean(within_vals)) if within_vals else 0.0
        rng = np.random.default_rng(cfg.seed)  # reset per-view for reproducibility
        between_mean = mean_between_concept_jaccard(concept_trials, rng)
        sparsity = mean_sparsity(
            (extractor(t) for t in result.trials),
            n_neurons=(readout_end - readout_start),
        )
        ratio = (within_mean / between_mean) if between_mean > 0 else float("inf")
        result.views.append(MetricView(
            name=view_name,
            within_concept_jaccard=within_mean,
            between_concept_jaccard=between_mean,
            separation_ratio=ratio,
            sparsity=sparsity,
        ))

    # GO/NO-GO is evaluated on the primary view (top_k_primary = 2% sparsity).
    primary = next(v for v in result.views if v.name == result.primary_view)
    gate_a = primary.within_concept_jaccard > 0.5
    gate_b = (
        primary.between_concept_jaccard <= 0.0
        or primary.within_concept_jaccard / primary.between_concept_jaccard > 1.5
    )
    result.go_nogo = "GO" if (gate_a and gate_b) else "NO-GO"

    print()
    print("=" * 72)
    print("PHASE 1 RESULTS: sim CA3 baseline")
    print("=" * 72)
    print(f"  readout window: neurons [{readout_start}, {readout_end})")
    print(f"  trials: {cfg.n_concepts} concepts × {cfg.n_trials} trials "
          f"= {len(result.trials)} total")
    print()
    print(f"  {'view':>16s} | {'within J':>8s} | {'between J':>9s} | {'sep ratio':>9s} | {'sparsity':>8s}")
    print(f"  {'-'*16}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}")
    for v in result.views:
        ratio_str = f"{v.separation_ratio:8.2f}" if v.separation_ratio != float("inf") else "     inf"
        print(f"  {v.name:>16s} |  {v.within_concept_jaccard:6.4f} |   {v.between_concept_jaccard:6.4f} |  {ratio_str} |  {v.sparsity:6.4f}")
    print()
    print(f"  PRIMARY VIEW: {result.primary_view}")
    print(f"  gate (a) within > 0.5:                 {'PASS' if gate_a else 'FAIL'} "
          f"(got {primary.within_concept_jaccard:.4f})")
    print(f"  gate (b) sep ratio > 1.5:              {'PASS' if gate_b else 'FAIL'} "
          f"(got {primary.separation_ratio:.2f})")
    print(f"  VERDICT:                               {result.go_nogo}")
    print("=" * 72)

    return result


def main() -> int:
    _self_test()

    parser = argparse.ArgumentParser(description="Path B Phase 1: sim CA3 measurement")
    parser.add_argument("--n-neurons", type=int, default=5000)
    parser.add_argument("--n-input", type=int, default=1000)
    parser.add_argument("--n-concepts", type=int, default=50)
    parser.add_argument("--concept-size", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--stim-ms", type=int, default=100)
    parser.add_argument("--rest-ms", type=int, default=200)
    parser.add_argument("--stim-amplitude-pA", type=float, default=500.0)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connectivity-k", type=int, default=50,
                        help="Outgoing synapses per neuron (overrides CA3 profile's k=10)")
    parser.add_argument("--readout-offset-ms", type=int, default=0,
                        help="Start counting spikes this many ms after stim onset")
    parser.add_argument("--readout-window-ms", type=int, default=None,
                        help="Spike-count window length in ms (default: until stim ends)")
    parser.add_argument("--sim-path", type=str,
                        default=os.environ.get("SIM_PATH", "E:/Documents/Projects/sim"))
    parser.add_argument("--out", type=str,
                        default="research/developmental/results/sim_ca3_baseline.json")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny smoke-test config: 5 concepts × 2 trials (~3s sim)")
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
        connectivity_k=args.connectivity_k,
        readout_offset_ms=args.readout_offset_ms,
        readout_window_ms=args.readout_window_ms,
    )
    if args.quick:
        cfg = MeasurementConfig(
            n_neurons=cfg.n_neurons,
            n_input=cfg.n_input,
            n_concepts=5,
            concept_size=cfg.concept_size,
            n_trials=2,
            stim_ms=cfg.stim_ms,
            rest_ms=cfg.rest_ms,
            stim_amplitude_pA=cfg.stim_amplitude_pA,
            dt_ms=cfg.dt_ms,
            seed=cfg.seed,
            connectivity_k=cfg.connectivity_k,
            readout_offset_ms=cfg.readout_offset_ms,
            readout_window_ms=cfg.readout_window_ms,
        )
        print("[sim_ca3] QUICK mode: 5 concepts × 2 trials")

    result = run_measurement(cfg, args.sim_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result.to_json_dict(), f, indent=2)
    print(f"[sim_ca3] results written to {out_path}")

    # Exit code signals the GO/NO-GO gate for CI / orchestration.
    return 0 if result.go_nogo == "GO" else 2


if __name__ == "__main__":
    sys.exit(main())
