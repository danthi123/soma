"""v0.5 spatial-distillation multi-seed sanity check — Direction 4b Phase 2.

Gates whether the spatial-distillation stack is stable on the v0.5
capacity task before committing to the Phase 3 LoCoMo benchmark. Not
a performance test (synthetic 16-dim vectors with pseudo-text "regime-N"
descriptions have no natural semantic signal); this is a stability +
mechanism-fires sanity check.

Variants (2):
- synap_only_local          reference (frozen projections, no distill)
- synap_only_local_spatial  Direction 4b target:
                              projection_mode=learnable,
                              projection_distillation_target=llm_spatial,
                              position_mode=learnable,
                              projection_distillation_winners=3,
                              position_coupling_weight=1.0

Pseudo-text: one per regime (8 unique strings total) so the teacher
cache stays tiny (8 embeds) while distillation still has a stable
per-regime target.

Pass criteria (per plan §Phase 2):
1. All runs finite (no NaN in MSE or projections or positions).
2. Spatial MSE within 1.5x of reference.
3. Position-distance distribution shifts meaningfully during training
   — evidence that the coupling loss is actually reshaping positions.
   Measured as KS distance between initial and final pairwise-distance
   distributions across associator positions.

Seeds {0, 1, 42}. 2 variants * 3 seeds = 6 runs.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.environments import SequenceEnv, make_capacity_schedule
from soma.llm.embedders import CachedEmbedder, OllamaEmbedder


@dataclass
class RunResult:
    seed: int
    variant: str
    mse_per_step: list[float]
    regime_per_step: list[int]
    synap_events: int
    wall_time: float
    final_projections_finite: bool
    final_positions_finite: bool
    # Direction 4b position-shift telemetry
    initial_pairwise_distances: list[float]
    final_pairwise_distances: list[float]


def build_config(dim: int, variant: str, seed: int) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim,
        text_embed_dim=dim,
        associator_input_dim=dim,
        associator_hidden_dim=dim * 2,
        associator_output_dim=dim,
        integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2,
        integrator_output_dim=dim,
        pruning_interval=0,
        neurogenesis_interval=0,
        synaptogenesis_max_distance=0.5,
        seed=seed,
    )
    if variant == "synap_only_local":
        # Frozen projections, no distillation (reference baseline)
        pass
    elif variant == "synap_only_local_spatial":
        overrides.update(
            projection_mode="learnable",
            projection_distillation_target="llm_spatial",
            position_mode="learnable",
            projection_distillation_winners=3,
            position_coupling_weight=1.0,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return SOMAConfig.developmental(**overrides)


def _associator_pairwise_distances(ps: PredictiveSOMA) -> list[float]:
    """All pairwise L2 distances between associator positions. Fed to
    the KS-distance check in the findings. Uses detached tensors."""
    from soma.core.node import NodeType

    positions: list[torch.Tensor] = []
    for node in ps.soma.graph.all_nodes():
        if node.node_type != NodeType.ASSOCIATOR:
            continue
        if node.position is None:
            continue
        positions.append(node.position.detach().cpu().flatten())

    out: list[float] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            out.append((positions[i] - positions[j]).norm().item())
    return out


def run_variant(
    seed: int,
    variant: str,
    device: torch.device,
    dim: int,
    teacher: CachedEmbedder | None,
) -> RunResult:
    config = build_config(dim, variant, seed)
    ps = PredictiveSOMA(config, device=device)
    # Spatial variant needs a teacher; reference does not.
    if variant == "synap_only_local_spatial":
        assert teacher is not None, "spatial variant needs a teacher"
        ps.attach_teacher(teacher)

    # Record initial pairwise distances BEFORE any training.
    initial_dists = _associator_pairwise_distances(ps)

    schedule = make_capacity_schedule(dim=dim, steps_per_regime=500)
    env = SequenceEnv(dim=dim, schedule=schedule, seed=seed)
    env.reset()

    mses: list[float] = []
    regimes: list[int] = []
    total = env.schedule.total_steps
    t0 = time.perf_counter()
    for step in range(total):
        obs = env.step().to(device)
        regime_idx = env.current_regime
        pseudo_text = (
            f"regime-{regime_idx}"
            if variant == "synap_only_local_spatial"
            else None
        )
        result = ps.process_input(obs, source_text=pseudo_text)
        mses.append(float(result["prediction_error"]))
        regimes.append(regime_idx)
    wall = time.perf_counter() - t0

    final_dists = _associator_pairwise_distances(ps)

    synap_events = sum(
        1 for e in ps.soma.growth_log if e.get("event_type") == "synaptogenesis"
    )

    # Sanity: all projection weights finite?
    all_proj_finite = True
    for p in ps._input_projections.values():
        if not torch.isfinite(p).all():
            all_proj_finite = False
            break

    # Sanity: all positions finite?
    all_pos_finite = True
    from soma.core.node import NodeType

    for node in ps.soma.graph.all_nodes():
        if node.node_type != NodeType.ASSOCIATOR or node.position is None:
            continue
        if not torch.isfinite(node.position).all():
            all_pos_finite = False
            break

    return RunResult(
        seed=seed,
        variant=variant,
        mse_per_step=mses,
        regime_per_step=regimes,
        synap_events=synap_events,
        wall_time=wall,
        final_projections_finite=all_proj_finite,
        final_positions_finite=all_pos_finite,
        initial_pairwise_distances=initial_dists,
        final_pairwise_distances=final_dists,
    )


def regime_mean_mse(
    mses: list[float], regimes: list[int], warmup: int = 100,
) -> dict[int, float]:
    by_regime: dict[int, list[float]] = defaultdict(list)
    prev = -1
    start = 0
    for i, r in enumerate(regimes):
        if r != prev:
            start = i
            prev = r
        if i - start >= warmup:
            by_regime[r].append(mses[i])
    return {r: sum(v) / len(v) if v else 0.0 for r, v in by_regime.items()}


def ks_distance(a: list[float], b: list[float]) -> float:
    """Empirical 1D Kolmogorov-Smirnov distance between two samples.
    Range [0, 1]; 0 = identical distributions, 1 = fully disjoint.
    Implemented inline to avoid a scipy dep."""
    if not a or not b:
        return 0.0
    merged = sorted(set(a) | set(b))
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    na = len(a_sorted)
    nb = len(b_sorted)
    ia = 0
    ib = 0
    max_d = 0.0
    for v in merged:
        while ia < na and a_sorted[ia] <= v:
            ia += 1
        while ib < nb and b_sorted[ib] <= v:
            ib += 1
        d = abs(ia / na - ib / nb)
        if d > max_d:
            max_d = d
    return max_d


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    DIM = 16
    SEEDS = [0, 1, 42]
    variants = [
        "synap_only_local",
        "synap_only_local_spatial",
    ]
    print(f"Seeds: {SEEDS}, Variants: {variants}")

    cache_dir = Path("research/developmental/.teacher_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = CachedEmbedder(
        teacher=OllamaEmbedder(model="mxbai-embed-large"),
        cache_dir=str(cache_dir),
    )

    results: dict[int, dict[str, RunResult]] = {s: {} for s in SEEDS}
    for seed in SEEDS:
        print(f"\n=== seed={seed} ===")
        for variant in variants:
            r = run_variant(seed, variant, device, DIM, teacher)
            mse_overall = sum(r.mse_per_step) / max(1, len(r.mse_per_step))
            ks = ks_distance(r.initial_pairwise_distances, r.final_pairwise_distances)
            print(
                f"  seed={seed} {variant:32s} mse={mse_overall:.4f} "
                f"synap={r.synap_events:3d} "
                f"proj_ok={r.final_projections_finite} "
                f"pos_ok={r.final_positions_finite} "
                f"pos_ks={ks:.3f} "
                f"wall={r.wall_time:.0f}s"
            )
            results[seed][variant] = r

    # Per-regime mean across seeds
    print("\n" + "=" * 110)
    print("PER-REGIME MEAN MSE (across 3 seeds)")
    print("=" * 110)
    regime_names = [
        "mlp_2x16", "mlp_2x32", "mlp_3x16", "mlp_3x32",
        "mlp_2x64", "mlp_3x64", "mlp_4x32", "mlp_4x64",
    ]
    header = f"  {'Regime':12s}" + " ".join(f"{v:32s}" for v in variants)
    print(header)
    for regime_idx, name in enumerate(regime_names):
        row = f"  {name:12s}"
        for variant in variants:
            vals = [
                regime_mean_mse(
                    results[s][variant].mse_per_step,
                    results[s][variant].regime_per_step,
                ).get(regime_idx, 0.0)
                for s in SEEDS
            ]
            mean = sum(vals) / len(vals)
            row += f" {mean:.4f}+-{max(vals) - min(vals):.4f}       "
        print(row)

    # Spatial-vs-reference gates
    print("\n" + "=" * 110)
    print("PASS GATES (Direction 4b Phase 2)")
    print("=" * 110)
    any_nan = False
    for seed in SEEDS:
        for variant in variants:
            r = results[seed][variant]
            if not r.final_projections_finite:
                print(f"  FAIL: seed={seed} {variant}: non-finite projections")
                any_nan = True
            if not r.final_positions_finite:
                print(f"  FAIL: seed={seed} {variant}: non-finite positions")
                any_nan = True
            if any((v != v) or (v == float("inf")) for v in r.mse_per_step):
                print(f"  FAIL: seed={seed} {variant}: NaN/Inf in MSE")
                any_nan = True
    gate1_pass = not any_nan
    print(f"\n  Gate 1 (all finite): {'PASS' if gate1_pass else 'FAIL'}")

    # MSE ratio gate
    ref_mse_per_seed = {
        s: sum(results[s]["synap_only_local"].mse_per_step)
        / max(1, len(results[s]["synap_only_local"].mse_per_step))
        for s in SEEDS
    }
    spatial_mse_per_seed = {
        s: sum(results[s]["synap_only_local_spatial"].mse_per_step)
        / max(1, len(results[s]["synap_only_local_spatial"].mse_per_step))
        for s in SEEDS
    }
    ratios = [
        spatial_mse_per_seed[s] / max(1e-9, ref_mse_per_seed[s]) for s in SEEDS
    ]
    max_ratio = max(ratios)
    gate2_pass = max_ratio <= 1.5
    print(
        f"  Gate 2 (spatial/ref MSE <= 1.5x): "
        f"{'PASS' if gate2_pass else 'FAIL'} "
        f"(per-seed ratios={[round(r, 3) for r in ratios]}, max={max_ratio:.3f})"
    )

    # Position-shift gate
    ks_per_seed = [
        ks_distance(
            results[s]["synap_only_local_spatial"].initial_pairwise_distances,
            results[s]["synap_only_local_spatial"].final_pairwise_distances,
        )
        for s in SEEDS
    ]
    mean_ks = sum(ks_per_seed) / len(ks_per_seed)
    # Meaningful shift threshold: >= 0.1 mean KS (arbitrary but non-trivial).
    gate3_pass = mean_ks >= 0.1
    print(
        f"  Gate 3 (position-distance KS >= 0.1): "
        f"{'PASS' if gate3_pass else 'FAIL'} "
        f"(per-seed={[round(k, 3) for k in ks_per_seed]}, mean={mean_ks:.3f})"
    )

    overall = gate1_pass and gate2_pass and gate3_pass
    print(f"\n  OVERALL: {'PASS — proceed to Phase 3' if overall else 'FAIL — investigate'}")

    # Save raw data
    out_path = Path(
        "research/developmental/results/env_sequence_v05_spatial_multiseed.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = {
        "schedule": regime_names,
        "seeds": SEEDS,
        "variants": variants,
        "gates": {
            "gate1_all_finite": gate1_pass,
            "gate2_mse_ratio_le_1_5": gate2_pass,
            "gate2_per_seed_ratios": ratios,
            "gate3_position_ks_ge_0_1": gate3_pass,
            "gate3_per_seed_ks": ks_per_seed,
            "overall": overall,
        },
        "runs": {
            str(s): {
                v: {
                    "mse_mean": (
                        sum(results[s][v].mse_per_step)
                        / max(1, len(results[s][v].mse_per_step))
                    ),
                    "synap_events": results[s][v].synap_events,
                    "wall_time": results[s][v].wall_time,
                    "finite_projections": results[s][v].final_projections_finite,
                    "finite_positions": results[s][v].final_positions_finite,
                    "regime_means": regime_mean_mse(
                        results[s][v].mse_per_step,
                        results[s][v].regime_per_step,
                    ),
                    "position_ks_distance": ks_distance(
                        results[s][v].initial_pairwise_distances,
                        results[s][v].final_pairwise_distances,
                    ),
                    "n_initial_pairs": len(results[s][v].initial_pairwise_distances),
                    "n_final_pairs": len(results[s][v].final_pairwise_distances),
                }
                for v in variants
            }
            for s in SEEDS
        },
    }
    with open(out_path, "w") as f:
        json.dump(serialized, f, indent=2, default=str)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
