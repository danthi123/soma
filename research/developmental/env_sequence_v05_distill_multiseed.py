"""v0.5 distillation multi-seed sanity check — Direction 4a Phase 2.

Tests that adding LLM distillation to learnable projections doesn't
destabilize the v0.5 capacity task. Not a performance test (semantic
distillation on synthetic random vectors has no natural signal); this
is a sanity gate before Phase 3 LoCoMo.

Variants (4):
- synap_only_local              reference (frozen random projections)
- synap_only_local_learnable    learnable projections, no distillation
- synap_only_local_distilled    learnable + teacher-distillation (alpha=0.5)
- synap_only_local_distilled10  learnable + teacher-distillation (alpha=1.0)

Pseudo-text: each 16-dim synthetic vector gets a description
"regime-{regime_idx}-step-{step}" that the teacher embeds. This gives
the distillation loss SOMETHING to train against, even though it has
no natural relationship to the v0.5 prediction task.

Success criteria:
- All variants complete without NaN
- Distilled variants' MSE within 2x of synap_only_local
- Final MSE ordering: distillation shouldn't catastrophically hurt

Seeds {0, 1, 42}. 4 variants x 3 seeds = 12 runs ~ 40-60 min.
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


def build_config(dim: int, variant: str, seed: int) -> SOMAConfig:
    overrides: dict[str, Any] = dict(
        sensor_output_dim=dim, text_embed_dim=dim,
        associator_input_dim=dim, associator_hidden_dim=dim * 2,
        associator_output_dim=dim, integrator_input_dim=dim,
        integrator_hidden_dim=dim * 2, integrator_output_dim=dim,
        pruning_interval=0,
        neurogenesis_interval=0,
        synaptogenesis_max_distance=0.5,
        seed=seed,
    )
    if variant == "synap_only_local":
        # Frozen projections, no distillation (reference baseline)
        pass
    elif variant == "synap_only_local_learnable":
        overrides.update(projection_mode="learnable")
    elif variant == "synap_only_local_distilled":
        overrides.update(
            projection_mode="learnable",
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=0.5,
        )
    elif variant == "synap_only_local_distilled10":
        overrides.update(
            projection_mode="learnable",
            projection_distillation_target="llm_embedding",
            projection_distillation_weight=1.0,
        )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return SOMAConfig.developmental(**overrides)


def run_variant(
    seed: int,
    variant: str,
    device: torch.device,
    dim: int,
    teacher: CachedEmbedder | None,
) -> RunResult:
    config = build_config(dim, variant, seed)
    ps = PredictiveSOMA(config, device=device)
    if "distilled" in variant:
        assert teacher is not None, "distilled variant needs a teacher"
        ps.attach_teacher(teacher)

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
        # Pseudo-text: 8 unique strings total (one per regime). Keeps
        # teacher-cache size sane (8 embeds per model) and still gives
        # distillation a stable, per-regime target to converge to.
        pseudo_text = f"regime-{regime_idx}" if teacher is not None else None
        result = ps.process_input(obs, source_text=pseudo_text)
        mses.append(float(result["prediction_error"]))
        regimes.append(regime_idx)
    wall = time.perf_counter() - t0

    synap_events = sum(
        1 for e in ps.soma.growth_log
        if e.get("event_type") == "synaptogenesis"
    )
    # Sanity: all projection weights still finite?
    all_finite = True
    for p in ps._input_projections.values():
        if not torch.isfinite(p).all():
            all_finite = False
            break

    return RunResult(
        seed=seed,
        variant=variant,
        mse_per_step=mses,
        regime_per_step=regimes,
        synap_events=synap_events,
        wall_time=wall,
        final_projections_finite=all_finite,
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    DIM = 16
    SEEDS = [0, 1, 42]
    variants = [
        "synap_only_local",
        "synap_only_local_learnable",
        "synap_only_local_distilled",
        "synap_only_local_distilled10",
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
            t0 = time.perf_counter()
            r = run_variant(seed, variant, device, DIM, teacher)
            mse_overall = sum(r.mse_per_step) / max(1, len(r.mse_per_step))
            print(
                f"  seed={seed} {variant:34s} mse={mse_overall:.4f} "
                f"synap={r.synap_events:3d} "
                f"finite={r.final_projections_finite} "
                f"wall={r.wall_time:.0f}s"
            )
            results[seed][variant] = r

    # Per-regime mean across seeds
    print("\n" + "=" * 110)
    print("PER-REGIME MEAN MSE (across 3 seeds)")
    print("=" * 110)
    regime_names = ["mlp_2x16", "mlp_2x32", "mlp_3x16", "mlp_3x32",
                    "mlp_2x64", "mlp_3x64", "mlp_4x32", "mlp_4x64"]
    header = f"  {'Regime':12s}" + " ".join(f"{v:28s}" for v in variants)
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
            row += f" {mean:.4f}+-{max(vals)-min(vals):.4f}       "
        print(row)

    # Sanity flags
    print("\nSanity checks:")
    any_nan = False
    for seed in SEEDS:
        for variant in variants:
            r = results[seed][variant]
            if not r.final_projections_finite:
                print(f"  !!! seed={seed} {variant}: NON-FINITE projections !!!")
                any_nan = True
            if any(
                (v != v) or (v == float("inf"))  # NaN check
                for v in r.mse_per_step
            ):
                print(f"  !!! seed={seed} {variant}: NaN/Inf in MSE !!!")
                any_nan = True
    if not any_nan:
        print("  All runs: projections finite, MSE finite. PASS.")

    # Save raw data
    out_path = Path(
        "research/developmental/results/env_sequence_v05_distill_multiseed.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = {
        "schedule": regime_names,
        "seeds": SEEDS,
        "variants": variants,
        "runs": {
            str(s): {
                v: {
                    "mse_mean": (
                        sum(results[s][v].mse_per_step)
                        / max(1, len(results[s][v].mse_per_step))
                    ),
                    "synap_events": results[s][v].synap_events,
                    "wall_time": results[s][v].wall_time,
                    "finite": results[s][v].final_projections_finite,
                    "regime_means": regime_mean_mse(
                        results[s][v].mse_per_step,
                        results[s][v].regime_per_step,
                    ),
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
