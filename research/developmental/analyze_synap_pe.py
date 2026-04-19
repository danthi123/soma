"""Analysis helper for env_sequence_v05_synap_pe.py results.

Prints a side-by-side comparison of (a) this run's five variants and
(b) the prior 2x2 baseline from env_sequence_v05_growth_2x2.json. The
comparison against the 2x2 anchors the interpretation: a _pe variant
is only meaningful relative to its non-_pe counterpart with everything
else held fixed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def regime_mean_mse(
    mses: list[float], regimes: list[int], warmup: int = 100
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


def print_table(
    label: str,
    variants: list[str],
    data: dict[str, dict],
    regime_names: list[str],
) -> None:
    print("=" * (14 + 16 * len(variants)))
    print(f"{label}")
    print("=" * (14 + 16 * len(variants)))
    print(f"  {'regime':<12}", end="")
    for v in variants:
        print(f"{v:<16}", end="")
    print()
    for regime_idx, regime in enumerate(regime_names):
        print(f"  {regime:<12}", end="")
        for v in variants:
            vd = data[v]
            means = regime_mean_mse(vd["mse_per_step"], vd["regime_per_step"])
            print(f"{means.get(regime_idx, 0.0):<16.4f}", end="")
        print()


def mean_regime_improvement(
    baseline: dict, treatment: dict, regime_names: list[str]
) -> dict[int, float]:
    """Per-regime (treatment - baseline) in mse terms; negative is better."""
    b_means = regime_mean_mse(
        baseline["mse_per_step"], baseline["regime_per_step"]
    )
    t_means = regime_mean_mse(
        treatment["mse_per_step"], treatment["regime_per_step"]
    )
    return {
        i: t_means.get(i, 0.0) - b_means.get(i, 0.0)
        for i in range(len(regime_names))
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1].parent / "research" / "developmental" / "results"
    synap_pe_path = root / "env_sequence_v05_synap_pe.json"
    two_by_two_path = root / "env_sequence_v05_growth_2x2.json"

    if not synap_pe_path.exists():
        print(f"No synap_pe results yet at {synap_pe_path}")
        return

    synap_pe = json.loads(synap_pe_path.read_text())
    regime_names = synap_pe["config"]["regimes"]

    # Main table: all five synap_pe variants side by side.
    variants = list(synap_pe["variants"].keys())
    print()
    print_table(
        "SYNAP_PE RUN — REGIME MEAN MSE (lower better)",
        variants,
        synap_pe["variants"],
        regime_names,
    )

    print()
    print("FINAL GRAPH STATE:")
    for v in variants:
        vd = synap_pe["variants"][v]
        neuro = len(vd["neurogenesis_event_steps"])
        synap = len(vd["synaptogenesis_event_steps"])
        print(
            f"  {v:<16} nodes={int(vd['n_nodes'][-1]):3d} "
            f"edges={int(vd['n_edges'][-1]):4d} "
            f"synap_events={synap:3d} neuro_events={neuro:2d} "
            f"ema_pairs={vd['final_pe_ema_count']:3d} "
            f"ema_neg%={vd['final_pe_ema_negative_fraction'] * 100:3.0f}"
        )

    # Treatment deltas: _pe vs non-_pe arm.
    print()
    print("=" * 85)
    print("TREATMENT DELTA: synap_only_pe - synap_only (negative = _pe helps)")
    print("=" * 85)
    if "synap_only" in variants and "synap_only_pe" in variants:
        delta = mean_regime_improvement(
            synap_pe["variants"]["synap_only"],
            synap_pe["variants"]["synap_only_pe"],
            regime_names,
        )
        for i, name in enumerate(regime_names):
            sign = "  "
            if delta[i] < -0.0001:
                sign = "+ "
            elif delta[i] > 0.0001:
                sign = "- "
            print(f"  {name:<12} delta={delta[i]:+.4f}  {sign}")

    print()
    print("=" * 85)
    print("TREATMENT DELTA: full_pe - full (negative = _pe helps)")
    print("=" * 85)
    if "full" in variants and "full_pe" in variants:
        delta = mean_regime_improvement(
            synap_pe["variants"]["full"],
            synap_pe["variants"]["full_pe"],
            regime_names,
        )
        for i, name in enumerate(regime_names):
            sign = "  "
            if delta[i] < -0.0001:
                sign = "+ "
            elif delta[i] > 0.0001:
                sign = "- "
            print(f"  {name:<12} delta={delta[i]:+.4f}  {sign}")

    # Compare to prior 2x2 as a sanity check on the no_growth / synap_only
    # / full baselines.
    if two_by_two_path.exists():
        tbt = json.loads(two_by_two_path.read_text())
        print()
        print("=" * 85)
        print("SANITY: common variants vs. prior 2x2 run (same schedule, same seed)")
        print("=" * 85)
        for v in ("no_growth", "synap_only", "full"):
            if v in variants and v in tbt["variants"]:
                cur = regime_mean_mse(
                    synap_pe["variants"][v]["mse_per_step"],
                    synap_pe["variants"][v]["regime_per_step"],
                )
                prev = regime_mean_mse(
                    tbt["variants"][v]["mse_per_step"],
                    tbt["variants"][v]["regime_per_step"],
                )
                # Report mean across all regimes
                cur_mean = sum(cur.values()) / len(cur)
                prev_mean = sum(prev.values()) / len(prev)
                print(
                    f"  {v:<14} overall_mean_mse: cur={cur_mean:.4f} "
                    f"prev={prev_mean:.4f} delta={cur_mean - prev_mean:+.4f}"
                )

    print()
    print("INTERPRETATION GUIDE:")
    print("  ema_neg% stays near 100 in _pe runs => global PE dropping each step,")
    print("    filter almost never fires (supervision ≈ no supervision).")
    print("  ema_neg% below 50 => filter regularly suppresses admissions,")
    print("    likely during regime transitions where PE spikes.")
    print("  synap_only_pe between synap_only and no_growth => filter partial help.")
    print("  synap_only_pe == no_growth => filter too aggressive (rejects all).")
    print("  synap_only_pe < no_growth => synap genuinely useful once filtered.")


if __name__ == "__main__":
    main()
