#!/usr/bin/env python3
"""Calibrate numpy sparse-code primitives against the Phase 1 sim baseline.

Reads sim measurements from
``research/developmental/results/sim_ca3_baseline.json`` and runs the
same 50-concept stimulus protocol through the numpy k-WTA +
pattern-separation primitives. Reports within/between Jaccard at each
top-k view so we can compare directly to the sim's numbers.

Also supports standalone mode (no sim baseline) that sweeps k, dim, and
separation strength on synthetic concept embeddings — this is the
useful-when-sim-baseline-is-NO-GO fallback.

Usage:
    # Calibrate against sim baseline
    python -m benchmarks.sparse_codes_calibration \
        --sim-baseline research/developmental/results/sim_ca3_baseline.json \
        --out research/developmental/results/sparse_codes_calibration.json

    # Synthetic sweep (no sim baseline needed)
    python -m benchmarks.sparse_codes_calibration --synthetic \
        --out research/developmental/results/sparse_codes_synthetic_sweep.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from soma.memory.sparse_codes import (
    SparseCode,
    code_similarity,
    kwta,
    pattern_separate,
)


# ---------------------------------------------------------------------------
# Concept embedding generation
# ---------------------------------------------------------------------------


def generate_concept_embeddings(
    n_concepts: int,
    n_trials: int,
    embed_dim: int,
    noise_std: float,
    overlap: float,
    seed: int,
) -> list[list[np.ndarray]]:
    """Synthetic dense concept embeddings with per-trial noise.

    Each concept has a fixed "mean vector" in ``R^embed_dim``. Each trial
    is ``mean + N(0, noise_std)`` noise. Concept means share a common
    component of weight ``overlap`` — small overlap means concepts are
    near-orthogonal, large overlap means similar concepts.

    Returns ``concept_trials[c][t]`` = np.ndarray, shape ``(embed_dim,)``.
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(embed_dim).astype(np.float32)
    common /= np.linalg.norm(common) + 1e-12

    concept_means = []
    for _ in range(n_concepts):
        v = rng.standard_normal(embed_dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        mean = (1.0 - overlap) * v + overlap * common
        concept_means.append(mean)

    concept_trials: list[list[np.ndarray]] = []
    for mean in concept_means:
        trials = [
            (mean + rng.standard_normal(embed_dim).astype(np.float32) * noise_std)
            for _ in range(n_trials)
        ]
        concept_trials.append(trials)
    return concept_trials


# ---------------------------------------------------------------------------
# Metrics (same shapes as sim_ca3_measurement.py)
# ---------------------------------------------------------------------------


def mean_pairwise_jaccard_codes(codes: list[SparseCode]) -> float:
    pairs = list(itertools.combinations(range(len(codes)), 2))
    if not pairs:
        return 1.0
    total = sum(code_similarity(codes[i], codes[j]) for i, j in pairs)
    return total / len(pairs)


def mean_between_concept_jaccard_codes(
    concept_codes: list[list[SparseCode]],
    rng: np.random.Generator,
) -> float:
    n = len(concept_codes)
    pairs = list(itertools.combinations(range(n), 2))
    if not pairs:
        return 0.0
    total = 0.0
    for i, j in pairs:
        ci = concept_codes[i][rng.integers(0, len(concept_codes[i]))]
        cj = concept_codes[j][rng.integers(0, len(concept_codes[j]))]
        total += code_similarity(ci, cj)
    return total / len(pairs)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    k: int
    dim: int
    sep_strength: float
    within_jaccard: float
    between_jaccard: float
    separation_ratio: float
    sparsity: float


def run_single_config(
    concept_trials: list[list[np.ndarray]],
    k: int,
    dim: int,
    sep_strength: float,
    seed: int,
) -> SweepResult:
    """Encode with kwta, apply pattern_separate to concept-level stored
    codes, and measure retrieval metrics.

    Model of the retrieval path:
      - First trial of each concept produces the STORED code.
      - pattern_separate is applied across stored codes (ingest-time
        orthogonalisation).
      - Remaining trials are QUERIES — we measure how well they match
        their own concept's stored code vs. other concepts' stored codes.

    Metrics are then:
      - within_jaccard  = mean Jaccard(query_t, stored_c) for t in 1..N-1,
                          per concept, averaged across concepts
      - between_jaccard = mean Jaccard(query from concept i, stored c
                          for concept j ≠ i), averaged
    """
    rng = np.random.default_rng(seed)
    embed_dim = concept_trials[0][0].shape[0]
    projection = rng.standard_normal((dim, embed_dim)).astype(np.float32)

    # Encode trial 0 of each concept as the stored code; trials 1+ as queries.
    stored_raw = [
        kwta(trials[0], k=k, dim=dim, projection=projection)
        for trials in concept_trials
    ]
    stored = pattern_separate(stored_raw, strength=sep_strength)

    query_codes: list[list[SparseCode]] = [
        [kwta(t, k=k, dim=dim, projection=projection) for t in trials[1:]]
        for trials in concept_trials
    ]

    # within: how well does a concept's query trial match its own stored code?
    within_per_concept = []
    for c in range(len(concept_trials)):
        if not query_codes[c]:
            continue
        sims = [code_similarity(q, stored[c]) for q in query_codes[c]]
        within_per_concept.append(float(np.mean(sims)))
    within = float(np.mean(within_per_concept)) if within_per_concept else 0.0

    # between: how much does a concept's query match a DIFFERENT concept's stored code?
    # Sample one query per concept pair for fair comparison.
    rng_metric = np.random.default_rng(seed)
    pairs = list(itertools.combinations(range(len(concept_trials)), 2))
    between_sum = 0.0
    n_between_pairs = 0
    for i, j in pairs:
        if not query_codes[i] or not query_codes[j]:
            continue
        # Cross: query from i vs stored j, and query from j vs stored i.
        q_from_i = query_codes[i][rng_metric.integers(0, len(query_codes[i]))]
        q_from_j = query_codes[j][rng_metric.integers(0, len(query_codes[j]))]
        between_sum += code_similarity(q_from_i, stored[j])
        between_sum += code_similarity(q_from_j, stored[i])
        n_between_pairs += 2
    between = between_sum / n_between_pairs if n_between_pairs else 0.0

    sparsity = k / dim
    ratio = within / between if between > 0 else float("inf")
    return SweepResult(
        k=k, dim=dim, sep_strength=sep_strength,
        within_jaccard=within, between_jaccard=between,
        separation_ratio=ratio, sparsity=sparsity,
    )


def run_sweep(
    concept_trials: list[list[np.ndarray]],
    k_values: list[int],
    dim_values: list[int],
    sep_values: list[float],
    seed: int,
) -> list[SweepResult]:
    results = []
    for k, dim, sep in itertools.product(k_values, dim_values, sep_values):
        if k > dim:
            continue
        result = run_single_config(concept_trials, k, dim, sep, seed)
        results.append(result)
        print(f"  k={k:4d} dim={dim:5d} sep={sep:.2f}  "
              f"within={result.within_jaccard:.4f} "
              f"between={result.between_jaccard:.4f} "
              f"ratio={result.separation_ratio:5.2f} "
              f"sparsity={result.sparsity:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Sparse code calibration")
    parser.add_argument("--sim-baseline", type=str, default=None,
                        help="Path to sim_ca3_baseline.json (for calibration mode)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Synthetic sweep mode (no sim baseline needed)")
    parser.add_argument("--n-concepts", type=int, default=50)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=1024,
                        help="Simulated dense embedding dimension")
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="Per-trial noise std (controls within-concept stability)")
    parser.add_argument("--overlap", type=float, default=0.1,
                        help="Shared mean component weight (controls between-concept overlap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--dim-values", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--sep-values", type=float, nargs="+", default=[0.0, 0.3, 0.7])
    parser.add_argument("--out", type=str,
                        default="research/developmental/results/sparse_codes_calibration.json")
    args = parser.parse_args()

    print(f"[calib] generating {args.n_concepts} concept embeddings "
          f"× {args.n_trials} trials, embed_dim={args.embed_dim}...")
    concept_trials = generate_concept_embeddings(
        n_concepts=args.n_concepts,
        n_trials=args.n_trials,
        embed_dim=args.embed_dim,
        noise_std=args.noise_std,
        overlap=args.overlap,
        seed=args.seed,
    )

    t0 = time.time()
    print(f"[calib] sweeping k={args.k_values} dim={args.dim_values} "
          f"sep={args.sep_values}")
    results = run_sweep(
        concept_trials,
        k_values=args.k_values,
        dim_values=args.dim_values,
        sep_values=args.sep_values,
        seed=args.seed,
    )
    wall = time.time() - t0

    # Optional: load sim baseline for side-by-side comparison.
    sim_summary = None
    if args.sim_baseline:
        path = Path(args.sim_baseline)
        if path.exists():
            with path.open("r") as f:
                sim_data = json.load(f)
            sim_summary = {
                "views": sim_data.get("views", []),
                "go_nogo": sim_data.get("go_nogo", "unknown"),
            }

    out = {
        "config": {
            "n_concepts": args.n_concepts,
            "n_trials": args.n_trials,
            "embed_dim": args.embed_dim,
            "noise_std": args.noise_std,
            "overlap": args.overlap,
            "seed": args.seed,
            "k_values": args.k_values,
            "dim_values": args.dim_values,
            "sep_values": args.sep_values,
        },
        "wall_time_s": wall,
        "results": [
            {
                "k": r.k, "dim": r.dim, "sep_strength": r.sep_strength,
                "within_jaccard": r.within_jaccard,
                "between_jaccard": r.between_jaccard,
                "separation_ratio": r.separation_ratio,
                "sparsity": r.sparsity,
            }
            for r in results
        ],
        "sim_baseline": sim_summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    # Summary picks the single best-separation config.
    best = max(results, key=lambda r: r.separation_ratio)
    print()
    print("=" * 64)
    print("CALIBRATION SUMMARY")
    print("=" * 64)
    print(f"  concepts={args.n_concepts} × trials={args.n_trials}  embed_dim={args.embed_dim}")
    print(f"  noise_std={args.noise_std}  overlap={args.overlap}")
    print(f"  wall time: {wall:.1f}s across {len(results)} configs")
    print()
    print(f"  BEST SEPARATION: k={best.k} dim={best.dim} sep={best.sep_strength:.2f}")
    print(f"    within={best.within_jaccard:.4f}  between={best.between_jaccard:.4f}  "
          f"ratio={best.separation_ratio:.2f}  sparsity={best.sparsity:.4f}")
    if sim_summary:
        print()
        print("  SIM BASELINE (for comparison):")
        for v in sim_summary["views"]:
            print(f"    {v['name']}: within={v['within_concept_jaccard']:.4f} "
                  f"between={v['between_concept_jaccard']:.4f} "
                  f"ratio={v['separation_ratio']:.2f} "
                  f"sparsity={v['sparsity']:.4f}")
        print(f"    verdict: {sim_summary['go_nogo']}")
    print("=" * 64)
    print(f"  results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
