"""Run LongMemEval benchmark: SOMA vs baseline comparison.

Evaluates both SOMA (memory-augmented retrieval) and baseline (raw
context window) on the LongMemEval benchmark and produces a side-by-side
comparison report.

Usage::

    # Quick smoke test (5 items, oracle variant)
    python -m benchmarks.industry.longmemeval.run_longmemeval --limit 5

    # Full oracle evaluation
    python -m benchmarks.industry.longmemeval.run_longmemeval

    # Specific model
    python -m benchmarks.industry.longmemeval.run_longmemeval --model qwen3:14b-q8_0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from benchmarks.industry.longmemeval.evaluate_baseline import (
    run_evaluation as run_baseline,
)
from benchmarks.industry.longmemeval.evaluate_soma import (
    run_evaluation as run_soma,
)
from benchmarks.industry.longmemeval.metrics import LongMemEvalScores

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("benchmarks/industry/longmemeval/results")


def _print_comparison(
    soma_scores: LongMemEvalScores,
    baseline_scores: LongMemEvalScores,
    model: str,
    variant: str,
) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'=' * 70}")
    print(f"LongMemEval Comparison — {variant} variant, model={model}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<12s} {'SOMA':>10s} {'Baseline':>10s} {'Delta':>10s}")
    print("-" * 44)

    metrics = [
        ("F1", soma_scores.f1, baseline_scores.f1),
        ("EM", soma_scores.em, baseline_scores.em),
        ("ROUGE-1", soma_scores.rouge1, baseline_scores.rouge1),
        ("ROUGE-2", soma_scores.rouge2, baseline_scores.rouge2),
        ("ROUGE-L", soma_scores.rougeL, baseline_scores.rougeL),
    ]

    for name, soma_val, base_val in metrics:
        delta = soma_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"{name:<12s} {soma_val:>10.4f} {base_val:>10.4f} {sign}{delta:>9.4f}")

    print(f"\nN = {soma_scores.n_total}")

    # Per-type breakdown
    all_types = set(soma_scores.per_type.keys()) | set(baseline_scores.per_type.keys())
    if all_types:
        print(f"\n{'Type':<25s} {'SOMA F1':>10s} {'Base F1':>10s} {'Delta':>10s}")
        print("-" * 57)
        for qtype in sorted(all_types):
            s = soma_scores.per_type.get(qtype, {})
            b = baseline_scores.per_type.get(qtype, {})
            sf1 = s.get("f1", 0.0)
            bf1 = b.get("f1", 0.0)
            d = sf1 - bf1
            sign = "+" if d >= 0 else ""
            print(f"{qtype:<25s} {sf1:>10.4f} {bf1:>10.4f} {sign}{d:>9.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval: SOMA vs baseline comparison"
    )
    parser.add_argument(
        "--variant", default="oracle", choices=["oracle", "small", "medium"],
    )
    parser.add_argument("--model", default="qwen3.5:4b-q8_0")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--soma-only", action="store_true", help="Skip baseline evaluation",
    )
    parser.add_argument(
        "--baseline-only", action="store_true", help="Skip SOMA evaluation",
    )
    parser.add_argument(
        "--soma-context-tokens", type=int, default=3800,
        help="Max context tokens for SOMA retrieval (default: 3800)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    soma_scores = None
    baseline_scores = None

    if not args.baseline_only:
        print(f"\n{'#' * 60}")
        print("# SOMA Evaluation")
        print(f"{'#' * 60}")
        soma_out = RESULTS_DIR / f"soma_{args.variant}_{args.model.replace(':', '_')}.json"
        soma_scores = run_soma(
            variant=args.variant,
            api_base=args.api_base,
            model=args.model,
            max_context_tokens=args.soma_context_tokens,
            limit=args.limit,
            output_path=soma_out,
            data_dir=args.data_dir,
        )
        print(f"\nSOMA: F1={soma_scores.f1:.4f} EM={soma_scores.em:.4f}")

    if not args.soma_only:
        print(f"\n{'#' * 60}")
        print("# Baseline Evaluation")
        print(f"{'#' * 60}")
        base_out = RESULTS_DIR / f"baseline_{args.variant}_{args.model.replace(':', '_')}.json"
        baseline_scores = run_baseline(
            variant=args.variant,
            api_base=args.api_base,
            model=args.model,
            limit=args.limit,
            output_path=base_out,
            data_dir=args.data_dir,
        )
        print(f"\nBaseline: F1={baseline_scores.f1:.4f} EM={baseline_scores.em:.4f}")

    total = time.perf_counter() - t0

    if soma_scores and baseline_scores:
        _print_comparison(soma_scores, baseline_scores, args.model, args.variant)

    print(f"\nTotal wall-clock: {total:.1f}s")


if __name__ == "__main__":
    main()
