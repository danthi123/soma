"""Phase 2 (audit plan 4.3): Consolidation survival test.

Tests whether SOMA's consolidation cycle protects old memories from
interference by new ones.

Protocol:
  1. Store 50 phase-1 facts
  2. Consolidate (attach SOMA, run growth pass + stable capture)
  3. Flood with 200 unrelated facts
  4. Query for the original 50 facts
  5. Compare recall WITH consolidation vs WITHOUT (two runs)

No LLM needed -- measures retrieval recall directly.
Runs on CPU (sentence-transformers + small SOMA graph).

Usage::

    python -m research.audit.phase2_consolidation_survival
    python -m research.audit.phase2_consolidation_survival --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from soma.core.config import SOMAConfig
from research.audit.synthetic_corpus import (
    ADVERSARIAL_FLOOD,
    FLOOD_FACTS,
    PHASE1_FACTS,
    QUERIES,
    build_memory_layer,
    build_soma_stack,
    measure_recall,
)

logger = logging.getLogger(__name__)
RESULTS_DIR = Path("research/audit/results")


def run_condition(
    label: str,
    *,
    do_consolidate: bool,
    do_flood: bool,
    graph_rerank_alpha: float = 0.0,
    consolidation_interval: int = 1000,
    consolidation_replay_steps: int = 100,
    flood_data: list[str] | None = None,
    k: int = 5,
    device: torch.device | None = None,
) -> dict:
    """Run one experimental condition and return results."""
    t0 = time.perf_counter()

    mem = build_memory_layer(graph_rerank_alpha=graph_rerank_alpha)

    # Store phase-1 facts
    mem.store_batch(PHASE1_FACTS)

    # Attach SOMA and optionally consolidate
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=8,
        initial_associator_count=16,
        consolidation_interval=consolidation_interval,
        consolidation_replay_steps=consolidation_replay_steps,
        seed=42,
    )
    soma, tokenizer, encoder = build_soma_stack(config, device=device)
    mem.attach_soma(soma, tokenizer, encoder)

    consolidation_time = 0.0
    if do_consolidate:
        tc = time.perf_counter()
        mem.consolidate()
        mem.stable_capture()
        consolidation_time = time.perf_counter() - tc

    # Record pre-flood recall
    pre_flood_recall = measure_recall(mem, k=k)

    # Flood with interference
    if do_flood:
        mem.store_batch(flood_data or FLOOD_FACTS)

    # Post-flood recall
    post_flood_recall = measure_recall(mem, k=k)

    node_count = len(soma.graph.nodes)
    edge_count = len(soma.graph.edges)
    total_time = time.perf_counter() - t0

    return {
        "label": label,
        "do_consolidate": do_consolidate,
        "do_flood": do_flood,
        "graph_rerank_alpha": graph_rerank_alpha,
        "consolidation_interval": consolidation_interval,
        "consolidation_replay_steps": consolidation_replay_steps,
        "pre_flood_recall": round(pre_flood_recall, 4),
        "post_flood_recall": round(post_flood_recall, 4),
        "recall_delta": round(post_flood_recall - pre_flood_recall, 4),
        "consolidation_time_s": round(consolidation_time, 4),
        "total_time_s": round(total_time, 4),
        "node_count": node_count,
        "edge_count": edge_count,
        "memory_count_final": len(mem._ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Consolidation survival test"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Smoke test with fewer conditions",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0
        else "cpu"
    )
    print(f"Device: {device}")

    # Define conditions
    # (label, do_consolidate, do_flood, alpha, flood_data)
    if args.quick:
        conditions = [
            ("no_consol+adversarial", False, True, 0.0, ADVERSARIAL_FLOOD),
            ("consol+adversarial", True, True, 0.0, ADVERSARIAL_FLOOD),
        ]
    else:
        conditions = [
            # Baseline: no flood
            ("baseline_no_flood", False, False, 0.0, None),
            # Unrelated flood (should be easy)
            ("no_consol+unrelated", False, True, 0.0, FLOOD_FACTS),
            ("consol+unrelated_a0.0", True, True, 0.0, FLOOD_FACTS),
            # Adversarial flood (semantically similar, wrong details)
            ("no_consol+adversarial", False, True, 0.0, ADVERSARIAL_FLOOD),
            ("consol+adversarial_a0.0", True, True, 0.0, ADVERSARIAL_FLOOD),
            ("consol+adversarial_a0.3", True, True, 0.3, ADVERSARIAL_FLOOD),
            ("consol+adversarial_a0.5", True, True, 0.5, ADVERSARIAL_FLOOD),
            ("consol+adversarial_a0.7", True, True, 0.7, ADVERSARIAL_FLOOD),
            # Consolidation only (no flood)
            ("consol_no_flood", True, False, 0.0, None),
        ]

    results = []
    for label, do_consolidate, do_flood, alpha, fdata in conditions:
        print(f"\n--- Running: {label} ---")
        r = run_condition(
            label,
            do_consolidate=do_consolidate,
            do_flood=do_flood,
            graph_rerank_alpha=alpha,
            flood_data=fdata,
            k=args.k,
            device=device,
        )
        results.append(r)
        print(f"  Pre-flood recall:  {r['pre_flood_recall']:.4f}")
        print(f"  Post-flood recall: {r['post_flood_recall']:.4f}")
        print(f"  Delta:             {r['recall_delta']:+.4f}")
        print(f"  Consolidation:     {r['consolidation_time_s']:.2f}s")
        print(f"  Graph: {r['node_count']} nodes, {r['edge_count']} edges")

    # Summary table
    print(f"\n{'=' * 72}")
    print("PHASE 2: CONSOLIDATION SURVIVAL TEST")
    print(f"{'=' * 72}")
    print(
        f"{'Condition':<30s} {'Pre-flood':>10s} {'Post-flood':>10s} "
        f"{'Delta':>8s} {'Consol(s)':>10s}"
    )
    print("-" * 72)
    for r in results:
        print(
            f"{r['label']:<30s} {r['pre_flood_recall']:>10.4f} "
            f"{r['post_flood_recall']:>10.4f} {r['recall_delta']:>+8.4f} "
            f"{r['consolidation_time_s']:>9.2f}s"
        )

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase2_consolidation_survival.json"
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
