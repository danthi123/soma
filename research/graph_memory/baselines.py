"""Pure-vector ``MemoryLayer`` baseline for the C1 retrieval study.

This is what graph-blended retrieval has to beat on the transitive
queries. It's deliberately the *exact same* MemoryLayer the harness
uses for the SOMA-attached run — only the ``attach_soma`` call and
the ``graph_rerank_alpha`` setting differ — so the comparison isn't
contaminated by adapter or codepath drift.

There's no SOMA construction, no consolidation, and no
stable-capture pass. Cosine over the supplied embedder is the entire
retrieval signal. ``HarnessResult.consolidation_iterations`` is fixed
at -1 to make "this run had no consolidation step at all" textually
distinct from a SOMA run with N=0 in the report tables.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from research.graph_memory.synthetic_corpus import GroundTruth

if TYPE_CHECKING:
    from research.graph_memory.harness import HarnessResult

EmbedFn = Callable[[str], torch.Tensor]


def run_baseline(
    gt: GroundTruth,
    *,
    embed_fn: EmbedFn,
    embed_dim: int,
    k: int = 5,
    method_label: str = "pure-vector",
) -> HarnessResult:
    """Run the pure-vector baseline harness.

    See :func:`research.graph_memory.harness.run_soma` for the parallel
    SOMA-attached pipeline. This function intentionally avoids any
    growth/consolidation step; the SOMA-side run at α=0 with
    consolidation_iterations=0 should reproduce these numbers exactly
    (modulo float tie-breaking) — that's the no-regression test in
    ``tests/test_research/test_graph_memory_c1.py``.
    """
    # Local import to keep the module-level import graph free of cycles
    # — harness imports baselines, baselines imports HarnessResult.
    from research.graph_memory.harness import HarnessResult, evaluate_memory
    from soma.memory.api import MemoryLayer

    started = time.monotonic()
    mem = MemoryLayer(embed_fn=embed_fn, embed_dim=embed_dim)
    snippet_ids: list[str] = [mem.store(s.text) for s in gt.snippets]
    metrics = evaluate_memory(mem, gt, snippet_ids, k=k)
    return HarnessResult(
        method=method_label,
        consolidation_iterations=-1,
        metrics=metrics,
        integrator_count=0,
        edge_count=0,
        edge_weight_cooccurrence_spearman=None,
        elapsed_seconds=time.monotonic() - started,
    )
