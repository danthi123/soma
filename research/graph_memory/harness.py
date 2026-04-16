"""C1 retrieval harness: train MemoryLayer + (optionally) SOMA, then eval.

Two entry points:

* :func:`run_baseline` — pure-vector ``MemoryLayer`` (no SOMA attach).
  Implemented in :mod:`research.graph_memory.baselines` to keep the
  module roles obvious; re-exported here only so the harness tests can
  import everything from one place.
* :func:`run_soma` — ``MemoryLayer`` with a SOMA graph attached, with
  the graph blend re-rank enabled at the requested α and a sweep of
  consolidation iterations. The plan asks N ∈ {0, 10, 100, 1000};
  ``run_soma`` runs ONE N value per call and the orchestrator script
  loops the sweep.

Metric helpers (R@k, MRR) live in this module so the unit tests can
import them in isolation.

Design constraints (do NOT change without re-reading the C1 plan):

* The eval is dataset-agnostic — it asks the harness for a list of
  (query_text, relevant_node_ids) tuples and computes metrics. The
  test fixtures use very small corpora to keep CI fast.
* No "SOMA wins" baked in. ``run_soma`` returns metrics; the
  orchestrator's report is the only place we compare.
* The integrator-count check is asserted by the harness (raises if 0)
  so a regression of the seed-graph bug fix can't silently make a
  whole experiment moot.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from research.graph_memory.synthetic_corpus import (
    GroundTruth,
    cooccurrence_counts,
)

EmbedFn = Callable[[str], torch.Tensor]


# ---------------------------------------------------------------------------
# Metric helpers (pure functions, easy to unit-test)
# ---------------------------------------------------------------------------
def recall_at_k(ranked: list[str], relevant: set[str], *, k: int) -> float:
    """Standard recall: fraction of *relevant* items found in the top-k.

    Returns 0.0 when ``relevant`` is empty (no ground truth = no
    score). ``k`` is clamped to ``len(ranked)`` implicitly via slicing.
    """
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    hits = len(top & relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(queries: list[tuple[list[str], set[str]]], *, k: int) -> float:
    """Mean reciprocal rank across ``queries`` (list of ``(ranked, relevant)``).

    Reciprocal rank for a single query = ``1 / (1 + first_hit_index)``,
    or 0 if no relevant item is in the top-``k``. Empty query list
    returns 0.0.
    """
    if not queries:
        return 0.0
    total = 0.0
    for ranked, relevant in queries:
        rr = 0.0
        for i, node_id in enumerate(ranked[:k]):
            if node_id in relevant:
                rr = 1.0 / (i + 1)
                break
        total += rr
    return total / len(queries)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class HarnessResult:
    """Outcome of one harness run.

    ``metrics[query_class]`` has ``r_at_1``, ``r_at_5``, ``mrr`` plus a
    ``num_queries`` count for sanity. ``query_class`` is one of
    ``"direct"``, ``"transitive"``, ``"unrelated"``.

    SOMA-only fields (zero / empty for the pure-vector baseline):

    * ``integrator_count`` — number of INTEGRATOR nodes in the seed
      graph; the 860cc82 fix is required for this to be > 0 when the
      config asks for integrators.
    * ``edge_weight_cooccurrence_spearman`` — Spearman ρ between the
      learned edge strength on the SOMA graph and the empirical
      co-occurrence counts in the corpus. ``None`` when SOMA is not
      attached or the graph has too few edges to be meaningful.
    * ``elapsed_seconds`` — wall-clock for the full run (store +
      consolidate + eval). Useful for the report.
    """

    method: str
    consolidation_iterations: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    integrator_count: int = 0
    edge_count: int = 0
    edge_weight_cooccurrence_spearman: float | None = None
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Eval runner — shared by baseline + SOMA
# ---------------------------------------------------------------------------
def _build_query_set(
    gt: GroundTruth,
    *,
    snippet_ids: list[str],
) -> dict[str, list[tuple[str, set[str]]]]:
    """Build the three query classes from the ground truth.

    Returns ``{class_name: [(query_text, relevant_node_id_set), ...]}``.
    Each query's text is just the entity token alone; the corpus uses
    those tokens verbatim so the pure-vector baseline gets a fair shot.

    * ``direct`` — query each entity that appears in the corpus,
      relevant = all snippet ids that mention that entity.
    * ``transitive`` — query the B (or C) of every transitive triple,
      relevant = snippets mentioning the OTHER non-A member (which
      never co-occur with the query in text). Pure-vector retrieval
      cannot reach those without a structural prior.
    * ``unrelated`` — query each unrelated token; relevant = empty
      set (so R@k must be 0; we still compute MRR which will also be
      0 — the diagnostic value is the score *distribution*, not the
      absolute number).
    """
    direct: list[tuple[str, set[str]]] = []
    for entity, indices in gt.snippets_by_entity.items():
        relevant = {snippet_ids[i] for i in indices}
        direct.append((entity, relevant))

    transitive: list[tuple[str, set[str]]] = []
    for (_triple_idx, source, _target), target_indices in gt.transitive_targets.items():
        relevant = {snippet_ids[i] for i in target_indices}
        transitive.append((source, relevant))

    unrelated: list[tuple[str, set[str]]] = [(q, set()) for q in gt.unrelated_queries]

    return {
        "direct": direct,
        "transitive": transitive,
        "unrelated": unrelated,
    }


def _eval_queries(
    queries: list[tuple[str, set[str]]],
    *,
    retrieve_fn: Callable[[str, int], list[str]],
    k: int,
) -> dict[str, float]:
    """Run ``retrieve_fn`` on each query and aggregate metrics."""
    if not queries:
        return {"r_at_1": 0.0, "r_at_5": 0.0, "mrr": 0.0, "num_queries": 0.0}

    results: list[tuple[list[str], set[str]]] = []
    r1_total = 0.0
    rk_total = 0.0
    for q_text, relevant in queries:
        ranked = retrieve_fn(q_text, k)
        results.append((ranked, relevant))
        r1_total += recall_at_k(ranked, relevant, k=1)
        rk_total += recall_at_k(ranked, relevant, k=k)
    return {
        "r_at_1": r1_total / len(queries),
        "r_at_5": rk_total / len(queries),
        "mrr": mean_reciprocal_rank(results, k=k),
        "num_queries": float(len(queries)),
    }


def evaluate_memory(
    mem: Any,
    gt: GroundTruth,
    snippet_ids: list[str],
    *,
    k: int,
) -> dict[str, dict[str, float]]:
    """Score ``mem.retrieve`` against the three query classes."""
    queries = _build_query_set(gt, snippet_ids=snippet_ids)

    def _retrieve(q: str, kk: int) -> list[str]:
        hits = mem.retrieve(q, k=kk)
        return [h.node_id for h in hits]

    return {cls: _eval_queries(qs, retrieve_fn=_retrieve, k=k) for cls, qs in queries.items()}


# ---------------------------------------------------------------------------
# Edge-weight ↔ co-occurrence Spearman (bonus diagnostic)
# ---------------------------------------------------------------------------
def edge_weight_vs_cooccurrence_spearman(
    soma: Any,
    gt: GroundTruth,
) -> float | None:
    """Spearman rank correlation between edge strength and pair count.

    The substrate's edges are between SOMA nodes (sensor / associator
    / integrator / output), NOT between corpus entities — so a direct
    "edge for entity-pair X-Y" lookup is impossible. What we CAN do is
    correlate the *distribution* of edge strengths with the
    *distribution* of pair counts, both reduced to rank vectors. This
    answers: "do the strongest edges show up at all when the corpus
    has high co-occurrence pairs?".

    Returns ``None`` when there are fewer than 5 edges (not enough
    signal) or when the helper can't import scipy.
    """
    edges = list(soma.graph.all_edges())
    if len(edges) < 5:
        return None
    counts = cooccurrence_counts(gt)
    if not counts:
        return None
    edge_strengths = sorted(float(e.strength) for e in edges)
    pair_counts = sorted(float(c) for c in counts.values())
    # Truncate the longer list to compare equal-length rank vectors.
    n = min(len(edge_strengths), len(pair_counts))
    if n < 5:
        return None
    es = edge_strengths[-n:]
    pc = pair_counts[-n:]
    # Both arrays must have variance > 0 for Spearman to be defined.
    if len(set(es)) < 2 or len(set(pc)) < 2:
        return 0.0
    try:
        from scipy.stats import spearmanr
    except ImportError:
        # Fall back to a small numpy-based Spearman.
        import numpy as np

        def _rank(x: list[float]) -> list[float]:
            order = np.argsort(x)
            ranks = np.empty(len(x), dtype=float)
            ranks[order] = np.arange(len(x))
            return ranks.tolist()

        rx = _rank(es)
        ry = _rank(pc)
        rx_arr = np.asarray(rx)
        ry_arr = np.asarray(ry)
        if rx_arr.std() == 0 or ry_arr.std() == 0:
            return 0.0
        return float(np.corrcoef(rx_arr, ry_arr)[0, 1])
    rho, _ = spearmanr(es, pc)
    if rho != rho:  # NaN
        return 0.0
    return float(rho)


# ---------------------------------------------------------------------------
# SOMA harness
# ---------------------------------------------------------------------------
def _build_soma(
    embed_dim: int,
    *,
    initial_integrator_count: int = 8,
    initial_associator_count: int = 16,
    seed: int = 1234,
) -> Any:
    """Construct a SOMA configured for the C1 study.

    The 860cc82 fix is asserted at construction time so a regression
    can't silently produce a graph with zero integrators (which would
    silently null the whole experiment).
    """
    from soma.core.config import SOMAConfig
    from soma.core.node import NodeType
    from soma.system import SOMA

    config = SOMAConfig(
        vocab_size=512,
        text_embed_dim=embed_dim,
        sensor_output_dim=embed_dim,
        associator_input_dim=embed_dim,
        associator_hidden_dim=embed_dim * 2,
        associator_output_dim=embed_dim,
        integrator_input_dim=embed_dim,
        integrator_hidden_dim=embed_dim * 2,
        integrator_output_dim=embed_dim,
        max_input_tokens=128,
        initial_integrator_count=initial_integrator_count,
        initial_associator_count=initial_associator_count,
        seed=seed,
    )
    soma = SOMA(config)
    n_int = len(soma.graph.nodes_by_type(NodeType.INTEGRATOR))
    if initial_integrator_count > 0 and n_int == 0:
        raise RuntimeError(
            "SOMA seed graph has 0 integrators despite "
            f"initial_integrator_count={initial_integrator_count}. "
            "The 860cc82 seed-graph fix is missing — bail out before "
            "running any further experiments on this checkout."
        )
    return soma


def run_soma(
    gt: GroundTruth,
    *,
    embed_fn: EmbedFn,
    embed_dim: int,
    consolidation_iterations: int,
    k: int = 5,
    graph_rerank_alpha: float = 0.3,
    initial_integrator_count: int = 8,
    initial_associator_count: int = 16,
    method_label: str | None = None,
) -> HarnessResult:
    """Run the SOMA-attached harness for one consolidation-N value.

    Pipeline:

    1. Build the MemoryLayer with the requested ``embed_fn``.
    2. Construct a SOMA via :func:`_build_soma` (post-bug-fix, asserted).
    3. Use a tiny BPE-trained TextEncoder for the SOMA side (independent
       of the cosine embedder so the substrate sees its own dimensional
       view). ``MemoryLayer.attach_soma`` accepts that pair.
    4. Ingest every snippet via :meth:`MemoryLayer.store`.
    5. Call :meth:`MemoryLayer.consolidate` ``consolidation_iterations``
       times (always ≥ 0; zero means "skip — measure the substrate
       cold").
    6. Set ``_graph_rerank_alpha`` to the requested blend and run
       :meth:`stable_capture` so retrieve uses the post-growth state.
    7. Score against the three query classes from
       :func:`_build_query_set` and return :class:`HarnessResult`.

    Parameters
    ----------
    method_label:
        Label propagated into ``HarnessResult.method`` so the report
        table can distinguish e.g. "soma-stub-a0.30" from "soma-sbert-a0.30".
        Defaults to ``f"soma-a{alpha:.2f}"``.
    """
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.memory.api import MemoryLayer

    started = time.monotonic()
    label = method_label or f"soma-a{graph_rerank_alpha:.2f}"

    mem = MemoryLayer(
        embed_fn=embed_fn,
        embed_dim=embed_dim,
        graph_rerank_alpha=graph_rerank_alpha,
        graph_rerank_stable_capture=True,
    )
    soma = _build_soma(
        embed_dim,
        initial_integrator_count=initial_integrator_count,
        initial_associator_count=initial_associator_count,
    )
    # Independent BPE tokenizer for the SOMA side. Trained on the
    # corpus so the entity tokens are in vocabulary; otherwise they'd
    # all OOV-collapse and the substrate would see identical input
    # vectors for every snippet.
    soma_tokenizer = train_bpe_tokenizer([s.text for s in gt.snippets], vocab_size=512)
    soma_encoder = TextEncoder(soma_tokenizer, embed_dim=embed_dim, max_seq_len=128)
    mem.attach_soma(soma, soma_tokenizer, soma_encoder)

    snippet_ids: list[str] = [mem.store(s.text) for s in gt.snippets]

    for _ in range(consolidation_iterations):
        mem.consolidate()
    # Even at iterations=0 we want stable-capture so the retrieve
    # path's blend formula sees a coherent activation set rather than
    # all-Nones (which would silently fall back to pure cosine and
    # mask whether the blend works).
    mem.stable_capture()

    metrics = evaluate_memory(mem, gt, snippet_ids, k=k)
    spearman = edge_weight_vs_cooccurrence_spearman(soma, gt)

    from soma.core.node import NodeType

    return HarnessResult(
        method=label,
        consolidation_iterations=consolidation_iterations,
        metrics=metrics,
        integrator_count=len(soma.graph.nodes_by_type(NodeType.INTEGRATOR)),
        edge_count=soma.graph.num_edges,
        edge_weight_cooccurrence_spearman=spearman,
        elapsed_seconds=time.monotonic() - started,
    )


# Re-export for convenience so ``from research.graph_memory.harness import
# run_baseline`` works (the orchestrator script imports both from here).
from research.graph_memory.baselines import run_baseline  # noqa: E402, F401
