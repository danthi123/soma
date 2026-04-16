"""Unit tests for the C1 synthetic-corpus + retrieval harness.

Scope (per the C1 plan, biased toward HONEST tests):

* Corpus: shape, determinism, structural invariants, ground-truth
  consistency.
* Metrics: R@1 / R@5 / MRR are computed correctly on canned inputs.
* Harness: the pure-vector baseline runs end-to-end on a stub embedder
  and produces non-empty results; the SOMA path actually creates
  integrators (post-bug-fix substrate); no-regression on R@5 against
  the baseline at consolidation N=0 (the SOMA path with α=0 must equal
  the pure-vector path).

We deliberately do NOT test that SOMA outperforms — that's what the
experiment is supposed to discover. Tests just guard against
infrastructure bugs that would silently corrupt that finding.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from research.graph_memory.synthetic_corpus import (
    DEFAULT_SEED,
    NUM_TRANSITIVE_TRIPLES,
    NUM_TRIANGLE_TRIPLES,
    cooccurrence_counts,
    generate,
)


# ---------------------------------------------------------------------------
# Tiny deterministic stub embedder — gives the harness a fast embed_fn that
# stays consistent across runs without needing sbert in unit tests.
# ---------------------------------------------------------------------------
def stub_embed(text: str, dim: int = 32) -> torch.Tensor:
    """Hash-based bag-of-words vector. Deterministic, fast, no model."""
    vec = torch.zeros(dim)
    for tok in text.split():
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    norm = float(vec.norm())
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Corpus tests
# ---------------------------------------------------------------------------
def test_generate_returns_correct_total_snippets() -> None:
    gt = generate()
    assert len(gt.snippets) == 1000


def test_generate_is_deterministic() -> None:
    a = generate(seed=DEFAULT_SEED)
    b = generate(seed=DEFAULT_SEED)
    assert [s.text for s in a.snippets] == [s.text for s in b.snippets]
    assert a.triples == b.triples


def test_generate_has_correct_triple_counts() -> None:
    gt = generate()
    n_trans = sum(1 for t in gt.triples if t.kind == "transitive")
    n_tri = sum(1 for t in gt.triples if t.kind == "triangle")
    assert n_trans == NUM_TRANSITIVE_TRIPLES
    assert n_tri == NUM_TRIANGLE_TRIPLES


def test_transitive_triples_never_have_bc_cooccurrence() -> None:
    """The B-C "no direct co-occurrence" invariant is the whole point."""
    gt = generate()
    for snip in gt.snippets:
        triple = gt.triples[snip.triple_idx]
        if triple.kind != "transitive":
            continue
        members = set(triple.members)
        snip_members = members.intersection(snip.pair)
        # Any snippet for a transitive triple must include A.
        assert triple.a in snip.pair, (
            f"transitive triple {triple} has snippet without A: {snip.text!r}"
        )
        # B and C must never appear together in the same snippet.
        if triple.b in snip_members and triple.c in snip_members:
            raise AssertionError(f"transitive invariant broken — B+C in {snip.text!r}")


def test_triangle_triples_can_have_all_pairs() -> None:
    """Triangles SHOULD include the BC pair somewhere — sanity check."""
    gt = generate()
    triangle_indices = [i for i, t in enumerate(gt.triples) if t.kind == "triangle"]
    for ti in triangle_indices:
        triple = gt.triples[ti]
        bc_seen = any(
            (snip.triple_idx == ti and {triple.b, triple.c}.issubset(set(snip.pair)))
            for snip in gt.snippets
        )
        assert bc_seen, f"triangle {triple} never produced a B-C snippet"


def test_unrelated_queries_never_appear_in_corpus() -> None:
    gt = generate()
    body = " ".join(s.text for s in gt.snippets)
    for q in gt.unrelated_queries:
        assert q not in body


def test_snippets_by_entity_indexes_into_snippets() -> None:
    gt = generate()
    for ent, indices in gt.snippets_by_entity.items():
        for idx in indices:
            assert ent in gt.snippets[idx].pair


def test_transitive_targets_only_for_transitive_triples() -> None:
    gt = generate()
    for triple_idx, _, _ in gt.transitive_targets:
        assert gt.triples[triple_idx].kind == "transitive"


def test_cooccurrence_counts_symmetric() -> None:
    gt = generate(total_snippets=120)
    counts = cooccurrence_counts(gt)
    for (a, b), n in counts.items():
        assert counts[(b, a)] == n


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------
def test_recall_at_k_basic() -> None:
    from research.graph_memory.harness import recall_at_k

    # Single query, 3 relevant items, rank list contains 1 of them
    # within top-5: R@5 = 1/3 (one of three relevant items found).
    ranked = ["a", "b", "x", "y", "z"]
    relevant = {"a", "c", "d"}
    score = recall_at_k(ranked, relevant, k=5)
    # By the standard "found / total relevant" convention.
    assert score == pytest.approx(1.0 / 3.0)


def test_recall_at_k_zero_when_no_relevant() -> None:
    from research.graph_memory.harness import recall_at_k

    assert recall_at_k(["x", "y"], set(), k=5) == 0.0


def test_recall_at_k_clamps_to_k() -> None:
    from research.graph_memory.harness import recall_at_k

    ranked = ["a", "b", "c"]
    # k=1 — only the first item counts, even though "b" is relevant.
    assert recall_at_k(ranked, {"b"}, k=1) == 0.0
    assert recall_at_k(ranked, {"a"}, k=1) == 1.0


def test_mrr_uses_first_relevant_rank() -> None:
    from research.graph_memory.harness import mean_reciprocal_rank

    # Two queries: relevant at ranks 2 and 1 → MRR = (1/2 + 1/1) / 2.
    queries = [
        (["x", "a", "y"], {"a"}),
        (["b", "y"], {"b"}),
    ]
    mrr = mean_reciprocal_rank(queries, k=10)
    assert mrr == pytest.approx((0.5 + 1.0) / 2.0)


def test_mrr_zero_when_no_relevant_in_topk() -> None:
    from research.graph_memory.harness import mean_reciprocal_rank

    queries = [(["x", "y"], {"a"})]
    assert mean_reciprocal_rank(queries, k=2) == 0.0


# ---------------------------------------------------------------------------
# Baseline harness tests
# ---------------------------------------------------------------------------
def test_baseline_runs_end_to_end() -> None:
    from research.graph_memory.baselines import run_baseline

    gt = generate(total_snippets=60, num_transitive=2, num_triangle=2)
    result = run_baseline(
        gt,
        embed_fn=lambda t: stub_embed(t, dim=16),
        embed_dim=16,
        k=5,
    )
    # Three query classes always present.
    assert "direct" in result.metrics
    assert "transitive" in result.metrics
    assert "unrelated" in result.metrics
    for _cls, m in result.metrics.items():
        assert "r_at_1" in m
        assert "r_at_5" in m
        assert "mrr" in m


# ---------------------------------------------------------------------------
# SOMA harness tests
# ---------------------------------------------------------------------------
def test_attached_soma_actually_creates_integrators() -> None:
    """The 860cc82 fix must be active — config.initial_integrator_count=8
    has to actually produce 8 integrator nodes in the graph."""
    from soma.core.config import SOMAConfig
    from soma.core.node import NodeType
    from soma.system import SOMA

    config = SOMAConfig(
        vocab_size=128,
        text_embed_dim=32,
        sensor_output_dim=32,
        max_input_tokens=64,
        initial_integrator_count=8,
        initial_associator_count=4,
    )
    soma = SOMA(config)
    integrators = soma.graph.nodes_by_type(NodeType.INTEGRATOR)
    assert len(integrators) == 8, (
        f"expected 8 integrators (post-bug-fix), got {len(integrators)} — "
        "the seed-graph integrator bug fix is not active in this checkout"
    )


def test_soma_harness_runs_end_to_end_on_stub() -> None:
    """The full SOMA-attached harness path runs without exploding on
    a tiny stub embedder. We're not asserting accuracy here — that's
    the experiment's job — only that the substrate doesn't crash."""
    from research.graph_memory.harness import run_soma

    gt = generate(total_snippets=40, num_transitive=2, num_triangle=2)
    result = run_soma(
        gt,
        embed_fn=lambda t: stub_embed(t, dim=16),
        embed_dim=16,
        consolidation_iterations=0,
        k=5,
        graph_rerank_alpha=0.3,
    )
    assert "direct" in result.metrics
    assert result.integrator_count == 8


def test_soma_alpha_zero_matches_pure_vector_at_n_zero() -> None:
    """With graph_rerank_alpha=0 and no consolidation, SOMA's retrieval
    must reduce to the pure-vector baseline (numerically identical R@5
    on direct queries, modulo tie-breaking, on the same corpus +
    embedder)."""
    from research.graph_memory.baselines import run_baseline
    from research.graph_memory.harness import run_soma

    gt = generate(total_snippets=60, num_transitive=2, num_triangle=2)
    embed = lambda t: stub_embed(t, dim=16)  # noqa: E731

    base = run_baseline(gt, embed_fn=embed, embed_dim=16, k=5)
    soma_res = run_soma(
        gt,
        embed_fn=embed,
        embed_dim=16,
        consolidation_iterations=0,
        k=5,
        graph_rerank_alpha=0.0,
    )
    # On direct queries with α=0 the blend is exactly the pure cosine
    # ranking. Allow tiny float tolerance.
    assert soma_res.metrics["direct"]["r_at_5"] == pytest.approx(
        base.metrics["direct"]["r_at_5"], abs=1e-6
    )
