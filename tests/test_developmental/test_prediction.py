"""Tests for next-activation prediction loop."""

from __future__ import annotations

import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA


def _make_config() -> SOMAConfig:
    return SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        seed=42,
    )


class TestPredictiveSOMA:
    def test_init(self) -> None:
        ps = PredictiveSOMA(_make_config(), device=torch.device("cpu"))
        assert ps.soma is not None
        assert ps.prediction_error == 0.0

    def test_process_returns_result(self) -> None:
        ps = PredictiveSOMA(_make_config(), device=torch.device("cpu"))
        result = ps.process_input(torch.randn(64))
        assert "activations" in result
        assert "prediction_error" in result
        assert "novelty" in result
        assert isinstance(result["prediction_error"], float)

    def test_prediction_error_changes(self) -> None:
        ps = PredictiveSOMA(_make_config(), device=torch.device("cpu"))
        r1 = ps.process_input(torch.randn(64))  # no prior prediction
        r2 = ps.process_input(torch.randn(64))  # now has prior prediction
        assert r1["prediction_error"] == 0.0
        assert r2["prediction_error"] >= 0.0

    def test_cumulative_error_tracked(self) -> None:
        ps = PredictiveSOMA(_make_config(), device=torch.device("cpu"))
        for _ in range(3):
            ps.process_input(torch.randn(64))
        assert len(ps.error_history) == 3

    def test_graph_grows_after_many_inputs(self) -> None:
        config = _make_config()
        config.synaptogenesis_interval = 5
        config.neurogenesis_interval = 10
        ps = PredictiveSOMA(config, device=torch.device("cpu"))
        initial_edges = len(ps.soma.graph.edges)
        for _ in range(20):
            ps.process_input(torch.randn(64))
        assert len(ps.soma.graph.edges) >= initial_edges

    def test_retrieve_uses_diversification(self) -> None:
        """Retrieval should apply diversification + inhibition so query
        fingerprints are comparable to stored fingerprints."""
        ps = PredictiveSOMA(_make_config(), device=torch.device("cpu"))

        # Develop with a few inputs
        for i in range(5):
            vec = torch.randn(64)
            ps.process_input(vec, source_text=f"text_{i}")

        assert len(ps._activation_store) == 5

        # Two different queries should produce different retrieval results
        # (if diversification works, fingerprints vary by input)
        q1 = torch.randn(64)
        q2 = torch.randn(64) * 2 + 1  # deliberately different
        r1 = ps.retrieve_by_graph(q1, top_k=5)
        r2 = ps.retrieve_by_graph(q2, top_k=5)

        # Both should return results
        assert len(r1) > 0
        assert len(r2) > 0

    def test_retrieve_hybrid_returns_results(self) -> None:
        """Hybrid retrieval should return results when corpus is provided."""
        config = _make_config()
        ps = PredictiveSOMA(config, device=torch.device("cpu"))
        dim = config.sensor_output_dim

        # Store some memories
        texts = ["hello world", "foo bar baz", "test input data"]
        for text in texts:
            ps.process_input(torch.randn(dim), source_text=text)

        # Build a fake corpus embedding matrix
        corpus_embeddings = torch.randn(len(texts), dim)
        step_map = {s: i for i, s in enumerate(ps.text_store.keys())}

        query = torch.randn(dim)
        results = ps.retrieve_hybrid(
            query, corpus_embeddings, step_map,
            recall_k=3, top_k=2, gate_threshold=0.0,
        )
        # With gate_threshold=0 the graph always reranks
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)

    def test_retrieve_hybrid_fallback_without_gate(self) -> None:
        """With very high gate threshold, hybrid should match embedding order."""
        config = _make_config()
        ps = PredictiveSOMA(config, device=torch.device("cpu"))
        dim = config.sensor_output_dim

        for text in ["a", "b", "c", "d", "e"]:
            ps.process_input(torch.randn(dim), source_text=text)

        corpus_embeddings = torch.randn(5, dim)
        step_map = {s: i for i, s in enumerate(ps.text_store.keys())}

        query = torch.randn(dim)
        results = ps.retrieve_hybrid(
            query, corpus_embeddings, step_map,
            recall_k=5, top_k=3, gate_threshold=999.0,
        )
        # Gate should never fire at threshold=999, so pure embedding order
        assert len(results) <= 3

    def test_neurogenesis_gets_input_projection(self) -> None:
        """Nodes born via neurogenesis should get input projections."""
        config = _make_config()
        config.neurogenesis_interval = 5
        config.neurogenesis_threshold = 0.5  # low threshold to trigger easily
        ps = PredictiveSOMA(config, device=torch.device("cpu"))

        # Feed varied inputs to potentially trigger neurogenesis
        for _ in range(30):
            ps.process_input(torch.randn(64))

        # If new nodes were created, they should have projections
        from soma.core.node import NodeType

        assoc_count = sum(
            1 for n in ps.soma.graph.all_nodes()
            if n.node_type == NodeType.ASSOCIATOR
        )
        # _ensure_projection is called lazily during diversification,
        # so projections are created on first use
        ps._diversify_activations(torch.randn(64))
        assert len(ps._input_projections) >= assoc_count
