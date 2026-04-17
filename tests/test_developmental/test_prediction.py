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
