"""Tests for SOMA state verbalization."""

from __future__ import annotations

import torch

from soma.core.config import SOMAConfig
from soma.developmental.verbalize import verbalize_state
from soma.system import SOMA


def _make_soma(device: str = "cpu") -> SOMA:
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        seed=42,
    )
    return SOMA(config, device=torch.device(device))


class TestVerbalizeState:
    def test_returns_string(self) -> None:
        result = verbalize_state(_make_soma())
        assert isinstance(result, str) and len(result) > 0

    def test_contains_developmental_stage(self) -> None:
        assert "Developmental Stage" in verbalize_state(_make_soma())

    def test_contains_graph_stats(self) -> None:
        r = verbalize_state(_make_soma())
        assert "nodes" in r.lower() and "edges" in r.lower()

    def test_contains_novelty(self) -> None:
        assert "novelty" in verbalize_state(_make_soma()).lower()

    def test_after_step_has_activations(self) -> None:
        soma = _make_soma()
        soma.step({"text": torch.randn(64)}, eval_mode=True)
        assert "active" in verbalize_state(soma).lower()

    def test_working_memory_section(self) -> None:
        soma = _make_soma()
        soma.step({"text": torch.randn(64)}, eval_mode=True)
        assert "Working Memory" in verbalize_state(soma)
