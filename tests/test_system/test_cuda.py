"""GPU smoke tests for ``soma.system.SOMA``.

Guarded by ``pytest.mark.skipif(not torch.cuda.is_available(), ...)`` so
the suite remains green on CPU-only machines. Run with:

    pytest tests/test_system/test_cuda.py -v

on a box with a CUDA-capable GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.system import SOMA

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device not available",
)


@pytest.fixture
def small_config() -> SOMAConfig:
    return SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        wm_slots=4,
        wm_dim=8,
        key_dim=8,
        value_dim=16,
        text_embed_dim=8,
        initial_associator_count=3,
        initial_integrator_count=0,
        max_nodes=32,
        num_curiosity_domains=2,
        base_lr=0.01,
        hebbian_lr=0.0001,
        youth_lr_multiplier=1.0,
        activation_threshold=0.01,
        synaptogenesis_interval=50,
        neurogenesis_interval=200,
        pruning_interval=100,
        consolidation_interval=100,
        seed=0,
    )


class TestSOMAOnCUDA:
    def test_construction_on_cuda(self, small_config: SOMAConfig) -> None:
        device = torch.device("cuda")
        soma = SOMA(small_config, device=device)
        # Every learnable parameter should live on cuda.
        for name, param in soma.graph.named_parameters():
            assert param.device.type == "cuda", f"param {name} on {param.device}"

    def test_step_runs_on_cuda(self, small_config: SOMAConfig) -> None:
        device = torch.device("cuda")
        soma = SOMA(small_config, device=device)
        dim = small_config.sensor_output_dim
        inp = torch.randn(dim, device=device)
        tgt = torch.randn(dim, device=device)
        result = soma.step(inputs={"text": inp}, targets={"text": tgt})
        assert "text" in result["outputs"]
        assert result["outputs"]["text"].device.type == "cuda"
        assert result["loss"] == result["loss"]  # not NaN

    def test_short_training_loop_on_cuda(self, small_config: SOMAConfig) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        soma = SOMA(small_config, device=device)
        dim = small_config.sensor_output_dim
        inp = torch.randn(dim, device=device)
        tgt = torch.randn(dim, device=device) * 0.5
        losses: list[float] = []
        for _ in range(30):
            result = soma.step(inputs={"text": inp.clone()}, targets={"text": tgt.clone()})
            losses.append(float(result["loss"]))
        # Sanity: no NaNs, and late loss should not be dramatically worse than early.
        assert all(loss_val == loss_val for loss_val in losses)  # NaN check
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early * 2.0

    def test_checkpoint_round_trip_between_devices(
        self, small_config: SOMAConfig, tmp_path: Path
    ) -> None:
        """Save on CUDA, load into a fresh CPU instance. State must transfer."""
        soma_gpu = SOMA(small_config, device=torch.device("cuda"))
        dim = small_config.sensor_output_dim
        for _ in range(3):
            soma_gpu.step(
                inputs={"text": torch.randn(dim, device="cuda")},
                targets={"text": torch.randn(dim, device="cuda")},
            )
        path = tmp_path / "soma_gpu.pt"
        soma_gpu.save_state(path)

        soma_cpu = SOMA(small_config, device=torch.device("cpu"))
        soma_cpu.load_state(path)
        assert soma_cpu.global_step == soma_gpu.global_step
        assert soma_cpu.graph.num_nodes == soma_gpu.graph.num_nodes

    def test_load_state_realigns_graph_to_target_device(
        self, small_config: SOMAConfig, tmp_path: Path
    ) -> None:
        """Save on CPU, load into a fresh CUDA SOMA, step. Regression for the
        autonomous-loop restart path: train_service rebuilds SOMA(device=cuda)
        then calls load_state on a checkpoint that was serialized from a
        different session. Without realignment, Graph.deserialize creates the
        graph on CPU while other modules stay on CUDA, and the first step
        crashes with a cross-device addmm."""
        soma_cpu = SOMA(small_config, device=torch.device("cpu"))
        dim = small_config.sensor_output_dim
        for _ in range(2):
            soma_cpu.step(
                inputs={"text": torch.randn(dim)},
                targets={"text": torch.randn(dim)},
            )
        path = tmp_path / "soma_cpu.pt"
        soma_cpu.save_state(path)

        soma_gpu = SOMA(small_config, device=torch.device("cuda"))
        soma_gpu.load_state(path)
        for param in soma_gpu.graph.parameters():
            assert param.device.type == "cuda", f"graph param stayed on {param.device}"
        # Most-important check: can we step without a device-mismatch crash?
        soma_gpu.step(
            inputs={"text": torch.randn(dim, device="cuda")},
            targets={"text": torch.randn(dim, device="cuda")},
        )
