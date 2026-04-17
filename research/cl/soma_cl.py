"""SOMA adapter for continual-learning classification benchmarks.

Wraps a SOMA system as a feature extractor with a linear classification
head on top.  SOMA updates its graph weights via Hebbian learning during
every forward pass (online plasticity); only the linear head is trained
via backprop on cross-entropy loss.

Four ablation modes are supported:

* ``soma-plastic`` -- full SOMA with Hebbian + inter-task consolidation.
* ``soma-frozen`` -- SOMA weights frozen after construction (no Hebbian).
* ``soma-no-consolidation`` -- plastic but skip inter-task consolidation.
* ``soma-no-critical-periods`` -- plastic + consolidation but constant LR
  (critical-period schedule disabled).

The adapter integrates with the B1/B2 CL harness via ``model_factory``,
``train_one_epoch``, and ``on_task_end`` callables.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader

from soma.consolidation.cycle import consolidation_cycle
from soma.core.config import SOMAConfig
from soma.core.node import NodeType
from soma.system import SOMA

# ------------------------------------------------------------------
# SOMA CL classifier module
# ------------------------------------------------------------------


class SomaClassifier(nn.Module):
    """SOMA feature extractor + linear classifier head.

    ``forward(x)`` flattens input, runs it through SOMA's graph, and
    returns class logits from the linear head.

    SOMA is NOT an ``nn.Module`` subclass -- it's a composite object
    that manages its own ``Graph`` of ``nn.Module`` nodes. We hold it
    as a plain attribute and handle its updates manually.
    """

    def __init__(
        self,
        soma: SOMA,
        num_classes: int,
        *,
        input_dim: int = 784,
        frozen: bool = False,
        enable_consolidation: bool = True,
    ) -> None:
        super().__init__()
        self.soma = soma
        self.frozen = frozen
        self.enable_consolidation = enable_consolidation

        # Determine the sensor and output dims
        sensor_dim = soma.config.sensor_output_dim

        # Determine the output dim from the OUTPUT node
        output_nodes = soma.graph.nodes_by_type(NodeType.OUTPUT)
        if not output_nodes:
            raise RuntimeError("SOMA graph has no OUTPUT nodes")
        self._output_dim = output_nodes[0].output_dim

        # Modality keys for input/output
        self._input_modality = soma.config.input_modalities[0]
        self._output_modality = soma.config.output_modalities[0]

        # Input projection: raw pixels -> sensor_output_dim
        # The sensor node expects input matching its output_dim.
        self.input_proj = nn.Linear(input_dim, sensor_dim, bias=False)

        # Linear classification head (only this gets backprop)
        self.head = nn.Linear(self._output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> SOMA features -> logits.

        Parameters
        ----------
        x:
            Flattened input tensor, shape ``(batch, input_dim)``.

        Returns
        -------
        Logits tensor, shape ``(batch, num_classes)``.
        """
        batch_size = x.size(0)
        features = []

        # Project all samples to sensor dim at once
        projected = self.input_proj(x)

        for i in range(batch_size):
            sample = projected[i]  # (sensor_dim,)

            # SOMA.step() expects dict[str, Tensor] inputs.
            # The sensor node's set_input takes a 1-d tensor.
            inputs = {self._input_modality: sample}

            # When frozen, run in eval_mode to skip weight updates
            result = self.soma.step(
                inputs,
                targets=None,
                eval_mode=self.frozen,
            )

            outputs = result["outputs"]
            if self._output_modality in outputs:
                feat = outputs[self._output_modality]
            else:
                # Fallback: zero features if output node didn't fire
                feat = torch.zeros(
                    self._output_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
            features.append(feat)

        # Stack into (batch, feature_dim)
        features_t = torch.stack(features, dim=0)

        # Only the head is trained via backprop -- detach SOMA features
        logits = self.head(features_t.detach())
        return logits

    def consolidate(self) -> None:
        """Run one consolidation cycle (artificial sleep).

        No-op if consolidation is disabled for this ablation.
        """
        if not self.enable_consolidation:
            return
        if self.soma.episodic_memory.num_valid == 0:
            return
        consolidation_cycle(
            self.soma.graph,
            self.soma.episodic_memory,
            current_step=self.soma.global_step,
            config=self.soma.config,
            experience_unpacker=self.soma.experience_unpacker,
            rng=None,
        )


# ------------------------------------------------------------------
# Config builders
# ------------------------------------------------------------------


def _mnist_soma_config(
    *,
    disable_critical_periods: bool = False,
    seed: int = 42,
) -> SOMAConfig:
    """Build a SOMAConfig suitable for Permuted-MNIST."""
    config = SOMAConfig(
        text_embed_dim=784,  # sensor input = pixel count
        sensor_output_dim=256,
        associator_input_dim=256,
        associator_hidden_dim=512,
        associator_output_dim=256,
        integrator_input_dim=256,
        integrator_hidden_dim=512,
        integrator_output_dim=128,
        initial_integrator_count=8,
        initial_associator_count=16,
        vocab_size=1,
        max_input_tokens=1,
        # Tighten intervals for the CL setting (fewer steps per task)
        consolidation_interval=500,
        consolidation_replay_steps=50,
        synaptogenesis_interval=200,
        neurogenesis_interval=1000,
        pruning_interval=2000,
        seed=seed,
    )
    return config


def _cifar_soma_config(
    *,
    disable_critical_periods: bool = False,
    seed: int = 42,
) -> SOMAConfig:
    """Build a SOMAConfig suitable for Split-CIFAR-10."""
    config = SOMAConfig(
        text_embed_dim=3072,  # 32*32*3 flattened
        sensor_output_dim=256,
        associator_input_dim=256,
        associator_hidden_dim=512,
        associator_output_dim=256,
        integrator_input_dim=256,
        integrator_hidden_dim=512,
        integrator_output_dim=128,
        initial_integrator_count=8,
        initial_associator_count=16,
        vocab_size=1,
        max_input_tokens=1,
        consolidation_interval=500,
        consolidation_replay_steps=50,
        synaptogenesis_interval=200,
        neurogenesis_interval=1000,
        pruning_interval=2000,
        seed=seed,
    )
    return config


# ------------------------------------------------------------------
# Build SOMA + assert integrators
# ------------------------------------------------------------------


def _build_soma(
    config: SOMAConfig,
    device: torch.device | str | None,
) -> SOMA:
    """Build a SOMA system and assert the seed-graph integrator fix."""
    soma = SOMA(config, device=device)
    n_int = len(soma.graph.nodes_by_type(NodeType.INTEGRATOR))
    if config.initial_integrator_count > 0 and n_int == 0:
        raise RuntimeError(
            f"SOMA seed graph has 0 integrators despite "
            f"initial_integrator_count={config.initial_integrator_count}. "
            "The 860cc82 seed-graph fix is missing."
        )
    return soma


# ------------------------------------------------------------------
# CL harness integration
# ------------------------------------------------------------------


class SomaCLAdapter:
    """Wraps SomaClassifier for use with the CL harness.

    Provides ``train_one_epoch`` and ``on_task_end`` callables
    matching the harness callback signatures.
    """

    def __init__(
        self,
        *,
        frozen: bool = False,
        enable_consolidation: bool = True,
        disable_critical_periods: bool = False,
        consolidation_replay_steps: int = 50,
    ) -> None:
        self.frozen = frozen
        self.enable_consolidation = enable_consolidation
        self.disable_critical_periods = disable_critical_periods
        self.consolidation_replay_steps = consolidation_replay_steps
        self._model: SomaClassifier | None = None

    def train_one_epoch(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        """Train one epoch: SOMA does Hebbian, head does SGD."""
        model.train(True)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def on_task_end(
        self,
        model: nn.Module,
        task_idx: int,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """Post-task hook: run consolidation cycle."""
        if isinstance(model, SomaClassifier):
            model.consolidate()


def make_soma_cl_components(
    *,
    dataset: str = "mnist",
    frozen: bool = False,
    enable_consolidation: bool = True,
    disable_critical_periods: bool = False,
    seed: int = 42,
) -> tuple[
    Callable[[torch.device], tuple[nn.Module, Optimizer]],
    Callable[..., float],
    Callable[..., None],
]:
    """Return ``(model_factory, train_one_epoch, on_task_end)`` for the harness.

    Parameters
    ----------
    dataset:
        ``"mnist"`` or ``"cifar"`` -- selects the SOMA config.
    frozen:
        If True, SOMA weights are frozen (eval_mode on every step).
    enable_consolidation:
        If True, run consolidation cycle between tasks.
    disable_critical_periods:
        If True, disable the development schedule (constant LR).
    seed:
        Random seed for SOMA construction.
    """

    config_fn = _mnist_soma_config if dataset == "mnist" else _cifar_soma_config
    config = config_fn(
        disable_critical_periods=disable_critical_periods,
        seed=seed,
    )

    num_classes = 10  # both MNIST and CIFAR-10
    input_dim = 784 if dataset == "mnist" else 3072

    adapter = SomaCLAdapter(
        frozen=frozen,
        enable_consolidation=enable_consolidation,
        disable_critical_periods=disable_critical_periods,
    )

    def model_factory(device: torch.device) -> tuple[nn.Module, Optimizer]:
        soma = _build_soma(config, device=device)

        # Disable critical-period schedule if requested
        if disable_critical_periods:
            soma.development.periods = ()

        classifier = SomaClassifier(
            soma,
            num_classes=num_classes,
            input_dim=input_dim,
            frozen=frozen,
            enable_consolidation=enable_consolidation,
        ).to(device)

        # Optimize both input projection and linear head
        trainable = list(classifier.input_proj.parameters()) + list(
            classifier.head.parameters()
        )
        optimizer = SGD(trainable, lr=0.01)
        return classifier, optimizer

    return model_factory, adapter.train_one_epoch, adapter.on_task_end
