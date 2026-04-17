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

        # Classification head takes SOMA features concatenated with
        # projected input. SOMA features are detached (Hebbian only);
        # the projected input carries gradients from backprop.
        # This lets the head learn from both the trainable projection
        # and the SOMA representation.
        self._head_dim = sensor_dim + self._output_dim
        self.head = nn.Linear(self._head_dim, num_classes)

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
        soma_features = []

        # Project: raw pixels -> sensor_output_dim.
        # Keep grad graph alive for the classification head's backprop.
        projected = self.input_proj(x)  # (batch, sensor_dim)
        # Detach copy for SOMA (its internal backward must not collide)
        projected_detached = projected.detach()

        is_eval = self.frozen or not self.training

        for i in range(batch_size):
            sample = projected_detached[i]  # (sensor_dim,)
            inputs = {self._input_modality: sample}

            # Self-supervised target for SOMA's Hebbian learning.
            # Skip in eval mode (no_grad context breaks backward).
            targets = (
                None if is_eval
                else {self._output_modality: sample.detach()}
            )

            result = self.soma.step(
                inputs,
                targets=targets,
                eval_mode=is_eval,
            )

            outputs = result["outputs"]
            if self._output_modality in outputs:
                feat = outputs[self._output_modality]
            else:
                feat = torch.zeros(
                    self._output_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
            soma_features.append(feat)

        # SOMA features: detached (Hebbian only, no backprop through graph)
        soma_t = torch.stack(soma_features, dim=0).detach()

        # Concatenate: [projected (grad-carrying), soma_features (detached)]
        combined = torch.cat([projected, soma_t], dim=1)
        logits = self.head(combined)
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
        head_replay: bool = False,
        replay_buffer_size: int = 200,
        replay_mix_ratio: float = 0.5,
    ) -> None:
        self.frozen = frozen
        self.enable_consolidation = enable_consolidation
        self.disable_critical_periods = disable_critical_periods
        self.consolidation_replay_steps = consolidation_replay_steps
        self.head_replay = head_replay
        self.replay_buffer_size = replay_buffer_size
        self.replay_mix_ratio = replay_mix_ratio
        self._model: SomaClassifier | None = None
        # Replay buffer: list of (x_batch, y_batch) from past tasks
        self._replay_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

    def train_one_epoch(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
        """Train one epoch: SOMA does Hebbian, head does SGD.

        When ``head_replay=True``, each batch also replays a random
        subset of past-task samples through the head, mixing the
        current-task loss with replay loss to prevent forgetting.
        """
        model.train(True)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            # Mix in replay loss from past tasks
            if self.head_replay and self._replay_buffer:
                replay_x, replay_y = self._sample_replay(
                    x.size(0), device
                )
                replay_logits = model(replay_x)
                replay_loss = criterion(replay_logits, replay_y)
                # Weighted combination: current + replay
                ratio = self.replay_mix_ratio
                loss = (1.0 - ratio) * loss + ratio * replay_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _sample_replay(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from the replay buffer (uniform across tasks)."""
        all_x = torch.cat([x for x, _y in self._replay_buffer], dim=0)
        all_y = torch.cat([_y for _x, _y in self._replay_buffer], dim=0)
        n = all_x.size(0)
        idx = torch.randint(0, n, (min(batch_size, n),))
        return all_x[idx].to(device), all_y[idx].to(device)

    def on_task_end(
        self,
        model: nn.Module,
        task_idx: int,
        train_loader: DataLoader,
        device: torch.device,
    ) -> None:
        """Post-task hook: run consolidation + update replay buffer."""
        if isinstance(model, SomaClassifier):
            model.consolidate()

        # Store samples from this task for future replay
        if self.head_replay:
            self._update_replay_buffer(train_loader)

    def _update_replay_buffer(self, train_loader: DataLoader) -> None:
        """Add a random subset of this task's data to the replay buffer.

        Keeps ``replay_buffer_size`` samples per task. Stored on CPU
        to avoid GPU memory growth across tasks.
        """
        all_x = []
        all_y = []
        for x, y in train_loader:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        n = all_x.size(0)
        k = min(self.replay_buffer_size, n)
        idx = torch.randperm(n)[:k]
        # Store on CPU to avoid VRAM accumulation
        self._replay_buffer.append((all_x[idx].cpu(), all_y[idx].cpu()))


def make_soma_cl_components(
    *,
    dataset: str = "mnist",
    frozen: bool = False,
    enable_consolidation: bool = True,
    disable_critical_periods: bool = False,
    seed: int = 42,
    integrator_count: int = 8,
    head_replay: bool = False,
    replay_buffer_size: int = 200,
    replay_mix_ratio: float = 0.5,
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
    integrator_count:
        Number of initial integrator nodes. Associators = 2x this.
    head_replay:
        If True, mix past-task replay into each training batch to
        prevent head forgetting. Uses SOMA's episodic buffer concept.
    replay_buffer_size:
        Samples stored per past task for replay (default 200).
    replay_mix_ratio:
        Fraction of loss from replay vs current task (default 0.5).
    """

    config_fn = _mnist_soma_config if dataset == "mnist" else _cifar_soma_config
    config = config_fn(
        disable_critical_periods=disable_critical_periods,
        seed=seed,
    )
    config.initial_integrator_count = integrator_count
    config.initial_associator_count = integrator_count * 2

    num_classes = 10  # both MNIST and CIFAR-10
    input_dim = 784 if dataset == "mnist" else 3072

    adapter = SomaCLAdapter(
        frozen=frozen,
        enable_consolidation=enable_consolidation,
        disable_critical_periods=disable_critical_periods,
        head_replay=head_replay,
        replay_buffer_size=replay_buffer_size,
        replay_mix_ratio=replay_mix_ratio,
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
        optimizer = SGD(trainable, lr=0.1)
        return classifier, optimizer

    return model_factory, adapter.train_one_epoch, adapter.on_task_end
