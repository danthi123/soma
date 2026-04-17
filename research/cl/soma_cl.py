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
        use_layer_norm: bool = False,
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
        self.input_proj = nn.Linear(input_dim, sensor_dim, bias=False)

        # Classification head
        self._head_dim = sensor_dim + self._output_dim
        self.layer_norm: nn.LayerNorm | None = None
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self._head_dim)
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
        if self.layer_norm is not None:
            combined = self.layer_norm(combined)
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
        use_herding: bool = False,
        head_ewc: bool = False,
        ewc_lambda: float = 1000.0,
    ) -> None:
        self.frozen = frozen
        self.enable_consolidation = enable_consolidation
        self.disable_critical_periods = disable_critical_periods
        self.consolidation_replay_steps = consolidation_replay_steps
        self.head_replay = head_replay
        self.replay_buffer_size = replay_buffer_size
        self.replay_mix_ratio = replay_mix_ratio
        self.use_herding = use_herding
        self.head_ewc = head_ewc
        self.ewc_lambda = ewc_lambda
        self._model: SomaClassifier | None = None
        # Replay buffer: list of (x_batch, y_batch) from past tasks
        self._replay_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Head-only EWC state
        self._fisher: dict[str, torch.Tensor] = {}
        self._prev_params: dict[str, torch.Tensor] = {}

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
        When ``head_ewc=True``, adds Fisher-weighted penalty on head
        parameter drift from previous task solutions.
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
                ratio = self.replay_mix_ratio
                loss = (1.0 - ratio) * loss + ratio * replay_loss

            # Head-only EWC penalty
            if self.head_ewc and self._fisher:
                ewc_loss = self._compute_ewc_penalty(model, device)
                loss = loss + ewc_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_ewc_penalty(
        self, model: nn.Module, device: torch.device
    ) -> torch.Tensor:
        """Fisher-weighted L2 penalty on head + input_proj drift."""
        penalty = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in self._fisher:
                fisher = self._fisher[name].to(device)
                prev = self._prev_params[name].to(device)
                penalty = penalty + (fisher * (param - prev) ** 2).sum()
        return (self.ewc_lambda / 2.0) * penalty

    def _compute_fisher(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> None:
        """Compute diagonal Fisher on head + input_proj only."""
        model.train(False)
        criterion = nn.CrossEntropyLoss()
        fisher: dict[str, torch.Tensor] = {}
        # Only track head and input_proj parameters
        head_params = {
            n: p for n, p in model.named_parameters()
            if "head" in n or "input_proj" in n or "layer_norm" in n
        }
        for name in head_params:
            fisher[name] = torch.zeros_like(head_params[name])

        n_samples = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            model.zero_grad()
            loss.backward()
            for name, param in head_params.items():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            n_samples += x.size(0)

        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        # Accumulate Fisher across tasks (online EWC)
        for name in fisher:
            if name in self._fisher:
                self._fisher[name] = self._fisher[name].cpu() + fisher[name].cpu()
            else:
                self._fisher[name] = fisher[name].cpu()

        # Snapshot current params
        for name, param in head_params.items():
            self._prev_params[name] = param.data.clone().cpu()

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
        """Post-task hook: consolidation + replay buffer + Fisher."""
        if isinstance(model, SomaClassifier):
            model.consolidate()

        if self.head_replay:
            self._update_replay_buffer(train_loader)

        if self.head_ewc:
            self._compute_fisher(model, train_loader, device)

    def _update_replay_buffer(self, train_loader: DataLoader) -> None:
        """Add samples from this task to the replay buffer.

        Uses herding (nearest-to-class-mean) if ``use_herding=True``,
        otherwise random selection. Stored on CPU.
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

        idx = (
            self._herd_select(all_x, all_y, k)
            if self.use_herding
            else torch.randperm(n)[:k]
        )

        self._replay_buffer.append((all_x[idx].cpu(), all_y[idx].cpu()))

    @staticmethod
    def _herd_select(
        x: torch.Tensor, y: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Select k samples nearest to per-class means (iCaRL-style)."""
        classes = y.unique()
        per_class = max(k // len(classes), 1)
        selected: list[int] = []

        for c in classes:
            mask = y == c
            class_x = x[mask]
            class_idx = mask.nonzero(as_tuple=True)[0]
            if len(class_x) == 0:
                continue
            mean = class_x.mean(dim=0, keepdim=True)
            dists = torch.cdist(class_x, mean).squeeze(-1)
            _, top_idx = dists.topk(min(per_class, len(class_x)), largest=False)
            selected.extend(class_idx[top_idx].tolist())

        return torch.tensor(selected[:k], dtype=torch.long)


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
    use_herding: bool = False,
    head_ewc: bool = False,
    ewc_lambda: float = 1000.0,
    use_layer_norm: bool = False,
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
        use_herding=use_herding,
        head_ewc=head_ewc,
        ewc_lambda=ewc_lambda,
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
            use_layer_norm=use_layer_norm,
        ).to(device)

        # Optimize head, input projection, and layer norm (if present)
        trainable = list(classifier.input_proj.parameters()) + list(
            classifier.head.parameters()
        )
        if classifier.layer_norm is not None:
            trainable += list(classifier.layer_norm.parameters())
        optimizer = SGD(trainable, lr=0.1)
        return classifier, optimizer

    return model_factory, adapter.train_one_epoch, adapter.on_task_end
