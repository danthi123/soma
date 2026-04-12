"""Consolidation cycle — offline replay of episodic experiences.

Whitepaper Section 6. Artificial sleep: every
``config.consolidation_interval`` steps, SOMA samples experiences from
episodic memory and replays them through the graph with a reduced
learning rate. The whitepaper also lists structural maintenance
(pruning, myelination, optional neurogenesis) during this phase — those
are added in Stage 4 (Unit 23). Unit 12 implements replay only.

``consolidation_cycle`` is intentionally decoupled from the SOMA class:
callers provide an ``ExperienceUnpacker`` that converts a stored
experience vector into ``(inputs, targets)`` — the usual convention is
"first half of the vector is sensor input, second half is target
output" but any modality split works.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as F  # noqa: N812

from soma.core.config import SOMAConfig
from soma.core.execution import execute_graph
from soma.core.graph import Graph
from soma.memory.episodic_memory import EpisodicMemory

# (experience_vector) -> (inputs_by_modality, targets_by_modality)
ExperienceUnpacker = Callable[
    [torch.Tensor],
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
]


@dataclass(frozen=True)
class ConsolidationResult:
    """Summary of one consolidation cycle."""

    num_replayed: int
    replay_losses: list[float]

    @property
    def mean_loss(self) -> float:
        if not self.replay_losses:
            return 0.0
        return sum(self.replay_losses) / len(self.replay_losses)


def consolidation_cycle(
    graph: Graph,
    episodic_memory: EpisodicMemory,
    current_step: int,
    config: SOMAConfig,
    experience_unpacker: ExperienceUnpacker,
    *,
    num_replay_steps: int | None = None,
    rng: torch.Generator | None = None,
) -> ConsolidationResult:
    """Run one consolidation cycle of replay-only learning.

    Parameters
    ----------
    graph:
        Processing graph whose params get updated. The graph must be
        executable (sensors/outputs registered) for the experience
        inputs/targets the unpacker produces.
    episodic_memory:
        Source of experiences; replay sampling is prioritized by
        surprise * age (see ``EpisodicMemory.sample_for_replay``).
    current_step:
        Global step at the start of the cycle. Used for replay-sampling
        age weighting and for ``execute_graph`` bookkeeping.
    config:
        Supplies ``base_lr``, ``consolidation_lr_ratio``, and the
        replay batch size default.
    experience_unpacker:
        Callable that splits a stored experience vector into the dict
        of sensor inputs + the dict of per-modality targets.
    num_replay_steps:
        Number of experiences to replay. Defaults to
        ``config.consolidation_replay_steps``.
    rng:
        Optional ``torch.Generator`` used for replay sampling.

    Returns
    -------
    ConsolidationResult with per-replay losses and the count actually
    replayed (may be less than ``num_replay_steps`` when memory is
    sparsely populated or surprise is all-zero).
    """
    if current_step < 0:
        raise ValueError(f"current_step must be non-negative, got {current_step}")

    batch_size = (
        num_replay_steps if num_replay_steps is not None else config.consolidation_replay_steps
    )
    if batch_size <= 0:
        raise ValueError(f"num_replay_steps must be positive, got {batch_size}")

    replay_batch = episodic_memory.sample_for_replay(
        current_step=current_step,
        batch_size=batch_size,
        rng=rng,
    )
    if not replay_batch:
        return ConsolidationResult(num_replayed=0, replay_losses=[])

    consolidation_lr = config.base_lr * config.consolidation_lr_ratio
    replay_losses: list[float] = []

    # Replay is *not* supposed to look like a normal forward pass for
    # pruning — we don't want it to keep edges alive that would otherwise
    # be eligible for removal.
    for experience in replay_batch:
        inputs, targets = experience_unpacker(experience)
        outputs, activations = execute_graph(
            graph,
            inputs=inputs,
            current_step=current_step,
            record_edge_activity=False,
        )

        loss = _replay_loss(outputs, targets)
        if loss is None:
            # No output / target overlap this replay step — skip.
            continue

        loss.backward()  # type: ignore[no-untyped-call]

        _apply_consolidation_sgd(
            graph,
            active_node_ids=list(activations.keys()),
            lr=consolidation_lr,
        )

        replay_losses.append(float(loss.item()))

    return ConsolidationResult(num_replayed=len(replay_losses), replay_losses=replay_losses)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _replay_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    """Mean of per-modality MSE losses; None if no modality pairs up."""
    components: list[torch.Tensor] = []
    for modality, target in targets.items():
        out = outputs.get(modality)
        if out is None:
            continue
        components.append(F.mse_loss(out, target))
    if not components:
        return None
    return torch.stack(components).mean()


def _apply_consolidation_sgd(
    graph: Graph,
    active_node_ids: list[str],
    *,
    lr: float,
) -> None:
    """SGD step over active nodes; LR scaled by ``0.5 + 0.5 * maturity``.

    Unlike ``update_step``, consolidation favours MATURE nodes (full LR);
    young nodes only get half-LR so new structure isn't overwritten
    before it has a chance to settle through online learning.
    """
    with torch.no_grad():
        for nid in active_node_ids:
            if nid not in graph.nodes:
                continue
            node = graph.nodes[nid]
            node_lr = lr * (0.5 + 0.5 * node.maturity)
            for param in node.parameters():
                if param.grad is None:
                    continue
                param.data.add_(param.grad, alpha=-node_lr)
                param.grad.zero_()
        # Edges learn at the base consolidation rate.
        for edge in graph.all_edges():
            for param in edge.parameters():
                if param.grad is None:
                    continue
                param.data.add_(param.grad, alpha=-lr)
                param.grad.zero_()
