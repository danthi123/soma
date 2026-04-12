"""Stage 1 integration test: end-to-end core loop stability.

Exercises: ``execute_graph`` + ``update_step`` + ``synaptogenesis`` +
``pruning`` in a single loop against a deterministic synthetic
regression task.

Asserts:
- No exceptions for the full run.
- Loss trends downward (recent mean < initial mean).
- Graph stays within hard caps (no explosion).
- Boundary nodes (SENSOR/OUTPUT) are never removed.
- Edge weights stay within ``config.max_edge_weight``.

Two variants:
- Default run (fast): 500 steps — runs on every CI invocation.
- Slow marker: 10 000 steps — matches whitepaper's stability criterion.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph
from soma.core.graph import Graph
from soma.core.learning import update_step
from soma.core.node import Node, NodeType
from soma.growth.pruning import pruning
from soma.growth.synaptogenesis import synaptogenesis


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _stage1_config(max_nodes: int = 64, max_edges_per_node: float = 20.0) -> SOMAConfig:
    """Small config tuned for fast integration testing."""
    return SOMAConfig(
        base_lr=0.01,
        youth_lr_multiplier=1.0,  # damp aggressive newborn LR for the test
        hebbian_lr=0.0001,
        maturity_increment=0.001,
        activation_threshold=0.01,
        max_edge_weight=2.0,  # tighter clamp so activation chain stays bounded
        synaptogenesis_rate=0.05,  # lower so we don't flood with edges in 500 steps
        synaptogenesis_interval=50,
        pruning_interval=100,
        pruning_grace_period=50,
        edge_strength_threshold=0.001,
        inactivity_threshold=100,
        max_nodes=max_nodes,
        max_edges_per_node=max_edges_per_node,
    )


def _build_seed_graph(
    config: SOMAConfig, dim: int = 8, num_assoc: int = 4, *, seed: int = 0
) -> Graph:
    """Build SENSOR -> [N associators] -> OUTPUT seed graph, all matched dims."""
    torch.manual_seed(seed)
    graph = Graph()
    sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    graph.add_node(sensor, modality="text")
    graph.add_node(out, modality="text")

    assoc_ids: list[str] = []
    for _ in range(num_assoc):
        node = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        graph.add_node(node)
        assoc_ids.append(node.id)

    for aid in assoc_ids:
        graph.add_edge(
            Edge(
                source_id=sensor.id,
                target_id=aid,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
                initial_weight=0.5,
            )
        )
        graph.add_edge(
            Edge(
                source_id=aid,
                target_id=out.id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
                initial_weight=0.5,
            )
        )
    return graph


def _synthetic_target(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Deterministic linear target: ``y = weight @ x + bias``."""
    return weight @ x + bias


def _run_training(
    graph: Graph,
    config: SOMAConfig,
    num_steps: int,
    *,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Run one training loop and return per-step metrics."""
    torch.manual_seed(seed)
    dim = graph.get_sensor("text").output_dim
    # Fixed linear target the graph has to learn.
    target_weight = torch.eye(dim) * 0.5
    target_bias = torch.zeros(dim)
    rng = torch.Generator().manual_seed(seed + 1)

    losses: list[float] = []
    node_counts: list[float] = []
    edge_counts: list[float] = []
    max_weight_abs: list[float] = []

    for step in range(num_steps):
        # Deterministic but varied inputs to avoid trivial memorization.
        x = torch.randn(dim, generator=rng)
        target = _synthetic_target(x, target_weight, target_bias)

        outputs, activations = execute_graph(graph, inputs={"text": x}, current_step=step)
        if "text" not in outputs:
            # Dormant output — treat as zero prediction.
            losses.append(float("nan"))
            node_counts.append(float(graph.num_nodes))
            edge_counts.append(float(graph.num_edges))
            max_weight_abs.append(0.0)
            continue

        loss = F.mse_loss(outputs["text"], target)
        losses.append(float(loss.item()))
        update_step(graph, loss, activations, config)

        # Structural mutations on schedule.
        if step % config.synaptogenesis_interval == 0 and step > 0:
            synaptogenesis(graph, activations, step, config, rng=rng)
        if step % config.pruning_interval == 0 and step > 0:
            pruning(graph, step, config)

        node_counts.append(float(graph.num_nodes))
        edge_counts.append(float(graph.num_edges))
        if graph.num_edges > 0:
            max_weight_abs.append(max(abs(float(e.weight.item())) for e in graph.all_edges()))
        else:
            max_weight_abs.append(0.0)

    return {
        "losses": losses,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "max_weight_abs": max_weight_abs,
    }


def _loss_mean(losses: list[float]) -> float:
    valid = [loss_val for loss_val in losses if loss_val == loss_val]  # NaN-safe
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
class TestFastLoop:
    """Fast 500-step integration sweep — runs on every CI invocation."""

    def test_loop_runs_without_exceptions(self) -> None:
        config = _stage1_config()
        graph = _build_seed_graph(config, seed=0)
        metrics = _run_training(graph, config, num_steps=500, seed=42)
        # No NaNs in the loss trace — the loop is stable.
        assert all(loss_val == loss_val for loss_val in metrics["losses"])

    def test_loss_trends_down(self) -> None:
        config = _stage1_config()
        graph = _build_seed_graph(config, seed=0)
        metrics = _run_training(graph, config, num_steps=500, seed=42)
        early = _loss_mean(metrics["losses"][:50])
        late = _loss_mean(metrics["losses"][-50:])
        assert late == late  # not NaN
        assert late < early * 0.75

    def test_graph_stays_within_caps(self) -> None:
        config = _stage1_config(max_nodes=64)
        graph = _build_seed_graph(config, seed=0)
        metrics = _run_training(graph, config, num_steps=500, seed=42)
        assert max(metrics["node_counts"]) <= config.max_nodes
        # Sanity: edge count shouldn't blow up beyond a reasonable bound
        # (max_nodes * max_edges_per_node).
        bound = config.max_nodes * config.max_edges_per_node
        assert max(metrics["edge_counts"]) <= bound

    def test_boundary_nodes_survive(self) -> None:
        config = _stage1_config()
        graph = _build_seed_graph(config, seed=0)
        _run_training(graph, config, num_steps=500, seed=42)
        # Both SENSOR and OUTPUT for "text" must still be present.
        assert "text" in graph.sensor_nodes
        assert "text" in graph.output_nodes

    def test_edge_weights_stay_clamped(self) -> None:
        config = _stage1_config()
        graph = _build_seed_graph(config, seed=0)
        metrics = _run_training(graph, config, num_steps=500, seed=42)
        # Every observed max |weight| must respect the clamp (allow tiny
        # floating-point slack).
        assert max(metrics["max_weight_abs"]) <= config.max_edge_weight + 1e-5


@pytest.mark.slow
class TestLongLoop:
    """10 000-step stability run — checkpoint spec criterion."""

    def test_stable_over_10k_steps(self) -> None:
        config = _stage1_config(max_nodes=96)
        graph = _build_seed_graph(config)
        metrics = _run_training(graph, config, num_steps=10_000)
        assert all(loss_val == loss_val for loss_val in metrics["losses"])
        early = _loss_mean(metrics["losses"][:200])
        late = _loss_mean(metrics["losses"][-200:])
        assert late < early  # any net improvement is enough for 10K
        assert max(metrics["node_counts"]) <= config.max_nodes
        assert max(metrics["max_weight_abs"]) <= config.max_edge_weight + 1e-5
        assert "text" in graph.sensor_nodes
        assert "text" in graph.output_nodes
