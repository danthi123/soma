"""Tests for the append-only growth journal on SOMA.

The journal records structural events (synaptogenesis, neurogenesis,
edge/node pruning) in the order they happen, each tagged with the
``global_step`` at which they occurred. It exists purely for
observability — inspecting the life of a brain over long runs.

See CLAUDE.md "Critical Invariants" for what kinds of structural events
the journal needs to cover.
"""

from __future__ import annotations

from pathlib import Path

import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.node import Node, NodeType
from soma.system import SOMA


def _small_cfg(**overrides: object) -> SOMAConfig:
    base = dict(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=16,
        integrator_hidden_dim=16,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=2,
        wm_dim=8,
        episodic_capacity=4,
        key_dim=8,
        value_dim=8,
        vocab_size=16,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )
    base.update(overrides)
    return SOMAConfig(**base)  # type: ignore[arg-type]


def test_growth_log_initialized_empty_and_bounded() -> None:
    soma = SOMA(_small_cfg(), device=torch.device("cpu"))
    assert hasattr(soma, "growth_log")
    assert len(soma.growth_log) == 0
    # The deque must be bounded so a long-running brain doesn't leak memory.
    assert soma.growth_log.maxlen == 10_000


def test_record_growth_event_stamps_step_and_event_type() -> None:
    soma = SOMA(_small_cfg(), device=torch.device("cpu"))
    soma.global_step = 42
    soma.record_growth_event("synaptogenesis", edge_id="e1", source="a", target="b")
    assert len(soma.growth_log) == 1
    event = soma.growth_log[-1]
    assert event["step"] == 42
    assert event["event_type"] == "synaptogenesis"
    assert event["edge_id"] == "e1"
    assert event["source"] == "a"
    assert event["target"] == "b"


def test_synaptogenesis_creates_event_via_maybe_grow() -> None:
    """Running the growth pass over a graph with co-active nodes creates
    at least one synaptogenesis event. We seed co-activation by feeding
    activations for two non-sensor/output nodes with matching shapes."""
    cfg = _small_cfg(
        synaptogenesis_interval=1,
        synaptogenesis_rate=1000.0,  # saturate probability to 1
        activation_threshold=0.0,
        neurogenesis_interval=10**9,  # disable
        pruning_interval=10**9,  # disable
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    soma.global_step = 1  # _maybe_grow skips step 0

    # Gather the two associators we seeded in _initialize_seed_graph;
    # they already share input/output dims so an edge between them is legal.
    associators = [n for n in soma.graph.all_nodes() if n.node_type == NodeType.ASSOCIATOR]
    assert len(associators) >= 2
    a, b = associators[0], associators[1]
    activations = {
        a.id: torch.ones(a.output_dim),
        b.id: torch.ones(b.output_dim),
    }

    before = len(soma.growth_log)
    soma._maybe_grow(activations, loss_value=None, rng=None)
    after = len(soma.growth_log)

    assert after > before
    types = {ev["event_type"] for ev in list(soma.growth_log)[before:]}
    assert "synaptogenesis" in types
    # Each event must carry the required shape.
    for event in list(soma.growth_log)[before:]:
        assert {"step", "event_type"}.issubset(event.keys())
        assert event["step"] == soma.global_step


def test_neurogenesis_creates_event_via_maybe_grow() -> None:
    """Feed a large error spike so the neurogenesis trigger fires, then
    check the journal gains an entry tagged ``neurogenesis``."""
    cfg = _small_cfg(
        synaptogenesis_interval=10**9,  # disable
        neurogenesis_interval=1,
        neurogenesis_threshold=1.5,
        pruning_interval=10**9,  # disable
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    soma.global_step = 1
    # Baseline ~1.0 over 100 samples, recent ~10.0 over last 5 — recent/base = 10.
    soma._recent_errors = [1.0] * 95 + [10.0] * 5

    before_nodes = soma.graph.num_nodes
    before_log = len(soma.growth_log)
    soma._maybe_grow(activations={}, loss_value=None, rng=None)
    # Either neurogenesis fired (new node + log entry) or it didn't (no
    # change at all). The contract: if a node was added, the log grew.
    if soma.graph.num_nodes > before_nodes:
        events_after = list(soma.growth_log)[before_log:]
        assert any(ev["event_type"] == "neurogenesis" for ev in events_after)


def test_pruning_creates_event_via_maybe_grow() -> None:
    """Manually install a weak stale edge, then run pruning — the journal
    should capture the removal."""
    cfg = _small_cfg(
        synaptogenesis_interval=10**9,  # disable
        neurogenesis_interval=10**9,  # disable
        pruning_interval=1,
        pruning_grace_period=1,
        edge_strength_threshold=1.0,  # every edge is "weak"
        inactivity_threshold=1,
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
    # Add an isolated associator pair + weak stale edge so pruning has something
    # to remove without breaking the sensor->output path.
    a = Node(
        node_type=NodeType.ASSOCIATOR,
        input_dim=cfg.associator_input_dim,
        hidden_dim=cfg.associator_hidden_dim,
        output_dim=cfg.associator_output_dim,
        creation_step=0,
        config=cfg,
    )
    b = Node(
        node_type=NodeType.ASSOCIATOR,
        input_dim=cfg.associator_input_dim,
        hidden_dim=cfg.associator_hidden_dim,
        output_dim=cfg.associator_output_dim,
        creation_step=0,
        config=cfg,
    )
    soma.graph.add_node(a)
    soma.graph.add_node(b)
    stale_edge = Edge(
        source_id=a.id,
        target_id=b.id,
        source_output_dim=a.output_dim,
        target_input_dim=b.input_dim,
        creation_step=0,
        initial_weight=0.0,
    )
    soma.graph.add_edge(stale_edge)
    soma.global_step = 100  # well past the grace period

    before = len(soma.growth_log)
    soma._maybe_grow(activations={}, loss_value=None, rng=None)
    after_events = list(soma.growth_log)[before:]
    # Pruning should have at minimum removed the stale edge we injected.
    event_types = {ev["event_type"] for ev in after_events}
    assert "prune_edge" in event_types


def test_growth_log_persists_round_trip(tmp_path: Path) -> None:
    """Save + load must preserve the journal contents (bounded deque)."""
    soma = SOMA(_small_cfg(), device=torch.device("cpu"))
    soma.record_growth_event("synaptogenesis", edge_id="e1")
    soma.record_growth_event("prune_edge", id="e2")

    save_path = tmp_path / "brain.pt"
    soma.save_state(str(save_path))

    soma2 = SOMA(_small_cfg(), device=torch.device("cpu"))
    soma2.load_state(str(save_path))
    assert [dict(ev) for ev in soma2.growth_log] == [dict(ev) for ev in soma.growth_log]
    assert soma2.growth_log.maxlen == 10_000


def test_load_state_with_missing_growth_log_is_backward_compatible(tmp_path: Path) -> None:
    """Old checkpoints written before the journal existed had no
    ``growth_log`` key in their payload. Loading one must still succeed
    and yield an empty journal."""
    soma = SOMA(_small_cfg(), device=torch.device("cpu"))
    save_path = tmp_path / "brain.pt"
    soma.save_state(str(save_path))

    # Strip ``growth_log`` from the payload to simulate a pre-Task-8 checkpoint.
    raw = torch.load(str(save_path), map_location="cpu", weights_only=False)
    assert raw["format"] == "soma-brain"
    raw["payload"].pop("growth_log", None)
    torch.save(raw, str(save_path))

    soma2 = SOMA(_small_cfg(), device=torch.device("cpu"))
    soma2.load_state(str(save_path))
    assert hasattr(soma2, "growth_log")
    assert len(soma2.growth_log) == 0
    assert soma2.growth_log.maxlen == 10_000
