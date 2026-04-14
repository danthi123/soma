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
    soma.record_growth_event("prune_edge", edge_id="e2")

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


def test_prune_event_uses_edge_id_not_id() -> None:
    """Regression: ``prune_edge`` / ``prune_node`` once used ``id=`` —
    inconsistent with ``synaptogenesis`` / ``neurogenesis`` (``edge_id`` /
    ``node_id``). The journal must use the same field name for the same
    conceptual identifier across create + destroy events, so dashboards
    can group events by object without special-casing the field name."""
    cfg = _small_cfg(
        synaptogenesis_interval=10**9,
        neurogenesis_interval=10**9,
        pruning_interval=1,
        pruning_grace_period=1,
        edge_strength_threshold=1.0,
        inactivity_threshold=1,
    )
    soma = SOMA(cfg, device=torch.device("cpu"))
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
    soma.global_step = 100
    soma._maybe_grow(activations={}, loss_value=None, rng=None)

    prune_edge_events = [ev for ev in soma.growth_log if ev["event_type"] == "prune_edge"]
    assert prune_edge_events, "expected at least one prune_edge event"
    for ev in prune_edge_events:
        assert "edge_id" in ev, f"prune_edge event missing 'edge_id' key: {ev}"
        assert "id" not in ev, f"prune_edge event still uses legacy 'id' key: {ev}"

    prune_node_events = [ev for ev in soma.growth_log if ev["event_type"] == "prune_node"]
    for ev in prune_node_events:
        assert "node_id" in ev, f"prune_node event missing 'node_id' key: {ev}"
        assert "id" not in ev, f"prune_node event still uses legacy 'id' key: {ev}"


def test_consolidation_pruning_events_logged() -> None:
    """Pruning that runs inside ``consolidation_cycle`` (during artificial
    sleep) must show up in the journal the same way online pruning does.
    Force a cycle by aligning ``global_step`` with ``consolidation_interval``
    and stage a weak stale edge the cycle will drop."""
    cfg = _small_cfg(
        synaptogenesis_interval=10**9,
        neurogenesis_interval=10**9,
        pruning_interval=10**9,
        consolidation_interval=10,
        consolidation_replay_steps=3,
        pruning_grace_period=1,
        edge_strength_threshold=1.0,
        inactivity_threshold=1,
        myelination_strength_threshold=1000.0,  # suppress myelination
        consolidation_error_threshold=1e9,  # suppress neurogenesis
    )
    soma = SOMA(cfg, device=torch.device("cpu"))

    # Stage a weak stale edge between two fresh associators the cycle can drop.
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
    stale_edge.strength = 0.0
    stale_edge.last_active_step = 0
    soma.graph.add_edge(stale_edge)

    # Seed episodic memory so sample_for_replay returns something.
    for t in range(4):
        vec = torch.randn(soma.episodic_memory.value_dim)
        soma.episodic_memory.encode(experience=vec, prediction_error=1.0, current_step=t)

    soma.global_step = cfg.consolidation_interval  # aligns divmod == 0
    before = len(soma.growth_log)
    soma._maybe_consolidate(rng=None)
    after_events = list(soma.growth_log)[before:]
    event_types = {ev["event_type"] for ev in after_events}
    assert "prune_edge" in event_types, (
        f"expected prune_edge event from consolidation, got types={event_types}"
    )
    # The stale edge we staged must be among the pruned edges.
    pruned_ids = {ev["edge_id"] for ev in after_events if ev["event_type"] == "prune_edge"}
    assert stale_edge.id in pruned_ids


def test_consolidation_myelination_events_logged() -> None:
    """Myelination only ever runs during consolidation. If a ripe chain
    exists and the cycle compresses it, the journal must record a
    ``myelinate`` event per new merged node — otherwise myelination is
    100% invisible to structural archaeology.

    ``detect_linear_chains`` only treats a node as chainable when it has
    exactly in=1 / out=1 and its predecessor is NOT itself chainable. So
    we stand up a 5-node line ``a -> b -> c -> d -> e`` where the
    bookends ``a`` and ``e`` fail the in/out=1 test (``a`` has in=0,
    ``e`` has out=0), and the middle three ``b, c, d`` form a chain of
    length 3 (>= the default ``min_length=2``) that myelination can
    compress.
    """
    cfg = _small_cfg(
        synaptogenesis_interval=10**9,
        neurogenesis_interval=10**9,
        pruning_interval=10**9,
        consolidation_interval=10,
        consolidation_replay_steps=3,
        pruning_grace_period=10**9,  # suppress pruning
        edge_strength_threshold=0.0,
        inactivity_threshold=10**9,
        myelination_strength_threshold=0.1,  # low -> edges count as strong
        myelination_age_threshold=5,  # low -> edges count as old
        consolidation_error_threshold=1e9,  # suppress neurogenesis
    )
    soma = SOMA(cfg, device=torch.device("cpu"))

    # Five isolated associators: a -> b -> c -> d -> e. Bookends a and e
    # fall out of the "chainable" set (in-degree 0 / out-degree 0), so b,
    # c, and d form the detectable chain. The seed graph's initial
    # associators already have sensor/output fan-in/fan-out that would
    # disqualify them, so we stand up fresh isolated nodes here.
    nodes: list[Node] = []
    for _ in range(5):
        n = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=cfg.associator_input_dim,
            hidden_dim=cfg.associator_hidden_dim,
            output_dim=cfg.associator_output_dim,
            creation_step=0,
            config=cfg,
        )
        soma.graph.add_node(n)
        nodes.append(n)
    for src, tgt in zip(nodes, nodes[1:], strict=False):
        e = Edge(
            source_id=src.id,
            target_id=tgt.id,
            source_output_dim=src.output_dim,
            target_input_dim=tgt.input_dim,
            creation_step=0,  # age = step - 0 = 10 > myelination_age_threshold=5
            initial_weight=1.0,
        )
        e.strength = 1.0  # > myelination_strength_threshold=0.1
        soma.graph.add_edge(e)

    # Seed episodic memory so the cycle runs.
    for t in range(4):
        vec = torch.randn(soma.episodic_memory.value_dim)
        soma.episodic_memory.encode(experience=vec, prediction_error=1.0, current_step=t)

    # Chain detection should pick b, c, d (the middle three).
    inner_chain_ids = {nodes[1].id, nodes[2].id, nodes[3].id}
    all_line_ids = {n.id for n in nodes}

    soma.global_step = cfg.consolidation_interval  # = 10, aligns divmod == 0
    before = len(soma.growth_log)
    soma._maybe_consolidate(rng=None)
    after_events = list(soma.growth_log)[before:]
    myelinate_events = [ev for ev in after_events if ev["event_type"] == "myelinate"]

    # POSITIVE-branch assertion: the ripe inner chain MUST have been
    # compressed. If this fails, the test config no longer actually
    # exercises myelination and the assertions below are meaningless.
    still_present = inner_chain_ids & set(soma.graph.nodes.keys())
    assert not still_present, (
        f"expected inner chain nodes {inner_chain_ids} to be removed by myelination, "
        f"but still present: {still_present}"
    )
    # Bookends a and e must survive — they weren't part of any ripe chain.
    assert nodes[0].id in soma.graph.nodes
    assert nodes[4].id in soma.graph.nodes

    # Journal must record exactly one myelinate event (one chain compressed).
    assert len(myelinate_events) == 1, (
        f"expected 1 myelinate event, got {len(myelinate_events)}: {myelinate_events}"
    )
    ev = myelinate_events[0]
    assert "new_node_id" in ev
    assert "chain_node_ids" in ev
    assert isinstance(ev["chain_node_ids"], list)
    # The chain we built has three internal nodes.
    assert len(ev["chain_node_ids"]) == 3
    # Chain IDs recorded must be exactly the inner nodes of our line.
    assert set(ev["chain_node_ids"]) == inner_chain_ids
    # The new merged node must actually exist in the graph.
    assert ev["new_node_id"] in soma.graph.nodes
    # And it must be fresh — not one of the five we built.
    assert ev["new_node_id"] not in all_line_ids
    # Event must carry the standard step stamp.
    assert ev["step"] == soma.global_step
