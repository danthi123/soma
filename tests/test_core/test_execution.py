"""Tests for ``soma.core.execution``."""

from __future__ import annotations

import copy

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import (
    _aggregate_bucket_inputs,
    _batched_node_forward,
    _record_batched_activations,
    bucket_wave_by_shape,
    compute_wave_layers,
    execute_graph,
    execute_graph_batched,
    topological_sort,
)
from soma.core.graph import Graph
from soma.core.node import Node, NodeType


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig()


def _matched_linear_graph(config: SOMAConfig) -> tuple[Graph, Node, Node, Node]:
    """Build SENSOR(text) -> ASSOC -> OUTPUT(text), all 64-dim (no projections)."""
    graph = Graph()
    # Use matching dims so we don't need projections.
    dim = config.sensor_output_dim
    sensor = Node(
        node_type=NodeType.SENSOR,
        input_dim=dim,
        hidden_dim=dim * 2,
        output_dim=dim,
        creation_step=0,
        config=config,
    )
    assoc = Node(
        node_type=NodeType.ASSOCIATOR,
        input_dim=dim,
        hidden_dim=dim * 2,
        output_dim=dim,
        creation_step=0,
        config=config,
    )
    out = Node(
        node_type=NodeType.OUTPUT,
        input_dim=dim,
        hidden_dim=dim * 2,
        output_dim=dim,
        creation_step=0,
        config=config,
    )
    graph.add_node(sensor, modality="text")
    graph.add_node(assoc)
    graph.add_node(out, modality="text")
    # edges
    graph.add_edge(
        Edge(
            source_id=sensor.id,
            target_id=assoc.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
            initial_weight=1.0,
        )
    )
    graph.add_edge(
        Edge(
            source_id=assoc.id,
            target_id=out.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
            initial_weight=1.0,
        )
    )
    return graph, sensor, assoc, out


class TestTopologicalSort:
    def test_empty_graph(self) -> None:
        order, back = topological_sort(Graph())
        assert order == []
        assert back == set()

    def test_linear_order(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _matched_linear_graph(config)
        order, back = topological_sort(graph)
        assert back == set()
        pos = {nid: idx for idx, nid in enumerate(order)}
        assert pos[sensor.id] < pos[assoc.id] < pos[out.id]

    def test_detects_back_edge(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = 8
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(
            Edge(
                source_id=a.id,
                target_id=b.id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
            )
        )
        # b -> a creates a cycle; this should be flagged as a back-edge.
        back_edge = Edge(
            source_id=b.id,
            target_id=a.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
        )
        graph.add_edge(back_edge)
        order, back = topological_sort(graph)
        assert len(order) == 2
        # Exactly one of the two edges is the back-edge.
        assert len(back) == 1

    def test_handles_disconnected_components(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = 4
        a = Node(NodeType.ASSOCIATOR, dim, 8, dim, 0, config)
        b = Node(NodeType.ASSOCIATOR, dim, 8, dim, 0, config)
        c = Node(NodeType.ASSOCIATOR, dim, 8, dim, 0, config)
        for n in (a, b, c):
            graph.add_node(n)
        # a -> b, c isolated.
        graph.add_edge(
            Edge(
                source_id=a.id,
                target_id=b.id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
            )
        )
        order, back = topological_sort(graph)
        assert set(order) == {a.id, b.id, c.id}
        pos = {nid: idx for idx, nid in enumerate(order)}
        assert pos[a.id] < pos[b.id]
        assert back == set()


class TestExecuteGraph:
    def test_linear_forward_pass(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _matched_linear_graph(config)
        dim = sensor.output_dim
        data = torch.randn(dim)
        outputs, activations = execute_graph(graph, inputs={"text": data}, current_step=1)
        assert "text" in outputs
        assert outputs["text"].shape == (dim,)
        assert sensor.id in activations
        assert assoc.id in activations
        assert out.id in activations
        # Sensor activation should equal the injected data.
        assert torch.equal(activations[sensor.id], data)

    def test_unknown_modality_raises(self, config: SOMAConfig) -> None:
        graph, _, _, _ = _matched_linear_graph(config)
        with pytest.raises(KeyError, match="SENSOR"):
            execute_graph(graph, inputs={"audio": torch.randn(64)}, current_step=0)

    def test_dormant_node_skipped(self, config: SOMAConfig) -> None:
        """A node with no active incoming edges should not produce an activation."""
        graph = Graph()
        dim = config.sensor_output_dim
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        dangling = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(dangling)
        graph.add_node(out, modality="text")
        # Only connect sensor -> out; dangling receives no input.
        graph.add_edge(
            Edge(
                source_id=sensor.id,
                target_id=out.id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
                initial_weight=1.0,
            )
        )
        _, activations = execute_graph(graph, inputs={"text": torch.randn(dim)}, current_step=0)
        assert sensor.id in activations
        assert out.id in activations
        assert dangling.id not in activations

    def test_edge_marked_active(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _matched_linear_graph(config)
        sensor_to_assoc = graph.get_edge(sensor.id, assoc.id)
        assert sensor_to_assoc.last_active_step == 0
        execute_graph(graph, inputs={"text": torch.randn(sensor.output_dim)}, current_step=7)
        assert sensor_to_assoc.last_active_step == 7

    def test_record_edge_activity_off(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _matched_linear_graph(config)
        sensor_to_assoc = graph.get_edge(sensor.id, assoc.id)
        execute_graph(
            graph,
            inputs={"text": torch.randn(sensor.output_dim)},
            current_step=7,
            record_edge_activity=False,
        )
        assert sensor_to_assoc.last_active_step == 0  # unchanged

    def test_output_omitted_when_dormant(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = config.sensor_output_dim
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(out, modality="text")
        # No edge connecting sensor -> out, so output is dormant.
        outputs, activations = execute_graph(
            graph, inputs={"text": torch.randn(dim)}, current_step=0
        )
        assert outputs == {}
        assert sensor.id in activations
        assert out.id not in activations

    def test_cycle_uses_previous_activations(self, config: SOMAConfig) -> None:
        """A back-edge should pick up signal from previous_activations, not current."""
        graph = Graph()
        dim = config.sensor_output_dim
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(out, modality="text")
        # Forward: sensor -> a -> b -> out; cycle: b -> a (back-edge).
        for src, tgt in [(sensor, a), (a, b), (b, out), (b, a)]:
            graph.add_edge(
                Edge(
                    source_id=src.id,
                    target_id=tgt.id,
                    source_output_dim=dim,
                    target_input_dim=dim,
                    creation_step=0,
                    initial_weight=1.0,
                )
            )
        # Step 1: no previous activations yet — cycle edge contributes nothing.
        _, acts1 = execute_graph(graph, inputs={"text": torch.ones(dim)}, current_step=1)
        # Step 2: pass activations from step 1 as previous_activations.
        outputs2, acts2 = execute_graph(
            graph,
            inputs={"text": torch.ones(dim)},
            current_step=2,
            previous_activations=acts1,
        )
        # With the back-edge now feeding b's previous activation into a, a's
        # step-2 activation should differ from a's step-1 activation.
        assert not torch.allclose(acts2[a.id], acts1[a.id])
        assert "text" in outputs2

    def test_multiple_modalities(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = config.sensor_output_dim
        s_text = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        s_img = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        assoc = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        o_text = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        o_img = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(s_text, modality="text")
        graph.add_node(s_img, modality="image")
        graph.add_node(assoc)
        graph.add_node(o_text, modality="text")
        graph.add_node(o_img, modality="image")
        for src, tgt in [
            (s_text, assoc),
            (s_img, assoc),
            (assoc, o_text),
            (assoc, o_img),
        ]:
            graph.add_edge(
                Edge(
                    source_id=src.id,
                    target_id=tgt.id,
                    source_output_dim=dim,
                    target_input_dim=dim,
                    creation_step=0,
                    initial_weight=1.0,
                )
            )
        outputs, _ = execute_graph(
            graph,
            inputs={"text": torch.randn(dim), "image": torch.randn(dim)},
            current_step=0,
        )
        assert set(outputs.keys()) == {"text", "image"}


class TestComputeWaveLayers:
    def test_linear_graph_has_three_waves(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _matched_linear_graph(config)
        waves = compute_wave_layers(graph)
        assert [sorted(w) for w in waves] == [[sensor.id], [assoc.id], [out.id]]

    def test_back_edge_does_not_increase_wave(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = config.sensor_output_dim
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(out, modality="text")
        for src, tgt in [(sensor, a), (a, b), (b, out), (b, a)]:
            graph.add_edge(
                Edge(
                    source_id=src.id,
                    target_id=tgt.id,
                    source_output_dim=dim,
                    target_input_dim=dim,
                    creation_step=0,
                    initial_weight=1.0,
                )
            )
        waves = compute_wave_layers(graph)
        flat = [nid for wave in waves for nid in wave]
        assert set(flat) == {sensor.id, a.id, b.id, out.id}
        idx = {nid: i for i, wave in enumerate(waves) for nid in wave}
        # Forward-edge constraints hold (back-edge b->a is ignored for wave).
        assert idx[sensor.id] < idx[a.id] < idx[b.id] < idx[out.id]

    def test_disconnected_component(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = 4
        nodes = [Node(NodeType.ASSOCIATOR, dim, 8, dim, 0, config) for _ in range(3)]
        for n in nodes:
            graph.add_node(n)
        graph.add_edge(
            Edge(
                source_id=nodes[0].id,
                target_id=nodes[1].id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
            )
        )
        waves = compute_wave_layers(graph)
        idx = {nid: i for i, w in enumerate(waves) for nid in w}
        # Two connected + one isolated. Isolated node sits at wave 0.
        assert idx[nodes[0].id] < idx[nodes[1].id]
        assert idx[nodes[2].id] == 0


class TestBucketWaveByShape:
    def test_uniform_wave_single_bucket(self, config: SOMAConfig) -> None:
        graph = Graph()
        nodes = [Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config) for _ in range(3)]
        for n in nodes:
            graph.add_node(n)
        wave = [n.id for n in nodes]
        buckets = bucket_wave_by_shape(graph, wave)
        assert list(buckets.keys()) == [(8, 16, 8)]
        assert buckets[(8, 16, 8)] == wave  # order preserved

    def test_mixed_wave_multiple_buckets(self, config: SOMAConfig) -> None:
        graph = Graph()
        a = Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config)
        b = Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config)
        c = Node(NodeType.INTEGRATOR, 8, 32, 16, 0, config)
        for n in (a, b, c):
            graph.add_node(n)
        wave = [a.id, c.id, b.id]
        buckets = bucket_wave_by_shape(graph, wave)
        assert buckets[(8, 16, 8)] == [a.id, b.id]  # a before b (input order)
        assert buckets[(8, 32, 16)] == [c.id]

    def test_sensor_excluded(self, config: SOMAConfig) -> None:
        graph = Graph()
        sensor = Node(NodeType.SENSOR, 8, 16, 8, 0, config)
        assoc = Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(assoc)
        buckets = bucket_wave_by_shape(graph, [sensor.id, assoc.id])
        # Sensor must not appear in any bucket — it's handled separately.
        all_ids = [nid for ids in buckets.values() for nid in ids]
        assert sensor.id not in all_ids
        assert assoc.id in all_ids


class TestBatchedNodeForward:
    def test_matches_sequential_same_shape(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        nodes = [Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config) for _ in range(4)]
        x = torch.randn(4, 8)  # pre-aggregated inputs
        # Sequential reference.
        seq_out = torch.stack([nodes[i].forward({"fake": x[i]}, current_step=0) for i in range(4)])
        # Reset state so the batched helper isn't comparing to nodes whose
        # stats have been mutated by the sequential forward above.
        from soma.core.ring_buffer import RingBuffer

        for n in nodes:
            n.activation_history = RingBuffer(n.activation_history.capacity)
            n.activation_ema = 0.0
            n.last_active_step = 0
        bat_out = _batched_node_forward(nodes, x)
        assert bat_out.shape == (4, 8)
        assert torch.allclose(bat_out, seq_out, atol=1e-5, rtol=1e-5)

    def test_gain_applied_per_node(self, config: SOMAConfig) -> None:
        nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(2)]
        nodes[0].gain = 2.0
        nodes[1].gain = 0.5
        # Zero-out weights so residual + gain is all that matters.
        for n in nodes:
            with torch.no_grad():
                n.linear1.weight.zero_()
                n.linear1.bias.zero_()
                n.linear2.weight.zero_()
                n.linear2.bias.zero_()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        out = _batched_node_forward(nodes, x)
        # h = 0 * gain + residual = x (since input_dim == output_dim).
        assert torch.allclose(out, x)

    def test_no_residual_when_dims_differ(self, config: SOMAConfig) -> None:
        nodes = [Node(NodeType.INTEGRATOR, 4, 8, 6, 0, config) for _ in range(2)]
        for n in nodes:
            with torch.no_grad():
                n.linear1.weight.zero_()
                n.linear1.bias.zero_()
                n.linear2.weight.zero_()
                n.linear2.bias.zero_()
        x = torch.randn(2, 4)
        out = _batched_node_forward(nodes, x)
        # All linear layers zeroed + no residual -> output is zero.
        assert torch.allclose(out, torch.zeros(2, 6))

    def test_gradient_routes_to_each_node(self, config: SOMAConfig) -> None:
        nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(3)]
        x = torch.randn(3, 4, requires_grad=False)
        out = _batched_node_forward(nodes, x)
        loss = out.sum()
        loss.backward()
        for n in nodes:
            assert n.linear1.weight.grad is not None
            assert n.linear1.bias.grad is not None
            assert n.linear2.weight.grad is not None
            assert n.linear2.bias.grad is not None
        # Distinct gradients — stack/unbind did not collapse them.
        assert not torch.allclose(nodes[0].linear1.weight.grad, nodes[1].linear1.weight.grad)


class TestRecordBatchedActivations:
    def test_matches_sequential_record(self, config: SOMAConfig) -> None:
        # Build two equivalent node lists. Run sequential on one, batched
        # record on the other. Compare final state.
        seq_nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(3)]
        bat_nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(3)]
        # Sync params so output magnitudes would match in the real flow
        # (this helper is agnostic but it documents intent).
        for s, b in zip(seq_nodes, bat_nodes, strict=True):
            b.load_state_dict(s.state_dict())
        outs = torch.stack([torch.randn(4) * 3.0 for _ in range(3)])
        for i, n in enumerate(seq_nodes):
            n._record_activation(outs[i], current_step=7)
        _record_batched_activations(bat_nodes, outs, current_step=7)
        for s, b in zip(seq_nodes, bat_nodes, strict=True):
            assert len(s.activation_history) == len(b.activation_history)
            assert s.activation_history.to_list() == pytest.approx(
                b.activation_history.to_list(), rel=1e-6, abs=1e-9
            )
            assert s.activation_ema == pytest.approx(b.activation_ema, rel=1e-6)
            assert s.last_active_step == b.last_active_step


class TestAggregateBucketInputs:
    def test_sums_multiple_incoming_edges(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _matched_linear_graph(config)
        dim = sensor.output_dim
        # Another parallel path: sensor -> out direct.
        graph.add_edge(
            Edge(
                source_id=sensor.id,
                target_id=out.id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
                initial_weight=0.5,
            )
        )
        # Prepare synthetic activations: sensor=ones, assoc=twos.
        activations = {
            sensor.id: torch.ones(dim),
            assoc.id: torch.full((dim,), 2.0),
        }
        x, active_nodes = _aggregate_bucket_inputs(
            graph=graph,
            nodes=[graph.nodes[out.id]],
            activations=activations,
            previous_activations={},
            back_edges=set(),
            current_step=5,
            record_edge_activity=False,
        )
        assert active_nodes == [graph.nodes[out.id]]
        assert x.shape == (1, dim)
        # assoc->out (w=1.0): 2.0 * 1.0 = 2.0
        # sensor->out (w=0.5): 1.0 * 0.5 = 0.5
        # Total: 2.5 per feature.
        assert torch.allclose(x[0], torch.full((dim,), 2.5), atol=1e-5)

    def test_dormant_node_excluded(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _matched_linear_graph(config)
        # No source activations -> every non-sensor node is dormant.
        x, active = _aggregate_bucket_inputs(
            graph=graph,
            nodes=[graph.nodes[assoc.id], graph.nodes[out.id]],
            activations={},
            previous_activations={},
            back_edges=set(),
            current_step=0,
            record_edge_activity=False,
        )
        assert active == []
        assert x.numel() == 0

    def test_mark_active_when_record_activity(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _matched_linear_graph(config)
        dim = sensor.output_dim
        activations = {sensor.id: torch.ones(dim)}
        s_to_a = graph.get_edge(sensor.id, assoc.id)
        assert s_to_a.last_active_step == 0
        _aggregate_bucket_inputs(
            graph=graph,
            nodes=[graph.nodes[assoc.id]],
            activations=activations,
            previous_activations={},
            back_edges=set(),
            current_step=42,
            record_edge_activity=True,
        )
        assert s_to_a.last_active_step == 42


class TestBatchedExecutorScaffold:
    def test_batched_symbol_exported(self) -> None:
        from soma.core import execution

        assert hasattr(execution, "execute_graph_batched")

    def test_batched_matches_sequential_on_linear_graph(self, config: SOMAConfig) -> None:
        graph, sensor, _, _ = _matched_linear_graph(config)
        data = torch.randn(sensor.output_dim)
        # Deepcopy preserves UUIDs + weights + buffers so the two graphs
        # are byte-identical before we run them.
        graph2 = copy.deepcopy(graph)
        out_seq, _ = execute_graph(graph, inputs={"text": data}, current_step=1)
        out_bat, _ = execute_graph_batched(graph2, inputs={"text": data}, current_step=1)
        assert set(out_seq.keys()) == set(out_bat.keys())
        for k in out_seq:
            assert torch.allclose(out_seq[k], out_bat[k], atol=1e-5, rtol=1e-5)


class TestBatchedExecutorParity:
    @staticmethod
    def _make_graph(config: SOMAConfig, seed: int = 0) -> Graph:
        torch.manual_seed(seed)
        g = Graph()
        dim = config.sensor_output_dim
        s = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        a1 = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        a2 = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        a3 = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        # Different-shape integrator to force a second bucket in its wave.
        integ = Node(NodeType.INTEGRATOR, dim, dim * 4, dim * 2, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        g.add_node(s, modality="text")
        for n in (a1, a2, a3, integ):
            g.add_node(n)
        g.add_node(out, modality="text")
        edges = [
            (s, a1),
            (s, a2),
            (s, a3),
            (s, integ),
            (a1, out),
            (a2, out),
            (a3, out),
        ]
        for src, tgt in edges:
            g.add_edge(
                Edge(
                    source_id=src.id,
                    target_id=tgt.id,
                    source_output_dim=src.output_dim,
                    target_input_dim=tgt.input_dim,
                    creation_step=0,
                    initial_weight=0.5,
                )
            )
        return g

    def test_batched_body_not_delegating(self) -> None:
        """Ensure the batched executor has a real body, not just a delegate.

        This is the concrete RED for Task 7 — it fails while
        execute_graph_batched is still a one-line wrapper around
        execute_graph and passes once we wire the real wave-by-wave
        implementation.
        """
        import inspect

        from soma.core import execution

        src = inspect.getsource(execution.execute_graph_batched)
        assert "compute_wave_layers" in src, (
            "execute_graph_batched still delegates; Task 7 must replace the body"
        )

    def test_batched_equals_sequential_outputs(self, config: SOMAConfig) -> None:
        g1 = self._make_graph(config, seed=123)
        g2 = copy.deepcopy(g1)
        data = torch.randn(config.sensor_output_dim)
        seq_out, seq_act = execute_graph(g1, inputs={"text": data}, current_step=1)
        bat_out, bat_act = execute_graph_batched(g2, inputs={"text": data}, current_step=1)
        assert set(seq_out.keys()) == set(bat_out.keys())
        for k in seq_out:
            assert torch.allclose(seq_out[k], bat_out[k], atol=1e-5, rtol=1e-5), (
                f"Output {k!r} diverges: max diff {(seq_out[k] - bat_out[k]).abs().max().item()}"
            )
        assert set(seq_act.keys()) == set(bat_act.keys())

    def test_batched_equals_sequential_gradients(self, config: SOMAConfig) -> None:
        g1 = self._make_graph(config, seed=7)
        g2 = copy.deepcopy(g1)
        data = torch.randn(config.sensor_output_dim)
        out_seq, _ = execute_graph(g1, inputs={"text": data}, current_step=1)
        out_bat, _ = execute_graph_batched(g2, inputs={"text": data}, current_step=1)
        out_seq["text"].sum().backward()
        out_bat["text"].sum().backward()
        for nid in g1.nodes:
            n1 = g1.nodes[nid]
            n2 = g2.nodes[nid]
            if n1.node_type is NodeType.SENSOR:
                continue
            for pname in (
                "linear1.weight",
                "linear1.bias",
                "linear2.weight",
                "linear2.bias",
            ):
                p1 = dict(n1.named_parameters())[pname].grad
                p2 = dict(n2.named_parameters())[pname].grad
                if p1 is None and p2 is None:
                    continue
                assert p1 is not None and p2 is not None, (
                    f"grad presence mismatch on {nid[:8]}.{pname}"
                )
                assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-5), (
                    f"Gradient diverges on {nid[:8]}.{pname}: "
                    f"max diff {(p1 - p2).abs().max().item()}"
                )

    def test_batched_preserves_per_node_state(self, config: SOMAConfig) -> None:
        g1 = self._make_graph(config, seed=11)
        g2 = copy.deepcopy(g1)
        data = torch.randn(config.sensor_output_dim)
        execute_graph(g1, inputs={"text": data}, current_step=42)
        execute_graph_batched(g2, inputs={"text": data}, current_step=42)
        for nid in g1.nodes:
            n1 = g1.nodes[nid]
            n2 = g2.nodes[nid]
            assert n1.last_active_step == n2.last_active_step
            assert n1.activation_ema == pytest.approx(n2.activation_ema, rel=1e-6, abs=1e-9)
            assert n1.activation_history.to_list() == pytest.approx(
                n2.activation_history.to_list(), rel=1e-6, abs=1e-9
            )

    def test_batched_handles_back_edges(self, config: SOMAConfig) -> None:
        def build() -> Graph:
            torch.manual_seed(99)
            g = Graph()
            dim = config.sensor_output_dim
            s = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
            a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
            b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
            o = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
            g.add_node(s, modality="text")
            g.add_node(a)
            g.add_node(b)
            g.add_node(o, modality="text")
            for src, tgt in [(s, a), (a, b), (b, o), (b, a)]:
                g.add_edge(
                    Edge(
                        source_id=src.id,
                        target_id=tgt.id,
                        source_output_dim=dim,
                        target_input_dim=dim,
                        creation_step=0,
                        initial_weight=1.0,
                    )
                )
            return g

        g1 = build()
        g2 = copy.deepcopy(g1)
        data = torch.ones(config.sensor_output_dim)
        _, prev_seq = execute_graph(g1, inputs={"text": data}, current_step=1)
        _, prev_bat = execute_graph_batched(g2, inputs={"text": data}, current_step=1)
        out_seq, _ = execute_graph(
            g1,
            inputs={"text": data},
            current_step=2,
            previous_activations=prev_seq,
        )
        out_bat, _ = execute_graph_batched(
            g2,
            inputs={"text": data},
            current_step=2,
            previous_activations=prev_bat,
        )
        assert torch.allclose(out_seq["text"], out_bat["text"], atol=1e-5, rtol=1e-5)
