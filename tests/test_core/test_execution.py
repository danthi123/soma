"""Tests for ``soma.core.execution``."""

from __future__ import annotations

import copy

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import (
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
        out_bat, _ = execute_graph_batched(
            graph2, inputs={"text": data}, current_step=1
        )
        assert set(out_seq.keys()) == set(out_bat.keys())
        for k in out_seq:
            assert torch.allclose(out_seq[k], out_bat[k], atol=1e-5, rtol=1e-5)
