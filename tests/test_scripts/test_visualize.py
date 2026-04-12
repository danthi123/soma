"""Tests for ``scripts.visualize`` — checkpoint summary + DOT export."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import visualize  # noqa: E402

from soma.core.config import SOMAConfig  # noqa: E402
from soma.core.node import NodeType  # noqa: E402
from soma.system import SOMA  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
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
        seed=0,
    )


@pytest.fixture
def soma_checkpoint(small_config: SOMAConfig, tmp_path: Path) -> Path:
    """Save a trained SOMA checkpoint for visualize tests."""
    import torch

    soma = SOMA(small_config)
    for _ in range(5):
        soma.step(
            inputs={"text": torch.randn(small_config.sensor_output_dim)},
            targets={"text": torch.randn(small_config.sensor_output_dim)},
        )
    path = tmp_path / "soma.pt"
    soma.save_state(path)
    return path


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
class TestLoadSoma:
    def test_loads_existing_checkpoint(self, soma_checkpoint: Path) -> None:
        soma = visualize.load_soma(soma_checkpoint)
        assert isinstance(soma, SOMA)
        assert soma.graph.num_nodes > 0

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            visualize.load_soma(tmp_path / "no.pt")


# ----------------------------------------------------------------------
# Summarization
# ----------------------------------------------------------------------
class TestSummary:
    def test_node_type_histogram_counts_types(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        hist = visualize.node_type_histogram(soma.graph)
        assert hist["sensor"] == len(small_config.input_modalities)
        assert hist["output"] == len(small_config.output_modalities)
        assert hist["associator"] == small_config.initial_associator_count

    def test_edge_weight_statistics_handles_edges(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        stats = visualize.edge_weight_statistics(soma.graph)
        assert stats["count"] > 0
        assert "mean" in stats and "abs_mean" in stats

    def test_edge_weight_statistics_handles_empty_graph(self, small_config: SOMAConfig) -> None:
        from soma.core.graph import Graph

        stats = visualize.edge_weight_statistics(Graph())
        assert stats == {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0}

    def test_most_active_nodes_clamps_k(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        # Asking for more than exist must still return a non-empty list
        # capped at num_nodes.
        top = visualize.most_active_nodes(soma.graph, 10_000)
        assert len(top) == soma.graph.num_nodes

    def test_format_summary_includes_expected_sections(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        text = visualize.format_summary(soma, top_k=3)
        assert "SOMA Checkpoint Summary" in text
        assert "Node counts by type" in text
        assert "Edge weight statistics" in text
        assert "most-active nodes" in text
        assert "Memory" in text

    def test_modality_resolver_for_boundary_nodes(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        sensor = soma.graph.get_sensor("text")
        assert visualize._modality_for_node(soma.graph, sensor) == "text"
        associator = next(n for n in soma.graph.all_nodes() if n.node_type is NodeType.ASSOCIATOR)
        assert visualize._modality_for_node(soma.graph, associator) is None


# ----------------------------------------------------------------------
# DOT export
# ----------------------------------------------------------------------
class TestDotExport:
    def test_includes_all_nodes_when_under_cap(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        dot = visualize.to_dot(soma, max_nodes=100)
        for node in soma.graph.all_nodes():
            assert node.id in dot
        assert dot.startswith("digraph SOMA")
        assert dot.rstrip().endswith("}")

    def test_truncates_when_over_cap_but_keeps_boundaries(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        dot = visualize.to_dot(soma, max_nodes=1)
        for sensor in soma.graph.sensor_nodes.values():
            assert sensor.id in dot
        for out in soma.graph.output_nodes.values():
            assert out.id in dot


# ----------------------------------------------------------------------
# networkx + JSON + PNG exports
# ----------------------------------------------------------------------
class TestNodeLinkJSON:
    def test_roundtrip_shape(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        data = visualize.to_node_link_json(soma, max_nodes=100)
        assert data["directed"] is True
        assert data["multigraph"] is False
        assert data["graph"]["num_nodes_total"] == soma.graph.num_nodes
        assert data["graph"]["num_edges_total"] == soma.graph.num_edges
        # All node entries carry our expected metadata keys.
        for node in data["nodes"]:
            assert {"id", "type", "activation_ema", "maturity", "modality"}.issubset(node)
        for link in data["links"]:
            assert {"source", "target", "weight", "strength"}.issubset(link)

    def test_truncation_flag_set_when_over_cap(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        data = visualize.to_node_link_json(soma, max_nodes=1)
        assert data["graph"]["truncated"] is True

    def test_truncation_keeps_boundary_nodes(self, small_config: SOMAConfig) -> None:
        soma = SOMA(small_config)
        data = visualize.to_node_link_json(soma, max_nodes=1)
        kept_ids = {n["id"] for n in data["nodes"]}
        for sensor in soma.graph.sensor_nodes.values():
            assert sensor.id in kept_ids
        for out in soma.graph.output_nodes.values():
            assert out.id in kept_ids


class TestNetworkx:
    def test_returns_digraph(self, small_config: SOMAConfig) -> None:
        pytest.importorskip("networkx")
        soma = SOMA(small_config)
        nx_graph = visualize.to_networkx(soma, max_nodes=100)
        assert nx_graph.is_directed()
        assert nx_graph.number_of_nodes() == soma.graph.num_nodes
        assert nx_graph.number_of_edges() == soma.graph.num_edges

    def test_node_attributes_preserved(self, small_config: SOMAConfig) -> None:
        pytest.importorskip("networkx")
        soma = SOMA(small_config)
        nx_graph = visualize.to_networkx(soma, max_nodes=100)
        # Pick any node and verify it has our metadata.
        node_id, attrs = next(iter(nx_graph.nodes(data=True)))
        assert "type" in attrs
        assert "activation_ema" in attrs


class TestRenderPNG:
    def test_writes_png_file(self, small_config: SOMAConfig, tmp_path: Path) -> None:
        pytest.importorskip("networkx")
        pytest.importorskip("matplotlib")
        soma = SOMA(small_config)
        out = tmp_path / "graph.png"
        visualize.render_png(soma, out, max_nodes=100)
        assert out.exists()
        # Spot-check the PNG magic header.
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ----------------------------------------------------------------------
# CLI entry
# ----------------------------------------------------------------------
class TestRunCLI:
    def test_writes_text_summary(self, soma_checkpoint: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "summary.txt"
        parser = visualize.build_arg_parser()
        args = parser.parse_args(["--checkpoint", str(soma_checkpoint), "--output", str(out_path)])
        result = visualize.run(args)
        assert out_path.exists()
        text = out_path.read_text(encoding="utf-8")
        assert "SOMA Checkpoint Summary" in text
        assert result["num_nodes"] > 0

    def test_writes_dot_file(self, soma_checkpoint: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "summary.txt"
        dot_path = tmp_path / "graph.dot"
        parser = visualize.build_arg_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                str(soma_checkpoint),
                "--output",
                str(out_path),
                "--dot",
                str(dot_path),
                "--max-dot-nodes",
                "50",
            ]
        )
        result = visualize.run(args)
        assert dot_path.exists()
        assert result["dot_path"] == str(dot_path)
        assert dot_path.read_text(encoding="utf-8").startswith("digraph SOMA")

    def test_main_returns_zero(self, soma_checkpoint: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "summary.txt"
        rc = visualize.main(["--checkpoint", str(soma_checkpoint), "--output", str(out_path)])
        assert rc == 0
