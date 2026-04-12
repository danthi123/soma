"""Graph: the dynamic container of Nodes and Edges.

The Graph holds all parametric memory of the system. It is itself an
``nn.Module`` so PyTorch can discover learnable parameters for backprop.
Internally it uses ``nn.ModuleDict`` for nodes and edges.

Responsibilities:
- Own the sets of nodes and edges.
- Maintain reverse indexes (incoming/outgoing edges per node, source-target
  lookup for ``has_edge``) so edge insertion and lookup are O(1).
- Track which nodes are the I/O boundary (sensor/output) per modality.
- Enforce CLAUDE.md invariants during mutation (SENSOR/OUTPUT never
  removed via ``remove_node``; see ``allow_boundary_removal`` escape hatch
  for explicit teardown during consolidation).
- Provide read/serialize helpers used by the executor, learning, growth,
  and pruning modules.

The Graph does not execute forward passes — that's the job of
``soma.core.execution.execute_graph`` (Unit 6).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, cast

import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.node import Node, NodeType


class Graph(nn.Module):
    """Directed graph of Nodes and Edges.

    Mutating methods (``add_node``, ``remove_node``, ``add_edge``,
    ``remove_edge``) keep the reverse indexes in sync. Read methods return
    mapping/iterator views that are safe to iterate during iteration of
    the graph itself (callers wanting to mutate mid-iteration should copy
    the iterator first).
    """

    def __init__(self) -> None:
        super().__init__()

        # Primary storage — nn.ModuleDict so params are discovered by PyTorch.
        self._nodes: nn.ModuleDict = nn.ModuleDict()
        self._edges: nn.ModuleDict = nn.ModuleDict()

        # Reverse indexes.
        self._incoming: dict[str, set[str]] = {}  # target_id -> {edge_id}
        self._outgoing: dict[str, set[str]] = {}  # source_id -> {edge_id}
        self._edges_by_pair: dict[tuple[str, str], str] = {}  # (src, tgt) -> edge_id

        # I/O boundary registry.
        self._sensor_by_modality: dict[str, str] = {}  # modality -> node_id
        self._output_by_modality: dict[str, str] = {}  # modality -> node_id

        # Device tracking. ``Graph.to(device)`` updates this so later
        # ``add_node`` / ``add_edge`` calls can move their arguments onto the
        # same device. ``None`` means "don't force anything; use the module's
        # own device," which is the CPU default.
        self._device: torch.device | None = None

    def _apply(self, fn, recurse: bool = True):  # type: ignore[no-untyped-def]
        """Override ``nn.Module._apply`` so ``.to(device)`` / ``.cpu()`` /
        ``.cuda()`` also update our tracked device. ``to(device)`` eventually
        calls ``_apply`` with a closure that carries the target device.
        """
        result = super()._apply(fn, recurse=recurse)  # type: ignore[no-untyped-call]
        # Peek at any surviving parameter/buffer to discover our effective device.
        for param in self.parameters():
            self._device = param.device
            break
        else:
            for buf in self.buffers():
                self._device = buf.device
                break
        return result

    @property
    def device(self) -> torch.device | None:
        """The device the graph currently lives on, or ``None`` if empty."""
        return self._device

    # ==================================================================
    # Node operations
    # ==================================================================
    def add_node(self, node: Node, *, modality: str | None = None) -> None:
        """Add a node to the graph.

        For SENSOR/OUTPUT nodes, ``modality`` identifies which I/O channel
        this node owns (e.g., ``"text"``, ``"image"``). Exactly one sensor
        and one output node may exist per modality — adding a second for
        the same modality raises ``ValueError``.
        """
        if node.id in self._nodes:
            raise ValueError(f"Node {node.id[:8]} is already in the graph")
        if self._device is not None:
            node.to(self._device)
        self._nodes[node.id] = node
        self._incoming[node.id] = set()
        self._outgoing[node.id] = set()

        if node.node_type is NodeType.SENSOR:
            if modality is None:
                raise ValueError("SENSOR nodes require an explicit modality")
            if modality in self._sensor_by_modality:
                raise ValueError(
                    f"SENSOR for modality {modality!r} already exists "
                    f"(id={self._sensor_by_modality[modality][:8]})"
                )
            self._sensor_by_modality[modality] = node.id
        elif node.node_type is NodeType.OUTPUT:
            if modality is None:
                raise ValueError("OUTPUT nodes require an explicit modality")
            if modality in self._output_by_modality:
                raise ValueError(
                    f"OUTPUT for modality {modality!r} already exists "
                    f"(id={self._output_by_modality[modality][:8]})"
                )
            self._output_by_modality[modality] = node.id
        elif modality is not None:
            # Optional: we don't reject, just ignore non-boundary modalities.
            raise ValueError(
                f"modality is only valid for SENSOR/OUTPUT nodes, got {node.node_type.value}"
            )

    def remove_node(self, node_id: str, *, allow_boundary_removal: bool = False) -> Node:
        """Remove a node and all edges incident to it.

        By default refuses to remove SENSOR/OUTPUT nodes (CLAUDE.md
        invariant). Pass ``allow_boundary_removal=True`` only when fully
        tearing down a graph (e.g., during ``deserialize``).

        Returns the removed node so the caller can inspect / dispose of it.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id[:8]} not in graph")
        node = self._nodes[node_id]
        assert isinstance(node, Node)

        if node.node_type.is_boundary and not allow_boundary_removal:
            raise ValueError(
                f"Cannot remove boundary node {node_id[:8]} "
                f"({node.node_type.value}) — invariant violation"
            )

        # Cascade-remove all incident edges (copy IDs first — we mutate).
        for edge_id in list(self._incoming[node_id] | self._outgoing[node_id]):
            self.remove_edge(edge_id)

        # Remove from boundary registry if applicable.
        if node.node_type is NodeType.SENSOR:
            self._sensor_by_modality = {
                m: nid for m, nid in self._sensor_by_modality.items() if nid != node_id
            }
        elif node.node_type is NodeType.OUTPUT:
            self._output_by_modality = {
                m: nid for m, nid in self._output_by_modality.items() if nid != node_id
            }

        self._incoming.pop(node_id)
        self._outgoing.pop(node_id)
        del self._nodes[node_id]
        return node

    # ==================================================================
    # Edge operations
    # ==================================================================
    def add_edge(self, edge: Edge) -> None:
        """Add an edge; both endpoints must already exist in the graph."""
        if edge.id in self._edges:
            raise ValueError(f"Edge {edge.id[:8]} is already in the graph")
        if edge.source_id not in self._nodes:
            raise ValueError(f"Edge source {edge.source_id[:8]} not in graph")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Edge target {edge.target_id[:8]} not in graph")
        pair = (edge.source_id, edge.target_id)
        if pair in self._edges_by_pair:
            raise ValueError(
                f"Edge {edge.source_id[:8]}->{edge.target_id[:8]} already exists "
                f"(id={self._edges_by_pair[pair][:8]})"
            )

        # Dimension sanity: edge source/target dims must match endpoints.
        src_node = self._nodes[edge.source_id]
        tgt_node = self._nodes[edge.target_id]
        assert isinstance(src_node, Node)
        assert isinstance(tgt_node, Node)
        if edge.source_output_dim != src_node.output_dim:
            raise ValueError(
                f"Edge source dim {edge.source_output_dim} != "
                f"source node output dim {src_node.output_dim}"
            )
        if edge.target_input_dim != tgt_node.input_dim:
            raise ValueError(
                f"Edge target dim {edge.target_input_dim} != "
                f"target node input dim {tgt_node.input_dim}"
            )

        if self._device is not None:
            edge.to(self._device)
        self._edges[edge.id] = edge
        self._outgoing[edge.source_id].add(edge.id)
        self._incoming[edge.target_id].add(edge.id)
        self._edges_by_pair[pair] = edge.id

    def remove_edge(self, edge_id: str) -> Edge:
        """Remove an edge and return it."""
        if edge_id not in self._edges:
            raise KeyError(f"Edge {edge_id[:8]} not in graph")
        edge = self._edges[edge_id]
        assert isinstance(edge, Edge)
        self._outgoing[edge.source_id].discard(edge_id)
        self._incoming[edge.target_id].discard(edge_id)
        self._edges_by_pair.pop((edge.source_id, edge.target_id), None)
        del self._edges[edge_id]
        return edge

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Return True if an edge ``source -> target`` already exists."""
        return (source_id, target_id) in self._edges_by_pair

    def get_edge(self, source_id: str, target_id: str) -> Edge:
        """Return the edge ``source -> target``. Raises if missing."""
        edge_id = self._edges_by_pair.get((source_id, target_id))
        if edge_id is None:
            raise KeyError(f"No edge {source_id[:8]}->{target_id[:8]} in graph")
        edge = self._edges[edge_id]
        assert isinstance(edge, Edge)
        return edge

    def get_incoming_edges(self, node_id: str) -> list[Edge]:
        """Edges whose target is ``node_id``."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id[:8]} not in graph")
        return [self._get_edge(eid) for eid in self._incoming[node_id]]

    def get_outgoing_edges(self, node_id: str) -> list[Edge]:
        """Edges whose source is ``node_id``."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id[:8]} not in graph")
        return [self._get_edge(eid) for eid in self._outgoing[node_id]]

    # ==================================================================
    # I/O boundary lookup
    # ==================================================================
    def get_sensor(self, modality: str) -> Node:
        """Return the sensor Node for ``modality``. Raises if missing."""
        node_id = self._sensor_by_modality.get(modality)
        if node_id is None:
            raise KeyError(f"No SENSOR registered for modality {modality!r}")
        return self._get_node(node_id)

    def get_output(self, modality: str) -> Node:
        """Return the output Node for ``modality``. Raises if missing."""
        node_id = self._output_by_modality.get(modality)
        if node_id is None:
            raise KeyError(f"No OUTPUT registered for modality {modality!r}")
        return self._get_node(node_id)

    @property
    def sensor_nodes(self) -> Mapping[str, Node]:
        """Modality -> sensor Node (read-only view)."""
        return {m: self._get_node(nid) for m, nid in self._sensor_by_modality.items()}

    @property
    def output_nodes(self) -> Mapping[str, Node]:
        """Modality -> output Node (read-only view)."""
        return {m: self._get_node(nid) for m, nid in self._output_by_modality.items()}

    # ==================================================================
    # Read views
    # ==================================================================
    @property
    def nodes(self) -> Mapping[str, Node]:
        """Read-only mapping of node_id -> Node. Mutate via add/remove only."""
        return _NodesView(self._nodes)

    @property
    def edges(self) -> Mapping[str, Edge]:
        """Read-only mapping of edge_id -> Edge."""
        return _EdgesView(self._edges)

    def all_nodes(self) -> list[Node]:
        """Return a snapshot list of all nodes (safe to iterate while mutating)."""
        return [self._get_node(nid) for nid in self._nodes]

    def all_edges(self) -> list[Edge]:
        return [self._get_edge(eid) for eid in self._edges]

    def nodes_by_type(self, node_type: NodeType) -> list[Node]:
        return [n for n in self.all_nodes() if n.node_type is node_type]

    def active_nodes(self, current_step: int, lookback: int = 1) -> list[Node]:
        """Nodes whose ``last_active_step`` is within the lookback window.

        A ``lookback`` of 1 means "active in the current step". Larger
        values (e.g., during pruning) let callers consider recently-active
        nodes.
        """
        if lookback < 0:
            raise ValueError(f"lookback must be non-negative, got {lookback}")
        cutoff = current_step - lookback
        return [n for n in self.all_nodes() if n.last_active_step >= cutoff]

    def active_edges(self, current_step: int, lookback: int = 1) -> list[Edge]:
        if lookback < 0:
            raise ValueError(f"lookback must be non-negative, got {lookback}")
        cutoff = current_step - lookback
        return [e for e in self.all_edges() if e.last_active_step >= cutoff]

    def most_active_nodes(self, k: int = 10) -> list[Node]:
        """Top-``k`` nodes by ``activation_ema`` (descending)."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        return sorted(self.all_nodes(), key=lambda n: n.activation_ema, reverse=True)[:k]

    def get_nearest_nodes(
        self,
        position: torch.Tensor,
        k: int = 5,
        *,
        exclude: Iterable[str] = (),
    ) -> list[Node]:
        """Top-``k`` nodes whose position is closest to ``position`` (L2).

        ``exclude`` can name node IDs that should be skipped (e.g., the
        new node itself during neurogenesis).
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        exclude_set = set(exclude)
        pos_cpu = _to_cpu_flat_float(position)
        scored: list[tuple[float, Node]] = []
        for node in self.all_nodes():
            if node.id in exclude_set:
                continue
            node_pos = _to_cpu_flat_float(cast(torch.Tensor, node.position))
            if node_pos.shape != pos_cpu.shape:
                # Position dim mismatch — skip (should not happen if positions
                # all come from the same config).
                continue
            dist = float(torch.linalg.vector_norm(node_pos - pos_cpu).item())
            scored.append((dist, node))
        scored.sort(key=lambda item: item[0])
        return [n for _, n in scored[:k]]

    # ==================================================================
    # Sizes
    # ==================================================================
    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    # ==================================================================
    # Serialization
    # ==================================================================
    def serialize(self) -> dict[str, Any]:
        """Capture full graph state (structure + learnable params).

        Produces a dict that can be passed to :meth:`deserialize`. The
        dict is ``torch.save``-friendly.
        """
        return {
            "nodes": {nid: self._get_node(nid).to_dict() for nid in self._nodes},
            "edges": {eid: self._get_edge(eid).to_dict() for eid in self._edges},
            "sensor_by_modality": dict(self._sensor_by_modality),
            "output_by_modality": dict(self._output_by_modality),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def deserialize(cls, data: Mapping[str, Any], config: SOMAConfig) -> Graph:
        """Reconstruct a Graph from :meth:`serialize` output.

        ``config`` is required so we can re-create Node/Edge modules with
        the right defaults for ring-buffer capacity, gain limits, etc.
        """
        graph = cls()

        # Nodes first.
        for node_state in data["nodes"].values():
            node_type = NodeType(node_state["node_type"])
            node = Node(
                node_type=node_type,
                input_dim=int(node_state["input_dim"]),
                hidden_dim=int(node_state["hidden_dim"]),
                output_dim=int(node_state["output_dim"]),
                creation_step=int(node_state["creation_step"]),
                config=config,
                node_id=node_state["id"],
            )
            node.load_scalar_state(node_state)
            graph._register_node(node)

        # Edges next (endpoints are guaranteed to exist).
        for edge_state in data["edges"].values():
            edge = Edge(
                source_id=edge_state["source_id"],
                target_id=edge_state["target_id"],
                source_output_dim=int(edge_state["source_output_dim"]),
                target_input_dim=int(edge_state["target_input_dim"]),
                creation_step=int(edge_state["creation_step"]),
                edge_id=edge_state["id"],
            )
            edge.load_scalar_state(edge_state)
            graph._register_edge(edge)

        # Modality registries.
        graph._sensor_by_modality.update(data["sensor_by_modality"])
        graph._output_by_modality.update(data["output_by_modality"])

        # Learnable params + buffers.
        graph.load_state_dict(data["state_dict"])
        return graph

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _register_node(self, node: Node) -> None:
        """Insert a node (and its index rows) without touching modality registry."""
        if node.id in self._nodes:
            raise ValueError(f"Node {node.id[:8]} already present")
        self._nodes[node.id] = node
        self._incoming[node.id] = set()
        self._outgoing[node.id] = set()

    def _register_edge(self, edge: Edge) -> None:
        """Insert an edge (used by deserialize — skips dimension sanity checks
        because the serialized edge already matched the stored endpoints)."""
        if edge.id in self._edges:
            raise ValueError(f"Edge {edge.id[:8]} already present")
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise ValueError(
                f"Edge {edge.id[:8]} references missing endpoints "
                f"{edge.source_id[:8]}->{edge.target_id[:8]}"
            )
        pair = (edge.source_id, edge.target_id)
        if pair in self._edges_by_pair:
            raise ValueError(f"Duplicate edge pair during deserialize: {pair}")
        self._edges[edge.id] = edge
        self._outgoing[edge.source_id].add(edge.id)
        self._incoming[edge.target_id].add(edge.id)
        self._edges_by_pair[pair] = edge.id

    def _get_node(self, node_id: str) -> Node:
        node = self._nodes[node_id]
        assert isinstance(node, Node)
        return node

    def _get_edge(self, edge_id: str) -> Edge:
        edge = self._edges[edge_id]
        assert isinstance(edge, Edge)
        return edge

    def __iter__(self) -> Iterator[Node]:
        return iter(self.all_nodes())


# ----------------------------------------------------------------------
# Thin typed views — forbid mutation, preserve dict-like access.
# ----------------------------------------------------------------------
class _NodesView(Mapping[str, Node]):
    def __init__(self, module_dict: nn.ModuleDict) -> None:
        self._backing = module_dict

    def __getitem__(self, key: str) -> Node:
        node = self._backing[key]
        assert isinstance(node, Node)
        return node

    def __iter__(self) -> Iterator[str]:
        return iter(self._backing)

    def __len__(self) -> int:
        return len(self._backing)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._backing


def _to_cpu_flat_float(tensor: torch.Tensor) -> torch.Tensor:
    """Detach, move to CPU, flatten, and cast to float32 — in that order.

    Used for position-distance math where we want a stable, contiguous
    CPU vector independent of the tensor's original device/dtype.
    """
    detached = tensor.detach()
    cpu = detached.cpu()
    flat = cpu.reshape(-1)
    return flat.to(torch.float32)


class _EdgesView(Mapping[str, Edge]):
    def __init__(self, module_dict: nn.ModuleDict) -> None:
        self._backing = module_dict

    def __getitem__(self, key: str) -> Edge:
        edge = self._backing[key]
        assert isinstance(edge, Edge)
        return edge

    def __iter__(self) -> Iterator[str]:
        return iter(self._backing)

    def __len__(self) -> int:
        return len(self._backing)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._backing
