"""Force-directed 2D layout for live graph visualization.

Uses :func:`networkx.spring_layout` seeded with previous positions so
nodes don't jump around when the graph topology mutates. Pure Python —
no DPG / torch dependency, so it's straightforward to unit-test.

The layout engine is stateful by design: it keeps the most recent
positions so new snapshots can incrementally update rather than
re-solving from scratch. When the graph topology changes (new or
removed nodes), unknown nodes get a small random position near the
centroid so the first solve nudges them into place.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LayoutResult:
    """Output of one layout pass.

    Coordinates are in the unit range (roughly -1..1) as spring_layout
    produces; the renderer scales them into pixels.
    """

    positions: dict[str, tuple[float, float]]
    global_step: int
    num_nodes: int
    num_edges: int


class ForceDirectedLayout:
    """Incremental spring-layout over a sequence of graph snapshots.

    Parameters
    ----------
    iterations:
        Number of Fruchterman-Reingold iterations per ``compute`` call.
        Smaller values give faster updates at the cost of more jitter.
    seed:
        Random seed for the initial positions of unseen nodes.
    k:
        Optimal spacing parameter forwarded to ``spring_layout``. If
        ``None``, networkx picks a default based on node count.
    """

    def __init__(
        self,
        *,
        iterations: int = 30,
        seed: int | None = 0,
        k: float | None = None,
    ) -> None:
        if iterations <= 0:
            raise ValueError(f"iterations must be positive, got {iterations}")
        self.iterations = iterations
        self.seed = seed
        self.k = k
        self._positions: dict[str, tuple[float, float]] = {}
        self._rng = random.Random(seed)
        self._last_result: LayoutResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(self, snapshot: dict[str, Any]) -> LayoutResult:
        """Compute positions for a ``graph_snapshot`` dict.

        Uses networkx internally. Returns a :class:`LayoutResult` and
        updates the internal position cache so the next call is seeded.
        """
        # Local import so tests without networkx can still import this module.
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - networkx is in [ui] extra
            raise ImportError("networkx is required for the graph layout") from exc

        nodes = snapshot.get("nodes", [])
        edges = snapshot.get("edges", [])

        if not nodes:
            self._positions.clear()
            result = LayoutResult(
                positions={},
                global_step=int(snapshot.get("global_step", 0)),
                num_nodes=0,
                num_edges=0,
            )
            self._last_result = result
            return result

        graph = nx.DiGraph()
        node_ids = {n["id"] for n in nodes}
        for node in nodes:
            graph.add_node(node["id"])
        for edge in edges:
            if edge["source"] in node_ids and edge["target"] in node_ids:
                graph.add_edge(edge["source"], edge["target"])

        # Seed positions: use prior ones where available; fresh random for new nodes.
        seed_positions = self._seed_positions_for(node_ids)

        kwargs: dict[str, Any] = {
            "pos": seed_positions,
            "iterations": self.iterations,
            "seed": self.seed,
        }
        if self.k is not None:
            kwargs["k"] = self.k
        pos_map = nx.spring_layout(graph, **kwargs)

        # Update cache: only keep current nodes.
        self._positions = {node_id: (float(x), float(y)) for node_id, (x, y) in pos_map.items()}
        result = LayoutResult(
            positions=self._positions.copy(),
            global_step=int(snapshot.get("global_step", 0)),
            num_nodes=len(nodes),
            num_edges=len(edges),
        )
        self._last_result = result
        return result

    def last(self) -> LayoutResult | None:
        return self._last_result

    def reset(self) -> None:
        self._positions.clear()
        self._last_result = None

    # ------------------------------------------------------------------
    # Seeding helpers
    # ------------------------------------------------------------------
    def _seed_positions_for(self, node_ids: set[str]) -> dict[str, tuple[float, float]]:
        seed: dict[str, tuple[float, float]] = {}
        # Start with remembered positions for nodes that still exist.
        for nid in node_ids:
            if nid in self._positions:
                seed[nid] = self._positions[nid]
        # For new nodes, drop them on a small ring near the centroid of the
        # known positions so spring_layout has something better than random
        # to relax from.
        if seed:
            cx = sum(p[0] for p in seed.values()) / len(seed)
            cy = sum(p[1] for p in seed.values()) / len(seed)
        else:
            cx = cy = 0.0
        for nid in node_ids - set(seed):
            angle = self._rng.uniform(0.0, 2 * math.pi)
            r = 0.05
            seed[nid] = (cx + r * math.cos(angle), cy + r * math.sin(angle))
        return seed
