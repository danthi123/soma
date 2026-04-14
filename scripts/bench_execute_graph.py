"""Benchmark sequential vs batched execute_graph.

Usage:
    python scripts/bench_execute_graph.py [--device cpu|cuda] [--steps 1000]

Builds a ~34-node / ~100-edge graph at embed_dim=64 (matches current
training config) and times both executors over (warmup + steps) forward
passes. Prints steps/sec for each path and the speedup ratio.

Acceptance criterion is 3x on CUDA. CPU typically runs at 1.2-2x because
Python overhead dominates — that's fine; this script is operator-run on
the RTX 3090 for the real number.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph, execute_graph_batched
from soma.core.graph import Graph
from soma.core.node import Node, NodeType


def build_benchmark_graph(config: SOMAConfig, device: torch.device) -> Graph:
    """34 nodes / ~100 edges at embed_dim=64 — matches current training config."""
    g = Graph()
    dim = config.sensor_output_dim  # 64
    # 2 sensors (text+image), 2 outputs, 30 associators. 34 total.
    s_text = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config, device=device)
    s_img = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config, device=device)
    g.add_node(s_text, modality="text")
    g.add_node(s_img, modality="image")
    assocs = [
        Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config, device=device) for _ in range(30)
    ]
    for a in assocs:
        g.add_node(a)
    o_text = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config, device=device)
    o_img = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config, device=device)
    g.add_node(o_text, modality="text")
    g.add_node(o_img, modality="image")
    # Edges: each sensor fans out to 15 assocs; each assoc fans out to 2 outputs.
    # 2*15 + 30*2 = 90 edges. Add ~10 intra-associator edges for variety.
    rng = torch.Generator().manual_seed(0)

    def _e(src: Node, tgt: Node) -> None:
        g.add_edge(
            Edge(
                source_id=src.id,
                target_id=tgt.id,
                source_output_dim=src.output_dim,
                target_input_dim=tgt.input_dim,
                creation_step=0,
                initial_weight=0.1,
                device=device,
            )
        )

    for a in assocs[:15]:
        _e(s_text, a)
    for a in assocs[15:]:
        _e(s_img, a)
    for a in assocs:
        _e(a, o_text)
        _e(a, o_img)
    # ~10 intra-associator edges.
    idx = torch.randperm(len(assocs), generator=rng).tolist()
    for i in range(0, 20, 2):
        src, tgt = assocs[idx[i]], assocs[idx[i + 1]]
        if not g.has_edge(src.id, tgt.id) and not g.has_edge(tgt.id, src.id):
            _e(src, tgt)
    return g


def run_bench(
    graph: Graph,
    data: torch.Tensor,
    fn,  # type: ignore[no-untyped-def]
    warmup: int,
    steps: int,
    device: torch.device,
) -> float:
    """Return steps/sec."""
    prev: dict[str, torch.Tensor] = {}
    for step in range(warmup):
        _, prev = fn(
            graph,
            inputs={"text": data, "image": data},
            current_step=step,
            previous_activations=prev,
        )
        # Detach previous_activations so the graph keeps growing tensors
        # forever under autograd. Benchmarks measure raw forward.
        prev = {k: v.detach() for k, v in prev.items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(warmup, warmup + steps):
        _, prev = fn(
            graph,
            inputs={"text": data, "image": data},
            current_step=step,
            previous_activations=prev,
        )
        prev = {k: v.detach() for k, v in prev.items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return steps / elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = torch.device("cpu")
    config = SOMAConfig()
    g_seq = build_benchmark_graph(config, device)
    # Deepcopy so weights are byte-identical; two run_bench calls operate
    # on distinct Graph instances so in-place edge.last_active_step
    # bookkeeping doesn't cross-contaminate.
    import copy  # local import keeps the top of this script tight

    g_bat = copy.deepcopy(g_seq)

    data = torch.randn(config.sensor_output_dim, device=device)
    print(f"[bench] device={device.type}  warmup={args.warmup}  steps={args.steps}")
    print(f"[bench] graph: {g_seq.num_nodes} nodes, {g_seq.num_edges} edges")
    sps_seq = run_bench(g_seq, data, execute_graph, args.warmup, args.steps, device)
    sps_bat = run_bench(g_bat, data, execute_graph_batched, args.warmup, args.steps, device)
    ratio = sps_bat / sps_seq if sps_seq > 0 else float("inf")
    print(f"[bench] sequential: {sps_seq:.1f} steps/sec")
    print(f"[bench] batched:    {sps_bat:.1f} steps/sec")
    print(f"[bench] speedup:    {ratio:.2f}x")
    if device.type == "cuda" and ratio < 3.0:
        print(f"[bench] WARN: speedup below target 3x (got {ratio:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
