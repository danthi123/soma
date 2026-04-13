"""Dump the current graph's edge weights, node gains, and per-associator
forward pass to identify where signal dies between SENSOR and OUTPUT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soma.core.config import SOMAConfig  # noqa: E402
from soma.core.execution import execute_graph  # noqa: E402
from soma.core.node import NodeType  # noqa: E402
from soma.system import SOMA  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/current.pt"))
    args = parser.parse_args(argv)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = SOMAConfig.from_dict(state["config"])
    soma = SOMA(cfg, device="cpu")
    soma.load_state(args.checkpoint)
    g = soma.graph

    print(f"[graph] step={soma.global_step} nodes={g.num_nodes} edges={g.num_edges}")

    nodes_by_type: dict[NodeType, list] = {t: [] for t in NodeType}
    for n in g.all_nodes():
        nodes_by_type[n.node_type].append(n)

    for t in (NodeType.SENSOR, NodeType.ASSOCIATOR, NodeType.INTEGRATOR, NodeType.OUTPUT):
        lst = nodes_by_type[t]
        if not lst:
            continue
        gains = [n.gain for n in lst]
        acts = [n.activation_ema for n in lst]
        mats = [n.maturity for n in lst]
        print(
            f"  {t.value:12s} count={len(lst):3d}  "
            f"gain min={min(gains):.3f} max={max(gains):.3f} mean={sum(gains)/len(gains):.3f}  "
            f"act_ema min={min(acts):.3f} max={max(acts):.3f}  "
            f"maturity min={min(mats):.3f} max={max(mats):.3f}"
        )

    weights = [float(e.weight.abs().max().item()) for e in g.all_edges()]
    if weights:
        sorted_w = sorted(weights)
        print(f"\n[edges] weight |w| distribution (N={len(weights)}):")
        print(f"  min        : {sorted_w[0]:.6e}")
        print(f"  p10        : {sorted_w[max(0, len(sorted_w)//10)]:.6e}")
        print(f"  median     : {sorted_w[len(sorted_w)//2]:.6e}")
        print(f"  p90        : {sorted_w[min(len(sorted_w)-1, 9*len(sorted_w)//10)]:.6e}")
        print(f"  max        : {sorted_w[-1]:.6e}")
        below_1e3 = sum(1 for w in weights if w < 1e-3)
        below_1e6 = sum(1 for w in weights if w < 1e-6)
        print(f"  |w| < 1e-3 : {below_1e3}/{len(weights)} ({100*below_1e3/len(weights):.1f}%)")
        print(f"  |w| < 1e-6 : {below_1e6}/{len(weights)} ({100*below_1e6/len(weights):.1f}%)")

    strengths = [float(e.strength) for e in g.all_edges()]
    if strengths:
        print(f"\n[edges] strength EMA: min={min(strengths):.4f} max={max(strengths):.4f} "
              f"mean={sum(strengths)/len(strengths):.4f}")

    # Now pipe a known nonzero signal through the graph and capture the
    # activation of every node, so we can see where magnitude dies.
    sensor = g.get_sensor("text")
    print(f"\n[trace] feeding random unit-norm input into sensor {sensor.id[:8]}...")
    inp = torch.randn(sensor.output_dim)
    inp = inp / inp.norm()
    _, activations = execute_graph(g, inputs={"text": inp}, current_step=soma.global_step)

    for t in (NodeType.SENSOR, NodeType.ASSOCIATOR, NodeType.OUTPUT):
        lst = nodes_by_type[t]
        norms = []
        for n in lst:
            a = activations.get(n.id)
            if a is None:
                continue
            norms.append(float(a.detach().norm().item()))
        if norms:
            print(
                f"  {t.value:12s} activation norms: min={min(norms):.4f} "
                f"max={max(norms):.4f} mean={sum(norms)/len(norms):.4f}"
            )

    # Output node bias magnitude — if input-to-output path is dead, the
    # bias dominates. Compare that to the observed constant OUTPUT norm.
    out_node = g.get_output("text")
    l2_bias = float(out_node.linear2.bias.detach().norm().item())
    zero_in = torch.zeros(out_node.input_dim)
    zero_fwd = out_node({sensor.id: zero_in}, current_step=soma.global_step).detach()
    print(
        f"\n[output node] gain={out_node.gain:.4f} linear2.bias |.|={l2_bias:.4f} "
        f"forward(zero)={float(zero_fwd.norm()):.4f}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
