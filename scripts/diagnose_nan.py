"""One-off diagnostic: inspect param distributions and reproduce inf/nan loss.

Throwaway. Loads the corrupt checkpoint preserved as
checkpoints/current.corrupt_1776082129.pt.bak and an early-step checkpoint,
prints distributions of edge weights / node gains / per-node param norms,
then attempts a forward pass on the same fixed prompt the train_service uses
to see if loss is non-finite.

Usage: python scripts/diagnose_nan.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from soma.core.config import SOMAConfig
from soma.io.text_encoder import TextEncoder
from soma.system import SOMA


def summarize(name: str, t: torch.Tensor) -> str:
    if t.numel() == 0:
        return f"{name}: empty"
    finite_mask = torch.isfinite(t)
    n_nonfinite = int((~finite_mask).sum().item())
    finite = t[finite_mask]
    if finite.numel() == 0:
        return f"{name}: all {t.numel()} non-finite"
    return (
        f"{name}: n={t.numel():>7}  min={float(finite.min()):+.3e}  "
        f"max={float(finite.max()):+.3e}  abs_max={float(finite.abs().max()):.3e}  "
        f"mean={float(finite.mean()):+.3e}  std={float(finite.std()):.3e}  "
        f"non_finite={n_nonfinite}"
    )


def inspect_state(state_path: Path, label: str, *, device: str = "cpu") -> SOMA:
    print(f"\n{'='*70}\n{label}: {state_path}\n{'='*70}")
    config = SOMAConfig()
    soma = SOMA(config, device=device)
    soma.load_state(state_path)

    n_nodes = soma.graph.num_nodes
    n_edges = soma.graph.num_edges
    print(f"global_step={soma.global_step}  nodes={n_nodes}  edges={n_edges}")

    edge_weights = torch.stack([e.weight.detach().flatten() for e in soma.graph.all_edges()])
    print(summarize("edge.weight (all)", edge_weights.flatten()))

    edge_w_max_per = torch.tensor(
        [float(e.weight.detach().abs().max()) for e in soma.graph.all_edges()]
    )
    n_at_clamp = int((edge_w_max_per >= config.max_edge_weight - 0.01).sum().item())
    print(f"edges at weight-clamp boundary (|w| >= {config.max_edge_weight - 0.01}): "
          f"{n_at_clamp}/{n_edges}  ({100.0*n_at_clamp/max(n_edges,1):.1f}%)")

    gains = torch.tensor([float(n.gain) for n in soma.graph.all_nodes()])
    print(summarize("node.gain", gains))
    n_at_gain_hi = int((gains >= 9.99).sum().item())
    n_at_gain_lo = int((gains <= 0.11).sum().item())
    print(f"nodes at gain-clamp: hi(>=9.99)={n_at_gain_hi}  lo(<=0.11)={n_at_gain_lo}  / {n_nodes}")

    activations = torch.tensor([float(n.activation_ema) for n in soma.graph.all_nodes()])
    print(summarize("node.activation_ema", activations))

    maturities = torch.tensor([float(n.maturity) for n in soma.graph.all_nodes()])
    print(summarize("node.maturity", maturities))

    all_node_param_norms: list[float] = []
    nan_param_nodes: list[str] = []
    for nid, node in soma.graph.nodes.items():
        for param in node.parameters():
            d = param.detach()
            if not torch.isfinite(d).all():
                nan_param_nodes.append(nid)
            all_node_param_norms.append(float(d.norm().item()))
    norms = torch.tensor(all_node_param_norms)
    print(summarize("node.parameters().norm()", norms))
    if nan_param_nodes:
        sample = nan_param_nodes[:5]
        print(f"  ! nodes with non-finite params: {len(nan_param_nodes)} (e.g. {sample})")

    return soma


def attempt_forward(soma: SOMA, encoder_path: Path, device: str = "cpu") -> None:
    print(f"\n--- forward-pass replay on {soma.global_step=} ---")
    encoder = TextEncoder(soma.config)
    if encoder_path.exists():
        encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        print(f"loaded encoder sidecar {encoder_path.name}")
    else:
        print(f"!! no encoder sidecar at {encoder_path}; using random init")
    encoder.to(device)

    text = "the quick brown fox jumps over the lazy dog"
    tokens = encoder.tokenizer.encode(text)
    print(f"input tokens: {len(tokens)}")
    if not tokens:
        print("  empty token sequence; skipping")
        return
    embeds = encoder.embed(torch.tensor(tokens, device=device))
    print(summarize("encoder.embed output", embeds))

    if not torch.isfinite(embeds).all():
        print("  !! encoder produced non-finite embeddings — root cause is encoder, not graph")
        return

    sensor_id = next(
        (n.id for n in soma.graph.all_nodes() if n.kind == "SENSOR"), None
    )
    output_id = next(
        (n.id for n in soma.graph.all_nodes() if n.kind == "OUTPUT"), None
    )
    print(f"sensor={sensor_id}  output={output_id}")

    inputs = {sensor_id: embeds[0]}
    targets = {output_id: embeds[1]} if len(embeds) > 1 else None
    try:
        result = soma.step(inputs=inputs, targets=targets)
    except Exception as exc:
        print(f"step raised: {type(exc).__name__}: {exc}")
        return
    loss = result.get("loss")
    out = result.get("outputs", {}).get(output_id)
    print(f"loss={loss!r}  finite={loss is not None and math.isfinite(loss)}")
    if out is not None:
        print(summarize(f"output[{output_id}]", out.detach()))


def fresh_train_sanity(n_steps: int = 200, device: str = "cpu") -> None:
    """Train a fresh SOMA for N steps and confirm no clamp saturation.

    Verifies the post-fix stability properties: edge weights stay below
    the clamp boundary, activations stay finite, and the skip counter
    doesn't blow up.
    """
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

    print(f"\n{'=' * 70}\nFRESH TRAIN ({n_steps} steps, device={device})\n{'=' * 70}")
    cfg = SOMAConfig(seed=42)
    soma = SOMA(cfg, device=device)
    corpus = ["the quick brown fox", "jumps over the lazy dog"] * 50
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=cfg.vocab_size)
    encoder = TextEncoder(tokenizer, embed_dim=cfg.text_embed_dim, device=device)
    skipped = 0
    for step in range(n_steps):
        text = corpus[step % len(corpus)]
        embeds = encoder.encode(text)
        if len(embeds) < 2:
            continue
        result = soma.step(
            inputs={"text": embeds[0]},
            targets={"text": embeds[1]},
        )
        if result.get("skipped"):
            skipped += 1
    print(f"completed {n_steps} steps; skipped={skipped}")
    n_edges = soma.graph.num_edges
    edge_weights = torch.stack([e.weight.detach().flatten() for e in soma.graph.all_edges()])
    n_at_clamp = int((edge_weights.abs() >= cfg.max_edge_weight - 0.01).sum().item())
    print(
        f"edges at weight-clamp boundary: {n_at_clamp}/{n_edges} "
        f"({100 * n_at_clamp / max(n_edges, 1):.1f}%)"
    )
    activations = torch.tensor([float(n.activation_ema) for n in soma.graph.all_nodes()])
    print(
        f"node activation_ema: max={float(activations.abs().max()):.3e}  "
        f"non_finite={int((~torch.isfinite(activations)).sum())}"
    )
    print(f"final loss_ema={soma.homeostasis.loss_ema:.4f}  "
          f"consecutive_skipped={soma._consecutive_skipped_steps}")
    assert n_at_clamp / max(n_edges, 1) < 0.5, (
        f"more than half of edges saturated within {n_steps} steps — fixes failed"
    )
    assert torch.isfinite(activations).all(), (
        f"non-finite activations within {n_steps} steps — fixes failed"
    )
    print("[OK] stability sanity passed")


def main() -> None:
    ckpt_dir = Path("checkpoints")
    corrupt = ckpt_dir / "current.corrupt_1776082129.pt.bak"
    if corrupt.exists():
        inspect_state(corrupt, "CORRUPT (was step ~53740)")
    fresh_train_sanity(n_steps=200)


if __name__ == "__main__":
    main()
