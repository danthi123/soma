"""Diagnose decoder collapse: is the model emitting 'GUE' because the
graph outputs a fixed vector, or because it outputs varying vectors
that all decode to the same token?

Runs several distinct prompts through the full pipeline and reports,
at each stage, whether inputs remain distinguishable:

  text -> encoder.embedding -> SENSOR node activation ->
    graph execution -> OUTPUT node activation ->
      decoder.logits -> argmax token

Where diversity collapses tells us where to aim the fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soma.core.config import SOMAConfig  # noqa: E402
from soma.core.execution import execute_graph  # noqa: E402
from soma.io.text_decoder import TextDecoder  # noqa: E402
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer  # noqa: E402
from soma.system import SOMA  # noqa: E402, I001

PROMPTS = [
    "To be, or not to be,",
    "hello",
    "O Romeo, Romeo",
    "The quick brown",
    "Shall I compare thee",
]


def _tensor_signature(vec: torch.Tensor) -> str:
    """Short printable signature — mean/std/first-3 values — for comparison."""
    v = vec.detach().float().cpu().flatten()
    mean = float(v.mean())
    std = float(v.std()) if v.numel() > 1 else 0.0
    head = ", ".join(f"{x:+.4f}" for x in v[:3].tolist())
    return f"mean={mean:+.4f} std={std:+.4f} head=[{head}] norm={float(v.norm()):.4f}"


def _diff_stack(stack: list[torch.Tensor]) -> tuple[float, float]:
    """Pairwise L2 distance stats between rows of a stacked tensor.

    Returns (min_dist, max_dist). If all rows identical, both are 0.
    """
    if len(stack) < 2:
        return 0.0, 0.0
    s = torch.stack([t.detach().float().cpu().flatten() for t in stack])
    dists: list[float] = []
    for i in range(s.shape[0]):
        for j in range(i + 1, s.shape[0]):
            dists.append(float((s[i] - s[j]).norm().item()))
    return min(dists), max(dists)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/current.pt"))
    parser.add_argument("--encoder", type=Path, default=Path("checkpoints/current.encoder.pt"))
    parser.add_argument("--corpus", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint {args.checkpoint} missing", file=sys.stderr)
        return 1

    device = torch.device(args.device)

    from soma.core.brain_bundle import peek_payload

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = peek_payload(raw)
    cfg = SOMAConfig.from_dict(state["config"])
    soma = SOMA(cfg, device=device)
    soma.load_state(args.checkpoint)
    print(
        f"[checkpoint] step={soma.global_step} nodes={soma.graph.num_nodes} "
        f"edges={soma.graph.num_edges}"
    )

    corpus_lines = [
        ln.strip() for ln in args.corpus.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    tokenizer = train_bpe_tokenizer(corpus_lines, vocab_size=cfg.vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
        device=device,
    )
    decoder = TextDecoder(tokenizer, embed_dim=cfg.text_embed_dim, device=device)
    decoder.tie_weights(encoder)

    if args.encoder.exists():
        encoder.load_state_dict(torch.load(args.encoder, map_location=device, weights_only=True))
        encoder.to(device)
        print(f"[encoder] loaded sidecar {args.encoder.name}")
    else:
        print(f"[encoder] WARNING: sidecar {args.encoder} missing, using fresh random embeddings")

    sensor = soma.graph.get_sensor("text")
    output_node = soma.graph.get_output("text")
    print(f"[graph] sensor_id={sensor.id[:8]}... output_id={output_node.id[:8]}...")

    # Collect per-stage tensors across prompts.
    first_input_embeds: list[torch.Tensor] = []
    sensor_activations: list[torch.Tensor] = []
    output_activations: list[torch.Tensor] = []
    decoder_logits: list[torch.Tensor] = []
    decoded_tokens: list[int] = []

    print()
    print("=" * 70)
    print("PER-PROMPT PIPELINE TRACE (first token only, non-autoregressive)")
    print("=" * 70)
    for prompt in PROMPTS:
        print(f"\n[prompt] {prompt!r}")
        embeds = encoder.encode_batch(prompt)
        if embeds.ndim != 2 or embeds.shape[0] == 0:
            print("  (skipped: encoder returned no tokens)")
            continue
        first_embed = embeds[0].detach()
        first_input_embeds.append(first_embed)
        print(f"  encoder[0] :  {_tensor_signature(first_embed)}")

        # Run one step of the graph with this single-token input (no target so
        # we don't accidentally update anything).
        outputs, activations = execute_graph(
            soma.graph, inputs={"text": first_embed}, current_step=soma.global_step
        )
        sensor_act = activations.get(sensor.id)
        if sensor_act is not None:
            sensor_activations.append(sensor_act)
            print(f"  sensor_act :  {_tensor_signature(sensor_act)}")
        else:
            print("  sensor_act :  (missing)")

        output_act = activations.get(output_node.id)
        if output_act is not None:
            output_activations.append(output_act)
            print(f"  output_act :  {_tensor_signature(output_act)}")
        else:
            print("  output_act :  (missing)")

        out_vec = outputs.get("text")
        if out_vec is not None:
            logits = decoder.logits(out_vec.detach())
            decoder_logits.append(logits)
            top5_vals, top5_idx = torch.topk(logits, k=5)
            top5_str = ", ".join(
                f"{int(i)}={ascii(tokenizer.decode([int(i)]))}({float(v):+.2f})"
                for i, v in zip(top5_idx.tolist(), top5_vals.tolist(), strict=False)
            )
            tok_id = int(logits.argmax().item())
            decoded_tokens.append(tok_id)
            print(f"  top5 logits:  {top5_str}")
            print(f"  argmax     :  id={tok_id} text={ascii(tokenizer.decode([tok_id]))}")

    # Summary of diversity at each stage.
    print()
    print("=" * 70)
    print("DIVERSITY SUMMARY (pairwise L2 distance across prompts)")
    print("=" * 70)
    stages = [
        ("encoder input embedding", first_input_embeds),
        ("SENSOR node activation", sensor_activations),
        ("OUTPUT node activation", output_activations),
        ("decoder logits", decoder_logits),
    ]
    for name, stack in stages:
        if not stack:
            print(f"  {name:30s}: (no data)")
            continue
        dmin, dmax = _diff_stack(stack)
        verdict = "COLLAPSED" if dmax < 1e-4 else "varies"
        print(f"  {name:30s}: min={dmin:.6f}  max={dmax:.6f}  -> {verdict}")

    uniq_tokens = sorted(set(decoded_tokens))
    print(f"\n  decoded tokens              : {decoded_tokens}")
    print(f"  unique decoded token count  : {len(uniq_tokens)}")
    if len(uniq_tokens) == 1 and decoded_tokens:
        tok = uniq_tokens[0]
        text = ascii(tokenizer.decode([tok]))
        print(f"  COLLAPSE CONFIRMED -> every prompt decodes to id={tok} text={text}")

    # Cross-check: what does the decoder think of a zero vector?
    zero = torch.zeros(cfg.text_embed_dim, device=device)
    zero_logits = decoder.logits(zero)
    zero_top5_val, zero_top5_idx = torch.topk(zero_logits, k=5)
    zero_top5 = ", ".join(
        f"{int(i)}={ascii(tokenizer.decode([int(i)]))}({float(v):+.2f})"
        for i, v in zip(zero_top5_idx.tolist(), zero_top5_val.tolist(), strict=False)
    )
    print(f"\n  decoder(zero_vec) top5      : {zero_top5}")
    print(
        f"  decoder(zero_vec) argmax    : id={int(zero_logits.argmax())} "
        f"text={ascii(tokenizer.decode([int(zero_logits.argmax())]))}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
