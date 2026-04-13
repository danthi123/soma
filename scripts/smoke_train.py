"""Smoke-train guard for the safety gate.

Runs a tiny ephemeral SOMA for N steps to verify the training loop, gradient
flow, and graph ops still work. Uses a HARDCODED minimal config independent of
``configs/current.yaml``, so shape-mismatch between corpus and vocab size is
not a concern.

Exit 0 on success; non-zero on any exception or non-finite loss.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SMOKE_CONFIG: dict[str, Any] = {
    "sensor_output_dim": 16,
    "associator_input_dim": 16,
    "associator_hidden_dim": 32,
    "associator_output_dim": 16,
    "integrator_input_dim": 32,
    "integrator_hidden_dim": 32,
    "integrator_output_dim": 32,
    "position_dim": 4,
    "wm_slots": 4,
    "wm_dim": 16,
    "episodic_capacity": 100,
    "key_dim": 16,
    "value_dim": 16,
    "vocab_size": 64,
    "text_embed_dim": 16,
    "max_nodes": 1000,
    "max_edges_per_node": 10.0,
    "initial_associator_count": 4,
    "initial_integrator_count": 2,
    "max_input_tokens": 16,
    "max_output_tokens": 8,
    "checkpoint_interval": 10_000,  # effectively disabled during smoke
}

SMOKE_CORPUS: list[str] = [
    "hello world foo bar baz quick brown fox lazy dog",
    "the quick brown fox jumps over the lazy dog again",
    "lorem ipsum dolor sit amet consectetur adipiscing elit",
    "ping pong abc xyz one two three four five six seven",
    "alpha beta gamma delta epsilon zeta eta theta iota",
    "shall we compare thee to a summer day thou art more",
    "to be or not to be that is the question of the day",
    "once upon a time in a land far far away there lived",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOMA smoke-train sanity check.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args(argv)

    start = time.time()

    import torch

    from soma.core.config import SOMAConfig
    from soma.io.dataset_feeders import TextDatasetFeeder
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    from soma.system import SOMA

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = SOMAConfig(**SMOKE_CONFIG)
    tokenizer = train_bpe_tokenizer(SMOKE_CORPUS, vocab_size=config.vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=config.text_embed_dim,
        max_seq_len=config.max_input_tokens,
        device=device,
    )
    feeder = TextDatasetFeeder(encoder, SMOKE_CORPUS, chunk_size=4)
    soma = SOMA(config, device=device)

    first_output = config.output_modalities[0]
    steps_done = 0
    for sample in feeder:
        if time.time() - start > args.timeout:
            print(
                f"smoke_train: TIMEOUT after {args.timeout}s (steps={steps_done})",
                file=sys.stderr,
            )
            return 2
        if steps_done >= args.steps:
            break
        if sample.target.numel() == 0:
            continue
        inputs = {m: t[0].detach() for m, t in sample.inputs.items() if t.numel() > 0}
        if not inputs:
            continue
        target = {first_output: sample.target[0].detach()}
        try:
            result = soma.step(inputs=inputs, targets=target)
        except Exception as exc:  # noqa: BLE001 — any exception here is a smoke failure
            print(f"smoke_train: step raised: {exc}", file=sys.stderr)
            return 1
        loss = result.get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            print(
                f"smoke_train: non-finite loss at step {steps_done}: {loss}",
                file=sys.stderr,
            )
            return 1
        steps_done += 1

    print(f"smoke_train: OK ({steps_done} steps, {time.time() - start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
