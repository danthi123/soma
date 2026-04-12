"""CLI training script — run a SOMA model against a text corpus.

Usage::

    python scripts/train.py --config configs/default.yaml --corpus data/corpus.txt \
        --steps 5000 --checkpoint-dir checkpoints/

The script wires together:
- ``SOMAConfig.from_yaml`` for hyperparameters.
- ``train_bpe_tokenizer`` + ``TextEncoder``/``TextDecoder`` for I/O.
- ``TextDatasetFeeder`` to produce training pairs from the corpus.
- ``SOMA.step`` for the full interaction loop.
- Periodic checkpointing at ``config.checkpoint_interval`` steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch

from soma.core.config import SOMAConfig
from soma.io.dataset_feeders import Sample, TextDatasetFeeder
from soma.io.text_decoder import TextDecoder
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.system import SOMA

LOGGER = logging.getLogger("soma.train")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the training CLI."""
    parser = argparse.ArgumentParser(description="Train a SOMA model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to a YAML config file (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to a UTF-8 text corpus (one training document per line).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5_000,
        help="Total number of training steps to run.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where checkpoints are written.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint file to resume from.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Emit a progress log line every N steps.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Optional JSONL file to append per-log-window metrics to.",
    )
    parser.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=None,
        help="Override tokenizer vocabulary size (default: config.vocab_size).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (cpu / cuda / cuda:0 / ...).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the config's random seed.",
    )
    return parser


# ----------------------------------------------------------------------
# Corpus + tokenizer
# ----------------------------------------------------------------------
def read_corpus(path: Path) -> list[str]:
    """Read a corpus file into a list of non-empty lines."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus file {path} not found")
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped:
            lines.append(stripped)
    if not lines:
        raise ValueError(f"Corpus file {path} has no non-empty lines")
    return lines


def build_encoders(
    config: SOMAConfig,
    corpus: Iterable[str],
    *,
    vocab_size: int | None = None,
) -> tuple[TextEncoder, TextDecoder]:
    """Train a BPE tokenizer over the corpus and wire encoder + decoder."""
    effective_vocab = vocab_size if vocab_size is not None else config.vocab_size
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=effective_vocab)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=config.text_embed_dim,
        max_seq_len=config.max_input_tokens,
    )
    decoder = TextDecoder(tokenizer, embed_dim=config.text_embed_dim)
    return encoder, decoder


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def _iter_samples(feeder: TextDatasetFeeder) -> Iterator[Sample]:
    """Wrap a feeder as an iterator; feeder's ``cycle=True`` default handles refills."""
    yield from feeder


def train(
    soma: SOMA,
    feeder: TextDatasetFeeder,
    *,
    num_steps: int,
    log_every: int,
    checkpoint_dir: Path,
    metrics_file: Path | None = None,
) -> dict[str, Any]:
    """Run the main training loop; returns a summary dict."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples = _iter_samples(feeder)

    window_losses: list[float] = []
    total_losses: list[float] = []

    for step_idx in range(num_steps):
        sample = next(samples)
        # The feeder emits token sequences; we use the first input/target token
        # per sample so the whitepaper's per-step semantics hold. Callers who
        # want full-sequence training should loop in their feeder.
        inputs = {m: tensor[0] for m, tensor in sample.inputs.items() if tensor.numel() > 0}
        if sample.target.numel() == 0:
            continue
        target = sample.target[0]
        first_out_modality = soma.config.output_modalities[0]

        result = soma.step(inputs=inputs, targets={first_out_modality: target})
        if result["loss"] is not None:
            window_losses.append(float(result["loss"]))
            total_losses.append(float(result["loss"]))

        if (step_idx + 1) % log_every == 0:
            mean_loss = sum(window_losses) / max(1, len(window_losses))
            LOGGER.info(
                "step=%d loss=%.6f curiosity=%.4f nodes=%d edges=%d lr_mul=%.3f",
                soma.global_step,
                mean_loss,
                result["curiosity"],
                result["num_nodes"],
                result["num_edges"],
                result["lr_multiplier"],
            )
            if metrics_file is not None:
                _append_metrics(
                    metrics_file,
                    {
                        "step": soma.global_step,
                        "window_mean_loss": mean_loss,
                        "curiosity": result["curiosity"],
                        "num_nodes": result["num_nodes"],
                        "num_edges": result["num_edges"],
                        "lr_multiplier": result["lr_multiplier"],
                    },
                )
            window_losses.clear()

        if (step_idx + 1) % soma.config.checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"soma_step_{soma.global_step}.pt"
            soma.save_state(ckpt_path)
            LOGGER.info("checkpoint written: %s", ckpt_path)

    # Final checkpoint.
    final_path = checkpoint_dir / "soma_final.pt"
    soma.save_state(final_path)
    LOGGER.info("final checkpoint written: %s", final_path)

    return {
        "final_step": soma.global_step,
        "mean_loss": (sum(total_losses) / len(total_losses)) if total_losses else float("nan"),
        "num_nodes": soma.graph.num_nodes,
        "num_edges": soma.graph.num_edges,
        "checkpoint_path": str(final_path),
    }


def _append_metrics(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = SOMAConfig.from_yaml(args.config)
    if args.seed is not None:
        config.seed = args.seed

    corpus = read_corpus(args.corpus)
    encoder, _decoder = build_encoders(config, corpus, vocab_size=args.tokenizer_vocab_size)

    feeder = TextDatasetFeeder(
        encoder,
        corpus,
        chunk_size=min(16, config.max_input_tokens // 2),
    )

    device = torch.device(args.device)
    soma = SOMA(config, device=device)

    if args.resume is not None:
        if not args.resume.exists():
            LOGGER.error("resume path %s does not exist", args.resume)
            return 1
        soma.load_state(args.resume)
        LOGGER.info("resumed from %s at step %d", args.resume, soma.global_step)

    summary = train(
        soma,
        feeder,
        num_steps=args.steps,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        metrics_file=args.metrics_file,
    )
    LOGGER.info("training complete: %s", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry
    sys.exit(main())
