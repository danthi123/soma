"""Interactive CLI — converse with a trained SOMA model.

Usage::

    python scripts/interactive.py \
        --checkpoint checkpoints/soma_final.pt \
        --tokenizer checkpoints/tokenizer.json

The script loads a SOMA checkpoint plus a matching tokenizer, then drops
into a REPL. Each user line is fed to ``SOMA.interactive_session``; the
generated text is printed back.

Side-channel commands (all start with ``:``) let the user save the session,
reset working memory, print graph stats, or quit — without touching
anything via eval/exec.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO

import torch

from soma.core.config import SOMAConfig
from soma.io.text_decoder import TextDecoder
from soma.io.text_encoder import TextEncoder, load_tokenizer, train_bpe_tokenizer
from soma.system import SOMA

LOGGER = logging.getLogger("soma.interactive")

PROMPT = "soma> "


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with a trained SOMA model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a SOMA checkpoint (torch.save format).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help=(
            "Path to a saved tokenizers.Tokenizer JSON. "
            "Required unless --train-tokenizer-from is set."
        ),
    )
    parser.add_argument(
        "--train-tokenizer-from",
        type=Path,
        default=None,
        help="If no --tokenizer is given, train a BPE tokenizer from this corpus file.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override config.max_output_tokens for each exchange.",
    )
    parser.add_argument(
        "--save-on-exit",
        type=Path,
        default=None,
        help="Write the (possibly updated) SOMA state here when the session ends.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (cpu / cuda / cuda:0 / ...).",
    )
    return parser


# ----------------------------------------------------------------------
# Bootstrapping
# ----------------------------------------------------------------------
def load_soma(checkpoint_path: Path, device: torch.device | str | None = None) -> SOMA:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    state = torch.load(str(checkpoint_path), weights_only=False)
    config = SOMAConfig.from_dict(state["config"])
    soma = SOMA(config, device=device)
    soma.load_state(checkpoint_path)
    return soma


def prepare_encoders(
    config: SOMAConfig,
    tokenizer_path: Path | None,
    corpus_path: Path | None,
) -> tuple[TextEncoder, TextDecoder]:
    """Load (or train) a tokenizer, then wire up encoder + decoder."""
    if tokenizer_path is not None:
        tokenizer = load_tokenizer(tokenizer_path)
    elif corpus_path is not None:
        if not corpus_path.exists():
            raise FileNotFoundError(f"Tokenizer corpus {corpus_path} not found")
        corpus = [
            line.strip()
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not corpus:
            raise ValueError(f"Tokenizer corpus {corpus_path} is empty")
        tokenizer = train_bpe_tokenizer(corpus, vocab_size=config.vocab_size)
    else:
        raise ValueError("Either --tokenizer or --train-tokenizer-from must be provided")
    encoder = TextEncoder(
        tokenizer,
        embed_dim=config.text_embed_dim,
        max_seq_len=config.max_input_tokens,
    )
    decoder = TextDecoder(tokenizer, embed_dim=config.text_embed_dim)
    return encoder, decoder


# ----------------------------------------------------------------------
# Session commands (pure functions, easy to unit-test)
# ----------------------------------------------------------------------
HELP_TEXT = (
    "Commands:\n"
    "  :help               show this message\n"
    "  :stats              print graph / memory statistics\n"
    "  :reset_memory       clear working memory\n"
    "  :save <path>        save current SOMA state to <path>\n"
    "  :quit / :exit       end the session\n"
    "Any other line is sent to the model as input."
)


def handle_command(command: str, soma: SOMA) -> tuple[str, bool]:
    """Parse and execute a ``:command``. Returns (output, should_continue).

    ``should_continue`` is False only when the user asks to quit.
    """
    parts = command.strip().split(maxsplit=1)
    head = parts[0].lower()
    if head in (":quit", ":exit"):
        return ("Goodbye.", False)
    if head == ":help":
        return (HELP_TEXT, True)
    if head == ":stats":
        return (_format_stats(soma), True)
    if head == ":reset_memory":
        soma.working_memory.clear()
        return ("Working memory cleared.", True)
    if head == ":save":
        if len(parts) < 2:
            return ("Usage: :save <path>", True)
        path = Path(parts[1])
        soma.save_state(path)
        return (f"Saved to {path}", True)
    return (f"Unknown command: {head}. Type :help for options.", True)


def _format_stats(soma: SOMA) -> str:
    return (
        f"global_step={soma.global_step} "
        f"nodes={soma.graph.num_nodes} edges={soma.graph.num_edges} "
        f"wm_occupancy={soma.working_memory.occupancy():.3f} "
        f"episodic_entries={soma.episodic_memory.num_valid} "
        f"last_curiosity={soma.last_curiosity:.4f}"
    )


# ----------------------------------------------------------------------
# REPL
# ----------------------------------------------------------------------
def _iter_default_input(stream: IO[str]) -> Iterable[str]:
    for line in stream:
        yield line.rstrip("\n")


def run_session(
    soma: SOMA,
    encoder: TextEncoder,
    decoder: TextDecoder,
    *,
    input_lines: Iterable[str] | None = None,
    output_stream: IO[str] | None = None,
    max_output_tokens: int | None = None,
    prompt: str = PROMPT,
    banner: str | None = "SOMA interactive session. Type :help for commands.",
) -> None:
    """Drive the REPL. ``input_lines`` and ``output_stream`` make this testable."""
    out = output_stream if output_stream is not None else sys.stdout
    lines: Iterable[str]
    lines = input_lines if input_lines is not None else _iter_default_input(sys.stdin)

    if banner:
        print(banner, file=out)

    def text_encoder_fn(text: str) -> torch.Tensor:
        vec = encoder.encode_batch(text)
        if vec.numel() == 0:
            return torch.zeros((1, encoder.embed_dim), device=encoder.embedding.weight.device)
        return vec

    def text_decoder_fn(vec: torch.Tensor) -> str:
        # Shape is (embed_dim,) — decoder.decode handles a single activation.
        return decoder.decode(vec)

    for raw in lines:
        line = raw.strip()
        if not line:
            print(prompt, end="", file=out, flush=True)
            continue
        if line.startswith(":"):
            reply, keep_going = handle_command(line, soma)
            print(reply, file=out)
            if not keep_going:
                break
            continue
        response = soma.interactive_session(
            line,
            text_encoder=text_encoder_fn,
            text_decoder=text_decoder_fn,
            max_output_tokens=max_output_tokens,
        )
        print(response, file=out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        soma = load_soma(args.checkpoint, device=torch.device(args.device))
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        return 1

    try:
        encoder, decoder = prepare_encoders(
            soma.config,
            tokenizer_path=args.tokenizer,
            corpus_path=args.train_tokenizer_from,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error(str(exc))
        return 1

    run_session(
        soma,
        encoder,
        decoder,
        max_output_tokens=args.max_output_tokens,
    )

    if args.save_on_exit is not None:
        soma.save_state(args.save_on_exit)
        LOGGER.info("session state saved to %s", args.save_on_exit)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry
    sys.exit(main())


# ----------------------------------------------------------------------
# Convenience helper for tests
# ----------------------------------------------------------------------
def build_callable_pair(
    encoder: TextEncoder, decoder: TextDecoder
) -> tuple[Callable[[str], torch.Tensor], Callable[[torch.Tensor], str]]:
    """Return (encoder_fn, decoder_fn) as the CLI would wire them.

    Useful for unit tests that want to exercise the same callables without
    running the whole REPL.
    """

    def text_encoder_fn(text: str) -> torch.Tensor:
        vec = encoder.encode_batch(text)
        if vec.numel() == 0:
            return torch.zeros((1, encoder.embed_dim), device=encoder.embedding.weight.device)
        return vec

    def text_decoder_fn(vec: torch.Tensor) -> str:
        return decoder.decode(vec)

    return text_encoder_fn, text_decoder_fn
