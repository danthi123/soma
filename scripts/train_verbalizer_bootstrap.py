"""Bootstrap-train SOMA's verbalizer against a frozen HF LLM.

Thin CLI wrapper around ``soma.training.verbalizer_bootstrap.VerbalizerTrainer``
that loads a SOMA brain, wraps a frozen HuggingFace causal LM in
``ChatHead``, constructs a fresh ``SomaVerbalizer`` matching the LLM's
hidden dim, and trains the verbalizer's projector on a UTF-8 text corpus.

Measures held-out LM loss before and after training so the caller can
see the projector actually learned something.

Usage::

    # Explicit HF model name (legacy backward-compat path):
    python scripts/train_verbalizer_bootstrap.py \\
        --soma-checkpoint checkpoints/current.pt \\
        --corpus data/tinyshakespeare.txt \\
        --llm-name HuggingFaceTB/SmolLM2-360M-Instruct \\
        --out-dir artifacts/verbalizer-bootstrap-001 \\
        --max-steps 2000

    # Zero-config: pick the tier matching available VRAM
    # (see soma.deploy.MODEL_TIERS for the tier -> model-name map):
    python scripts/train_verbalizer_bootstrap.py \\
        --soma-checkpoint checkpoints/current.pt \\
        --corpus data/tinyshakespeare.txt \\
        --tier auto \\
        --out-dir artifacts/verbalizer-bootstrap-001

The ``--soma-checkpoint`` flag accepts either a single ``.pt`` file
(legacy layout) or a bundle directory (from :meth:`SOMA.save_bundle`).
Bundles carry their own tokenizer + encoder; single-file checkpoints
fall back to training a fresh BPE tokenizer over the corpus.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soma.core.brain_bundle import peek_payload
from soma.core.config import SOMAConfig
from soma.deploy.chat_head_factory import build_chat_head
from soma.deploy.cli import (
    add_deploy_arguments,
    print_selection,
    resolve_device_dtype_tier,
    resolve_quantization,
)
from soma.io.chat_head import ChatHead
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.system import SOMA
from soma.training.verbalizer_bootstrap import VerbalizerTrainer


def _window_corpus(text: str, window_chars: int) -> list[str]:
    """Yield overlapping character-windows of ~window_chars size.

    Bootstrap is small-scale — char approximation (~4 chars per BPE token)
    is close enough. Future phases can tokenize-then-window.
    """
    stride = max(window_chars // 2, 1)
    if len(text) <= window_chars:
        return [text] if text else []
    return [text[i : i + window_chars] for i in range(0, len(text) - window_chars, stride)]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bootstrap-train SOMA's verbalizer against a frozen HF LLM.",
    )
    p.add_argument(
        "--soma-checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to a SOMA brain checkpoint. Accepts either a single .pt "
            "file (legacy layout) or a bundle directory (from SOMA.save_bundle)."
        ),
    )
    p.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to a UTF-8 text corpus used for training + held-out eval.",
    )
    add_deploy_arguments(p)
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write verbalizer checkpoints into.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override SOMAConfig.bootstrap_max_steps (default: use config value).",
    )
    p.add_argument(
        "--num-prefix-tokens",
        type=int,
        default=8,
        help="Number of soft-prompt tokens emitted by the verbalizer (default: 8).",
    )
    p.add_argument(
        "--eval-samples",
        type=int,
        default=10,
        help="Leading corpus windows held out for before/after eval (default: 10).",
    )
    p.add_argument(
        "--tokenizer-vocab-size",
        type=int,
        default=None,
        help=(
            "Vocab size when training a fresh BPE tokenizer from the corpus "
            "(single-file checkpoint path only; default: config.vocab_size)."
        ),
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help=(
            "Accumulate gradients over this many micro-batches before each "
            "optim.step. Default 1 (no accumulation)."
        ),
    )
    p.add_argument(
        "--cosine-lr",
        action="store_true",
        help=(
            "Enable half-cosine LR decay to zero with optional linear warmup "
            "(see --warmup-steps). Default: constant config.verbalizer_lr."
        ),
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Linear warmup length (only used with --cosine-lr). Default 0 "
            "(full base LR from step 1)."
        ),
    )
    p.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help=(
            "Micro-step period between held-out eval runs during training. "
            "Default 1000. Only fires when eval_texts are available."
        ),
    )
    p.add_argument(
        "--loss-log",
        type=Path,
        default=None,
        help=(
            "Optional CSV path for per-step loss logging "
            "(columns: step,train_loss,eval_loss,lr). Written incrementally."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of corpus windows per forward pass (near-linear speedup "
            "up to GPU saturation). Default 1 (one window per step). "
            "Padded windows are masked out of the loss."
        ),
    )
    p.add_argument(
        "--soma-max-tokens",
        type=int,
        default=None,
        help=(
            "Cap the number of token embeddings fed through soma.step per "
            "window; longer windows are evenly subsampled. Default None "
            "(process every embedding). Profiled overhead is ~6 ms per "
            "soma.step call, so capping at e.g. 16 gives a ~4x speedup on "
            "the bootstrap loop when SOMA was the bottleneck."
        ),
    )
    return p


def _parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def _load_soma(
    checkpoint_path: Path,
    corpus_text: str,
    device: torch.device,
    *,
    tokenizer_vocab_size: int | None = None,
) -> tuple[SOMA, SOMAConfig, Any, Any]:
    """Load SOMA from a checkpoint; return (soma, config, tokenizer, encoder).

    Two layouts are supported:

    1. **Bundle directory** (``--soma-checkpoint some/bundle_dir``):
       ``SOMA.load_bundle`` rehydrates the brain, tokenizer, and encoder
       from the sidecar files. This is the preferred path because it
       guarantees the tokenizer vocab matches what the graph was trained
       against.
    2. **Single ``.pt`` file** (legacy): the brain is restored via
       ``SOMA.load_state`` and a fresh BPE tokenizer + ``TextEncoder``
       are built from the provided corpus. This is fine for bootstrap
       since the verbalizer only reads the OUTPUT-node aggregate, but
       note that token ids won't match the SOMA's earlier training runs.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SOMA checkpoint {checkpoint_path} not found")

    if checkpoint_path.is_dir():
        # Bundle layout: peek at brain.pt to size the skeleton, then
        # let load_bundle rehydrate the graph + tokenizer + encoder.
        brain_file = checkpoint_path / "brain.pt"
        if not brain_file.exists():
            raise FileNotFoundError(f"Bundle directory {checkpoint_path} missing brain.pt sidecar")
        raw = torch.load(str(brain_file), map_location="cpu", weights_only=False)
        state = peek_payload(raw)
        config = SOMAConfig.from_dict(state["config"])
        soma = SOMA(config, device=device)
        tokenizer, encoder, _verb = soma.load_bundle(checkpoint_path)
        if tokenizer is None or encoder is None:
            raise ValueError(
                f"Bundle at {checkpoint_path} is missing tokenizer.json or encoder.pt; "
                "re-save the bundle with a tokenizer + encoder, or pass a single .pt "
                "checkpoint so a fresh BPE tokenizer gets trained from the corpus."
            )
        return soma, config, tokenizer, encoder

    # Single-file layout: peek config, construct empty SOMA, load_state.
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = peek_payload(raw)
    config = SOMAConfig.from_dict(state["config"])
    soma = SOMA(config, device=device)
    soma.load_state(checkpoint_path)

    # No bundled tokenizer → train one on the corpus. Using line-split
    # mirrors scripts/train.py so a fresh checkpoint trained there keeps
    # a similar vocab distribution.
    corpus_lines = [line for line in corpus_text.splitlines() if line.strip()]
    if not corpus_lines:
        corpus_lines = [corpus_text]
    vocab_size = tokenizer_vocab_size if tokenizer_vocab_size is not None else config.vocab_size
    tokenizer = train_bpe_tokenizer(corpus_lines, vocab_size=vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=config.text_embed_dim,
        max_seq_len=config.max_input_tokens,
        device=device,
    )
    return soma, config, tokenizer, encoder


def main() -> None:
    args = _parse_args()
    device, dtype, llm_name, tier = resolve_device_dtype_tier(args)
    quantization = resolve_quantization(args)
    print_selection(
        llm_name=llm_name, tier=tier, device=device, dtype=dtype, quantization=quantization
    )

    corpus_text = args.corpus.read_text(encoding="utf-8")

    # ----- Load SOMA + text IO --------------------------------------------
    soma, cfg, tokenizer, encoder = _load_soma(
        args.soma_checkpoint,
        corpus_text,
        device,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
    )

    if args.max_steps is not None:
        cfg = replace(cfg, bootstrap_max_steps=args.max_steps)

    # ----- Load frozen HF LLM --------------------------------------------
    # --tier path uses the shared factory so the hidden-dim validation and
    # any future download/caching policies are shared with chat_repl.
    # --llm-name path keeps the bespoke loader so operators can point at
    # arbitrary HF models not in MODEL_TIERS.
    if tier is None:
        hf_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if quantization == "none":
            hf_model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    llm_name,
                    torch_dtype=dtype,
                ),
            ).to(device)
        else:
            # Mirror chat_head_factory's bnb path: bnb handles placement via
            # device_map, so no .to(device) after load (would break quantised
            # buffers). Import locally to keep the vanilla path bnb-free.
            from soma.deploy.chat_head_factory import _build_bnb_config

            hf_model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    llm_name,
                    torch_dtype=dtype,
                    quantization_config=_build_bnb_config(quantization, dtype),  # type: ignore[arg-type]
                    device_map={"": device.type},
                ),
            )
        chat_head = ChatHead(model=hf_model, tokenizer=hf_tokenizer)
    else:
        chat_head = build_chat_head(
            tier=tier,
            device=device,
            dtype=dtype,
            quantization=quantization,  # type: ignore[arg-type]
        )

    # ----- Build verbalizer ----------------------------------------------
    # Match the verbalizer's input dim to SOMA's actual OUTPUT-node dim.
    # Per SOMA._initialize_seed_graph, OUTPUT nodes share sensor_output_dim.
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name=llm_name,
        llm_hidden_dim=chat_head.hidden_size,
        num_prefix_tokens=args.num_prefix_tokens,
    )
    verbalizer = SomaVerbalizer(spec).to(device)

    # ----- Trainer --------------------------------------------------------
    trainer = VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=tokenizer,
        encoder=encoder,
    )
    if args.soma_max_tokens is not None:
        trainer.soma_max_tokens = args.soma_max_tokens

    # ----- Corpus split ---------------------------------------------------
    window_chars = cfg.bootstrap_sample_tokens * 4  # ~4 chars per BPE token
    windows = _window_corpus(corpus_text, window_chars)
    if not windows:
        raise ValueError(f"Corpus {args.corpus} is empty or smaller than one window")

    if len(windows) <= args.eval_samples:
        print(
            f"[warn] corpus produced only {len(windows)} windows; "
            f"need > {args.eval_samples} for a held-out split — "
            "running train-only without eval."
        )
        eval_texts: list[str] = []
        train_texts = windows
    else:
        eval_texts = windows[: args.eval_samples]
        train_texts = windows[args.eval_samples :]

    # ----- Pre-training eval ----------------------------------------------
    pre_loss: float | None = None
    if eval_texts:
        pre_loss = trainer.eval_lm_loss(texts=eval_texts)
        print(f"Pre-training held-out loss: {pre_loss:.4f}")

    # ----- Train ----------------------------------------------------------
    losses = trainer.train(
        corpus=iter(train_texts),
        max_steps=cfg.bootstrap_max_steps,
        out_dir=args.out_dir,
        eval_texts=eval_texts if eval_texts else None,
        eval_interval=args.eval_interval,
        grad_accum_steps=args.grad_accum_steps,
        cosine_lr=args.cosine_lr,
        warmup_steps=args.warmup_steps,
        loss_log_path=args.loss_log,
        batch_size=args.batch_size,
    )

    if losses:
        print(
            f"Trained {len(losses)} steps. First loss: {losses[0]:.4f}, last loss: {losses[-1]:.4f}"
        )
    else:
        print("Trained 0 steps (empty training slice).")

    # ----- Post-training eval --------------------------------------------
    if eval_texts:
        post_loss = trainer.eval_lm_loss(texts=eval_texts)
        print(f"Post-training held-out loss: {post_loss:.4f}")
        if pre_loss is not None:
            print(f"Delta: {post_loss - pre_loss:+.4f}")

    print(f"Verbalizer saved to: {args.out_dir / 'verbalizer_final'}")


if __name__ == "__main__":
    main()
