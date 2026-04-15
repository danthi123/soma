"""Interactive multi-turn chat REPL with SOMA + frozen LLM.

Thin CLI wrapper around :class:`soma.session.chat_session.ChatSession`.
Loads a SOMA brain bundle (tokenizer + encoder come from the bundle), a
frozen HuggingFace causal LM wrapped in :class:`ChatHead`, and a
:class:`SomaVerbalizer` (from a saved checkpoint or freshly constructed),
then drops the caller into an interactive REPL.

Usage:
    # Zero-config: pick the best tier for this machine's hardware
    # (see soma.deploy.MODEL_TIERS for the tier -> model-name map):
    python scripts/chat_repl.py \\
        --soma-checkpoint artifacts/brain-bundle/ \\
        --tier auto

    # Fresh session with an explicit HF model:
    python scripts/chat_repl.py \\
        --soma-checkpoint artifacts/brain-bundle/ \\
        --llm-name HuggingFaceTB/SmolLM2-360M-Instruct \\
        --verbalizer-checkpoint artifacts/verbalizer-bootstrap-001/verbalizer_final/ \\
        --system-prompt "You are a helpful assistant." \\
        --out-dir artifacts/sessions/today/

    # Resume an existing session:
    python scripts/chat_repl.py \\
        --soma-checkpoint artifacts/sessions/today/ \\
        --llm-name HuggingFaceTB/SmolLM2-360M-Instruct \\
        --verbalizer-checkpoint artifacts/sessions/today/verbalizer/ \\
        --resume-from artifacts/sessions/today/ \\
        --out-dir artifacts/sessions/today/

REPL conventions:
    > user message
    Empty line -> ignored (prompt again)
    Ctrl-D / EOF / ``quit`` / ``exit`` -> save if ``--out-dir`` set, then exit

This script requires a **bundle directory** for ``--soma-checkpoint`` (a
directory containing ``brain.pt``, ``tokenizer.json``, ``encoder.pt``).
Single-file ``.pt`` checkpoints have no bundled tokenizer/encoder, which
makes interactive chat impossible; that layout stays supported only by
the verbalizer bootstrap trainer.
"""

from __future__ import annotations

import argparse
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
)
from soma.io.chat_head import ChatHead
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.session.chat_session import ChatSession
from soma.system import SOMA


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive multi-turn SOMA chat REPL (SOMA + frozen HF LLM).",
    )
    p.add_argument(
        "--soma-checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to a SOMA brain bundle directory (containing brain.pt, "
            "tokenizer.json, encoder.pt). Single-file .pt checkpoints are "
            "rejected because interactive chat needs a bundled tokenizer."
        ),
    )
    add_deploy_arguments(p)
    p.add_argument(
        "--verbalizer-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional path to a saved SomaVerbalizer directory (from "
            "SomaVerbalizer.save). If omitted, a fresh near-zero-init "
            "verbalizer is constructed matching the LLM's hidden dim."
        ),
    )
    p.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=(
            "Optional system prompt pre-fed through SOMA at session start. "
            "Ignored when --resume-from is set (the saved brain already "
            "reflects it; re-warming would double-feed)."
        ),
    )
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Directory containing chat_history.json from a prior session. "
            "Loads the history into the new session's log. The SOMA brain "
            "itself is expected to be loaded from --soma-checkpoint pointing "
            "at the same saved session."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "If set, save the full session bundle (brain + tokenizer + "
            "encoder + verbalizer + chat_history.json) to this directory on "
            "exit. Created if missing."
        ),
    )
    p.add_argument(
        "--num-prefix-tokens",
        type=int,
        default=8,
        help="Number of soft-prompt tokens emitted by the verbalizer (default: 8).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens per assistant response (default: 64).",
    )
    return p


def _parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def _load_soma(checkpoint_path: Path, device: torch.device) -> tuple[SOMA, SOMAConfig, Any, Any]:
    """Load SOMA + tokenizer + encoder from a bundle directory.

    Mirrors the bundle-branch of ``scripts/train_verbalizer_bootstrap.py``'s
    ``_load_soma`` helper. Interactive chat has no corpus to fall back on
    for training a fresh BPE tokenizer, so single-file ``.pt`` checkpoints
    are rejected with a clear error.

    Returns ``(soma, config, tokenizer, encoder)``. The bundled verbalizer
    (if any) is discarded here; the caller picks a verbalizer via
    ``--verbalizer-checkpoint`` or the fresh-init fallback.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SOMA checkpoint {checkpoint_path} not found")

    if checkpoint_path.is_file():
        raise ValueError(
            "chat_repl.py requires a directory-shaped bundle (with "
            "tokenizer/encoder sidecars). Pass --soma-checkpoint pointing "
            "at the bundle dir."
        )

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
            f"Bundle at {checkpoint_path} is missing tokenizer.json or "
            "encoder.pt; re-save the bundle with both sidecars present."
        )
    return soma, config, tokenizer, encoder


def _build_chat_head_explicit(
    llm_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> ChatHead:
    """Build a ChatHead for an explicitly-named HF model at the given dtype."""
    hf_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    hf_model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=dtype,
        ),
    ).to(device)
    return ChatHead(model=hf_model, tokenizer=hf_tokenizer)


def _build_verbalizer(
    *,
    soma_output_dim: int,
    chat_head: ChatHead,
    llm_name: str,
    num_prefix_tokens: int,
    checkpoint_path: Path | None,
    device: torch.device,
) -> SomaVerbalizer:
    if checkpoint_path is not None:
        return SomaVerbalizer.load(checkpoint_path).to(device)
    spec = VerbalizerSpec(
        soma_output_dim=soma_output_dim,
        llm_name=llm_name,
        llm_hidden_dim=chat_head.hidden_size,
        num_prefix_tokens=num_prefix_tokens,
    )
    return SomaVerbalizer(spec).to(device)


def _repl(session: ChatSession, max_new_tokens: int) -> None:
    """Read lines until EOF or a quit command; print session responses.

    Blank lines are ignored. ``quit`` / ``exit`` / ``:q`` / ``:quit``
    (case-insensitive) return cleanly. Ctrl-D / EOF is caught and also
    returns cleanly so the ``finally`` block in ``main`` can save.
    """
    print("\nSOMA chat session. Ctrl-D, quit, or exit to end.\n")
    while True:
        try:
            user = input("> ").strip()
        except EOFError:
            print()  # newline after ^D so the shell prompt lands cleanly
            return
        if not user:
            continue
        if user.lower() in {"quit", "exit", ":q", ":quit"}:
            return
        response = session.respond(user_text=user, max_new_tokens=max_new_tokens)
        print(response)
        print()  # blank line after each response


def main() -> None:
    args = _parse_args()
    device, dtype, llm_name, tier = resolve_device_dtype_tier(args)
    print_selection(llm_name=llm_name, tier=tier, device=device, dtype=dtype)

    soma, cfg, tokenizer, encoder = _load_soma(args.soma_checkpoint, device)
    if tier is None:
        chat_head = _build_chat_head_explicit(llm_name, device, dtype)
    else:
        chat_head = build_chat_head(tier=tier, device=device, dtype=dtype)
    verbalizer = _build_verbalizer(
        soma_output_dim=cfg.sensor_output_dim,
        chat_head=chat_head,
        llm_name=llm_name,
        num_prefix_tokens=args.num_prefix_tokens,
        checkpoint_path=args.verbalizer_checkpoint,
        device=device,
    )

    # Resume vs. fresh: when resuming, the saved brain.pt already encodes
    # the system prompt's effect on SOMA state — re-warming would double-
    # feed. So only pass system_prompt on a fresh session.
    system_prompt = None if args.resume_from else args.system_prompt

    session = ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder,
        system_prompt=system_prompt,
    )

    if args.resume_from is not None:
        session.load_history(out_dir=args.resume_from)
        print(f"Resumed with {len(session.history)} prior turns.")

    try:
        _repl(session, args.max_new_tokens)
    finally:
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            session.save(out_dir=args.out_dir)
            print(f"Session saved to {args.out_dir}")


if __name__ == "__main__":
    main()
