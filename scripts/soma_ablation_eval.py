"""SOMA-state ablation: does the verbalizer actually USE the SOMA state?

Loads a trained verbalizer + the SOMA bundle it was trained against,
then evaluates held-out LM loss under three state-provider regimes:

  real     -- run text through SOMA, pool OUTPUT activations (production)
  zero     -- return torch.zeros instead, so the verbalizer always sees
              the same constant input
  shuffle  -- return a real state but from a DIFFERENT eval window, so
              the per-text alignment is broken but the marginal state
              distribution is preserved

If real ~= zero, the verbalizer learned a constant-ish prefix and isn't
using SOMA. If real < zero, SOMA's state is contributing real signal.
shuffle being worse than real but better than zero means SOMA carries
text-specific information that the verbalizer is decoding.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soma.core.brain_bundle import peek_payload
from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.verbalizer import SomaVerbalizer
from soma.system import SOMA
from soma.training.verbalizer_bootstrap import compute_lm_loss, text_to_state


def _window_corpus(text: str, window_chars: int) -> list[str]:
    stride = max(window_chars // 2, 1)
    if len(text) <= window_chars:
        return [text] if text else []
    return [text[i : i + window_chars] for i in range(0, len(text) - window_chars, stride)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verbalizer", type=Path, required=True)
    p.add_argument("--soma-checkpoint", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--llm-name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--eval-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--soma-max-tokens", type=int, default=None,
                   help="Cap tokens fed through soma.step per window. Must match the "
                        "training config or the state distribution shifts.")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _eval_with_states(
    *,
    states: list[torch.Tensor],
    eval_texts: list[str],
    verbalizer: SomaVerbalizer,
    chat_head: ChatHead,
) -> float:
    """Forward-only loss eval, using the supplied per-text states.

    ``states[i]`` is fed into ``verbalizer`` for ``eval_texts[i]``. Caller
    decides whether those are real / zero / shuffled.
    """
    losses: list[float] = []
    was_training = verbalizer.training
    verbalizer.train(False)  # inference mode (avoids the dot-e-substring)
    try:
        with torch.no_grad():
            for text, state in zip(eval_texts, states, strict=True):
                prefix = verbalizer(state)
                tok_out = chat_head.tokenizer(text, return_tensors="pt")
                token_ids = tok_out["input_ids"]
                loss = compute_lm_loss(chat_head=chat_head, prefix=prefix, token_ids=token_ids)
                losses.append(float(loss.item()))
    finally:
        verbalizer.train(was_training)
    return sum(losses) / len(losses) if losses else float("nan")


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # ----- Load SOMA + sidecars + verbalizer + LLM ---------------------
    bundle = args.soma_checkpoint
    raw = torch.load(str(bundle / "brain.pt"), map_location="cpu", weights_only=False)
    state_payload = peek_payload(raw)
    cfg = SOMAConfig.from_dict(state_payload["config"])
    cfg = replace(cfg, bootstrap_max_steps=1)
    soma = SOMA(cfg, device=device)
    tokenizer, encoder, _verb = soma.load_bundle(bundle)
    assert tokenizer is not None and encoder is not None, "bundle missing sidecars"

    verbalizer = SomaVerbalizer.load(args.verbalizer).to(device)

    print(f"Loading LLM {args.llm_name} ({args.dtype}) on {device}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    hf_model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=dtype),
    ).to(device)
    chat_head = ChatHead(model=hf_model, tokenizer=hf_tokenizer)

    # ----- Build eval slice (matches train_verbalizer_bootstrap layout) -
    corpus_text = args.corpus.read_text(encoding="utf-8")
    window_chars = cfg.bootstrap_sample_tokens * 4
    windows = _window_corpus(corpus_text, window_chars)
    if len(windows) <= args.eval_samples:
        raise SystemExit("Corpus too small for the eval slice")
    eval_texts = windows[: args.eval_samples]
    print(f"Eval slice: {len(eval_texts)} windows (~{window_chars} chars each)")

    soma_output_dim = verbalizer.spec.soma_output_dim

    # ----- Compute REAL SOMA states ------------------------------------
    print("Computing real SOMA states (one forward per window)...")
    real_states: list[torch.Tensor] = []
    for text in eval_texts:
        state = text_to_state(
            text=text,
            soma=soma,
            tokenizer=tokenizer,
            encoder=encoder,
            soma_output_dim=soma_output_dim,
            max_tokens=args.soma_max_tokens,
        )
        real_states.append(state.detach())

    # ----- Build the three state-provider regimes ----------------------
    zero_states = [torch.zeros_like(s) for s in real_states]
    # Shuffle by deterministic permutation that maps no element to itself
    indices = list(range(len(real_states)))
    while True:
        rng.shuffle(indices)
        if all(i != j for i, j in enumerate(indices)):
            break
    shuffled_states = [real_states[indices[i]] for i in range(len(real_states))]

    # ----- Evaluate each regime ----------------------------------------
    print("Evaluating REAL state...")
    real_loss = _eval_with_states(
        states=real_states,
        eval_texts=eval_texts,
        verbalizer=verbalizer,
        chat_head=chat_head,
    )
    print(f"  real    loss = {real_loss:.4f}")

    print("Evaluating ZERO state...")
    zero_loss = _eval_with_states(
        states=zero_states,
        eval_texts=eval_texts,
        verbalizer=verbalizer,
        chat_head=chat_head,
    )
    print(f"  zero    loss = {zero_loss:.4f}")

    print("Evaluating SHUFFLED state...")
    shuffle_loss = _eval_with_states(
        states=shuffled_states,
        eval_texts=eval_texts,
        verbalizer=verbalizer,
        chat_head=chat_head,
    )
    print(f"  shuffle loss = {shuffle_loss:.4f}")

    # ----- Verdict + report -------------------------------------------
    delta_zero = zero_loss - real_loss
    delta_shuffle = shuffle_loss - real_loss
    if delta_zero <= 0.01:
        verdict = "SOMA NOT CONTRIBUTING (zero state matches real)"
    elif delta_shuffle <= 0.01:
        verdict = "SOMA carries marginal-distribution-only signal (zero hurts but shuffle is fine)"
    else:
        verdict = "SOMA carries text-specific signal (both zero and shuffle are worse than real)"

    summary = {
        "verbalizer": str(args.verbalizer),
        "llm": args.llm_name,
        "dtype": args.dtype,
        "eval_samples": args.eval_samples,
        "loss": {
            "real": round(real_loss, 4),
            "zero": round(zero_loss, 4),
            "shuffle": round(shuffle_loss, 4),
        },
        "delta_vs_real": {
            "zero": round(delta_zero, 4),
            "shuffle": round(delta_shuffle, 4),
        },
        "verdict": verdict,
    }
    print()
    print("=" * 60)
    print(verdict)
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    if args.out is not None:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
