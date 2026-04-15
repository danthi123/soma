"""Sample-generation A/B: fresh vs trained verbalizer on fixed prompts.

Loads the same SOMA bundle + LLM under both a freshly-initialised
verbalizer (zero training) and a trained one, then generates a response
to each prompt with both. Outputs side-by-side markdown for visual
comparison -- "did the verbalizer actually learn to bias the LLM toward
useful continuations?".

Both verbalizers see the same SOMA state per prompt (we run SOMA once
and reuse the pooled activations) so the only varying factor is the
verbalizer's projector weights.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

# Force UTF-8 stdout so the printed samples don't crash on the Windows
# cp1252 console when the LLM emits non-ASCII characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoModelForCausalLM, AutoTokenizer

from soma.core.brain_bundle import peek_payload
from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.session.chat_session import ChatSession
from soma.system import SOMA

PROMPTS = [
    "The history of",
    "In a typical wikipedia article,",
    "The capital city of France",
    "Quantum mechanics describes",
    "After the war ended,",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trained-verbalizer", type=Path, required=True)
    p.add_argument("--soma-checkpoint", type=Path, required=True)
    p.add_argument("--llm-name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def _build_session(*, soma, verbalizer, chat_head, tokenizer, encoder) -> ChatSession:
    return ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder,
    )


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # ---- Load SOMA + sidecars -----------------------------------------
    bundle = args.soma_checkpoint
    raw = torch.load(str(bundle / "brain.pt"), map_location="cpu", weights_only=False)
    state_payload = peek_payload(raw)
    cfg = SOMAConfig.from_dict(state_payload["config"])
    cfg = replace(cfg, bootstrap_max_steps=1)

    # Build TWO somas (one per session) so state is independent.
    soma_fresh = SOMA(cfg, device=device)
    tokenizer, encoder, _verb = soma_fresh.load_bundle(bundle)
    assert tokenizer is not None and encoder is not None

    soma_trained = SOMA(cfg, device=device)
    _tk, _enc, _v2 = soma_trained.load_bundle(bundle)
    # Reuse the same tokenizer/encoder objects -- they're stateless w.r.t.
    # SOMA, only the SOMA graph state needs to be independent.

    # ---- LLM ---------------------------------------------------------
    print(f"Loading LLM {args.llm_name} ({args.dtype}) on {device}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    hf_model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=dtype),
    ).to(device)
    chat_head = ChatHead(model=hf_model, tokenizer=hf_tokenizer)

    # ---- Verbalizers --------------------------------------------------
    trained_verbalizer = SomaVerbalizer.load(args.trained_verbalizer).to(device)
    spec = trained_verbalizer.spec
    fresh_spec = VerbalizerSpec(
        soma_output_dim=spec.soma_output_dim,
        llm_name=spec.llm_name,
        llm_hidden_dim=spec.llm_hidden_dim,
        num_prefix_tokens=spec.num_prefix_tokens,
    )
    fresh_verbalizer = SomaVerbalizer(fresh_spec).to(device)

    fresh_session = _build_session(
        soma=soma_fresh,
        verbalizer=fresh_verbalizer,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder,
    )
    trained_session = _build_session(
        soma=soma_trained,
        verbalizer=trained_verbalizer,
        chat_head=chat_head,
        tokenizer=tokenizer,
        encoder=encoder,
    )

    # ---- Generate samples --------------------------------------------
    rows: list[dict[str, str]] = []
    for prompt in PROMPTS:
        print(f"\n[prompt] {prompt!r}")
        fresh_resp = fresh_session.respond(
            user_text=prompt,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=3,
            do_sample=False,
        )
        trained_resp = trained_session.respond(
            user_text=prompt,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=3,
            do_sample=False,
        )
        print(f"  fresh:   {fresh_resp!r}")
        print(f"  trained: {trained_resp!r}")
        rows.append({"prompt": prompt, "fresh": fresh_resp, "trained": trained_resp})

    # ---- Markdown report ---------------------------------------------
    lines = [
        "# Verbalizer Sample Comparison (Fresh vs Trained)",
        "",
        f"**LLM**: `{args.llm_name}` ({args.dtype})",
        f"**SOMA bundle**: `{args.soma_checkpoint}`",
        f"**Trained verbalizer**: `{args.trained_verbalizer}`",
        f"**max_new_tokens**: {args.max_new_tokens}, do_sample=False (greedy)",
        "",
        "Both runs use the SAME SOMA state per prompt (each session feeds",
        "the prompt through SOMA, so SOMA-state evolution is identical).",
        "The only varying factor is the verbalizer's projector weights:",
        "fresh = randomly-initialised, trained = step_15500 (best held-out).",
        "",
    ]
    for row in rows:
        lines.append(f"## Prompt: `{row['prompt']}`")
        lines.append("")
        lines.append("**Fresh** (untrained verbalizer):")
        lines.append("```")
        lines.append(row["fresh"])
        lines.append("```")
        lines.append("")
        lines.append("**Trained** (step_15500 verbalizer):")
        lines.append("```")
        lines.append(row["trained"])
        lines.append("```")
        lines.append("")
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
