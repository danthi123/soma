"""Compute held-out LM loss for every verbalizer checkpoint in a bootstrap run.

Reads each ``verbalizer_step_N`` directory under ``--out-dir``, loads the
verbalizer weights, and measures mean LM loss on a fixed eval slice of the
same corpus the bootstrap used. Produces a CSV + markdown summary.

Usage::

    python scripts/eval_bootstrap_checkpoints.py \\
        --bootstrap-dir artifacts/bootstrap-2026-04-15-qwen3b-wikitext2 \\
        --soma-checkpoint checkpoints/bootstrap-src-fresh-2026-04-15 \\
        --corpus data/wikitext-2-train.txt \\
        --llm-name Qwen/Qwen2.5-3B-Instruct \\
        --device cuda --dtype fp16 \\
        --eval-samples 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
from soma.training.verbalizer_bootstrap import VerbalizerTrainer


def _window_corpus(text: str, window_chars: int) -> list[str]:
    stride = max(window_chars // 2, 1)
    if len(text) <= window_chars:
        return [text] if text else []
    return [text[i : i + window_chars] for i in range(0, len(text) - window_chars, stride)]


def _iter_checkpoint_dirs(bootstrap_dir: Path) -> list[tuple[int, Path]]:
    pat = re.compile(r"^verbalizer_step_(\d+)$")
    results: list[tuple[int, Path]] = []
    for child in bootstrap_dir.iterdir():
        if not child.is_dir():
            continue
        m = pat.match(child.name)
        if m is not None:
            results.append((int(m.group(1)), child))
    # Also include verbalizer_final if present (end-of-run checkpoint).
    final = bootstrap_dir / "verbalizer_final"
    if final.is_dir():
        # Treat final as a special step -- use max+1 for ordering, label "final".
        results.append((10**9, final))
    results.sort(key=lambda pair: pair[0])
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap-dir", type=Path, required=True)
    p.add_argument("--soma-checkpoint", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--llm-name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
    )
    p.add_argument("--eval-samples", type=int, default=20)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV (default: <bootstrap-dir>/loss_curve.csv)",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output markdown report (default: <bootstrap-dir>/loss_curve.md)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # ---- Load SOMA + sidecars from the bundle -------------------------
    bundle = args.soma_checkpoint
    brain_file = bundle / "brain.pt"
    raw = torch.load(str(brain_file), map_location="cpu", weights_only=False)
    state = peek_payload(raw)
    cfg = SOMAConfig.from_dict(state["config"])
    cfg = replace(cfg, bootstrap_max_steps=1)  # irrelevant for eval
    soma = SOMA(cfg, device=device)
    tokenizer, encoder, _verb = soma.load_bundle(bundle)
    assert tokenizer is not None and encoder is not None, "bundle missing sidecars"

    # ---- Load HF LLM --------------------------------------------------
    print(f"Loading LLM {args.llm_name} in {args.dtype} on {device}...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    hf_model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=dtype),
    ).to(device)
    chat_head = ChatHead(model=hf_model, tokenizer=hf_tokenizer)

    # ---- Corpus -> eval slice (SAME split the bootstrap used) ---------
    corpus_text = args.corpus.read_text(encoding="utf-8")
    window_chars = cfg.bootstrap_sample_tokens * 4
    windows = _window_corpus(corpus_text, window_chars)
    if len(windows) <= args.eval_samples:
        raise ValueError("Corpus too small for the eval slice")
    eval_texts = windows[: args.eval_samples]
    print(f"Eval slice: {len(eval_texts)} windows of ~{window_chars} chars each")

    # ---- Iterate checkpoints -----------------------------------------
    ckpts = _iter_checkpoint_dirs(args.bootstrap_dir)
    if not ckpts:
        raise SystemExit(f"No verbalizer_step_*/ checkpoints under {args.bootstrap_dir}")
    print(f"Found {len(ckpts)} checkpoints")

    rows: list[dict[str, Any]] = []
    for step, ck_dir in ckpts:
        # Build a trainer with THIS checkpoint's verbalizer.
        spec_json = json.loads((ck_dir / "spec.json").read_text(encoding="utf-8"))
        verb = SomaVerbalizer.load(ck_dir).to(device)
        trainer = VerbalizerTrainer(
            soma=soma,
            verbalizer=verb,
            chat_head=chat_head,
            config=cfg,
            tokenizer=tokenizer,
            encoder=encoder,
        )
        loss = trainer.eval_lm_loss(texts=eval_texts)
        step_label = "final" if step == 10**9 else str(step)
        print(f"  step={step_label:>6} loss={loss:.4f}")
        rows.append(
            {
                "step": step_label,
                "step_int": step if step != 10**9 else None,
                "loss": round(loss, 4),
                "checkpoint": ck_dir.name,
                "llm_hidden_dim": spec_json.get("llm_hidden_dim"),
            }
        )

    # ---- CSV ----------------------------------------------------------
    out_csv = args.out_csv or (args.bootstrap_dir / "loss_curve.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["step", "step_int", "loss", "checkpoint"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ("step", "step_int", "loss", "checkpoint")})
    print(f"Wrote {out_csv}")

    # ---- Markdown summary --------------------------------------------
    out_md = args.out_md or (args.bootstrap_dir / "loss_curve.md")
    first = rows[0]
    last = rows[-1]
    delta = last["loss"] - first["loss"]
    direction = "down" if delta < 0 else "up"
    lines = [
        "# Verbalizer Bootstrap Loss Curve",
        "",
        f"**Run**: {args.bootstrap_dir.name}",
        f"**LLM**: {args.llm_name} ({args.dtype})",
        f"**Corpus**: {args.corpus.name}",
        f"**Eval windows**: {len(eval_texts)} (first {args.eval_samples} of corpus)",
        f"**Checkpoints**: {len(rows)}",
        "",
        f"**Start**: step={first['step']}, loss={first['loss']:.4f}",
        f"**End**:   step={last['step']}, loss={last['loss']:.4f}",
        f"**Delta**: {delta:+.4f} ({direction})",
        "",
        "## Curve",
        "",
        "| step | loss |",
        "|------|------|",
    ]
    for row in rows:
        lines.append(f"| {row['step']} | {row['loss']:.4f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
