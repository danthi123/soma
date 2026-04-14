"""5000-step smoke for wave-batching: sequential vs batched training.

Runs two short training sessions from identical seeds / configs, one
using ``execute_graph``, one using ``execute_graph_batched``. Verifies:
  (a) no NaN / exception in either run
  (b) batched final held-out loss within 10% of sequential

Usage:
    python scripts/smoke_wave_batching.py --steps 5000

Per the plan this is operator-level validation, not a pytest test —
don't wire it into CI.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from soma.core.config import SOMAConfig
from soma.io.dataset_feeders import TextDatasetFeeder
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.system import SOMA

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "lorem ipsum dolor sit amet consectetur adipiscing elit",
    "to be or not to be that is the question",
    "once upon a time in a land far far away",
    "alpha beta gamma delta epsilon zeta eta theta iota",
]

HELDOUT = [
    "the fox and the hound are friends",
    "nothing gold can stay",
]


def run_training(config: SOMAConfig, steps: int, device: torch.device) -> dict:
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=config.vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=config.text_embed_dim,
        max_seq_len=config.max_input_tokens,
        device=device,
    )
    feeder = TextDatasetFeeder(encoder, CORPUS, chunk_size=4)
    soma = SOMA(config, device=device)
    first_out = config.output_modalities[0]
    step = 0
    last_loss = float("nan")
    t0 = time.perf_counter()
    for sample in feeder:
        if step >= steps:
            break
        if sample.target.numel() == 0:
            continue
        inputs = {m: t[0].detach() for m, t in sample.inputs.items() if t.numel() > 0}
        if not inputs:
            continue
        target = {first_out: sample.target[0].detach()}
        try:
            res = soma.step(inputs=inputs, targets=target)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"exception at step {step}: {exc}"}
        loss = res.get("loss")
        if loss is not None:
            if not math.isfinite(loss):
                return {
                    "ok": False,
                    "reason": f"non-finite loss at step {step}: {loss}",
                }
            last_loss = loss
        step += 1
    elapsed = time.perf_counter() - t0

    # Evaluate on HELDOUT.
    heldout_feeder = TextDatasetFeeder(encoder, HELDOUT, chunk_size=4, cycle=False)
    heldout_losses: list[float] = []
    for sample in heldout_feeder:
        if sample.target.numel() == 0:
            continue
        inputs = {m: t[0].detach() for m, t in sample.inputs.items() if t.numel() > 0}
        if not inputs:
            continue
        target = {first_out: sample.target[0].detach()}
        res = soma.step(inputs=inputs, targets=target, eval_mode=True)
        hl = res.get("loss")
        if hl is not None and math.isfinite(hl):
            heldout_losses.append(hl)
    heldout = sum(heldout_losses) / max(1, len(heldout_losses))
    return {
        "ok": True,
        "steps": step,
        "final_train_loss": last_loss,
        "heldout_loss": heldout,
        "elapsed_sec": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    base = SOMAConfig(
        sensor_output_dim=16,
        associator_input_dim=16,
        associator_hidden_dim=32,
        associator_output_dim=16,
        integrator_input_dim=32,
        integrator_hidden_dim=32,
        integrator_output_dim=32,
        position_dim=4,
        wm_slots=4,
        wm_dim=16,
        episodic_capacity=100,
        key_dim=16,
        value_dim=16,
        vocab_size=64,
        text_embed_dim=16,
        max_nodes=300,
        max_edges_per_node=10.0,
        initial_associator_count=4,
        initial_integrator_count=2,
        max_input_tokens=16,
        max_output_tokens=8,
        checkpoint_interval=10_000,
        seed=42,
    )
    seq_cfg = replace(base, use_batched_executor=False)
    bat_cfg = replace(base, use_batched_executor=True)
    print(f"[smoke] seq run (steps={args.steps})...")
    seq_res = run_training(seq_cfg, args.steps, device)
    print(f"[smoke]   -> {seq_res}")
    print(f"[smoke] bat run (steps={args.steps})...")
    bat_res = run_training(bat_cfg, args.steps, device)
    print(f"[smoke]   -> {bat_res}")
    if not seq_res["ok"] or not bat_res["ok"]:
        print("[smoke] FAILED: one of the runs crashed")
        return 1
    seq_h = seq_res["heldout_loss"]
    bat_h = bat_res["heldout_loss"]
    # "within 10%" (batched may be worse OR better by random-seed noise;
    # we care that it's not wildly off).
    rel = abs(bat_h - seq_h) / max(abs(seq_h), 1e-8)
    print(f"[smoke] heldout seq={seq_h:.5f}  bat={bat_h:.5f}  rel-diff={rel:.3%}")
    if rel > 0.10:
        print(f"[smoke] FAILED: heldout relative diff {rel:.1%} > 10%")
        return 1
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
