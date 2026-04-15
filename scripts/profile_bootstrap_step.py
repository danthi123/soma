"""Profile where time actually goes in a bootstrap train step.

Isolates SOMA forward, verbalizer forward, LLM forward, and LLM backward
so we can tell whether the batching speedup is being eaten by a serial
SOMA loop or by LLM overhead.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from soma.core.brain_bundle import peek_payload
from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.system import SOMA
from soma.training.verbalizer_bootstrap import compute_lm_loss, text_to_state


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--soma-checkpoint", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--llm-name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--quantization", choices=["none", "int4"], default="none")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--soma-max-tokens", type=int, default=None)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--measure-iters", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = _parse()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Load SOMA
    raw = torch.load(
        str(args.soma_checkpoint / "brain.pt"), map_location="cpu", weights_only=False
    )
    payload = peek_payload(raw)
    cfg = SOMAConfig.from_dict(payload["config"])
    cfg = replace(cfg, bootstrap_max_steps=1)
    soma = SOMA(cfg, device=device)
    tokenizer, encoder, _ = soma.load_bundle(args.soma_checkpoint)
    assert tokenizer is not None and encoder is not None

    # Load LLM
    print(f"Loading {args.llm_name} ({args.dtype}, {args.quantization}) on {device}...")
    hf_tok = AutoTokenizer.from_pretrained(args.llm_name)
    if args.quantization == "none":
        hf_model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(args.llm_name, torch_dtype=dtype),
        ).to(device)
    else:
        from soma.deploy.chat_head_factory import _build_bnb_config

        hf_model = cast(
            Any,
            AutoModelForCausalLM.from_pretrained(
                args.llm_name,
                torch_dtype=dtype,
                quantization_config=_build_bnb_config(args.quantization, dtype),
                device_map={"": device.type},
            ),
        )
    chat_head = ChatHead(model=hf_model, tokenizer=hf_tok)

    # Verbalizer
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name=args.llm_name,
        llm_hidden_dim=chat_head.hidden_size,
        num_prefix_tokens=8,
    )
    verbalizer = SomaVerbalizer(spec).to(device)
    optim = torch.optim.Adam(verbalizer.parameters(), lr=1e-4)

    # Corpus slice
    corpus_text = args.corpus.read_text(encoding="utf-8")
    window_chars = cfg.bootstrap_sample_tokens * 4
    stride = max(window_chars // 2, 1)
    windows = [
        corpus_text[i : i + window_chars]
        for i in range(0, len(corpus_text) - window_chars, stride)
    ]
    batch = windows[: args.batch_size]
    print(f"batch_size={args.batch_size}, window_chars={window_chars}")
    # Show per-window token count for reference.
    sample_ids = hf_tok(batch, padding=True, return_tensors="pt")["input_ids"]
    print(f"tokenized batch shape: {tuple(sample_ids.shape)}")

    # Warmup
    for _ in range(args.warmup_iters):
        states = [
            text_to_state(
                text=t,
                soma=soma,
                tokenizer=tokenizer,
                encoder=encoder,
                soma_output_dim=cfg.sensor_output_dim,
                max_tokens=args.soma_max_tokens,
            )
            for t in batch
        ]
        stacked = torch.cat(states, dim=0)
        prefix = verbalizer(stacked)
        tok_out = hf_tok(batch, padding=True, return_tensors="pt")
        loss = compute_lm_loss(
            chat_head=chat_head,
            prefix=prefix,
            token_ids=tok_out["input_ids"],
            token_attention_mask=tok_out.get("attention_mask"),
        )
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Measured runs
    def _t() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    soma_times: list[float] = []
    verb_times: list[float] = []
    tok_times: list[float] = []
    llm_fwd_times: list[float] = []
    llm_bwd_times: list[float] = []
    optim_times: list[float] = []

    for _ in range(args.measure_iters):
        t0 = _t()
        states = [
            text_to_state(
                text=t,
                soma=soma,
                tokenizer=tokenizer,
                encoder=encoder,
                soma_output_dim=cfg.sensor_output_dim,
                max_tokens=args.soma_max_tokens,
            )
            for t in batch
        ]
        stacked = torch.cat(states, dim=0)
        t1 = _t()
        prefix = verbalizer(stacked)
        t2 = _t()
        tok_out = hf_tok(batch, padding=True, return_tensors="pt")
        token_ids = tok_out["input_ids"]
        attn_mask = tok_out.get("attention_mask")
        t3 = _t()
        optim.zero_grad()
        llm_device = chat_head.model.get_input_embeddings().weight.device
        if token_ids.device != llm_device:
            token_ids = token_ids.to(llm_device)
        if attn_mask is not None and attn_mask.device != llm_device:
            attn_mask = attn_mask.to(llm_device)
        embeds = chat_head.model.get_input_embeddings()(token_ids)
        if prefix.dtype != embeds.dtype:
            prefix = prefix.to(dtype=embeds.dtype)
        inputs_embeds = torch.cat([prefix, embeds], dim=1)
        B, K, _ = prefix.shape
        _, T = token_ids.shape
        prefix_labels = torch.full((B, K), -100, dtype=torch.long, device=llm_device)
        if attn_mask is not None:
            token_labels = token_ids.masked_fill(attn_mask == 0, -100)
            full_attn = torch.cat(
                [torch.ones(B, K, dtype=torch.long, device=llm_device), attn_mask.long()],
                dim=1,
            )
        else:
            token_labels = token_ids
            full_attn = torch.ones(B, K + T, dtype=torch.long, device=llm_device)
        labels = torch.cat([prefix_labels, token_labels], dim=1)
        out = chat_head.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attn,
            labels=labels,
        )
        loss = cast(torch.Tensor, out.loss)
        t4 = _t()
        loss.backward()
        t5 = _t()
        optim.step()
        t6 = _t()

        soma_times.append(t1 - t0)
        verb_times.append(t2 - t1)
        tok_times.append(t3 - t2)
        llm_fwd_times.append(t4 - t3)
        llm_bwd_times.append(t5 - t4)
        optim_times.append(t6 - t5)

    def _summ(vals: list[float], name: str) -> None:
        mean = sum(vals) / len(vals)
        print(f"  {name:12s}: {mean * 1000:7.1f} ms  (range {min(vals) * 1000:6.1f} - {max(vals) * 1000:6.1f})")

    print(f"\nPer-iter timing ({args.measure_iters} iters, batch={args.batch_size}):")
    _summ(soma_times, "SOMA x B")
    _summ(verb_times, "verbalizer")
    _summ(tok_times, "tokenize")
    _summ(llm_fwd_times, "LLM fwd")
    _summ(llm_bwd_times, "LLM bwd")
    _summ(optim_times, "optim.step")
    total = [sum(x) for x in zip(soma_times, verb_times, tok_times, llm_fwd_times, llm_bwd_times, optim_times, strict=True)]
    _summ(total, "TOTAL step")
    iter_mean = sum(total) / len(total)
    print(f"\nThroughput: {1.0 / iter_mean:.2f} steps/sec = {60.0 / iter_mean:.1f} steps/min")
    print(f"            {args.batch_size / iter_mean:.2f} windows/sec = {60.0 * args.batch_size / iter_mean:.1f} windows/min")


if __name__ == "__main__":
    main()
