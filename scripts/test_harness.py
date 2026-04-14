"""B-tier test harness for the autonomous improvement loop.

Loads a SOMA checkpoint and produces a tick report summarizing:
- Held-out loss + perplexity
- Output-distribution KL vs corpus reference
- Graph structure (nodes, edges, avg degree)
- Memory occupancy (WM, episodic)
- Recent training EMAs (loss, curiosity)
- Growth event counts over the last 1k steps
- Health flags: observations like ``nan_detected``, ``graph_collapsed``,
  ``wm_pinned_high``

Output JSON schema: see docs/plans/2026-04-12-autonomous-loop-design.md
Appendix B.4. A companion JSONL file captures per-prompt chat traces
(``tick-<tick_id>/chat.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---- Pure helpers (unit-tested) --------------------------------------------


def compute_kl_divergence(
    out_freq: Mapping[str, int | float], ref_freq: Mapping[str, int | float]
) -> float:
    """Compute ``KL(out || ref)`` after normalizing both sides.

    Uses a small epsilon floor for zero-probability reference tokens (not full
    additive smoothing) so that proportionally-identical distributions yield
    exactly zero divergence. Returns 0.0 when either input is empty.
    """
    if not out_freq or not ref_freq:
        return 0.0
    vocab = set(out_freq) | set(ref_freq)
    total_out = float(sum(out_freq.values()))
    total_ref = float(sum(ref_freq.values()))
    if total_out <= 0.0 or total_ref <= 0.0:
        return 0.0
    eps = 1e-10
    kl = 0.0
    for k in vocab:
        p = float(out_freq.get(k, 0)) / total_out
        q = float(ref_freq.get(k, 0)) / total_ref
        if p > 0.0:
            kl += p * math.log(p / max(q, eps))
    return max(0.0, kl)


def compute_ema(values: Sequence[float], *, alpha: float = 0.1) -> float:
    """Exponential moving average of ``values`` with factor ``alpha``.

    Returns ``nan`` on empty input. Uses ``y_t = alpha*x_t + (1-alpha)*y_{t-1}``
    seeded with the first sample.
    """
    if not values:
        return float("nan")
    ema = float(values[0])
    for v in values[1:]:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        ema = alpha * float(v) + (1.0 - alpha) * ema
    return ema


def summarize_growth(records: Sequence[dict[str, Any]], *, lookback_steps: int) -> dict[str, int]:
    """Count neurogenesis / synaptogenesis / pruning events in the last
    ``lookback_steps`` of training.

    Classifies each step-to-step delta the same way TrainingController does:
    - ``delta_nodes > 0``        → neurogenesis
    - ``delta_nodes < 0`` OR
      ``delta_edges < 0``        → pruning
    - ``delta_edges > 0``        → synaptogenesis
    """
    counts = {"syn": 0, "neuro": 0, "prune": 0}
    if len(records) < 2:
        return counts

    max_step = max(int(r.get("step", 0)) for r in records)
    window_start = max_step - lookback_steps
    prev: dict[str, Any] | None = None
    for rec in records:
        step = int(rec.get("step", 0))
        if step < window_start:
            prev = rec
            continue
        if prev is None:
            prev = rec
            continue
        delta_nodes = int(rec.get("num_nodes", 0)) - int(prev.get("num_nodes", 0))
        delta_edges = int(rec.get("num_edges", 0)) - int(prev.get("num_edges", 0))
        if delta_nodes > 0:
            counts["neuro"] += 1
        elif delta_nodes < 0 or delta_edges < 0:
            counts["prune"] += 1
        elif delta_edges > 0:
            counts["syn"] += 1
        prev = rec
    return counts


def compute_health_flags(report: dict[str, Any]) -> list[str]:
    """Return a list of health flag strings observed in ``report``."""
    flags: list[str] = []
    loss = report.get("heldout_loss_mean")
    if loss is not None and isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss)):
        flags.append("nan_detected")
    graph = report.get("graph", {}) or {}
    nodes = int(graph.get("nodes", 0))
    edges = int(graph.get("edges", 0))
    if nodes < 10 or edges < 5:
        flags.append("graph_collapsed")
    memory = report.get("memory", {}) or {}
    wm_occ = float(memory.get("wm_occupancy", 0.0))
    if wm_occ >= 0.95:
        flags.append("wm_pinned_high")
    ep_pct = float(memory.get("episodic_fill_pct", 0.0))
    if ep_pct >= 0.99:
        flags.append("episodic_saturated")
    return flags


# ---- Integration helpers ---------------------------------------------------


def load_token_freq(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def read_recent_metrics(path: Path, *, max_records: int = 2000) -> list[dict[str, Any]]:
    """Read the last ``max_records`` lines of the metrics JSONL file."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()
    tail = lines[-max_records:]
    records: list[dict[str, Any]] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _load_corpus(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _build_encoder_decoder(
    corpus: Sequence[str], *, embed_dim: int, max_seq_len: int, vocab_size: int, device: Any
) -> tuple[Any, Any]:
    from soma.io.text_decoder import TextDecoder
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

    tokenizer = train_bpe_tokenizer(corpus, vocab_size=vocab_size)
    encoder = TextEncoder(tokenizer, embed_dim=embed_dim, max_seq_len=max_seq_len, device=device)
    decoder = TextDecoder(tokenizer, embed_dim=embed_dim, device=device)
    decoder.tie_weights(encoder)
    return encoder, decoder


def _evaluate_heldout(
    soma: Any,
    encoder: Any,
    heldout_lines: Sequence[str],
    *,
    first_out: str,
    max_tokens_per_line: int,
) -> tuple[float, int]:
    """Run heldout lines through SOMA and average the token losses.

    Returns (mean_loss, num_token_pairs). Caller is responsible for ensuring
    ``soma`` is the harness-owned copy (mutation during eval is acceptable).
    """
    import torch  # noqa: F401

    losses: list[float] = []
    for text in heldout_lines:
        ids = encoder.tokenize(text)
        if len(ids) < 2:
            continue
        embeds = encoder.encode_batch(text)
        if embeds.ndim != 2 or embeds.shape[0] < 2:
            continue
        first_in = soma.config.input_modalities[0]
        limit = min(max_tokens_per_line, embeds.shape[0] - 1)
        for t in range(limit):
            inputs = {first_in: embeds[t].detach()}
            target = {first_out: embeds[t + 1].detach()}
            try:
                result = soma.step(inputs=inputs, targets=target, eval_mode=True)
            except Exception as exc:  # noqa: BLE001 — eval failure is a harness bug
                print(f"test_harness: step raised during eval: {exc}", file=sys.stderr)
                return float("nan"), 0
            loss = result.get("loss")
            if loss is None:
                continue
            if math.isnan(loss) or math.isinf(loss):
                return float(loss), len(losses)
            losses.append(float(loss))
    if not losses:
        return float("nan"), 0
    return sum(losses) / len(losses), len(losses)


def _output_token_distribution(
    soma: Any,
    encoder: Any,
    decoder: Any,
    seed_prompts: Sequence[str],
    *,
    max_out_tokens: int,
) -> dict[str, int]:
    """Generate short continuations from ``seed_prompts`` and count decoded tokens."""
    counter: Counter[str] = Counter()
    for prompt in seed_prompts:
        if not prompt:
            continue
        try:
            out = soma.interactive_session(
                prompt,
                text_encoder=encoder.encode_batch,
                text_decoder=decoder.decode,
                max_output_tokens=max_out_tokens,
            )
        except Exception:  # noqa: BLE001 — individual prompt failure isn't fatal
            continue
        for tok in out.split():
            if tok:
                counter[tok] += 1
    return dict(counter)


def _run_fixed_prompts(
    soma: Any,
    encoder: Any,
    decoder: Any,
    prompts: Sequence[str],
    *,
    max_out_tokens: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        entry: dict[str, Any] = {
            "index": idx,
            "prompt": prompt,
            "response": None,
            "error": None,
            "duration_s": None,
        }
        if not prompt:
            entry["response"] = ""
            records.append(entry)
            continue
        start = time.time()
        try:
            entry["response"] = soma.interactive_session(
                prompt,
                text_encoder=encoder.encode_batch,
                text_decoder=decoder.decode,
                max_output_tokens=max_out_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)
        entry["duration_s"] = round(time.time() - start, 3)
        records.append(entry)
    return records


def _read_base_commit_sha() -> str:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


# ---- Main -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOMA B-tier test harness.")
    parser.add_argument("--out", type=Path, required=True, help="Report JSON output path.")
    parser.add_argument("--chat-out", type=Path, required=True, help="Chat JSONL output path.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/current.pt"))
    parser.add_argument("--corpus", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--heldout", type=Path, default=Path("data/heldout.txt"))
    parser.add_argument("--fixed-prompts", type=Path, default=Path("data/fixed_prompts.txt"))
    parser.add_argument(
        "--token-freq",
        type=Path,
        default=Path("data/corpus_token_freq.json"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(".soma-loop/metrics/metrics.current.jsonl"),
    )
    parser.add_argument("--max-gen-tokens", type=int, default=32)
    parser.add_argument("--heldout-max-lines", type=int, default=200)
    parser.add_argument("--heldout-max-tokens-per-line", type=int, default=16)
    parser.add_argument("--tick-id", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=(
            "Eval device. Defaults to cpu: harness inference is cheap "
            "and running on cpu avoids (a) contention with the training "
            "service on GPU and (b) CUDA device-side asserts on NaN/Inf "
            "losses that would otherwise kill the harness process "
            "asynchronously at the next sync point."
        ),
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint {args.checkpoint} missing", file=sys.stderr)
        return 1

    import torch

    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    corpus_lines = _load_corpus(args.corpus)
    if not corpus_lines:
        print(f"ERROR: corpus {args.corpus} has no lines", file=sys.stderr)
        return 1

    # Load the checkpoint first to learn config parameters (notably vocab_size
    # and embed_dim), then rebuild encoder/decoder with matching shapes.
    from soma.core.brain_bundle import peek_payload

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = peek_payload(raw)
    ckpt_config = SOMAConfig.from_dict(state["config"])
    soma = SOMA(ckpt_config, device=device)
    soma.load_state(args.checkpoint)
    checkpoint_step = int(soma.global_step)

    encoder, decoder = _build_encoder_decoder(
        corpus_lines,
        embed_dim=ckpt_config.text_embed_dim,
        max_seq_len=ckpt_config.max_input_tokens,
        vocab_size=ckpt_config.vocab_size,
        device=device,
    )

    # Load trained encoder weights from the sidecar train_service wrote
    # alongside the checkpoint. Without this, the harness's fresh-random
    # embeddings have no relationship to the ones SOMA was trained
    # against and heldout_loss is effectively noise. Decoder weights are
    # tied to the encoder via tie_weights() in _build_encoder_decoder,
    # so loading the encoder state dict also fixes decode-time logits.
    encoder_sidecar = args.checkpoint.with_suffix("").with_suffix(".encoder.pt")
    if not encoder_sidecar.exists():
        encoder_sidecar = args.checkpoint.parent / "current.encoder.pt"
    if encoder_sidecar.exists():
        try:
            encoder.load_state_dict(
                torch.load(encoder_sidecar, map_location=device, weights_only=True)
            )
            encoder.to(device)
            print(
                f"test_harness: loaded encoder sidecar {encoder_sidecar.name}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"test_harness: encoder sidecar load failed ({exc}); "
                f"heldout_loss will reflect random-init encoder drift",
                file=sys.stderr,
            )

    first_out = ckpt_config.output_modalities[0]

    # Heldout evaluation
    heldout_lines = _load_corpus(args.heldout)[: args.heldout_max_lines]
    heldout_loss, num_pairs = _evaluate_heldout(
        soma,
        encoder,
        heldout_lines,
        first_out=first_out,
        max_tokens_per_line=args.heldout_max_tokens_per_line,
    )
    heldout_perplexity = (
        math.exp(heldout_loss)
        if not math.isnan(heldout_loss) and not math.isinf(heldout_loss) and heldout_loss < 700
        else float("nan")
    )

    # Output-distribution KL
    ref_freq = load_token_freq(args.token_freq)
    seed_prompts = [ln for ln in corpus_lines[:20] if ln]
    output_freq = _output_token_distribution(
        soma, encoder, decoder, seed_prompts, max_out_tokens=args.max_gen_tokens
    )
    kl = compute_kl_divergence(output_freq, ref_freq)

    # Fixed prompts
    fixed_prompts: list[str] = []
    if args.fixed_prompts.exists():
        fixed_prompts = args.fixed_prompts.read_text(encoding="utf-8").splitlines()
    chat_records = _run_fixed_prompts(
        soma, encoder, decoder, fixed_prompts, max_out_tokens=args.max_gen_tokens
    )

    # Training-series metrics
    metrics_records = read_recent_metrics(args.metrics)
    recent_losses = [float(r["loss"]) for r in metrics_records if r.get("loss") is not None]
    recent_curs = [float(r.get("curiosity", 0.0)) for r in metrics_records]
    recent_lr = [float(r.get("lr_multiplier", 1.0)) for r in metrics_records]
    growth = summarize_growth(metrics_records, lookback_steps=1000)

    graph = soma.graph
    edges_count = int(graph.num_edges)
    nodes_count = int(graph.num_nodes)
    avg_degree = (2.0 * edges_count / nodes_count) if nodes_count > 0 else 0.0

    wm_occ = float(soma.working_memory.occupancy())
    ep_valid = int(soma.episodic_memory.num_valid)
    ep_cap = int(ckpt_config.episodic_capacity)
    ep_fill = (ep_valid / ep_cap) if ep_cap > 0 else 0.0

    last_loss = recent_losses[-1] if recent_losses else float("nan")

    report: dict[str, Any] = {
        "tick_id": int(args.tick_id),
        "ts": datetime.now(UTC).isoformat(),
        "base_commit_sha": _read_base_commit_sha(),
        "global_step": checkpoint_step,
        "heldout_loss_mean": (float(heldout_loss) if not math.isnan(heldout_loss) else None),
        "heldout_loss_num_pairs": int(num_pairs),
        "heldout_perplexity": (
            float(heldout_perplexity) if not math.isnan(heldout_perplexity) else None
        ),
        "output_distribution_kl": float(kl),
        "graph": {
            "nodes": nodes_count,
            "edges": edges_count,
            "avg_degree": float(avg_degree),
        },
        "memory": {
            "wm_occupancy": wm_occ,
            "episodic_entries": ep_valid,
            "episodic_fill_pct": float(ep_fill),
        },
        "training": {
            "last_loss": last_loss if recent_losses else None,
            "loss_ema_500": compute_ema(recent_losses[-500:], alpha=0.1) if recent_losses else None,
            "curiosity_ema_500": compute_ema(recent_curs[-500:], alpha=0.1)
            if recent_curs
            else None,
            "lr_multiplier": recent_lr[-1] if recent_lr else 1.0,
        },
        "growth_last_1k_steps": growth,
    }
    report["health_flags"] = compute_health_flags(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    args.chat_out.parent.mkdir(parents=True, exist_ok=True)
    with args.chat_out.open("w", encoding="utf-8") as fh:
        for rec in chat_records:
            fh.write(json.dumps(rec) + "\n")

    print(
        f"test_harness: OK tick={args.tick_id} step={checkpoint_step} "
        f"loss={heldout_loss:.4f} kl={kl:.4f} prompts={len(chat_records)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
