"""Parameter sweep: find optimal node count, inhibition, training length, dims.

Tests cross-topic differentiation and retrieval precision across the
parameter space. All CPU, no LLM calls.

Usage::

    python -m research.developmental.parameter_sweep
    python -m research.developmental.parameter_sweep --quick
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

RESULTS_DIR = Path("research/developmental/results")

# 10 distinct topics with 3 variants each
TOPICS = {
    "cooking": [
        "I love cooking Italian food especially pasta and risotto",
        "Cooking relaxes me after a long day at work",
        "Italian food is all about fresh ingredients and simplicity",
    ],
    "travel": [
        "I want to visit Japan next spring for cherry blossoms",
        "Rome was amazing the food there was incredible",
        "Barcelona architecture blew my mind especially Gaudi",
    ],
    "music": [
        "Learning guitar has been my hobby for six months",
        "Jazz music is incredibly complex but beautiful",
        "The concert last weekend was one of the best",
    ],
    "cat": [
        "My cat Nebula loves sitting by the window all day",
        "Nebula knocked over my coffee again this morning",
        "Nebula has a funny habit of chirping at birds",
    ],
    "science": [
        "Quantum computing could revolutionize cryptography",
        "The James Webb telescope images are breathtaking",
        "Neuroscience is finally understanding consciousness",
    ],
    "exercise": [
        "I started running three times a week this month",
        "Yoga has really helped with my back pain",
        "Training for a half marathon in October",
    ],
    "grandmother": [
        "My grandmother taught me her secret recipes",
        "Grandmother made the best tomato sauce ever",
        "Grandmother always said cooking is an act of love",
    ],
    "books": [
        "I read a great book about neuroscience yesterday",
        "The novel I finished last week was gripping",
        "Reading before bed helps me wind down",
    ],
}

# Queries per topic (question form)
QUERIES = {
    "cooking": "What do I like to cook",
    "travel": "Where do I want to travel",
    "music": "What musical instrument am I learning",
    "cat": "What is my cat called",
    "science": "What science topics interest me",
    "exercise": "What exercise do I do",
    "grandmother": "Tell me about my grandmother",
    "books": "What books have I read",
}


def build_corpus() -> list[tuple[str, str]]:
    """Return (text, topic) pairs, interleaved across topics."""
    pairs: list[tuple[str, str]] = []
    max_variants = max(len(v) for v in TOPICS.values())
    for vi in range(max_variants):
        for topic, texts in TOPICS.items():
            pairs.append((texts[vi % len(texts)], topic))
    return pairs


def run_config(
    node_count: int,
    inhibition: float,
    train_steps: int,
    embed_dim: int,
) -> dict:
    """Run one configuration and return metrics."""
    # Split node_count into 2/3 associators + 1/3 integrators
    n_integ = max(2, node_count // 3)
    n_assoc = node_count - n_integ

    config = SOMAConfig.developmental(
        text_embed_dim=embed_dim,
        sensor_output_dim=embed_dim,
        initial_associator_count=n_assoc,
        initial_integrator_count=n_integ,
    )

    corpus = build_corpus()
    all_texts = [t for t, _ in corpus]
    tokenizer = train_bpe_tokenizer(all_texts, vocab_size=256)
    encoder = TextEncoder(
        tokenizer, embed_dim=embed_dim, device=torch.device("cpu"),
    )
    ps = PredictiveSOMA(config, device=torch.device("cpu"))

    # Override inhibition ratio
    ps._apply_lateral_inhibition = (
        lambda keep_ratio=inhibition, **kw: PredictiveSOMA._apply_lateral_inhibition(ps, keep_ratio=keep_ratio)
    )
    orig_fp_fn = PredictiveSOMA._get_node_fingerprint
    ps._get_node_fingerprint = (
        lambda inhibition_ratio=inhibition, **kw: orig_fp_fn(ps, inhibition_ratio=inhibition_ratio)
    )

    # Develop
    t0 = time.perf_counter()
    step = 0
    while step < train_steps:
        for text, topic in corpus:
            if step >= train_steps:
                break
            vec = encoder.encode_batch(text).mean(dim=0)
            ps.process_input(vec, source_text=text)
            step += 1
    train_time = time.perf_counter() - t0

    # Measure cross-topic differentiation
    fingerprints: dict[str, torch.Tensor] = {}
    for topic, query_text in QUERIES.items():
        qvec = encoder.encode_batch(query_text).mean(dim=0)
        ps.soma.step({config.input_modalities[0]: qvec}, eval_mode=True)
        fingerprints[topic] = ps._get_node_fingerprint()

    # Pairwise cosine similarities
    topics = sorted(fingerprints.keys())
    sims = []
    for i, a in enumerate(topics):
        for b in topics[i + 1 :]:
            sim = float(
                torch.nn.functional.cosine_similarity(
                    fingerprints[a].unsqueeze(0),
                    fingerprints[b].unsqueeze(0),
                ).item()
            )
            sims.append(sim)
    avg_cross_sim = sum(sims) / len(sims) if sims else 1.0

    # Same-topic similarity (query vs stored statement)
    same_sims = []
    for topic in topics:
        query_fp = fingerprints[topic]
        # Find stored fingerprints for this topic
        for step_num, text in ps.text_store.items():
            if step_num in ps._activation_store:
                # Check if this text belongs to this topic
                if any(
                    t.lower()[:20] in text.lower()
                    for t in TOPICS.get(topic, [])
                ):
                    stored_fp = ps._activation_store[step_num]
                    if stored_fp.shape == query_fp.shape:
                        sim = float(
                            torch.nn.functional.cosine_similarity(
                                query_fp.unsqueeze(0),
                                stored_fp.unsqueeze(0),
                            ).item()
                        )
                        same_sims.append(sim)
                        break
    avg_same_sim = sum(same_sims) / len(same_sims) if same_sims else 0.0

    # Retrieval precision: for each topic query, check if top-3
    # retrieved texts match the topic
    hits = 0
    total = 0
    per_topic_precision: dict[str, float] = {}
    for topic, query_text in QUERIES.items():
        qvec = encoder.encode_batch(query_text).mean(dim=0)
        results = ps.retrieve_by_graph(qvec, top_k=3)
        topic_hits = 0
        for _, text, _ in results:
            total += 1
            topic_words = set()
            for t in TOPICS.get(topic, []):
                topic_words.update(t.lower().split()[:5])
            if any(w in text.lower() for w in topic_words):
                hits += 1
                topic_hits += 1
        per_topic_precision[topic] = topic_hits / max(len(results), 1)

    precision = hits / total if total > 0 else 0.0

    # Discrimination score: how much more similar is same-topic
    # vs cross-topic? Higher = better differentiation.
    discrimination = avg_same_sim - avg_cross_sim

    n_final = len(ps.soma.graph.nodes)
    e_final = len(ps.soma.graph.edges)

    return {
        "node_count": node_count,
        "inhibition": inhibition,
        "train_steps": train_steps,
        "embed_dim": embed_dim,
        "final_nodes": n_final,
        "final_edges": e_final,
        "avg_cross_sim": round(avg_cross_sim, 4),
        "avg_same_sim": round(avg_same_sim, 4),
        "discrimination": round(discrimination, 4),
        "retrieval_precision": round(precision, 4),
        "per_topic_precision": {
            k: round(v, 2) for k, v in per_topic_precision.items()
        },
        "train_time_s": round(train_time, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Developmental parameter sweep",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Reduced sweep for testing")
    args = parser.parse_args()

    if args.quick:
        node_counts = [14, 50]
        inhibitions = [0.2, 0.3]
        train_lengths = [100, 500]
        embed_dims = [64]
    else:
        node_counts = [14, 50, 100, 200]
        inhibitions = [0.1, 0.2, 0.3, 0.5]
        train_lengths = [100, 300, 1000, 3000]
        embed_dims = [64, 128]

    combos = list(product(node_counts, inhibitions, train_lengths, embed_dims))
    print(f"Sweep: {len(combos)} configurations", flush=True)
    print(f"  Nodes: {node_counts}", flush=True)
    print(f"  Inhibition: {inhibitions}", flush=True)
    print(f"  Train steps: {train_lengths}", flush=True)
    print(f"  Embed dims: {embed_dims}", flush=True)
    print(flush=True)

    results: list[dict] = []
    t0 = time.perf_counter()

    for i, (nodes, inhib, steps, dim) in enumerate(combos):
        label = f"n={nodes:<3d} inh={inhib:.1f} steps={steps:<4d} dim={dim}"
        print(f"[{i + 1:>3d}/{len(combos)}] {label} ... ", end="", flush=True)

        try:
            r = run_config(nodes, inhib, steps, dim)
            results.append(r)
            print(
                f"cross={r['avg_cross_sim']:.4f} "
                f"disc={r['discrimination']:+.4f} "
                f"prec={r['retrieval_precision']:.2f} "
                f"({r['train_time_s']:.1f}s)",
                flush=True,
            )
        except Exception as exc:
            print(f"FAILED: {exc}", flush=True)

    total_time = time.perf_counter() - t0

    # Sort by retrieval precision, then discrimination
    results.sort(key=lambda r: (-r["retrieval_precision"], -r["discrimination"]))

    # Top 10
    print(f"\n{'=' * 80}", flush=True)
    print("TOP 10 CONFIGURATIONS", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(
        f"{'Rank':>4s} {'Nodes':>5s} {'Inhib':>5s} {'Steps':>5s} "
        f"{'Dim':>3s} {'Cross':>6s} {'Discrim':>8s} {'Prec':>5s} "
        f"{'Edges':>5s} {'Time':>5s}",
        flush=True,
    )
    print("-" * 60, flush=True)
    for rank, r in enumerate(results[:10], 1):
        print(
            f"{rank:>4d} {r['node_count']:>5d} {r['inhibition']:>5.1f} "
            f"{r['train_steps']:>5d} {r['embed_dim']:>3d} "
            f"{r['avg_cross_sim']:>6.4f} {r['discrimination']:>+8.4f} "
            f"{r['retrieval_precision']:>5.2f} "
            f"{r['final_edges']:>5d} {r['train_time_s']:>5.1f}s",
            flush=True,
        )

    # Per-variable analysis
    print(f"\n{'=' * 80}", flush=True)
    print("PER-VARIABLE ANALYSIS (avg retrieval precision)", flush=True)
    print(f"{'=' * 80}", flush=True)

    for var_name, var_values, getter in [
        ("Nodes", node_counts, lambda r: r["node_count"]),
        ("Inhibition", inhibitions, lambda r: r["inhibition"]),
        ("Train steps", train_lengths, lambda r: r["train_steps"]),
        ("Embed dim", embed_dims, lambda r: r["embed_dim"]),
    ]:
        print(f"\n{var_name}:")
        for val in var_values:
            subset = [r for r in results if getter(r) == val]
            if subset:
                avg_prec = sum(r["retrieval_precision"] for r in subset) / len(subset)
                avg_disc = sum(r["discrimination"] for r in subset) / len(subset)
                print(f"  {val:>6}: precision={avg_prec:.3f}, discrimination={avg_disc:+.4f}")

    print(f"\nTotal sweep time: {total_time:.0f}s")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "parameter_sweep.json"
    with open(out_path, "w") as f:
        json.dump({
            "sweep_config": {
                "node_counts": node_counts,
                "inhibitions": inhibitions,
                "train_lengths": train_lengths,
                "embed_dims": embed_dims,
            },
            "results": results,
            "total_time_s": round(total_time, 1),
        }, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
