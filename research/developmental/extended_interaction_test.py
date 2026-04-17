"""Extended interaction test — measure SOMA development over 200+ inputs.

Feeds diverse text through PredictiveSOMA (no LLM calls) and tracks:
- Prediction error trajectory (should decrease for recurring topics)
- Graph growth (nodes, edges over time)
- Working memory usage
- Episodic memory accumulation
- Novelty response (should decrease for familiar topics, spike for new ones)

Usage::

    python -m research.developmental.extended_interaction_test
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop
from soma.developmental.verbalize import verbalize_state

RESULTS_DIR = Path("research/developmental/results")

# Diverse conversation corpus — 5 topic clusters + noise
TOPIC_COOKING = [
    "I love making pasta from scratch, especially fettuccine.",
    "My grandmother taught me her secret tomato sauce recipe.",
    "Italian cooking is all about fresh ingredients and simplicity.",
    "I tried making risotto last night but it turned out too thick.",
    "The best pizza I ever had was in Naples, so simple but perfect.",
    "Cooking relaxes me after a long day at work.",
    "I need to buy more olive oil, I use it in everything.",
    "Fresh basil makes such a difference in any Italian dish.",
    "I'm experimenting with homemade pasta shapes this week.",
    "My carbonara is getting better, the key is tempering the eggs.",
]

TOPIC_TRAVEL = [
    "I want to visit Japan next spring for the cherry blossoms.",
    "Rome was incredible, the history is everywhere you look.",
    "I prefer traveling slow, staying in one city for a week.",
    "The best part of travel is trying local street food.",
    "I'm saving up for a trip to New Zealand next year.",
    "Barcelona's architecture blew my mind, especially Gaudi.",
    "Traveling solo taught me so much about myself.",
    "I always try to learn basic phrases in the local language.",
    "The train system in Switzerland is amazingly efficient.",
    "My dream is to see the Northern Lights in Iceland.",
]

TOPIC_MUSIC = [
    "I've been learning guitar for about six months now.",
    "Jazz piano is incredibly complex but beautiful.",
    "I went to an amazing concert last weekend, live music is special.",
    "My playlist is mostly indie rock and electronic.",
    "Learning music theory changed how I listen to songs.",
    "I practice scales every morning before work.",
    "The Beatles really did change music forever.",
    "I want to try writing my own songs this year.",
    "Vinyl records have such a warm sound compared to digital.",
    "I joined a local jam session group, it's so fun.",
]

TOPIC_SCIENCE = [
    "Quantum computing could revolutionize cryptography.",
    "The James Webb telescope images are breathtaking.",
    "I've been reading about CRISPR gene editing technology.",
    "Climate models are getting more accurate every year.",
    "Neuroscience is finally understanding consciousness better.",
    "The discovery of gravitational waves was historic.",
    "Machine learning is transforming drug discovery.",
    "I find evolutionary biology absolutely fascinating.",
    "The ocean floor is less explored than Mars.",
    "Fusion energy might actually be viable within decades.",
]

TOPIC_EXERCISE = [
    "I started running three times a week this month.",
    "Yoga has really helped with my back pain.",
    "I'm training for a half marathon in October.",
    "Swimming is the best full-body workout in my opinion.",
    "I bought a new pair of trail running shoes.",
    "Strength training twice a week has made a huge difference.",
    "Morning workouts give me energy for the whole day.",
    "I tried rock climbing for the first time and loved it.",
    "Recovery days are just as important as training days.",
    "My running pace has improved a lot since I started intervals.",
]

NOVEL_INPUTS = [
    "I just adopted a rescue cat named Nebula.",
    "The aurora borealis was visible from my backyard last night.",
    "I'm considering switching careers to marine biology.",
    "My neighbor builds model trains in his garage.",
    "I discovered a hidden waterfall on a hike yesterday.",
]


def build_interaction_sequence() -> list[tuple[str, str]]:
    """Build a 200+ input sequence that tests development.

    Structure:
    - Phase 1 (0-50): Introduce cooking and travel topics
    - Phase 2 (50-100): Add music, revisit cooking/travel
    - Phase 3 (100-150): Add science, revisit all previous
    - Phase 4 (150-200): Add exercise, inject novel inputs
    - Phase 5 (200-220): Revisit earliest topics (test long-term memory)
    """
    sequence: list[tuple[str, str]] = []

    # Phase 1: cooking + travel (interleaved)
    for i in range(25):
        sequence.append((TOPIC_COOKING[i % len(TOPIC_COOKING)], "cooking"))
        sequence.append((TOPIC_TRAVEL[i % len(TOPIC_TRAVEL)], "travel"))

    # Phase 2: add music, revisit cooking/travel
    for i in range(25):
        sequence.append((TOPIC_MUSIC[i % len(TOPIC_MUSIC)], "music"))
        sequence.append((TOPIC_COOKING[(i + 5) % len(TOPIC_COOKING)], "cooking"))

    # Phase 3: add science, mix in travel + music
    for i in range(25):
        sequence.append((TOPIC_SCIENCE[i % len(TOPIC_SCIENCE)], "science"))
        sequence.append((TOPIC_TRAVEL[(i + 3) % len(TOPIC_TRAVEL)], "travel"))

    # Phase 4: add exercise, inject novel inputs
    for i in range(25):
        sequence.append((TOPIC_EXERCISE[i % len(TOPIC_EXERCISE)], "exercise"))
        if i < len(NOVEL_INPUTS):
            sequence.append((NOVEL_INPUTS[i], "novel"))
        else:
            sequence.append((TOPIC_MUSIC[(i + 2) % len(TOPIC_MUSIC)], "music"))

    # Phase 5: revisit earliest topics
    for i in range(10):
        sequence.append((TOPIC_COOKING[i % len(TOPIC_COOKING)], "cooking"))
        sequence.append((TOPIC_TRAVEL[i % len(TOPIC_TRAVEL)], "travel"))

    return sequence


def main() -> None:
    print("=== Extended Interaction Test ===", flush=True)

    device = torch.device("cpu")  # GPU is busy with per-item analysis
    print(f"Device: {device}", flush=True)

    config = SOMAConfig.developmental()

    loop = InteractionLoop(
        config=config,
        llm_model="unused",
        llm_api_base="http://localhost:11434",
        device=device,
    )

    # Build corpus for tokenizer from all topics
    all_texts = (
        TOPIC_COOKING + TOPIC_TRAVEL + TOPIC_MUSIC
        + TOPIC_SCIENCE + TOPIC_EXERCISE + NOVEL_INPUTS
    )
    loop.train_tokenizer(all_texts)

    sequence = build_interaction_sequence()
    print(f"Sequence length: {len(sequence)} inputs", flush=True)
    print(flush=True)

    # Track per-phase metrics
    phase_boundaries = [
        (0, 50, "Phase 1: cooking+travel"),
        (50, 100, "Phase 2: +music, revisit"),
        (100, 150, "Phase 3: +science, mix"),
        (150, 200, "Phase 4: +exercise, novel"),
        (200, len(sequence), "Phase 5: revisit earliest"),
    ]

    t0 = time.perf_counter()
    topic_errors: dict[str, list[float]] = {}

    for i, (text, topic) in enumerate(sequence):
        result = loop.process_input(text, call_llm=False)

        # Track per-topic prediction error
        if topic not in topic_errors:
            topic_errors[topic] = []
        topic_errors[topic].append(result["prediction_error"])

        # Print progress every 25 inputs
        if (i + 1) % 25 == 0:
            soma = loop.predictive_soma.soma
            n_nodes = len(soma.graph.nodes)
            n_edges = len(soma.graph.edges)
            wm_used = int((soma.working_memory.usage > 0.1).sum().item())
            ep_count = int(soma.episodic_memory.valid.sum().item())
            recent_errors = list(loop.predictive_soma.error_history)[-25:]
            avg_error = sum(recent_errors) / len(recent_errors)

            print(
                f"[{i + 1:>3d}/{len(sequence)}] "
                f"nodes={n_nodes:>3d} edges={n_edges:>3d} "
                f"wm={wm_used:>2d}/32 ep={ep_count:>3d} "
                f"avg_pred_err={avg_error:.4f} "
                f"novelty={result['novelty']:.3f}",
                flush=True,
            )

    total_time = time.perf_counter() - t0

    # Summary
    print(flush=True)
    print(f"{'=' * 60}", flush=True)
    print("DEVELOPMENT SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Total time: {total_time:.1f}s ({total_time / len(sequence):.2f}s/input)")
    print(flush=True)

    # Final graph state
    soma = loop.predictive_soma.soma
    print(f"Final graph: {len(soma.graph.nodes)} nodes, {len(soma.graph.edges)} edges")
    print(f"Working memory: {int((soma.working_memory.usage > 0.1).sum().item())}/32 slots")
    print(f"Episodic memory: {int(soma.episodic_memory.valid.sum().item())}/{soma.episodic_memory.capacity}")
    print(flush=True)

    # Per-phase prediction error
    print("Per-phase avg prediction error:")
    for start, end, label in phase_boundaries:
        errors = list(loop.predictive_soma.error_history)[start:end]
        if errors:
            avg = sum(errors) / len(errors)
            print(f"  {label}: {avg:.4f}")
    print(flush=True)

    # Per-topic prediction error (first 5 vs last 5 exposures)
    print("Per-topic prediction error (first 5 vs last 5 exposures):")
    for topic in ["cooking", "travel", "music", "science", "exercise", "novel"]:
        errs = topic_errors.get(topic, [])
        if len(errs) >= 10:
            first5 = sum(errs[:5]) / 5
            last5 = sum(errs[-5:]) / 5
            delta = last5 - first5
            print(f"  {topic:<10s}: first5={first5:.4f} last5={last5:.4f} delta={delta:+.4f}")
        elif errs:
            avg = sum(errs) / len(errs)
            print(f"  {topic:<10s}: avg={avg:.4f} (only {len(errs)} exposures)")
    print(flush=True)

    # Verbalized state at end
    print("Final SOMA state:")
    print(verbalize_state(soma))
    print(flush=True)

    # Save detailed tracker data
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tracker_path = RESULTS_DIR / "extended_interaction_tracker.json"
    loop.tracker.save(tracker_path)

    # Save summary
    summary_path = RESULTS_DIR / "extended_interaction_summary.json"
    summary = {
        "total_inputs": len(sequence),
        "total_time_s": round(total_time, 1),
        "final_nodes": len(soma.graph.nodes),
        "final_edges": len(soma.graph.edges),
        "final_wm_used": int((soma.working_memory.usage > 0.1).sum().item()),
        "final_episodic": int(soma.episodic_memory.valid.sum().item()),
        "tracker_summary": loop.tracker.summary(),
        "per_topic_errors": {
            t: {"count": len(e), "mean": sum(e) / len(e)}
            for t, e in topic_errors.items()
        },
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
