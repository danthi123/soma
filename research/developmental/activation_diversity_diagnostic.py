"""Diagnostic: how diverse are SOMA's node activations across inputs?

Measures whether different inputs produce measurably different
activation patterns in the graph. If all inputs produce similar
patterns (low diversity), the graph can't discriminate -- and no
fingerprint design will help.

This checks the ROOT CAUSE of graph retrieval underperformance:
is the graph actually learning different representations for
different topics, or are random BPE embeddings too noisy for
the graph to differentiate?
"""
from __future__ import annotations

import time

import torch

from benchmarks.industry.locomo.data_loader import load_dataset
from soma.core.config import SOMAConfig
from soma.core.node import NodeType
from soma.developmental.interaction import InteractionLoop


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    conversations = load_dataset()
    print(f"Loaded {len(conversations)} convs")

    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(f"[{session.date_time}] {turn.speaker}: {turn.text}")

    config = SOMAConfig.developmental()
    loop = InteractionLoop(config=config, llm_model="unused", device=device)
    loop.train_tokenizer(corpus[:500])

    # Develop on all conversations
    print("Developing...", flush=True)
    t0 = time.perf_counter()
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                loop.process_input(text, call_llm=False)
    print(f"Done: {time.perf_counter() - t0:.1f}s")

    ps = loop.predictive_soma

    # 1. Measure activation pattern diversity
    print("\n--- Activation Pattern Analysis ---")

    # Collect per-node activation magnitudes for different topics
    topics = {
        "cooking": ["What do you like to cook?", "Tell me about your favorite recipe",
                     "Do you enjoy baking?"],
        "travel": ["Where have you traveled?", "What's your favorite vacation spot?",
                    "Tell me about your last trip"],
        "work": ["What do you do for work?", "How is your job going?",
                 "Tell me about your career"],
        "family": ["Tell me about your family", "How are your kids?",
                   "What does your spouse do?"],
        "hobbies": ["What are your hobbies?", "Do you play any sports?",
                    "What do you do for fun?"],
    }

    # For each topic, run 3 queries and record which nodes activate
    topic_patterns = {}
    for topic, queries in topics.items():
        patterns = []
        for q in queries:
            qvec = loop.encode_text(q, keep_grad=False)
            modality = ps.config.input_modalities[0]
            ps.soma.step({modality: qvec}, eval_mode=True)
            ps._diversify_activations(qvec)
            ps._apply_lateral_inhibition()

            # Record activation magnitudes per node
            node_mags = {}
            for node in ps.soma.graph.all_nodes():
                if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                    continue
                mag = 0.0
                if node.last_activation is not None:
                    mag = node.last_activation.norm().item()
                node_mags[node.id] = mag
            patterns.append(node_mags)
        topic_patterns[topic] = patterns

    # 2. Check winner consistency within topics
    print("\nWinner nodes per topic (should be consistent within, different across):")
    topic_winners = {}
    for topic, patterns in topic_patterns.items():
        winners = []
        for p in patterns:
            active = sorted(p.items(), key=lambda x: -x[1])[:3]
            winner_ids = tuple(nid for nid, _ in active)
            winners.append(winner_ids)
        topic_winners[topic] = winners

        # Check if same winners across queries in this topic
        unique_winner_sets = len(set(winners))
        consistency = "CONSISTENT" if unique_winner_sets == 1 else f"VARIES ({unique_winner_sets} patterns)"
        print(f"  {topic:10s}: {consistency}")
        for i, w in enumerate(winners):
            short = [nid[-6:] for nid in w]
            print(f"    query {i}: {short}")

    # 3. Cross-topic overlap: do different topics activate different nodes?
    print("\nCross-topic winner overlap:")
    topic_names = list(topic_winners.keys())
    for i in range(len(topic_names)):
        for j in range(i + 1, len(topic_names)):
            t1, t2 = topic_names[i], topic_names[j]
            # Use first query's winners from each topic
            w1 = set(topic_winners[t1][0])
            w2 = set(topic_winners[t2][0])
            overlap = len(w1 & w2)
            print(f"  {t1:10s} vs {t2:10s}: {overlap}/3 shared winners")

    # 4. Fingerprint cosine similarity within vs across topics
    print("\nFingerprint similarity (within-topic vs cross-topic):")
    topic_fps = {}
    for topic, queries in topics.items():
        fps = []
        for q in queries:
            qvec = loop.encode_text(q, keep_grad=False)
            modality = ps.config.input_modalities[0]
            ps.soma.step({modality: qvec}, eval_mode=True)
            ps._diversify_activations(qvec)
            ps._apply_lateral_inhibition()
            fp = ps._get_node_fingerprint()
            fps.append(fp)
        topic_fps[topic] = fps

    # Within-topic similarities
    within_sims = []
    for topic, fps in topic_fps.items():
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = torch.nn.functional.cosine_similarity(
                    fps[i].unsqueeze(0), fps[j].unsqueeze(0)
                ).item()
                within_sims.append(sim)

    # Cross-topic similarities
    cross_sims = []
    for i, t1 in enumerate(topic_names):
        for j in range(i + 1, len(topic_names)):
            t2 = topic_names[j]
            sim = torch.nn.functional.cosine_similarity(
                topic_fps[t1][0].unsqueeze(0), topic_fps[t2][0].unsqueeze(0)
            ).item()
            cross_sims.append(sim)

    avg_within = sum(within_sims) / len(within_sims) if within_sims else 0
    avg_cross = sum(cross_sims) / len(cross_sims) if cross_sims else 0
    print(f"  Within-topic avg similarity: {avg_within:.4f}")
    print(f"  Cross-topic avg similarity:  {avg_cross:.4f}")
    print(f"  Discrimination ratio:        {avg_within / (avg_cross + 1e-8):.2f}x")

    # 5. Raw embedding similarity (is the problem at the input level?)
    print("\nRaw embedding similarity (before graph):")
    topic_embeds = {}
    for topic, queries in topics.items():
        embeds = [loop.encode_text(q, keep_grad=False) for q in queries]
        topic_embeds[topic] = embeds

    within_raw = []
    for topic, embeds in topic_embeds.items():
        for i in range(len(embeds)):
            for j in range(i + 1, len(embeds)):
                sim = torch.nn.functional.cosine_similarity(
                    embeds[i].unsqueeze(0), embeds[j].unsqueeze(0)
                ).item()
                within_raw.append(sim)

    cross_raw = []
    for i, t1 in enumerate(topic_names):
        for j in range(i + 1, len(topic_names)):
            t2 = topic_names[j]
            sim = torch.nn.functional.cosine_similarity(
                topic_embeds[t1][0].unsqueeze(0), topic_embeds[t2][0].unsqueeze(0)
            ).item()
            cross_raw.append(sim)

    avg_within_raw = sum(within_raw) / len(within_raw) if within_raw else 0
    avg_cross_raw = sum(cross_raw) / len(cross_raw) if cross_raw else 0
    print(f"  Within-topic avg similarity: {avg_within_raw:.4f}")
    print(f"  Cross-topic avg similarity:  {avg_cross_raw:.4f}")
    print(f"  Discrimination ratio:        {avg_within_raw / (avg_cross_raw + 1e-8):.2f}x")

    if avg_within / (avg_cross + 1e-8) > avg_within_raw / (avg_cross_raw + 1e-8):
        print("\n  >> Graph AMPLIFIES discrimination vs raw embeddings")
    else:
        print("\n  >> Graph REDUCES discrimination vs raw embeddings")


if __name__ == "__main__":
    main()
