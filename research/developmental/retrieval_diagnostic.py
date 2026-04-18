"""Diagnose retrieval quality at various development lengths.

Measures:
1. Fingerprint diversity — are fingerprints for different topics distinguishable?
2. Retrieval accuracy — does the right memory come back for a related query?
3. How many development steps are needed for reliable retrieval?
"""
from __future__ import annotations

import sys

import torch

from soma.core.config import SOMAConfig

# Topically distinct development corpus
TOPICS = {
    "cooking": [
        "I love cooking Italian pasta dishes",
        "my favorite recipe is homemade lasagna",
        "cooking with fresh herbs makes everything better",
        "I learned to make bread during the pandemic",
        "spaghetti carbonara is my comfort food",
    ],
    "sports": [
        "I play basketball every weekend",
        "running a marathon was my biggest achievement",
        "swimming is great exercise for the whole body",
        "our local soccer team won the championship",
        "tennis requires both skill and endurance",
    ],
    "music": [
        "I play guitar in a small band",
        "jazz music helps me relax after work",
        "learning piano takes years of practice",
        "classical music inspires my creativity",
        "I went to an amazing concert last night",
    ],
    "travel": [
        "visiting Japan was a life changing experience",
        "I love exploring ancient ruins in Greece",
        "backpacking through Europe taught me independence",
        "the beaches in Thailand are absolutely stunning",
        "road trips across America are the best vacations",
    ],
}

# Queries mapped to expected topic
QUERIES = {
    "What do I like to cook?": "cooking",
    "What sports do I play?": "sports",
    "Do I play any instruments?": "music",
    "Where have I traveled?": "travel",
    "What is my favorite recipe?": "cooking",
    "What exercise do I do?": "sports",
    "What kind of music do I listen to?": "music",
    "What countries have I visited?": "travel",
}


def run_diagnostic(n_repeats: int = 1, train_enc: bool = False) -> None:
    config = SOMAConfig.developmental()

    # Build corpus
    all_texts = []
    text_topics: dict[str, str] = {}
    for topic, texts in TOPICS.items():
        for t in texts:
            all_texts.append(t)
            text_topics[t] = topic

    # Use InteractionLoop to test with trainable encoder
    from soma.developmental.interaction import InteractionLoop

    loop = InteractionLoop(
        config=config,
        llm_model="unused",
        device=torch.device("cpu"),
        train_encoder=train_enc,
        encoder_lr=0.001,
    )
    loop.train_tokenizer(all_texts, vocab_size=2000)
    ps = loop.predictive_soma

    print(f"Encoder training: {'ON' if train_enc else 'OFF'}")
    print(f"Graph: {len(ps.soma.graph.nodes)} nodes, {len(ps.soma.graph.edges)} edges")
    print(f"Corpus: {len(all_texts)} texts across {len(TOPICS)} topics")
    print()

    # Develop on the corpus, grouped by topic.
    # In a real conversation, topics are temporally clustered —
    # a cooking discussion has multiple cooking sentences in sequence.
    # This temporal structure lets the encoder learn that words
    # appearing in similar contexts should have similar embeddings.
    topic_texts = list(TOPICS.values())
    total_inputs = len(all_texts) * n_repeats
    for _rep in range(n_repeats):
        for texts_in_topic in topic_texts:
            for text in texts_in_topic:
                loop.process_input(text, call_llm=False)

    print(f"After {total_inputs} development steps:")
    print(f"  Graph: {len(ps.soma.graph.nodes)} nodes, {len(ps.soma.graph.edges)} edges")
    print(f"  Memories: {len(ps.text_store)}")
    print(f"  Pred error (recent): {ps.prediction_error:.6f}")
    print()

    # Measure fingerprint diversity
    print("=== Fingerprint Diversity ===")
    topic_fps: dict[str, list[torch.Tensor]] = {t: [] for t in TOPICS}
    for step, text in ps.text_store.items():
        topic = text_topics.get(text)
        if topic and step in ps._activation_store:
            topic_fps[topic].append(ps._activation_store[step])

    # Within-topic vs between-topic similarity
    within_sims = []
    between_sims = []
    topics_list = list(TOPICS.keys())
    for i, t1 in enumerate(topics_list):
        fps1 = topic_fps[t1]
        for j in range(len(fps1)):
            for k in range(j + 1, len(fps1)):
                sim = torch.nn.functional.cosine_similarity(
                    fps1[j].unsqueeze(0), fps1[k].unsqueeze(0)
                ).item()
                within_sims.append(sim)
        for t2 in topics_list[i + 1:]:
            fps2 = topic_fps[t2]
            for fp1 in fps1:
                for fp2 in fps2:
                    sim = torch.nn.functional.cosine_similarity(
                        fp1.unsqueeze(0), fp2.unsqueeze(0)
                    ).item()
                    between_sims.append(sim)

    if within_sims and between_sims:
        avg_within = sum(within_sims) / len(within_sims)
        avg_between = sum(between_sims) / len(between_sims)
        print(f"  Within-topic similarity:  {avg_within:.4f} (n={len(within_sims)})")
        print(f"  Between-topic similarity: {avg_between:.4f} (n={len(between_sims)})")
        print(f"  Discrimination ratio:     {avg_within / max(avg_between, 1e-8):.2f}x")
        if avg_within > avg_between:
            print("  -> Fingerprints CAN discriminate topics")
        else:
            print("  -> Fingerprints CANNOT discriminate topics (problem!)")
    print()

    # Retrieval accuracy
    print("=== Retrieval Accuracy ===")
    correct = 0
    total = 0
    for query, expected_topic in QUERIES.items():
        qvec = loop.encode_text(query, keep_grad=False)
        results = ps.retrieve_by_graph(qvec, top_k=3)
        if results:
            top_text = results[0][1]
            top_topic = text_topics.get(top_text, "unknown")
            hit = "HIT" if top_topic == expected_topic else "MISS"
            if top_topic == expected_topic:
                correct += 1
            total += 1
            print(f"  [{hit}] \"{query}\"")
            print(f"        -> \"{top_text}\" (topic={top_topic}, sim={results[0][2]:.3f})")
            for _, text, sim in results[1:]:
                t = text_topics.get(text, "?")
                print(f"           \"{text[:60]}\" ({t}, sim={sim:.3f})")
        else:
            total += 1
            print(f"  [MISS] \"{query}\" -> no results")
        print()

    print(f"Accuracy: {correct}/{total} ({100*correct/max(total,1):.0f}%)")

    # Consolidation test
    print("\n=== After Consolidation (Sleep) ===")
    ps.soma._maybe_consolidate(rng=None)

    correct_post = 0
    total_post = 0
    for query, expected_topic in QUERIES.items():
        qvec = loop.encode_text(query, keep_grad=False)
        results = ps.retrieve_by_graph(qvec, top_k=1)
        if results:
            top_topic = text_topics.get(results[0][1], "unknown")
            if top_topic == expected_topic:
                correct_post += 1
        total_post += 1

    pct = 100 * correct_post / max(total_post, 1)
    print(f"Accuracy after sleep: {correct_post}/{total_post} ({pct:.0f}%)")


if __name__ == "__main__":
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    train = "--train" in sys.argv

    print("=" * 60)
    print(f"RETRIEVAL DIAGNOSTIC (repeats={repeats}, train_encoder={train})")
    print("=" * 60)
    print()

    run_diagnostic(n_repeats=repeats, train_enc=train)
