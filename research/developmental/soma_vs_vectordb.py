"""SOMA + pretrained embeddings vs plain vector retrieval.

The product question: does SOMA's graph add retrieval value on top of
good embeddings, compared to plain cosine similarity (what every
vector DB / RAG system does)?

Strategies compared:
  1. Vector DB baseline: encode all texts with sentence-transformers,
     retrieve by cosine similarity (no SOMA at all)
  2. SOMA + pretrained: same encoder, but SOMA processes embeddings
     through its graph, retrieves via graph fingerprints
  3. SOMA + random BPE: current PoC baseline (random embeddings)

If SOMA + pretrained beats vector DB baseline, that proves the graph
adds genuine retrieval value beyond what embeddings alone provide.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-per-conv", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load LoCoMo
    conversations = load_dataset()
    total_turns = sum(c.total_turns for c in conversations)
    print(f"LoCoMo: {len(conversations)} convs, {total_turns} turns")

    # Build corpus
    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(
                    f"[{session.date_time}] {turn.speaker}: {turn.text}"
                )
    print(f"Corpus: {len(corpus)} texts")

    # Load pretrained encoder
    print("\nLoading sentence-transformers model...", flush=True)
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()
    print(f"Model dim: {st_dim}")

    # Build QA list
    qa_pairs = []
    for conv in conversations:
        for qa in conv.qa_pairs[: args.qa_per_conv]:
            qa_pairs.append((qa.question, str(qa.answer)))
    print(f"QA pairs: {len(qa_pairs)}")

    # ================================================================
    # Strategy 1: Vector DB baseline (cosine similarity, no SOMA)
    # ================================================================
    print("\n=== 1. VECTOR DB BASELINE (embeddings only) ===", flush=True)
    t0 = time.perf_counter()

    # Encode all corpus texts
    corpus_embeddings = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )
    encode_time = time.perf_counter() - t0
    print(f"  Encoded {len(corpus)} texts in {encode_time:.1f}s")

    # Retrieve by cosine similarity
    t1 = time.perf_counter()
    vectordb_scores = []
    for question, answer in qa_pairs:
        q_emb = st_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), corpus_embeddings, dim=1,
        )
        top_indices = torch.topk(sims, min(5, len(corpus))).indices
        best = max(
            (token_f1(corpus[idx.item()], answer) for idx in top_indices),
            default=0.0,
        )
        vectordb_scores.append(best)

    vectordb_time = time.perf_counter() - t1
    vectordb_hits = sum(1 for s in vectordb_scores if s > 0.05)
    vectordb_f1 = sum(vectordb_scores) / len(vectordb_scores)
    print(f"  F1={vectordb_f1:.4f}  hits={vectordb_hits}/{len(qa_pairs)}  "
          f"({vectordb_time:.1f}s)")

    # ================================================================
    # Strategy 2: SOMA + pretrained embeddings
    # ================================================================
    print("\n=== 2. SOMA + PRETRAINED EMBEDDINGS ===", flush=True)

    # Configure SOMA with pretrained embedding dim
    # Project 384-dim to SOMA's 128-dim via a learned linear layer
    config = SOMAConfig.developmental()
    loop = InteractionLoop(
        config=config, llm_model="unused", device=device,
    )
    # We need a tokenizer for the token cache, use corpus for that
    loop.train_tokenizer(corpus[:500])

    # Create a projection from st_dim (384) to soma_dim (128)
    soma_dim = config.sensor_output_dim
    proj = torch.nn.Linear(st_dim, soma_dim, bias=False).to(device)
    torch.nn.init.orthogonal_(proj.weight)

    # Development: encode with sentence-transformers, project, feed to SOMA
    print("  Developing...", flush=True)
    t0 = time.perf_counter()
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                # Encode with pretrained model
                with torch.no_grad():
                    st_emb = st_model.encode(
                        text, convert_to_tensor=True, device=str(device),
                    )
                    soma_input = proj(st_emb)
                # Feed projected embedding to SOMA
                loop.predictive_soma.process_input(
                    soma_input, source_text=text,
                )
        ps = loop.predictive_soma
        print(f"    Conv {ci + 1}/10: nodes={len(ps.soma.graph.nodes)}, "
              f"mem={len(ps.text_store)}", flush=True)

    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s")

    # Retrieve via SOMA graph fingerprint
    ps = loop.predictive_soma
    t1 = time.perf_counter()
    soma_pretrained_scores = []
    for question, answer in qa_pairs:
        with torch.no_grad():
            q_emb = st_model.encode(
                question, convert_to_tensor=True, device=str(device),
            )
            q_soma = proj(q_emb)
        results = ps.retrieve_by_graph(q_soma, top_k=5)
        best = max(
            (token_f1(t, answer) for _, t, _ in results),
            default=0.0,
        )
        soma_pretrained_scores.append(best)

    soma_time = time.perf_counter() - t1
    soma_hits = sum(1 for s in soma_pretrained_scores if s > 0.05)
    soma_f1 = sum(soma_pretrained_scores) / len(soma_pretrained_scores)
    print(f"  graph_fingerprint: F1={soma_f1:.4f}  "
          f"hits={soma_hits}/{len(qa_pairs)}  ({soma_time:.1f}s)")

    # Also test topology
    t2 = time.perf_counter()
    soma_topo_scores = []
    for question, answer in qa_pairs:
        with torch.no_grad():
            q_emb = st_model.encode(
                question, convert_to_tensor=True, device=str(device),
            )
            q_soma = proj(q_emb)
        results = ps.retrieve_by_topology(q_soma, top_k=5)
        best = max(
            (token_f1(t, answer) for _, t, _ in results),
            default=0.0,
        )
        soma_topo_scores.append(best)

    topo_time = time.perf_counter() - t2
    topo_hits = sum(1 for s in soma_topo_scores if s > 0.05)
    topo_f1 = sum(soma_topo_scores) / len(soma_topo_scores)
    print(f"  graph_topology:    F1={topo_f1:.4f}  "
          f"hits={topo_hits}/{len(qa_pairs)}  ({topo_time:.1f}s)")

    # ================================================================
    # Strategy 3: SOMA + random BPE (current baseline)
    # ================================================================
    print("\n=== 3. SOMA + RANDOM BPE (baseline) ===", flush=True)
    config3 = SOMAConfig.developmental()
    loop3 = InteractionLoop(
        config=config3, llm_model="unused", device=device,
    )
    loop3.train_tokenizer(corpus[:500])

    t0 = time.perf_counter()
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                loop3.process_input(text, call_llm=False)
        print(f"    Conv {ci + 1}/10: nodes="
              f"{len(loop3.predictive_soma.soma.graph.nodes)}", flush=True)

    dev_time3 = time.perf_counter() - t0
    print(f"  Development: {dev_time3:.1f}s")

    ps3 = loop3.predictive_soma
    t1 = time.perf_counter()
    bpe_scores = []
    for question, answer in qa_pairs:
        qvec = loop3.encode_text(question, keep_grad=False)
        results = ps3.retrieve_by_graph(qvec, top_k=5)
        best = max(
            (token_f1(t, answer) for _, t, _ in results),
            default=0.0,
        )
        bpe_scores.append(best)

    bpe_time = time.perf_counter() - t1
    bpe_hits = sum(1 for s in bpe_scores if s > 0.05)
    bpe_f1 = sum(bpe_scores) / len(bpe_scores)
    print(f"  graph_fingerprint: F1={bpe_f1:.4f}  "
          f"hits={bpe_hits}/{len(qa_pairs)}  ({bpe_time:.1f}s)")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Vector DB (cosine sim):     F1={vectordb_f1:.4f}  "
          f"hits={vectordb_hits}/{len(qa_pairs)}")
    print(f"  SOMA + pretrained (fp):     F1={soma_f1:.4f}  "
          f"hits={soma_hits}/{len(qa_pairs)}")
    print(f"  SOMA + pretrained (topo):   F1={topo_f1:.4f}  "
          f"hits={topo_hits}/{len(qa_pairs)}")
    print(f"  SOMA + random BPE (fp):     F1={bpe_f1:.4f}  "
          f"hits={bpe_hits}/{len(qa_pairs)}")

    if soma_hits > vectordb_hits:
        print(f"\n  >> SOMA + pretrained BEATS vector DB by "
              f"+{soma_hits - vectordb_hits} hits!")
    elif soma_hits == vectordb_hits:
        print(f"\n  >> SOMA + pretrained TIES with vector DB")
    else:
        print(f"\n  >> Vector DB wins by +{vectordb_hits - soma_hits} hits")

    # Per-query comparison: where does SOMA win over vector DB?
    print("\n--- Where SOMA + Pretrained Beats Vector DB ---")
    soma_wins = 0
    for i, (q, a) in enumerate(qa_pairs):
        if soma_pretrained_scores[i] > vectordb_scores[i] + 0.01:
            soma_wins += 1
            if soma_wins <= 5:
                print(f"  Q: {q[:70]}")
                print(f"    soma={soma_pretrained_scores[i]:.3f} "
                      f"vecdb={vectordb_scores[i]:.3f}")
    print(f"  SOMA wins: {soma_wins}/{len(qa_pairs)} queries")

    vectordb_wins = sum(
        1 for i in range(len(qa_pairs))
        if vectordb_scores[i] > soma_pretrained_scores[i] + 0.01
    )
    print(f"  VecDB wins: {vectordb_wins}/{len(qa_pairs)} queries")


if __name__ == "__main__":
    main()
