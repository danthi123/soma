"""Phase 2: Full-dim sensor input — no 384->128 projection loss.

Tests whether SOMA's graph performs better when it receives the full
384-dim pretrained embedding instead of a lossy 128-dim projection.

Compares:
  1. Vector DB baseline (384-dim cosine sim)
  2. SOMA with 384-dim sensor (no projection)
  3. SOMA with 128-dim sensor (384->128 projection, from Phase 1)
"""
from __future__ import annotations

import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.core.node import NodeType
from soma.developmental.prediction import PredictiveSOMA


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-per-conv", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    conversations = load_dataset()
    total_turns = sum(c.total_turns for c in conversations)
    print(f"LoCoMo: {len(conversations)} convs, {total_turns} turns")

    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(
                    f"[{session.date_time}] {turn.speaker}: {turn.text}"
                )

    # Load pretrained encoder
    print("Loading sentence-transformers...", flush=True)
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()
    print(f"Embedding dim: {st_dim}")

    # Encode corpus
    print("Encoding corpus...", flush=True)
    corpus_embeddings = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    qa_pairs = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            qa_pairs.append((qa.question, str(qa.answer)))

    # ================================================================
    # Vector DB baseline
    # ================================================================
    print("\n--- Vector DB baseline ---", flush=True)
    vecdb_scores = []
    for question, answer in qa_pairs:
        q_emb = st_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), corpus_embeddings, dim=1,
        )
        top5 = torch.topk(sims, 5).indices
        best = max(
            (token_f1(corpus[i.item()], answer) for i in top5),
            default=0.0,
        )
        vecdb_scores.append(best)
    vecdb_hits = sum(1 for s in vecdb_scores if s > 0.05)
    vecdb_f1 = sum(vecdb_scores) / len(vecdb_scores)
    print(f"  F1={vecdb_f1:.4f}  hits={vecdb_hits}/100")

    # ================================================================
    # SOMA with full 384-dim sensor
    # ================================================================
    print("\n--- SOMA 384-dim sensor ---", flush=True)
    config384 = SOMAConfig.developmental(
        sensor_output_dim=st_dim,
        text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2,
        associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2,
        integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2,
        integrator_output_dim=st_dim,
    )
    ps384 = PredictiveSOMA(config384, device=device)

    t0 = time.perf_counter()
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                with torch.no_grad():
                    emb = st_model.encode(
                        text, convert_to_tensor=True, device=str(device),
                    )
                ps384.process_input(emb, source_text=text)
        print(f"  Conv {ci + 1}/10: nodes={len(ps384.soma.graph.nodes)}",
              flush=True)
    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s")

    # Retrieve via fingerprint
    t0 = time.perf_counter()
    soma384_scores = []
    for question, answer in qa_pairs:
        with torch.no_grad():
            q_emb = st_model.encode(
                question, convert_to_tensor=True, device=str(device),
            )
        results = ps384.retrieve_by_graph(q_emb, top_k=5)
        best = max(
            (token_f1(t, answer) for _, t, _ in results),
            default=0.0,
        )
        soma384_scores.append(best)
    t384 = time.perf_counter() - t0
    soma384_hits = sum(1 for s in soma384_scores if s > 0.05)
    soma384_f1 = sum(soma384_scores) / len(soma384_scores)
    print(f"  graph_fp: F1={soma384_f1:.4f}  hits={soma384_hits}/100  ({t384:.1f}s)")

    # Topology
    t0 = time.perf_counter()
    topo384_scores = []
    for question, answer in qa_pairs:
        with torch.no_grad():
            q_emb = st_model.encode(
                question, convert_to_tensor=True, device=str(device),
            )
        results = ps384.retrieve_by_topology(q_emb, top_k=5)
        best = max(
            (token_f1(t, answer) for _, t, _ in results),
            default=0.0,
        )
        topo384_scores.append(best)
    topo384_time = time.perf_counter() - t0
    topo384_hits = sum(1 for s in topo384_scores if s > 0.05)
    topo384_f1 = sum(topo384_scores) / len(topo384_scores)
    print(f"  graph_topo: F1={topo384_f1:.4f}  hits={topo384_hits}/100  ({topo384_time:.1f}s)")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    print(f"  Vector DB (384-dim cosine):   hits={vecdb_hits}/100  F1={vecdb_f1:.4f}")
    print(f"  SOMA 384-dim (fingerprint):   hits={soma384_hits}/100  F1={soma384_f1:.4f}")
    print(f"  SOMA 384-dim (topology):      hits={topo384_hits}/100  F1={topo384_f1:.4f}")
    print(f"  (Phase 1: SOMA 128-dim fp:    hits=49/100)")

    # Per-query comparison
    wins = sum(1 for i in range(len(qa_pairs))
               if soma384_scores[i] > vecdb_scores[i] + 0.01)
    losses = sum(1 for i in range(len(qa_pairs))
                 if vecdb_scores[i] > soma384_scores[i] + 0.01)
    print(f"\n  SOMA 384 vs VecDB: +{wins} wins, -{losses} losses")


if __name__ == "__main__":
    main()
