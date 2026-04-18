"""Phase 1: Hybrid retrieval — embeddings for recall, graph for reranking.

Hypothesis: SOMA's graph captures structural patterns (temporal,
co-occurrence) that pure cosine similarity misses. By using embeddings
to get top-K candidates and then reranking with graph topology scores,
we can combine the strengths of both.

The graph is NOT dead weight here — it provides a learned structural
signal that adjusts the ranking. If reranking doesn't change results,
the graph truly adds nothing. If it improves, the graph is learning
useful patterns that embeddings alone can't capture.
"""
from __future__ import annotations

import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.core.node import NodeType
from soma.developmental.interaction import InteractionLoop


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-per-conv", type=int, default=10)
    parser.add_argument("--recall-k", type=int, default=20,
                        help="Number of candidates from embedding retrieval")
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

    # Encode full corpus for embedding retrieval
    print("Encoding corpus...", flush=True)
    corpus_embeddings = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    # Build QA
    qa_pairs = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            qa_pairs.append((qa.question, str(qa.answer)))

    # Set up SOMA with pretrained embeddings
    config = SOMAConfig.developmental()
    soma_dim = config.sensor_output_dim  # 128
    loop = InteractionLoop(config=config, llm_model="unused", device=device)
    loop.train_tokenizer(corpus[:500])

    # Projection 384 -> 128
    proj = torch.nn.Linear(st_dim, soma_dim, bias=False).to(device)
    torch.nn.init.orthogonal_(proj.weight)

    # Develop SOMA on the corpus
    print("Developing SOMA...", flush=True)
    t0 = time.perf_counter()
    # Map step_num -> corpus index for retrieval cross-referencing
    step_to_corpus_idx: dict[int, int] = {}
    for ci, conv in enumerate(conversations):
        idx = 0
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                # Find this text's index in corpus
                corpus_idx = sum(
                    c.total_turns for c in conversations[:ci]
                ) + idx

                with torch.no_grad():
                    st_emb = st_model.encode(
                        text, convert_to_tensor=True, device=str(device),
                    )
                    soma_input = proj(st_emb)

                result = loop.predictive_soma.process_input(
                    soma_input, source_text=text,
                )
                step_num = result["global_step"]
                step_to_corpus_idx[step_num] = corpus_idx
                idx += 1
        print(f"  Conv {ci + 1}/10", flush=True)

    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s")

    ps = loop.predictive_soma

    # ================================================================
    # Strategy 1: Pure vector DB (baseline, same as before)
    # ================================================================
    print("\n--- Strategy 1: Vector DB (cosine sim) ---", flush=True)
    t0 = time.perf_counter()
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
    vecdb_time = time.perf_counter() - t0
    vecdb_hits = sum(1 for s in vecdb_scores if s > 0.05)
    vecdb_f1 = sum(vecdb_scores) / len(vecdb_scores)
    print(f"  F1={vecdb_f1:.4f}  hits={vecdb_hits}/100  ({vecdb_time:.1f}s)")

    # ================================================================
    # Strategy 2: Hybrid — embedding recall + graph reranking
    # ================================================================
    for rerank_weight in [0.1, 0.2, 0.3, 0.5]:
        label = f"hybrid_w{rerank_weight}"
        print(f"\n--- Strategy 2: {label} (recall={args.recall_k}) ---",
              flush=True)
        t0 = time.perf_counter()
        hybrid_scores = []
        for question, answer in qa_pairs:
            # Step 1: Embedding recall — get top-K candidates
            q_emb = st_model.encode(
                question, convert_to_tensor=True, device=str(device),
            )
            sims = torch.nn.functional.cosine_similarity(
                q_emb.unsqueeze(0), corpus_embeddings, dim=1,
            )
            top_k_sims, top_k_indices = torch.topk(
                sims, min(args.recall_k, len(corpus)),
            )

            # Step 2: Graph reranking — run query through SOMA,
            # compute topology overlap for each candidate
            with torch.no_grad():
                q_soma = proj(q_emb)
            modality = ps.config.input_modalities[0]
            ps.soma.step({modality: q_soma}, eval_mode=True)
            ps._diversify_activations(q_soma)
            ps._apply_lateral_inhibition()

            # Find active nodes for query
            active_nodes: set[str] = set()
            for node in ps.soma.graph.all_nodes():
                if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                    continue
                if (node.last_activation is not None
                        and node.last_activation.norm().item() > 1e-8):
                    active_nodes.add(node.id)

            # Score each candidate by node overlap
            n_active = len(active_nodes) or 1
            rerank_scores = []
            for i in range(len(top_k_indices)):
                corpus_idx = top_k_indices[i].item()
                # Find which SOMA step this corpus entry corresponds to
                step_num = None
                for s, cidx in step_to_corpus_idx.items():
                    if cidx == corpus_idx:
                        step_num = s
                        break

                topo_score = 0.0
                if step_num is not None and active_nodes:
                    overlap = 0
                    for nid in active_nodes:
                        if (nid in ps._node_memory_index
                                and step_num in ps._node_memory_index[nid]):
                            overlap += 1
                    topo_score = overlap / n_active

                # Combined score: (1-w)*embedding_sim + w*topo_score
                emb_sim = top_k_sims[i].item()
                combined = (1 - rerank_weight) * emb_sim + rerank_weight * topo_score
                rerank_scores.append((combined, corpus_idx))

            # Take top-5 after reranking
            rerank_scores.sort(key=lambda x: -x[0])
            top5_reranked = [idx for _, idx in rerank_scores[:5]]

            best = max(
                (token_f1(corpus[idx], answer) for idx in top5_reranked),
                default=0.0,
            )
            hybrid_scores.append(best)

        hybrid_time = time.perf_counter() - t0
        hybrid_hits = sum(1 for s in hybrid_scores if s > 0.05)
        hybrid_f1 = sum(hybrid_scores) / len(hybrid_scores)
        print(f"  F1={hybrid_f1:.4f}  hits={hybrid_hits}/100  "
              f"({hybrid_time:.1f}s)")

        # Where does reranking help vs hurt?
        wins = sum(1 for i in range(len(qa_pairs))
                   if hybrid_scores[i] > vecdb_scores[i] + 0.01)
        losses = sum(1 for i in range(len(qa_pairs))
                     if vecdb_scores[i] > hybrid_scores[i] + 0.01)
        print(f"  vs vecDB: +{wins} wins, -{losses} losses")

    # ================================================================
    # Strategy 3: Hybrid with fingerprint instead of topology
    # ================================================================
    print(f"\n--- Strategy 3: hybrid_fp (recall={args.recall_k}) ---",
          flush=True)
    t0 = time.perf_counter()
    hybrid_fp_scores = []
    for question, answer in qa_pairs:
        q_emb = st_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), corpus_embeddings, dim=1,
        )
        top_k_sims, top_k_indices = torch.topk(
            sims, min(args.recall_k, len(corpus)),
        )

        # Graph fingerprint for query
        with torch.no_grad():
            q_soma = proj(q_emb)
        modality = ps.config.input_modalities[0]
        ps.soma.step({modality: q_soma}, eval_mode=True)
        ps._diversify_activations(q_soma)
        ps._apply_lateral_inhibition()
        query_fp = ps._get_node_fingerprint()

        # Rerank by fingerprint similarity
        rerank_scores = []
        for i in range(len(top_k_indices)):
            corpus_idx = top_k_indices[i].item()
            step_num = None
            for s, cidx in step_to_corpus_idx.items():
                if cidx == corpus_idx:
                    step_num = s
                    break

            fp_sim = 0.0
            if step_num is not None and step_num in ps._activation_store:
                stored_fp = ps._activation_store[step_num]
                fp_sim = torch.nn.functional.cosine_similarity(
                    query_fp.unsqueeze(0), stored_fp.unsqueeze(0),
                ).item()

            emb_sim = top_k_sims[i].item()
            combined = 0.8 * emb_sim + 0.2 * fp_sim
            rerank_scores.append((combined, corpus_idx))

        rerank_scores.sort(key=lambda x: -x[0])
        top5 = [idx for _, idx in rerank_scores[:5]]
        best = max(
            (token_f1(corpus[idx], answer) for idx in top5),
            default=0.0,
        )
        hybrid_fp_scores.append(best)

    hybrid_fp_time = time.perf_counter() - t0
    hybrid_fp_hits = sum(1 for s in hybrid_fp_scores if s > 0.05)
    hybrid_fp_f1 = sum(hybrid_fp_scores) / len(hybrid_fp_scores)
    print(f"  F1={hybrid_fp_f1:.4f}  hits={hybrid_fp_hits}/100  "
          f"({hybrid_fp_time:.1f}s)")
    wins = sum(1 for i in range(len(qa_pairs))
               if hybrid_fp_scores[i] > vecdb_scores[i] + 0.01)
    losses = sum(1 for i in range(len(qa_pairs))
                 if vecdb_scores[i] > hybrid_fp_scores[i] + 0.01)
    print(f"  vs vecDB: +{wins} wins, -{losses} losses")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Vector DB baseline:    hits={vecdb_hits}/100  F1={vecdb_f1:.4f}")


if __name__ == "__main__":
    main()
