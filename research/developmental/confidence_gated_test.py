"""Phase 3: Confidence-gated hybrid retrieval.

The graph consistently improves 15-22 queries but hurts 19-48 others.
If we can detect WHEN the graph signal is reliable and only apply
reranking then, we get the best of both worlds.

Gate: the graph's internal confidence — how much its top-1 match
stands out from the rest. High confidence = the graph "knows"
something specific about this query. Low confidence = the graph
is guessing, fall back to embedding similarity.

This is SOMA adding genuine value: it selectively provides a
structural signal where it has learned useful patterns, and stays
out of the way where it hasn't.
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

    print("Loading sentence-transformers...", flush=True)
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    print("Encoding corpus...", flush=True)
    corpus_embeddings = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    qa_pairs = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            qa_pairs.append((qa.question, str(qa.answer)))

    # Set up SOMA with 384-dim (Phase 2 showed marginally better)
    config = SOMAConfig.developmental(
        sensor_output_dim=st_dim,
        text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2,
        associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2,
        integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2,
        integrator_output_dim=st_dim,
    )
    ps = PredictiveSOMA(config, device=device)

    # Develop
    print("Developing SOMA...", flush=True)
    t0 = time.perf_counter()
    step_to_corpus_idx: dict[int, int] = {}
    corpus_idx = 0
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                with torch.no_grad():
                    emb = st_model.encode(
                        text, convert_to_tensor=True, device=str(device),
                    )
                result = ps.process_input(emb, source_text=text)
                step_to_corpus_idx[result["global_step"]] = corpus_idx
                corpus_idx += 1
        print(f"  Conv {ci + 1}/10", flush=True)
    dev_time = time.perf_counter() - t0
    print(f"  Development: {dev_time:.1f}s")

    # Vector DB baseline
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
    print(f"  hits={vecdb_hits}/100")

    # Confidence-gated hybrid
    recall_k = 20
    for gate_threshold in [0.3, 0.5, 0.7, 0.9]:
        print(f"\n--- Confidence-gated (threshold={gate_threshold}) ---",
              flush=True)
        gated_scores = []
        graph_used_count = 0

        for qi, (question, answer) in enumerate(qa_pairs):
            q_emb = st_model.encode(
                question, convert_to_tensor=True, device=str(device),
            )

            # Embedding recall
            sims = torch.nn.functional.cosine_similarity(
                q_emb.unsqueeze(0), corpus_embeddings, dim=1,
            )
            top_k_sims, top_k_indices = torch.topk(
                sims, min(recall_k, len(corpus)),
            )

            # Graph confidence: run query through SOMA
            modality = ps.config.input_modalities[0]
            ps.soma.step({modality: q_emb}, eval_mode=True)
            ps._diversify_activations(q_emb)
            ps._apply_lateral_inhibition()

            # Compute node overlap for each candidate
            active_nodes: set[str] = set()
            for node in ps.soma.graph.all_nodes():
                if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                    continue
                if (node.last_activation is not None
                        and node.last_activation.norm().item() > 1e-8):
                    active_nodes.add(node.id)

            n_active = len(active_nodes) or 1
            topo_scores_list = []
            for i in range(len(top_k_indices)):
                cidx = top_k_indices[i].item()
                step_num = None
                for s, c in step_to_corpus_idx.items():
                    if c == cidx:
                        step_num = s
                        break
                topo = 0.0
                if step_num is not None and active_nodes:
                    overlap = sum(
                        1 for nid in active_nodes
                        if nid in ps._node_memory_index
                        and step_num in ps._node_memory_index[nid]
                    )
                    topo = overlap / n_active
                topo_scores_list.append(topo)

            # Confidence: how much does the top topology score
            # stand out from the rest?
            if topo_scores_list:
                topo_sorted = sorted(topo_scores_list, reverse=True)
                top_topo = topo_sorted[0]
                second_topo = topo_sorted[1] if len(topo_sorted) > 1 else 0
                # Confidence = gap between top and second
                confidence = top_topo - second_topo
            else:
                confidence = 0.0

            # Gate: only apply graph reranking if confidence exceeds threshold
            if confidence >= gate_threshold:
                graph_used_count += 1
                # Rerank with graph topology
                rerank_scores = []
                for i in range(len(top_k_indices)):
                    emb_sim = top_k_sims[i].item()
                    combined = 0.7 * emb_sim + 0.3 * topo_scores_list[i]
                    rerank_scores.append((combined, top_k_indices[i].item()))
                rerank_scores.sort(key=lambda x: -x[0])
                top5_final = [idx for _, idx in rerank_scores[:5]]
            else:
                # Pure embedding retrieval
                top5_final = [top_k_indices[i].item() for i in range(min(5, len(top_k_indices)))]

            best = max(
                (token_f1(corpus[idx], answer) for idx in top5_final),
                default=0.0,
            )
            gated_scores.append(best)

        gated_hits = sum(1 for s in gated_scores if s > 0.05)
        wins = sum(1 for i in range(len(qa_pairs))
                   if gated_scores[i] > vecdb_scores[i] + 0.01)
        losses = sum(1 for i in range(len(qa_pairs))
                     if vecdb_scores[i] > gated_scores[i] + 0.01)
        print(f"  hits={gated_hits}/100  graph_used={graph_used_count}/100")
        print(f"  vs vecDB: +{wins} wins, -{losses} losses")

    # Also try: fingerprint confidence (cosine sim gap)
    print(f"\n--- Fingerprint confidence gate ---", flush=True)
    for gate_threshold in [0.05, 0.1, 0.2]:
        gated_scores = []
        graph_used_count = 0

        for qi, (question, answer) in enumerate(qa_pairs):
            q_emb = st_model.encode(
                question, convert_to_tensor=True, device=str(device),
            )

            # Embedding recall
            sims = torch.nn.functional.cosine_similarity(
                q_emb.unsqueeze(0), corpus_embeddings, dim=1,
            )
            top_k_sims, top_k_indices = torch.topk(
                sims, min(recall_k, len(corpus)),
            )

            # Graph fingerprint for query
            modality = ps.config.input_modalities[0]
            ps.soma.step({modality: q_emb}, eval_mode=True)
            ps._diversify_activations(q_emb)
            ps._apply_lateral_inhibition()
            query_fp = ps._get_node_fingerprint()

            # Fingerprint similarity for each candidate
            fp_sims = []
            for i in range(len(top_k_indices)):
                cidx = top_k_indices[i].item()
                step_num = None
                for s, c in step_to_corpus_idx.items():
                    if c == cidx:
                        step_num = s
                        break
                fp_sim = 0.0
                if step_num is not None and step_num in ps._activation_store:
                    fp_sim = torch.nn.functional.cosine_similarity(
                        query_fp.unsqueeze(0),
                        ps._activation_store[step_num].unsqueeze(0),
                    ).item()
                fp_sims.append(fp_sim)

            # Confidence: gap between top fp score and embedding-predicted top
            if fp_sims:
                fp_sorted = sorted(fp_sims, reverse=True)
                fp_confidence = fp_sorted[0] - (fp_sorted[1] if len(fp_sorted) > 1 else 0)
            else:
                fp_confidence = 0.0

            if fp_confidence >= gate_threshold:
                graph_used_count += 1
                rerank_scores = []
                for i in range(len(top_k_indices)):
                    emb_sim = top_k_sims[i].item()
                    combined = 0.8 * emb_sim + 0.2 * fp_sims[i]
                    rerank_scores.append((combined, top_k_indices[i].item()))
                rerank_scores.sort(key=lambda x: -x[0])
                top5_final = [idx for _, idx in rerank_scores[:5]]
            else:
                top5_final = [top_k_indices[i].item() for i in range(min(5, len(top_k_indices)))]

            best = max(
                (token_f1(corpus[idx], answer) for idx in top5_final),
                default=0.0,
            )
            gated_scores.append(best)

        gated_hits = sum(1 for s in gated_scores if s > 0.05)
        wins = sum(1 for i in range(len(qa_pairs))
                   if gated_scores[i] > vecdb_scores[i] + 0.01)
        losses = sum(1 for i in range(len(qa_pairs))
                     if vecdb_scores[i] > gated_scores[i] + 0.01)
        print(f"  gate={gate_threshold}: hits={gated_hits}/100  "
              f"graph_used={graph_used_count}/100  "
              f"vs vecDB: +{wins}/-{losses}")

    print(f"\n  Vector DB baseline: {vecdb_hits}/100")


if __name__ == "__main__":
    main()
