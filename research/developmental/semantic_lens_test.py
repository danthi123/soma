"""Phase 5: Graph as semantic lens for retrieval.

Instead of using the graph's fingerprint/topology as a reranking
signal, use the graph's learned node weights as a query-dependent
projection. The graph identifies which nodes are most relevant for
a query (via activation), and those nodes' weights define a
"semantic lens" that emphasizes certain dimensions of the embedding
space when computing similarity.

This is different from fingerprint reranking:
- Fingerprint: compare query's graph activation pattern to stored patterns
- Semantic lens: use the graph to MODIFY the similarity computation itself

The graph is genuinely contributing here: it learned which dimensions
matter for which types of queries through its structural plasticity.
Combined with confidence gating from Phase 3.
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

    # Set up SOMA
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
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                with torch.no_grad():
                    emb = st_model.encode(
                        text, convert_to_tensor=True, device=str(device),
                    )
                ps.process_input(emb, source_text=text)
        print(f"  Conv {ci + 1}/10", flush=True)
    print(f"  Development: {time.perf_counter() - t0:.1f}s")

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
        vecdb_scores.append(max(
            (token_f1(corpus[i.item()], answer) for i in top5),
            default=0.0,
        ))
    vecdb_hits = sum(1 for s in vecdb_scores if s > 0.05)
    print(f"  hits={vecdb_hits}/100")

    # ================================================================
    # Semantic lens: use active node weights to create a query-dependent
    # importance weighting over embedding dimensions
    # ================================================================
    for lens_strength in [0.1, 0.3, 0.5]:
        print(f"\n--- Semantic lens (strength={lens_strength}) ---",
              flush=True)
        lens_scores = []
        lens_used = 0

        for question, answer in qa_pairs:
            q_emb = st_model.encode(
                question, convert_to_tensor=True, device=str(device),
            )

            # Run query through graph to find active nodes
            modality = ps.config.input_modalities[0]
            ps.soma.step({modality: q_emb}, eval_mode=True)
            ps._diversify_activations(q_emb)
            ps._apply_lateral_inhibition()

            # Build importance weights from active nodes' learned weights
            # Each active node's linear1 weight matrix encodes what
            # input dimensions that node has learned to respond to.
            # The column-wise norm of the weight matrix indicates
            # which input dimensions are most important to this node.
            importance = torch.zeros(st_dim, device=device)
            n_active = 0
            for node in ps.soma.graph.all_nodes():
                if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                    continue
                if (node.last_activation is None
                        or node.last_activation.norm().item() < 1e-8):
                    continue
                n_active += 1
                w = node.linear1.weight.detach()
                # Column norms: how much each input dim contributes
                # to this node's response
                col_norms = w.norm(dim=0)
                # Pad or truncate to match embedding dim
                d = min(col_norms.shape[0], st_dim)
                importance[:d] += col_norms[:d]

            if n_active > 0:
                importance = importance / n_active
                # Normalize to [0.5, 1.5] range so it modulates
                # rather than dominates
                imp_min = importance.min()
                imp_max = importance.max()
                if imp_max > imp_min:
                    importance = 0.5 + (importance - imp_min) / (imp_max - imp_min)
                else:
                    importance = torch.ones_like(importance)

                # Confidence: how peaked is the importance distribution?
                # High entropy = uniform = no useful signal
                imp_std = importance.std().item()
                if imp_std > 0.1:  # some differentiation exists
                    lens_used += 1
                    # Apply lens: weight query and corpus embeddings
                    # by importance before computing similarity
                    lens_weight = 1.0 + lens_strength * (importance - 1.0)
                    q_lensed = q_emb * lens_weight
                    c_lensed = corpus_embeddings * lens_weight.unsqueeze(0)
                    sims = torch.nn.functional.cosine_similarity(
                        q_lensed.unsqueeze(0), c_lensed, dim=1,
                    )
                else:
                    sims = torch.nn.functional.cosine_similarity(
                        q_emb.unsqueeze(0), corpus_embeddings, dim=1,
                    )
            else:
                sims = torch.nn.functional.cosine_similarity(
                    q_emb.unsqueeze(0), corpus_embeddings, dim=1,
                )

            top5 = torch.topk(sims, 5).indices
            best = max(
                (token_f1(corpus[i.item()], answer) for i in top5),
                default=0.0,
            )
            lens_scores.append(best)

        lens_hits = sum(1 for s in lens_scores if s > 0.05)
        wins = sum(1 for i in range(len(qa_pairs))
                   if lens_scores[i] > vecdb_scores[i] + 0.01)
        losses = sum(1 for i in range(len(qa_pairs))
                     if vecdb_scores[i] > lens_scores[i] + 0.01)
        print(f"  hits={lens_hits}/100  lens_used={lens_used}/100  "
              f"vs vecDB: +{wins}/-{losses}")

    # ================================================================
    # Combined: semantic lens + confidence-gated fingerprint reranking
    # ================================================================
    print(f"\n--- Combined: lens(0.3) + fp-gate(0.05) ---", flush=True)

    step_to_cidx: dict[int, int] = {}
    for step_num, text in ps.text_store.items():
        try:
            cidx = corpus.index(text)
            step_to_cidx[step_num] = cidx
        except ValueError:
            pass

    combined_scores = []
    for question, answer in qa_pairs:
        q_emb = st_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )

        # Semantic lens
        modality = ps.config.input_modalities[0]
        ps.soma.step({modality: q_emb}, eval_mode=True)
        ps._diversify_activations(q_emb)
        ps._apply_lateral_inhibition()

        importance = torch.zeros(st_dim, device=device)
        n_active = 0
        for node in ps.soma.graph.all_nodes():
            if node.node_type in (NodeType.SENSOR, NodeType.OUTPUT):
                continue
            if (node.last_activation is None
                    or node.last_activation.norm().item() < 1e-8):
                continue
            n_active += 1
            w = node.linear1.weight.detach()
            col_norms = w.norm(dim=0)
            d = min(col_norms.shape[0], st_dim)
            importance[:d] += col_norms[:d]

        if n_active > 0:
            importance = importance / n_active
            imp_min, imp_max = importance.min(), importance.max()
            if imp_max > imp_min:
                importance = 0.5 + (importance - imp_min) / (imp_max - imp_min)
            else:
                importance = torch.ones_like(importance)
            lens_weight = 1.0 + 0.3 * (importance - 1.0)
            q_lensed = q_emb * lens_weight
            c_lensed = corpus_embeddings * lens_weight.unsqueeze(0)
            sims = torch.nn.functional.cosine_similarity(
                q_lensed.unsqueeze(0), c_lensed, dim=1,
            )
        else:
            sims = torch.nn.functional.cosine_similarity(
                q_emb.unsqueeze(0), corpus_embeddings, dim=1,
            )

        # Recall top-20
        top_k_sims, top_k_indices = torch.topk(sims, 20)

        # Fingerprint confidence gate
        query_fp = ps._get_node_fingerprint()
        fp_sims = []
        for i in range(len(top_k_indices)):
            cidx = top_k_indices[i].item()
            step_num = None
            for s, c in step_to_cidx.items():
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

        fp_sorted = sorted(fp_sims, reverse=True)
        confidence = fp_sorted[0] - (fp_sorted[1] if len(fp_sorted) > 1 else 0)

        if confidence >= 0.05:
            rerank_scores = []
            for i in range(len(top_k_indices)):
                emb_sim = top_k_sims[i].item()
                combined_sim = 0.8 * emb_sim + 0.2 * fp_sims[i]
                rerank_scores.append((combined_sim, top_k_indices[i].item()))
            rerank_scores.sort(key=lambda x: -x[0])
            top5 = [idx for _, idx in rerank_scores[:5]]
        else:
            top5 = [top_k_indices[i].item() for i in range(min(5, len(top_k_indices)))]

        best = max(
            (token_f1(corpus[idx], answer) for idx in top5),
            default=0.0,
        )
        combined_scores.append(best)

    combined_hits = sum(1 for s in combined_scores if s > 0.05)
    wins = sum(1 for i in range(len(qa_pairs))
               if combined_scores[i] > vecdb_scores[i] + 0.01)
    losses = sum(1 for i in range(len(qa_pairs))
                 if vecdb_scores[i] > combined_scores[i] + 0.01)
    print(f"  hits={combined_hits}/100  vs vecDB: +{wins}/-{losses}")

    print(f"\n  Vector DB baseline: {vecdb_hits}/100")


if __name__ == "__main__":
    main()
