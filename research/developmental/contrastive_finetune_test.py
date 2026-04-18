"""Phase 4b: Contrastive encoder fine-tuning via graph topology.

Uses SOMA's node memory index as supervision: memories sharing
active nodes should have similar embeddings. The contrastive loss
has a proper grad_fn connected to the encoder, unlike Phase 4's
attempt which used SOMA's detached prediction loss.

Combined with Phase 3's confidence-gated retrieval to measure
whether fine-tuning expands the graph's effective range.
"""
from __future__ import annotations

import copy
import time

import torch
from sentence_transformers import SentenceTransformer

from benchmarks.industry.locomo.data_loader import load_dataset
from benchmarks.industry.longmemeval.metrics import token_f1
from soma.core.config import SOMAConfig
from soma.core.node import NodeType
from soma.developmental.prediction import PredictiveSOMA


def evaluate_gated(
    ps: PredictiveSOMA,
    st_model: SentenceTransformer,
    corpus: list[str],
    corpus_embeddings: torch.Tensor,
    step_to_cidx: dict[int, int],
    qa_pairs: list[tuple[str, str]],
    device: torch.device,
    gate_threshold: float = 0.05,
) -> tuple[int, int, list[float]]:
    """Run confidence-gated evaluation. Returns (hits, graph_used, scores)."""
    scores = []
    graph_used = 0
    cidx_to_step = {v: k for k, v in step_to_cidx.items()}

    for question, answer in qa_pairs:
        q_emb = st_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), corpus_embeddings, dim=1,
        )
        top_k_sims, top_k_indices = torch.topk(sims, 20)

        modality = ps.config.input_modalities[0]
        ps.soma.step({modality: q_emb}, eval_mode=True)
        ps._diversify_activations(q_emb)
        ps._apply_lateral_inhibition()
        query_fp = ps._get_node_fingerprint()

        fp_sims = []
        for i in range(len(top_k_indices)):
            cidx = top_k_indices[i].item()
            step_num = cidx_to_step.get(cidx)
            fp_sim = 0.0
            if step_num is not None and step_num in ps._activation_store:
                fp_sim = torch.nn.functional.cosine_similarity(
                    query_fp.unsqueeze(0),
                    ps._activation_store[step_num].unsqueeze(0),
                ).item()
            fp_sims.append(fp_sim)

        fp_sorted = sorted(fp_sims, reverse=True)
        confidence = fp_sorted[0] - (fp_sorted[1] if len(fp_sorted) > 1 else 0)

        if confidence >= gate_threshold:
            graph_used += 1
            rerank = [
                (0.8 * top_k_sims[i].item() + 0.2 * fp_sims[i],
                 top_k_indices[i].item())
                for i in range(len(top_k_indices))
            ]
            rerank.sort(key=lambda x: -x[0])
            top5 = [idx for _, idx in rerank[:5]]
        else:
            top5 = [top_k_indices[i].item() for i in range(min(5, len(top_k_indices)))]

        best = max(
            (token_f1(corpus[idx], answer) for idx in top5),
            default=0.0,
        )
        scores.append(best)

    hits = sum(1 for s in scores if s > 0.05)
    return hits, graph_used, scores


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-per-conv", type=int, default=10)
    parser.add_argument("--ft-lr", type=float, default=5e-6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    conversations = load_dataset()
    corpus = []
    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                corpus.append(
                    f"[{session.date_time}] {turn.speaker}: {turn.text}"
                )

    qa_pairs = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            qa_pairs.append((qa.question, str(qa.answer)))

    print("Loading model...", flush=True)
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    st_dim = st_model.get_sentence_embedding_dimension()

    # ================================================================
    # Frozen baseline (reproduce Phase 3)
    # ================================================================
    print("\n=== Frozen encoder baseline ===", flush=True)
    corpus_emb = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )
    config = SOMAConfig.developmental(
        sensor_output_dim=st_dim, text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2, associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2, integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2, integrator_output_dim=st_dim,
    )
    ps_frozen = PredictiveSOMA(config, device=device)
    step_to_cidx_frozen: dict[int, int] = {}
    cidx = 0
    t0 = time.perf_counter()
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                with torch.no_grad():
                    emb = st_model.encode(
                        text, convert_to_tensor=True, device=str(device),
                    )
                result = ps_frozen.process_input(emb, source_text=text)
                step_to_cidx_frozen[result["global_step"]] = cidx
                cidx += 1
        print(f"  Conv {ci + 1}/10", flush=True)
    print(f"  Dev: {time.perf_counter() - t0:.1f}s")

    frozen_hits, frozen_used, frozen_scores = evaluate_gated(
        ps_frozen, st_model, corpus, corpus_emb, step_to_cidx_frozen,
        qa_pairs, device,
    )
    vecdb_hits = sum(
        1 for q, a in qa_pairs
        for _ in [1]  # dummy loop
        if max(
            (token_f1(corpus[i.item()], a)
             for i in torch.topk(
                 torch.nn.functional.cosine_similarity(
                     st_model.encode(q, convert_to_tensor=True, device=str(device)).unsqueeze(0),
                     corpus_emb, dim=1,
                 ), 5,
             ).indices),
            default=0.0,
        ) > 0.05
    )
    print(f"  VecDB: {vecdb_hits}/100")
    print(f"  Gated (frozen): {frozen_hits}/100 (graph_used={frozen_used})")

    # ================================================================
    # Contrastive fine-tuning
    # ================================================================
    print(f"\n=== Contrastive fine-tuning (lr={args.ft_lr}) ===", flush=True)

    ft_model = copy.deepcopy(st_model)
    ft_model.to(device)
    ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=args.ft_lr)

    config_ft = SOMAConfig.developmental(
        sensor_output_dim=st_dim, text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2, associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2, integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2, integrator_output_dim=st_dim,
    )
    ps_ft = PredictiveSOMA(config_ft, device=device)
    step_to_cidx_ft: dict[int, int] = {}
    cidx = 0
    ft_updates = 0
    t0 = time.perf_counter()

    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"

                # Encode with gradient flow for contrastive loss
                ft_model.train()
                tokenized = ft_model.tokenize([text])
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                model_output = ft_model.forward(tokenized)
                emb_grad = model_output["sentence_embedding"][0]

                # Process through SOMA (uses detached embedding internally)
                with torch.no_grad():
                    emb_detached = emb_grad.detach()
                result = ps_ft.process_input(
                    emb_detached, source_text=text,
                )
                step_to_cidx_ft[result["global_step"]] = cidx

                # Contrastive loss: use graph topology as supervision
                loss = ps_ft.compute_contrastive_loss(emb_grad)
                if loss is not None and loss.item() > 0:
                    ft_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        ft_model.parameters(), max_norm=1.0,
                    )
                    ft_optimizer.step()
                    ft_updates += 1

                cidx += 1

        print(f"  Conv {ci + 1}/10: ft_updates={ft_updates}", flush=True)

    print(f"  Dev: {time.perf_counter() - t0:.1f}s")
    print(f"  Total FT updates: {ft_updates}")

    # Re-encode corpus with fine-tuned encoder
    ft_model.train(False)
    corpus_emb_ft = ft_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    ft_hits, ft_used, ft_scores = evaluate_gated(
        ps_ft, ft_model, corpus, corpus_emb_ft, step_to_cidx_ft,
        qa_pairs, device,
    )

    # VecDB with fine-tuned embeddings
    vecdb_ft_hits = sum(
        1 for q, a in qa_pairs
        for _ in [1]
        if max(
            (token_f1(corpus[i.item()], a)
             for i in torch.topk(
                 torch.nn.functional.cosine_similarity(
                     ft_model.encode(q, convert_to_tensor=True, device=str(device)).unsqueeze(0),
                     corpus_emb_ft, dim=1,
                 ), 5,
             ).indices),
            default=0.0,
        ) > 0.05
    )

    print(f"\n{'='*60}")
    print("PHASE 4b SUMMARY")
    print(f"{'='*60}")
    print(f"  VecDB (original):          {vecdb_hits}/100")
    print(f"  Gated (frozen):            {frozen_hits}/100 (used={frozen_used})")
    print(f"  VecDB (fine-tuned):        {vecdb_ft_hits}/100")
    print(f"  Gated (fine-tuned):        {ft_hits}/100 (used={ft_used})")
    print(f"  FT updates:                {ft_updates}")

    if ft_hits > frozen_hits:
        print(f"\n  >> Fine-tuning improves gated by +{ft_hits - frozen_hits}")
    elif ft_hits < frozen_hits:
        print(f"\n  >> Fine-tuning hurts gated by -{frozen_hits - ft_hits}")
    else:
        print(f"\n  >> Fine-tuning neutral for gated retrieval")


if __name__ == "__main__":
    main()
