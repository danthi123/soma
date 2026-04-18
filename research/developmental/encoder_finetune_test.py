"""Phase 4: Graph-driven encoder fine-tuning.

Use SOMA's prediction error to fine-tune the pretrained encoder.
When prediction error is high, the current embedding isn't well-
represented by the graph's learned structure. Fine-tuning reshapes
the embedding space to align with the graph's topology.

Combined with Phase 3's confidence-gated reranking, this should
expand the range of queries where the graph has useful signal
(currently ~32/100 pass the confidence gate).

Strategy: feed pretrained embeddings through SOMA with gradient
flow enabled. SOMA's prediction loss backprops into the encoder,
adapting it to produce embeddings the graph can better differentiate.
Then evaluate with the confidence-gated hybrid from Phase 3.
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


def confidence_gated_retrieve(
    ps: PredictiveSOMA,
    query_emb: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    corpus: list[str],
    step_to_corpus_idx: dict[int, int],
    recall_k: int = 20,
    gate_threshold: float = 0.05,
    rerank_weight: float = 0.2,
) -> list[int]:
    """Retrieve top-5 using confidence-gated hybrid."""
    device = query_emb.device

    # Embedding recall
    sims = torch.nn.functional.cosine_similarity(
        query_emb.unsqueeze(0), corpus_embeddings, dim=1,
    )
    top_k_sims, top_k_indices = torch.topk(
        sims, min(recall_k, len(corpus)),
    )

    # Graph fingerprint
    modality = ps.config.input_modalities[0]
    ps.soma.step({modality: query_emb}, eval_mode=True)
    ps._diversify_activations(query_emb)
    ps._apply_lateral_inhibition()
    query_fp = ps._get_node_fingerprint()

    # Fingerprint similarity for candidates
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

    # Confidence gate
    if fp_sims:
        fp_sorted = sorted(fp_sims, reverse=True)
        confidence = fp_sorted[0] - (fp_sorted[1] if len(fp_sorted) > 1 else 0)
    else:
        confidence = 0.0

    if confidence >= gate_threshold:
        rerank_scores = []
        for i in range(len(top_k_indices)):
            emb_sim = top_k_sims[i].item()
            combined = (1 - rerank_weight) * emb_sim + rerank_weight * fp_sims[i]
            rerank_scores.append((combined, top_k_indices[i].item()))
        rerank_scores.sort(key=lambda x: -x[0])
        return [idx for _, idx in rerank_scores[:5]]
    else:
        return [top_k_indices[i].item() for i in range(min(5, len(top_k_indices)))]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-per-conv", type=int, default=10)
    parser.add_argument("--ft-lr", type=float, default=1e-5,
                        help="Fine-tuning learning rate for encoder")
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

    qa_pairs = []
    for conv in conversations:
        for qa in conv.qa_pairs[:args.qa_per_conv]:
            qa_pairs.append((qa.question, str(qa.answer)))

    # ================================================================
    # Strategy A: No fine-tuning (Phase 3 baseline)
    # ================================================================
    print("\n=== A: No fine-tuning (Phase 3 baseline) ===", flush=True)

    # Encode corpus with frozen encoder
    corpus_embeddings = st_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

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
    ps_frozen = PredictiveSOMA(config, device=device)

    t0 = time.perf_counter()
    step_to_idx_frozen: dict[int, int] = {}
    cidx = 0
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"
                with torch.no_grad():
                    emb = st_model.encode(
                        text, convert_to_tensor=True, device=str(device),
                    )
                result = ps_frozen.process_input(emb, source_text=text)
                step_to_idx_frozen[result["global_step"]] = cidx
                cidx += 1
        print(f"  Conv {ci + 1}/10", flush=True)
    print(f"  Development: {time.perf_counter() - t0:.1f}s")

    # Evaluate frozen
    vecdb_scores = []
    gated_frozen_scores = []
    for question, answer in qa_pairs:
        q_emb = st_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )
        # VecDB
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), corpus_embeddings, dim=1,
        )
        top5 = torch.topk(sims, 5).indices
        vecdb_scores.append(max(
            (token_f1(corpus[i.item()], answer) for i in top5),
            default=0.0,
        ))
        # Gated
        top5_gated = confidence_gated_retrieve(
            ps_frozen, q_emb, corpus_embeddings, corpus,
            step_to_idx_frozen, gate_threshold=0.05,
        )
        gated_frozen_scores.append(max(
            (token_f1(corpus[idx], answer) for idx in top5_gated),
            default=0.0,
        ))

    vecdb_hits = sum(1 for s in vecdb_scores if s > 0.05)
    frozen_hits = sum(1 for s in gated_frozen_scores if s > 0.05)
    print(f"  VecDB: {vecdb_hits}/100")
    print(f"  Gated (frozen): {frozen_hits}/100")

    # ================================================================
    # Strategy B: Fine-tuned encoder
    # ================================================================
    print(f"\n=== B: Fine-tuned encoder (lr={args.ft_lr}) ===", flush=True)

    # Clone the encoder for fine-tuning
    import copy
    ft_model = copy.deepcopy(st_model)
    ft_model.to(device)

    # Set up optimizer for encoder fine-tuning
    ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=args.ft_lr)

    config_ft = SOMAConfig.developmental(
        sensor_output_dim=st_dim,
        text_embed_dim=st_dim,
        associator_input_dim=st_dim // 2,
        associator_hidden_dim=st_dim,
        associator_output_dim=st_dim // 2,
        integrator_input_dim=st_dim,
        integrator_hidden_dim=st_dim * 2,
        integrator_output_dim=st_dim,
    )
    ps_ft = PredictiveSOMA(config_ft, device=device)

    t0 = time.perf_counter()
    step_to_idx_ft: dict[int, int] = {}
    cidx = 0
    ft_losses = []
    for ci, conv in enumerate(conversations):
        for session in conv.sessions:
            for turn in session.turns:
                text = f"[{session.date_time}] {turn.speaker}: {turn.text}"

                # Encode with gradient flow
                ft_model.train()
                encoded = ft_model.encode(
                    text, convert_to_tensor=True, device=str(device),
                )
                # sentence-transformers encode() detaches by default
                # We need to re-encode with grad enabled
                tokenized = ft_model.tokenize([text])
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                model_output = ft_model.forward(tokenized)
                emb = model_output["sentence_embedding"][0]

                result = ps_ft.process_input(emb, source_text=text)
                pred_err = result.get("prediction_error", 0.0)

                # Fine-tune: backprop prediction error through encoder
                if pred_err > 1e-5 and emb.grad_fn is not None:
                    ft_optimizer.zero_grad()
                    # The prediction error from SOMA was already backpropped
                    # in process_input. But we need our own loss to fine-tune
                    # the encoder. Use the SOMA step loss if available.
                    loss = result.get("loss")
                    if loss is not None and hasattr(loss, "backward"):
                        try:
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                ft_model.parameters(), max_norm=1.0,
                            )
                            ft_optimizer.step()
                            ft_losses.append(loss.item())
                        except RuntimeError:
                            # Graph already freed
                            pass

                step_to_idx_ft[result["global_step"]] = cidx
                cidx += 1

        avg_loss = sum(ft_losses[-100:]) / max(len(ft_losses[-100:]), 1)
        print(f"  Conv {ci + 1}/10: avg_loss={avg_loss:.6f}", flush=True)

    print(f"  Development: {time.perf_counter() - t0:.1f}s")
    print(f"  Total FT updates: {len(ft_losses)}")

    # Re-encode corpus with fine-tuned encoder
    ft_model.train(False)
    corpus_embeddings_ft = ft_model.encode(
        corpus, batch_size=64, show_progress_bar=False,
        convert_to_tensor=True, device=str(device),
    )

    # Evaluate fine-tuned
    vecdb_ft_scores = []
    gated_ft_scores = []
    for question, answer in qa_pairs:
        q_emb = ft_model.encode(
            question, convert_to_tensor=True, device=str(device),
        )
        # VecDB with fine-tuned embeddings
        sims = torch.nn.functional.cosine_similarity(
            q_emb.unsqueeze(0), corpus_embeddings_ft, dim=1,
        )
        top5 = torch.topk(sims, 5).indices
        vecdb_ft_scores.append(max(
            (token_f1(corpus[i.item()], answer) for i in top5),
            default=0.0,
        ))
        # Gated with fine-tuned
        top5_gated = confidence_gated_retrieve(
            ps_ft, q_emb, corpus_embeddings_ft, corpus,
            step_to_idx_ft, gate_threshold=0.05,
        )
        gated_ft_scores.append(max(
            (token_f1(corpus[idx], answer) for idx in top5_gated),
            default=0.0,
        ))

    vecdb_ft_hits = sum(1 for s in vecdb_ft_scores if s > 0.05)
    ft_gated_hits = sum(1 for s in gated_ft_scores if s > 0.05)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)
    print(f"  VecDB (original encoder):     {vecdb_hits}/100")
    print(f"  Gated (frozen encoder):        {frozen_hits}/100")
    print(f"  VecDB (fine-tuned encoder):    {vecdb_ft_hits}/100")
    print(f"  Gated (fine-tuned encoder):    {ft_gated_hits}/100")

    if ft_gated_hits > frozen_hits:
        print(f"\n  >> Fine-tuning improves gated retrieval by "
              f"+{ft_gated_hits - frozen_hits} hits")
    elif ft_gated_hits == frozen_hits:
        print(f"\n  >> Fine-tuning has no effect on gated retrieval")
    else:
        print(f"\n  >> Fine-tuning hurts gated retrieval by "
              f"-{frozen_hits - ft_gated_hits} hits")


if __name__ == "__main__":
    main()
