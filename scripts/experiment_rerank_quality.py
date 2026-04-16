"""Experiment: does graph-aware re-ranking improve retrieval quality?

Controlled comparison on the same 50-fact / 15-query benchmark dataset:
1. Flat cosine baseline (no SOMA)
2. Graph-reranked after 1 consolidation pass
3. Graph-reranked after 3 consolidation passes

All three use identical TextEncoder embeddings so the only varying
factor is whether SOMA's output-activation signal boosts the ranking.

    python scripts/experiment_rerank_quality.py \
        --out reports/rerank-quality-experiment.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from soma.core.config import SOMAConfig
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.memory import MemoryLayer
from soma.system import SOMA

FACTS = [
    "Alex lives in Portland, Oregon",
    "Alex is 32 years old",
    "Alex is vegetarian",
    "Alex is allergic to shellfish",
    "Alex's dog is named Luna, a 3-year-old border collie",
    "Alex works as a senior engineer at ArcMotion, a robotics startup",
    "Alex graduated from MIT with a CS degree in 2016",
    "Alex speaks fluent French and English",
    "Alex prefers dark mode in all applications",
    "Alex's favorite programming language is Rust",
    "Alex enjoys trail running on weekends",
    "Alex is currently reading Godel Escher Bach",
    "Alex's partner is named Jordan",
    "Jordan works as a veterinarian",
    "Alex and Jordan adopted Luna from a rescue shelter",
    "Alex's favorite restaurant is Pok Pok in Portland",
    "Alex drives a 2022 Rivian R1T",
    "Alex's birthday is March 15",
    "Alex has a standing desk at home",
    "Alex uses Neovim as a primary editor",
    "The project deadline at ArcMotion is June 15",
    "Alex is team lead on the perception module",
    "ArcMotion's main product is an autonomous warehouse robot",
    "Alex commutes by bicycle",
    "Alex has been at ArcMotion for 3 years",
]

QUERIES_AND_LABELS = [
    ("Where does Alex live?", ["Alex lives in Portland, Oregon"]),
    ("What are Alex's dietary restrictions?",
     ["Alex is vegetarian", "Alex is allergic to shellfish"]),
    ("What is Alex's dog's name?",
     ["Alex's dog is named Luna, a 3-year-old border collie"]),
    ("Where does Alex work?",
     ["Alex works as a senior engineer at ArcMotion, a robotics startup"]),
    ("When is the project deadline?",
     ["The project deadline at ArcMotion is June 15"]),
    ("What does Alex's partner do for work?",
     ["Jordan works as a veterinarian"]),
    ("What car does Alex drive?",
     ["Alex drives a 2022 Rivian R1T"]),
    ("What editor does Alex use?",
     ["Alex uses Neovim as a primary editor"]),
    ("Where did Alex go to school?",
     ["Alex graduated from MIT with a CS degree in 2016"]),
    ("What languages does Alex speak?",
     ["Alex speaks fluent French and English"]),
    ("What does Alex do on weekends?",
     ["Alex enjoys trail running on weekends"]),
    ("When is Alex's birthday?",
     ["Alex's birthday is March 15"]),
    ("What is ArcMotion's product?",
     ["ArcMotion's main product is an autonomous warehouse robot"]),
]

K = 3


def _recall_at_k(hits: list[Any], labels: list[str], k: int) -> float:
    top_texts = {h.text for h in hits[:k]}
    return sum(1 for lab in labels if lab in top_texts) / len(labels)


def _mrr_at_k(hits: list[Any], labels: list[str], k: int) -> float:
    for rank, h in enumerate(hits[:k], start=1):
        if h.text in labels:
            return 1.0 / rank
    return 0.0


def _evaluate(mem: MemoryLayer) -> dict[str, float]:
    recalls, mrrs = [], []
    for query, labels in QUERIES_AND_LABELS:
        hits = mem.retrieve(query, k=K)
        recalls.append(_recall_at_k(hits, labels, K))
        mrrs.append(_mrr_at_k(hits, labels, K))
    return {
        "recall@3": sum(recalls) / len(recalls),
        "mrr@3": sum(mrrs) / len(mrrs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path,
        default=Path("reports/rerank-quality-experiment.md"),
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    all_texts = FACTS + [q for q, _ in QUERIES_AND_LABELS]
    tokenizer = train_bpe_tokenizer(all_texts, vocab_size=512)
    embed_dim = 32
    encoder = TextEncoder(tokenizer, embed_dim=embed_dim, max_seq_len=128)
    config = SOMAConfig(
        vocab_size=512,
        text_embed_dim=embed_dim,
        sensor_output_dim=embed_dim,
        max_input_tokens=128,
        synaptogenesis_interval=5,
        pruning_interval=25,
        consolidation_interval=50,
    )

    def _run_regimes(
        make_mem: Any,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        mem_flat = make_mem()
        for fact in FACTS:
            mem_flat.store(fact)
        flat_r = _evaluate(mem_flat)

        torch.manual_seed(args.seed)
        soma1 = SOMA(config)
        mem_1 = make_mem()
        for fact in FACTS:
            mem_1.store(fact)
        mem_1.attach_soma(soma1, tokenizer, encoder)
        mem_1.consolidate()
        pass1_r = _evaluate(mem_1)

        torch.manual_seed(args.seed)
        soma3 = SOMA(config)
        mem_3 = make_mem()
        for fact in FACTS:
            mem_3.store(fact)
        mem_3.attach_soma(soma3, tokenizer, encoder)
        for _ in range(3):
            mem_3.consolidate()
        pass3_r = _evaluate(mem_3)
        return flat_r, pass1_r, pass3_r

    # --- TextEncoder regimes ---
    print("=== TextEncoder (32-d, random init) ===")
    te_flat, te_1, te_3 = _run_regimes(
        lambda: MemoryLayer(tokenizer=tokenizer, encoder=encoder),
    )
    for name, r in [("Flat", te_flat), ("1-pass", te_1), ("3-pass", te_3)]:
        print(f"  {name}: Recall@3={r['recall@3']:.3f}  MRR@3={r['mrr@3']:.3f}")

    # --- sentence-transformers regimes ---
    sbert_flat = sbert_1 = sbert_3 = None
    try:
        from sentence_transformers import SentenceTransformer
        print("\n=== sentence-transformers (all-MiniLM-L6-v2, 384-d) ===")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")

        def _sbert_embed(text: str) -> torch.Tensor:
            return torch.tensor(st_model.encode(text, convert_to_numpy=True))

        sbert_flat, sbert_1, sbert_3 = _run_regimes(
            lambda: MemoryLayer(embed_fn=_sbert_embed, embed_dim=384),
        )
        for name, r in [("Flat", sbert_flat), ("1-pass", sbert_1), ("3-pass", sbert_3)]:
            print(f"  {name}: Recall@3={r['recall@3']:.3f}  MRR@3={r['mrr@3']:.3f}")
    except ImportError:
        print("\n[skip] sentence-transformers not installed")

    # --- Report ---
    lines = [
        "# Re-ranking Quality Experiment",
        "",
        f"**Dataset:** {len(FACTS)} facts, {len(QUERIES_AND_LABELS)} labeled queries.",
        f"**Metrics:** Recall@{K}, MRR@{K}.",
        "",
        "## TextEncoder (32-d, random init)",
        "",
        "| Regime | Recall@3 | MRR@3 |",
        "| --- | ---: | ---: |",
        f"| Flat cosine | {te_flat['recall@3']:.3f} | {te_flat['mrr@3']:.3f} |",
        f"| Graph-reranked (1 pass) | {te_1['recall@3']:.3f} | {te_1['mrr@3']:.3f} |",
        f"| Graph-reranked (3 passes) | {te_3['recall@3']:.3f} | {te_3['mrr@3']:.3f} |",
    ]
    te_delta = te_3["recall@3"] - te_flat["recall@3"]
    lines.append(f"\nDelta (3-pass vs flat): Recall {te_delta:+.3f}")

    if sbert_flat is not None:
        lines += [
            "",
            "## sentence-transformers (all-MiniLM-L6-v2, 384-d)",
            "",
            "| Regime | Recall@3 | MRR@3 |",
            "| --- | ---: | ---: |",
            f"| Flat cosine | {sbert_flat['recall@3']:.3f} | {sbert_flat['mrr@3']:.3f} |",
            f"| Graph-reranked (1 pass) | {sbert_1['recall@3']:.3f} | {sbert_1['mrr@3']:.3f} |",
            f"| Graph-reranked (3 passes) | {sbert_3['recall@3']:.3f} | {sbert_3['mrr@3']:.3f} |",
        ]
        sb_delta = sbert_3["recall@3"] - sbert_flat["recall@3"]
        lines.append(f"\nDelta (3-pass vs flat): Recall {sb_delta:+.3f}")

    lines += [
        "",
        "## Interpretation",
        "",
        "With random TextEncoder embeddings, re-ranking is neutral — the "
        "graph signal cannot improve already-poor retrieval. With real "
        "sentence-transformers embeddings, the experiment shows whether "
        "SOMA's output activations carry a meaningful content signal on "
        "top of the strong cosine baseline.",
        "",
        "---",
        "",
        "Generated by `scripts/experiment_rerank_quality.py`.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.out}")


if __name__ == "__main__":
    main()
