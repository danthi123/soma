"""Plastic-graph activation test.

Protocol:
  Phase 1: Ingest all 1000 facts into 3 systems (SOMA plastic, SOMA
           frozen, chroma). Record initial per-topic R@5.
  Phase 2: Focused session on Topic A. Fire 50 queries on different
           attributes of Topic A, interleaved with 50 cross-topic
           queries. After each mini-batch (every 10 queries) on the
           SOMA-plastic system, call consolidate() to trigger
           synaptogenesis/pruning.
  Phase 3: Held-out evaluation: 50 fresh queries on Topic A's
           remaining attributes, plus 50 cross-topic queries.
           Measure per-system R@5 change.

Success criterion: SOMA plastic's Topic A R@5 increases by >0.05
AND is greater than SOMA frozen's (which should be flat).

Null result: SOMA plastic flat (same as frozen). Documents that the
graph substrate doesn't activate on this workload.

Usage::

    python -m benchmarks.plastic_graph.run_activation \\
        --topic-a-index 0 --out-suffix _v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.plastic_graph.topic_data import (
    Fact,
    TOPICS,
    generate_facts,
    generate_questions,
    get_fact_text,
)

RESULTS_DIR = Path("benchmarks/plastic_graph/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def _partition_questions(
    facts: list[Fact],
    topic_a: str,
    session_size: int = 50,
    heldout_size: int = 50,
) -> tuple[list[Fact], list[Fact], list[Fact], list[Fact]]:
    """Split per-topic question facts into (phase2_A, phase2_other,
    phase3_A, phase3_other).

    Strategy: each topic has 20 unique attributes. Use 10 attrs of
    Topic A for phase 2 session, 10 for phase 3 heldout. For other
    topics, aggregate ~5 questions per topic for interleave/control.
    """
    all_questions = generate_questions(facts, per_topic=20)
    topic_a_q = all_questions[topic_a]
    # 10 for phase 2, 10 for phase 3
    phase2_a = topic_a_q[:10]
    phase3_a = topic_a_q[10:20]

    # Cross-topic: 5 questions from each of the 9 other topics = 45
    phase2_other: list[Fact] = []
    phase3_other: list[Fact] = []
    for t, qs in all_questions.items():
        if t == topic_a:
            continue
        phase2_other.extend(qs[:3])   # 3 * 9 = 27
        phase3_other.extend(qs[3:6])  # 3 * 9 = 27

    # Replicate phase2_a to reach session_size focus; interleave
    # with other topics to simulate realistic mixed session
    phase2_a_rep: list[Fact] = (phase2_a * 5)[:session_size - len(phase2_other)]
    return phase2_a_rep, phase2_other, phase3_a, phase3_other


class _SomaAdapter:
    """Wraps MemoryLayer for the protocol. plastic=True attaches a
    SOMA graph and calls consolidate() periodically.

    When plastic=True we build a small BPE tokenizer + TextEncoder on
    the corpus so SOMA has its own latent space. Retrieval still uses
    the outer sbert embed_fn (384-d) — SOMA's 32-d graph is for
    plasticity only.
    """

    def __init__(
        self,
        embed_fn,
        dim: int,
        plastic: bool,
        hybrid_alpha: float | None = 0.3,
        graph_rerank_alpha: float = 0.0,
        corpus: list[str] | None = None,
        attach_soma_even_when_frozen: bool = False,
    ) -> None:
        """
        plastic=True  -> attach SOMA AND call consolidate() each call.
        plastic=False + attach_soma_even_when_frozen=True -> attach SOMA but
            never consolidate during session (frozen plastic graph baseline;
            needed for the graph-rerank path to have any signal at all).
        plastic=False + attach_soma_even_when_frozen=False -> no SOMA (pure
            vector baseline, identical effective behaviour to chroma).
        """
        from soma.core.config import SOMAConfig
        from soma.system import SOMA
        from soma.memory import MemoryLayer
        from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

        self.plastic = plastic
        # hybrid_alpha=None + graph_rerank_alpha>0 = use graph re-rank.
        # hybrid_alpha set = pure hybrid (graph ignored by retrieve()).
        self.hybrid_alpha = hybrid_alpha
        self.mem = MemoryLayer.ephemeral(
            embed_fn=embed_fn,
            embed_dim=dim,
            graph_rerank_alpha=graph_rerank_alpha,
        )
        should_attach = plastic or attach_soma_even_when_frozen
        if should_attach:
            assert corpus is not None, "SOMA attach needs corpus for BPE training"
            tokenizer = train_bpe_tokenizer(corpus, vocab_size=512)
            encoder = TextEncoder(tokenizer, embed_dim=32, max_seq_len=64)
            config = SOMAConfig.memory_layer(
                vocab_size=512, text_embed_dim=32, sensor_output_dim=32,
                max_input_tokens=64, max_nodes=1500,
            )
            # Pin SOMA to CUDA when available — pre-consolidate runs
            # 20x+ faster there. Falls back to CPU automatically.
            import torch as _torch
            _device = _torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu")
            soma = SOMA(config=config, device=_device)
            # TextEncoder must be on the same device as SOMA so its
            # output embeddings feed SOMA.step without a host<->device
            # copy. Otherwise consolidate() crashes on the matmul.
            encoder = encoder.to(_device)
            self.mem.attach_soma(soma, tokenizer, encoder)

    def store(self, text: str, meta: dict) -> None:
        self.mem.store(text, metadata=meta)

    def consolidate(self) -> int:
        # Only plastic calls consolidate; frozen stays static even if SOMA
        # is attached (for graph-rerank baseline).
        if not self.plastic:
            return 0
        return self.mem.consolidate()

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, dict]]:
        if self.hybrid_alpha is None:
            hits = self.mem.retrieve(query, k=k)
        else:
            hits = self.mem.retrieve(query, k=k, hybrid_alpha=self.hybrid_alpha)
        return [(h.text, h.metadata) for h in hits]

    def graph_stats(self) -> dict[str, Any]:
        if not self.plastic:
            return {}
        try:
            soma = self.mem._soma
            graph = soma.graph if hasattr(soma, "graph") else None
            if graph is None:
                return {}
            return {
                "n_nodes": len(graph.nodes),
                "n_edges": len(graph.edges),
            }
        except Exception as exc:
            logger.debug("graph_stats failed: %s", exc)
            return {}


class _ChromaAdapter:
    """chroma.Client with sbert embeddings, same interface as _SomaAdapter."""

    def __init__(self, sbert_model, dim: int) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import \
            SentenceTransformerEmbeddingFunction

        self.client = chromadb.EphemeralClient()
        # chroma's embedding-fn expects sbert name; we pass the loaded model
        ef = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        self.collection = self.client.create_collection(
            name="plastic_test", embedding_function=ef,
        )
        self._counter = 0

    def store(self, text: str, meta: dict) -> None:
        self._counter += 1
        self.collection.add(
            ids=[str(self._counter)],
            documents=[text],
            metadatas=[{k: v for k, v in meta.items() if isinstance(
                v, (str, int, float, bool)
            )}],
        )

    def consolidate(self) -> int:
        return 0  # chroma has no plasticity

    def retrieve(self, query: str, k: int = 5) -> list[tuple[str, dict]]:
        res = self.collection.query(query_texts=[query], n_results=k)
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        return list(zip(docs, metas, strict=True))

    def graph_stats(self) -> dict[str, Any]:
        return {}


def _score_recall(hits: list[tuple[str, dict]], gold_answer: str) -> int:
    """Return 1 if any hit text contains the gold answer verbatim, else 0."""
    for text, _ in hits:
        if gold_answer.lower() in text.lower():
            return 1
    return 0


def evaluate(
    adapter,
    questions: list[Fact],
    k: int = 5,
) -> dict[str, Any]:
    hits_count = 0
    per_topic: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for q in questions:
        hits = adapter.retrieve(q.question, k=k)
        hit = _score_recall(hits, q.answer)
        hits_count += hit
        per_topic.setdefault(q.topic, []).append(hit)
    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "n": len(questions),
        "r_at_k": hits_count / max(1, len(questions)),
        "per_topic_r_at_k": {
            t: sum(v) / max(1, len(v)) for t, v in per_topic.items()
        },
        "elapsed_ms": total_ms,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--topic-a-index", type=int, default=0,
                   help="Index into TOPICS. 0=Python preferences, etc.")
    p.add_argument("--out-suffix", default="")
    p.add_argument("--consolidate-every", type=int, default=10,
                   help="Consolidate SOMA-plastic every N queries in phase 2")
    p.add_argument("--graph-rerank-alpha", type=float, default=0.0,
                   help="If >0, use graph re-rank instead of hybrid at retrieve time.")
    p.add_argument("--k", type=int, default=5,
                   help="Top-k evaluation (R@k). Lower k = stricter precision test.")
    p.add_argument("--phase1-frac", type=float, default=0.5,
                   help="Fraction of facts to store in Phase 1 (rest streamed "
                        "into Phase 2 as 'new' entries).")
    args = p.parse_args()

    topic_a = TOPICS[args.topic_a_index][0]
    logger.info("Topic A: %s", topic_a)

    # Generate facts + questions
    facts = generate_facts()
    logger.info("Generated %d facts across %d topics",
                len(facts), len(TOPICS))

    phase2_a, phase2_other, phase3_a, phase3_other = _partition_questions(
        facts, topic_a,
    )
    logger.info(
        "phase2_a=%d, phase2_other=%d, phase3_a=%d, phase3_other=%d",
        len(phase2_a), len(phase2_other), len(phase3_a), len(phase3_other),
    )

    # Build adapters
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    dim = sbert.get_sentence_embedding_dimension()

    def embed_fn(text: str) -> torch.Tensor:
        return torch.tensor(sbert.encode(text, convert_to_numpy=True))

    # Build corpus strings upfront so plastic SOMA can train its BPE
    corpus_texts = [get_fact_text(f) for f in facts]

    # If graph_rerank_alpha > 0, use graph-rerank path (hybrid_alpha=None).
    # If 0, fall back to hybrid retrieve (original behaviour).
    use_graph_rerank = args.graph_rerank_alpha > 0.0
    hybrid_alpha_val: float | None = None if use_graph_rerank else 0.3
    logger.info(
        "retrieve mode: %s (graph_rerank_alpha=%.2f, hybrid_alpha=%s, k=%d)",
        "graph-rerank" if use_graph_rerank else "hybrid",
        args.graph_rerank_alpha,
        "None" if hybrid_alpha_val is None else f"{hybrid_alpha_val:.2f}",
        args.k,
    )

    systems: dict[str, Any] = {
        "soma_plastic": _SomaAdapter(
            embed_fn, dim, plastic=True,
            hybrid_alpha=hybrid_alpha_val,
            graph_rerank_alpha=args.graph_rerank_alpha,
            corpus=corpus_texts,
        ),
        # Frozen: attach SOMA so graph-rerank path has a graph to read, but
        # never consolidate. Plastic vs frozen differ only in whether the
        # graph grows during Phase 2.
        "soma_frozen": _SomaAdapter(
            embed_fn, dim, plastic=False,
            hybrid_alpha=hybrid_alpha_val,
            graph_rerank_alpha=args.graph_rerank_alpha,
            corpus=corpus_texts,
            attach_soma_even_when_frozen=use_graph_rerank,
        ),
        "chroma": _ChromaAdapter(sbert, dim),
    }

    # === Partition facts: Phase 1 ingest vs Phase 2 trickle ===
    # Bias Phase 2 trickle toward Topic A so plastic's consolidate()
    # during Phase 2 sees Topic-A-heavy new content.
    phase1_facts: list[Fact] = []
    phase2_trickle: list[Fact] = []
    for f in facts:
        # Within topic A, reserve the last 20% of paraphrase-instances
        # for Phase 2 trickle. Outside topic A, keep Phase 1 proportion.
        frac = args.phase1_frac if f.topic == topic_a else max(0.75, args.phase1_frac)
        # Deterministic split by hash of text so re-runs are stable.
        h = hash(get_fact_text(f)) % 1000
        if h / 1000.0 < frac:
            phase1_facts.append(f)
        else:
            phase2_trickle.append(f)
    logger.info("  phase1_facts=%d, phase2_trickle=%d (Topic A share of "
                "trickle: %.2f)",
                len(phase1_facts), len(phase2_trickle),
                sum(1 for f in phase2_trickle if f.topic == topic_a)
                / max(1, len(phase2_trickle)))

    # === Phase 1: ingest (subset only) ===
    logger.info("Phase 1: ingesting %d facts into 3 systems...",
                len(phase1_facts))
    for name, adapter in systems.items():
        t0 = time.perf_counter()
        for f in phase1_facts:
            adapter.store(
                get_fact_text(f),
                meta={"topic": f.topic, "attribute": f.attribute},
            )
        logger.info("  %s ingest: %.1fs", name, time.perf_counter() - t0)

    # === Phase 1 baseline eval ===
    # For graph-rerank mode, pre-consolidate once so frozen's graph has stable
    # capture too (otherwise initial R@k is pure cosine for both).
    if use_graph_rerank:
        logger.info("Pre-consolidating for graph-rerank (both plastic and frozen)...")
        n_plastic = systems["soma_plastic"].mem.consolidate()
        n_frozen = systems["soma_frozen"].mem.consolidate()
        logger.info("  plastic: %d, frozen: %d entries consolidated", n_plastic, n_frozen)

    logger.info("Phase 1 baseline: evaluating initial R@%d...", args.k)
    initial: dict[str, dict] = {}
    for name, adapter in systems.items():
        initial[name] = {
            "topic_a": evaluate(adapter, phase3_a, k=args.k),
            "other": evaluate(adapter, phase3_other, k=args.k),
            "graph": adapter.graph_stats(),
        }
        logger.info("  %s initial Topic A R@%d=%.3f, other R@%d=%.3f",
                    name, args.k,
                    initial[name]["topic_a"]["r_at_k"],
                    args.k,
                    initial[name]["other"]["r_at_k"])

    # === Phase 2: focused session on Topic A ===
    logger.info("Phase 2: focused session on %s", topic_a)
    # Interleave phase2_a (topic focus) with phase2_other
    # Take pairs: (topic_a_q, topic_a_q, other_q) and cycle
    session_queries: list[Fact] = []
    ia, io = 0, 0
    while ia < len(phase2_a) or io < len(phase2_other):
        for _ in range(2):
            if ia < len(phase2_a):
                session_queries.append(phase2_a[ia])
                ia += 1
        if io < len(phase2_other):
            session_queries.append(phase2_other[io])
            io += 1
    logger.info("  total session queries: %d (ratio 2:1 topic-A:other)",
                len(session_queries))

    # Interleave trickle stores with retrieves: every other step is a
    # store (Topic-A-biased). Plastic consolidates periodically and
    # graphifies the new entries; frozen never consolidates.
    trickle_idx = 0
    for step, q in enumerate(session_queries):
        # Retrieve step
        for name, adapter in systems.items():
            adapter.retrieve(q.question, k=args.k)
        # Trickle store — store one new fact per retrieve into all systems
        if trickle_idx < len(phase2_trickle):
            tf = phase2_trickle[trickle_idx]
            for name, adapter in systems.items():
                adapter.store(
                    get_fact_text(tf),
                    meta={"topic": tf.topic, "attribute": tf.attribute},
                )
            trickle_idx += 1
        # Consolidate plastic SOMA periodically (incremental — only new
        # entries since the last pass get graphified).
        if (step + 1) % args.consolidate_every == 0:
            n = systems["soma_plastic"].consolidate()
            logger.info("  step %d/%d: consolidated %d entries",
                        step + 1, len(session_queries), n)
    logger.info("  Phase 2 stored %d new trickle facts", trickle_idx)

    # Drain any remaining trickle facts into all systems so Phase 3 eval
    # sees the same total corpus on all systems. Only plastic will
    # consolidate them.
    remaining = len(phase2_trickle) - trickle_idx
    if remaining > 0:
        logger.info("  Draining %d remaining trickle facts into all "
                    "systems", remaining)
        for tf in phase2_trickle[trickle_idx:]:
            for name, adapter in systems.items():
                adapter.store(
                    get_fact_text(tf),
                    meta={"topic": tf.topic, "attribute": tf.attribute},
                )

    # Final consolidation
    n_final = systems["soma_plastic"].consolidate()
    logger.info("  final consolidation: %d entries", n_final)

    # === Phase 3: held-out eval ===
    logger.info("Phase 3: held-out evaluation at R@%d...", args.k)
    # Per-item retrieval capture so we can measure plastic-vs-frozen
    # disagreement after the run.
    per_item_phase3: dict[str, list[dict[str, Any]]] = {}
    for name, adapter in systems.items():
        rows = []
        for q in phase3_a + phase3_other:
            hits = adapter.retrieve(q.question, k=args.k)
            top1_text = hits[0][0] if hits else ""
            hit = 1 if any(q.answer.lower() in t.lower() for t, _ in hits) else 0
            rows.append({
                "topic": q.topic,
                "attribute": q.attribute,
                "question": q.question,
                "answer": q.answer,
                "top1_text": top1_text,
                "top1_contains_answer": (1 if q.answer.lower()
                                         in top1_text.lower() else 0),
                "hit_at_k": hit,
            })
        per_item_phase3[name] = rows

    final: dict[str, dict] = {}
    for name, adapter in systems.items():
        final[name] = {
            "topic_a": evaluate(adapter, phase3_a, k=args.k),
            "other": evaluate(adapter, phase3_other, k=args.k),
            "graph": adapter.graph_stats(),
        }
        logger.info(
            "  %s final Topic A R@%d=%.3f (delta=%+.3f), other R@%d=%.3f (delta=%+.3f)",
            name, args.k,
            final[name]["topic_a"]["r_at_k"],
            final[name]["topic_a"]["r_at_k"] - initial[name]["topic_a"]["r_at_k"],
            args.k,
            final[name]["other"]["r_at_k"],
            final[name]["other"]["r_at_k"] - initial[name]["other"]["r_at_k"],
        )

    # === Report ===
    report = {
        "topic_a": topic_a,
        "n_facts": len(facts),
        "phase1_facts": len(phase1_facts),
        "phase2_trickle": len(phase2_trickle),
        "session_queries": len(session_queries),
        "consolidate_every": args.consolidate_every,
        "graph_rerank_alpha": args.graph_rerank_alpha,
        "k": args.k,
        "retrieve_mode": "graph-rerank" if use_graph_rerank else "hybrid",
        "initial": initial,
        "final": final,
        "per_item_phase3": per_item_phase3,
    }
    out_path = RESULTS_DIR / f"activation_test{args.out_suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written to %s", out_path)

    # Print markdown summary
    print("\n## Plastic-graph activation test\n")
    print(f"Topic A: **{topic_a}**  |  N facts: {len(facts)}  |  "
          f"session queries: {len(session_queries)}\n")
    print("| System | Initial Topic A R@5 | Final Topic A R@5 | "
          "delta | other topics delta |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name in ["soma_plastic", "soma_frozen", "chroma"]:
        i_a = initial[name]["topic_a"]["r_at_k"]
        f_a = final[name]["topic_a"]["r_at_k"]
        i_o = initial[name]["other"]["r_at_k"]
        f_o = final[name]["other"]["r_at_k"]
        print(f"| {name} | {i_a:.3f} | {f_a:.3f} | "
              f"{f_a - i_a:+.3f} | {f_o - i_o:+.3f} |")

    if initial["soma_plastic"]["graph"]:
        g_i = initial["soma_plastic"]["graph"]
        g_f = final["soma_plastic"]["graph"]
        print(f"\nSOMA plastic graph: {g_i.get('n_nodes', 0)} -> "
              f"{g_f.get('n_nodes', 0)} nodes, "
              f"{g_i.get('n_edges', 0)} -> "
              f"{g_f.get('n_edges', 0)} edges")


if __name__ == "__main__":
    main()
