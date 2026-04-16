# Research Direction C: Plastic Graph Under the Memory Layer

> **For Claude:** research-phase plan. Implementing starts with Sub-phase C1.

**Hypothesis:** the 2026-04-15 SOMA-content ablation showed the plastic graph contributed noise to next-token LM prediction. But *retrieval* is a different objective entirely — and has never been tested. The graph should help retrieval in ways pure vector lookup structurally cannot: learned co-occurrence via edges, centrality as a relevance prior, graph-traversal for composite queries, drift detection via homeostasis, age-out via pruning.

**Strategic framing:** this is the direction that pays engineering back into the shipping product. If it works, SOMA-the-memory-layer gets a concrete research-backed differentiator that no vector-DB-plus-RAG competitor has. If it doesn't, we learn the graph substrate isn't load-bearing for retrieval either, which decisively informs the product positioning.

**Target users:** the agent-memory product we just shipped Phases 16-37 for. Any research win lands directly in the next product release.

**Prerequisite:** seed-graph integrator bug fix (`860cc82`) in every eval. Pre-fix results are meaningless.

---

## Sub-phase C1: Retrieval-objective training harness + synthetic signal test

**Goal:** infrastructure + first "is there any signal?" measurement. Today SOMA's training objective is next-token LM; we need a retrieval-objective training signal and the harness to evaluate it.

**Experimental method:**
- **Synthetic dataset:** 1000 text snippets with controllable co-occurrence structure — e.g., 20 topic-pairs where "A-B" and "A-C" appear together in training but "B-C" never does. At eval, query "A" should retrieve both B and C (direct), but a model that uses co-occurrence edges should *also* retrieve B when querying C (one-hop transitive).
- **Pure-vector baseline:** current MemoryLayer retrieval. Cosine over SBERT embeddings. This is what C1's SOMA-enhanced version must beat on transitive queries.
- **SOMA-enhanced retrieval (C1 version):** when a retrieve(query) hits node X, also score X's graph neighbors via edge-strength. Top-K blended by α (learnable in later phase, fixed α=0.3 here).
- **Metrics:** R@1, R@5, MRR for (a) direct queries (should tie), (b) one-hop transitive queries (should favor SOMA if edges carry signal), (c) unrelated queries (should tie; SOMA shouldn't hurt).

**Files:**
- Create: `research/graph_memory/synthetic_corpus.py` — generates the controlled co-occurrence dataset
- Create: `research/graph_memory/harness.py` — trains MemoryLayer with graph attached, runs eval
- Create: `research/graph_memory/baselines.py` — pure-vector MemoryLayer reference
- Create: `research/graph_memory/reports/c1_synthetic_signal.md`
- Create: `tests/test_research/test_graph_memory_c1.py`

**Training loop specifics:**
1. Instantiate `MemoryLayer` with SOMA graph attached (`attach_soma`).
2. Ingest all 1000 snippets via `store`.
3. Run `consolidate` N times (N ∈ {0, 10, 100, 1000} — sweep).
4. After each N, run the eval query set.
5. Record how edge strength correlates with co-occurrence.

**Success criterion (C1 gate):** on one-hop transitive queries, SOMA-enhanced R@5 beats pure-vector by ≥5 points after 100 consolidation cycles. This is the "does the graph carry ANY retrieval signal at all?" question. If the gap is ≤1 point, the hypothesis is weak — proceed to C1b failure analysis instead of C2.

**Scope:** ~1 week (this phase is dispatched to an agent now).

---

## Sub-phase C1b (contingent): Failure mode analysis

Only runs if C1's gate fails. Ablate each hypothesized contributor:
- Edge strengths actually differentiating after consolidation? (We know from prior plasticity experiment they move from std 0 → 0.0006 → 0.0011, but is that signal or noise?)
- Graph structure correlating with co-occurrence at all? (Spearman rank between edge weight and empirical co-occurrence.)
- Retrieval blend formula broken? (Try pure-graph-rank, pure-vector, and the blend — three-way compare.)

If failure analysis says "graph captures no co-occurrence signal," research-direction C is dead and we fold back to A, B, or D. Honest result.

---

## Sub-phase C2: Real-data retrieval benchmark

**Goal:** move from synthetic to realistic. Use the LoCoMo-QA corpus (already loaded by existing benchmarks) and the threshold-calibration sweep bench (4×4 grid we have) as comparison frames.

**Experimental method:**
- Input: LoCoMo conversations, store via ConversationalMemory + graph attached, run QA eval harness.
- Compare:
  - Pure-vector (baseline, already measured in `benchmarks/reports/locomo_qa_smoke.md`).
  - Graph-blended retrieval with α swept over {0.0, 0.1, 0.3, 0.5, 0.7}.
  - Graph-traversal expansion (top-K vectors then 1-hop edge expansion).
- Metrics: R@k (retrieval recall, LLM-independent) + QA accuracy (LLM-dependent, but LoCoMo ceiling ≈ 24% so measure the ceiling delta, not absolute).

**Files:**
- Extend: `benchmarks/run_locomo.py` with `--retrieval-mode {pure_vector, graph_blend, graph_expand}`
- Create: `research/graph_memory/reports/c2_locomo_retrieval.md`

**Success criterion:** on R@5, graph-blend beats pure-vector by ≥3 points at best α, at any consolidation cycle count. If no α helps, proceed to C3 with a different retrieval signal (not blend).

**Scope:** ~1-1.5 weeks.

---

## Sub-phase C3: Learned blend + graph traversal strategies

**Goal:** replace hand-picked α with a learned gate. Also test multi-hop traversal and centrality-weighted blend.

**Experimental strategies:**
- **Learned α:** small 2-layer MLP that takes (query_embedding, top-K vector scores, top-K graph scores) → scalar α per query.
- **Multi-hop:** expand retrieved set by 1-hop and 2-hop neighbors, re-score.
- **Centrality prior:** nodes with higher PageRank contribute more to final ranking.

**Files:**
- Create: `research/graph_memory/learned_blend.py`
- Create: `research/graph_memory/reports/c3_strategies.md`

**Success criterion:** at least one strategy beats the best-α fixed blend from C2 on both R@5 and MRR. If not, the fixed-α is the ceiling and we publish that.

**Scope:** ~2 weeks.

---

## Sub-phase C4: Consolidation-during-retrieval experiment

**Goal:** the existing `MemoryLayer.retrieve` is a pure read. What if it ran a tiny consolidation pulse first, letting the graph briefly update its weights given the query context, then read activations? This would make retrieval *adaptive* to the query — a qualitatively different operation.

**Experimental method:**
- Compare: (a) retrieve without consolidation, (b) retrieve after N=1 consolidation step with query as seed, (c) retrieve after N=3 steps.
- Measure latency cost alongside accuracy gain. This matters for the product — a 10% R@5 gain at 50× latency isn't shippable.

**Files:**
- Create: `research/graph_memory/query_time_consolidation.py`
- Create: `research/graph_memory/reports/c4_adaptive_retrieval.md`

**Success criterion:** at least one (accuracy, latency) point Pareto-dominates the C3 best. Otherwise C4 is a negative result and we note it.

**Scope:** ~1-1.5 weeks.

---

## Sub-phase C5: Integration + product writeup

**Goal:** best-performing configuration from C1-C4 becomes an opt-in MemoryLayer feature (`retrieval_mode="graph"` or similar). Published as a research blog post + paper-draft section + cookbook recipe.

**Files:**
- Modify: `src/soma/memory/api.py` — add the retrieval mode.
- Modify: `docs/cookbook.md` — new recipe §24.
- Create: `docs/papers/2026-xx-graph-memory-retrieval.md`.

**Scope:** ~1-2 weeks.

---

## Total estimated scope

6-9 weeks. Highest strategic payoff (research that lifts the product). C1 is the load-bearing gate — dispatching now.

**Adjacency:** touches `src/soma/memory/api.py` in C5 (shipped code). C1-C4 live entirely under `research/graph_memory/`. No conflicts with the four production-readiness phases (38-41).
