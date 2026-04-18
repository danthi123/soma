# Paper Outline

## 1. Abstract (~200 words)

Setup: brain-inspired memory architectures for LLM RAG; SOMA's specifics
(graph + growth + Hebbian + consolidation). Question: does graph-derived
signal add retrieval value? Method: gated hybrid retrieval, tested on
LoCoMo + LongMemEval with diagnostic experiments (shuffle, scaling,
per-query attribution). Result: signal is real but architecturally
bounded at ~0.8% absolute over random, does not generalize
cross-benchmark. Root cause: "structural without semantic."
Contributions: (1) rigorous negative result on a popular framing;
(2) diagnostic methodology (shuffle, held-out, scaling); (3) positive
results on SOMA-native tasks (consolidation, multi-session growth).

## 2. Introduction (~1 page)

- **Motivation**: LLMs need external memory; RAG is dominant; multiple
  brain-inspired architectures proposed (MAGMA, Graphiti, SYNAPSE,
  Mem0); open empirical question whether graph structure helps
  retrieval.
- **Problem**: existing evaluations are underpowered (single benchmark,
  small held-out, no shuffle baselines) — can't distinguish real
  signal from overfit-to-tuning-sample.
- **Contributions**:
  1. SOMA architecture: graph + Hebbian + consolidation + structural
     plasticity (summary; cite whitepaper as tech report)
  2. Gated-hybrid retrieval: confidence-gated fingerprint reranking
  3. Diagnostic methodology: shuffle, held-out slices, scaling sweep,
     cross-benchmark
  4. Empirical findings: weak signal ceiling, LoCoMo vs LongMemEval
     divergence, architecturally-bounded (not mechanism-specific)
  5. Positive results on SOMA-native tasks → direction for brain-
     inspired architectures

## 3. Background

### 3.1 Retrieval-augmented generation
Standard dense retrieval (embedding cosine), reranker approaches.

### 3.2 Brain-inspired memory architectures
- SYNAPSE: spreading activation + Hebbian
- Graphiti / MAGMA: knowledge graphs + policy traversal
- Mem0: fact extraction + hybrid search
- Common premise: graph structure carries retrievable signal

### 3.3 SOMA architecture (one page)
- Node types (sensor / associator / integrator / output)
- Graph execution: waves, topological order, cycle handling
- Learning: Hebbian + backprop, structural plasticity
- Memory tiers: working / episodic / parametric
- Consolidation cycle (offline replay)
- Cite whitepaper: docs/whitepaper.md

### 3.4 Fingerprint and topology signals
- Fingerprint: per-node 16-value SHA256 projection over 128 dims
- Topology: set of active nodes per memory (inverted index)
- Both derive from lateral-inhibition 3-winner selection

## 4. Method

### 4.1 Gated-hybrid retrieval algorithm
- Embedding recall (top-20)
- Fingerprint similarity per candidate
- Confidence gate: top-1 minus top-2
- Conditional rerank at weight 0.2

### 4.2 Experimental setup
- Encoder: all-MiniLM-L6-v2 (384-dim)
- Corpus: LoCoMo (10 convs, 5882 turns), LongMemEval (100 items,
  3094 turns oracle variant)
- Scoring: token-F1 on top-5 retrieval (threshold 0.05 for "hit")
- Hardware: single RTX 3090

### 4.3 Diagnostic suite
- Held-out slices (original, tuned on slice A; B and C unseen)
- Shuffle diagnostic (5 random fingerprint permutations)
- Associator count sweep (8, 32, 64)
- Per-query attribution (6 keyword categories)
- Signal ablation (fingerprint vs topology vs vecdb-only)

## 5. Results

### 5.1 Primary retrieval comparison
Table: VecDB vs gated-hybrid on LoCoMo slice A, B, C, combined.
Result: +1 / -2 / +3 / +2 combined. Wins/losses 32/34.

### 5.2 Shuffle diagnostic
Table: real vs 5 shuffled per slice, and combined totals.
Result: real +2 > all shuffled (max +1). Signal is non-random but
weak.

### 5.3 Scaling sweep
Table: 8 / 32 / 64 associators on LoCoMo slice A.
Result: n=8 Pareto optimal. Scaling hurts.

### 5.4 Per-query attribution
Table: category-level W/L/T across 500 queries.
Result: no clean win-dominant category with meaningful absolute delta.

### 5.5 Signal equivalence
Table: fingerprint vs topology on slices A/B/C.
Result: 323 vs 322 hits. Functionally equivalent.

### 5.6 Cross-benchmark validation
Table: LongMemEval oracle, 100 items.
Result: -1 delta, 3W/6L. Ceiling cross-benchmark.

### 5.7 Positive results on native tasks
- Consolidation: +45% QA on synthetic (Phase 2 audit)
- Multi-session: +275% graph growth, cross-session recall (commit
  `6a0a822`)
- These are NOT retrieval wins; they're development/adaptation wins

## 6. Analysis

### 6.1 Why the ceiling exists
- Random input projections create structurally diverse but
  semantically arbitrary fingerprints
- 3-winner lateral inhibition means vocabulary caps at C(N, 3)
- But vocabulary is NOT the bottleneck — scaling hurts
- Real bottleneck: the fingerprint doesn't correlate with semantic
  structure the encoder already captures
- Shuffle diagnostic: real beats shuffle by ~4 hits/500 —
  sub-noise-floor on a single benchmark, insignificant at scale

### 6.2 Why contrastive fine-tuning fails
- Phase 4b tested: fine-tune encoder against graph co-activation
  topology
- Result: catastrophic (-13 hits on 100 queries after 316 updates)
- Explanation: graph topology is driven by frozen random projections
  that have no relation to semantic structure. Training the encoder
  toward an arbitrary target destroys its pretrained semantic
  structure.

### 6.3 What structural plasticity is for
- Developmental adaptation: neurogenesis for new concepts,
  consolidation for reorganization, pruning for outdated paths
- Evaluation axis: adaptation speed, survival time, novel-concept
  integration
- NOT: static retrieval on pre-existing embeddings

## 7. Discussion

- Implications for brain-inspired memory proposals: evaluate on tasks
  where mechanisms are load-bearing; use shuffle diagnostics;
  cross-benchmark
- Implications for SOMA specifically: pivot to open-ended learning
  environment; retrieval was a pragmatic-but-mismatched framing
- Open question: could a learned (not random) projection system
  change the picture? Phase 4b suggests no for the current
  contrastive objective. Phase 8 (learnable diversification) suggests
  modest but small gains.

## 8. Related work

- Brain-inspired retrieval (SYNAPSE, Graphiti, MAGMA, Mem0)
- Shuffle diagnostics in ML eval
- Negative-results literature in RAG
- Developmental AI (ACT-R, Leabra, SPAUN — earlier architectures)

## 9. Limitations

- Single encoder (all-MiniLM-L6-v2); would results differ with a
  larger / domain-tuned encoder?
- Single top-k (20 recall, 5 final); harder thresholds
- Only two benchmarks
- No user-LLM-generation loop; pure retrieval
- No consolidation stress test on LoCoMo scale

## 10. Conclusion

- Brain-inspired graph memory as a retrieval signal has an
  architectural ceiling that holds across benchmarks
- The "structural without semantic" problem is fundamental, not a
  tuning artifact
- The mechanisms DO work on tasks they were designed for
  (development, adaptation, consolidation)
- Call-to-action: evaluate brain-inspired architectures on the
  problems they are structurally suited to, not on problems where
  they're forced to compete with pretrained embeddings on the
  latter's home field
