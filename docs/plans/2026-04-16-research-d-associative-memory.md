# Research Direction D: Associative / Hopfield-Style Recall

> **For Claude:** research-phase plan. Sub-phases are experiments with pass/fail criteria.

**Hypothesis:** SOMA's plastic graph with Hebbian learning is structurally a generalization of a Hopfield network — continuous activation, learned connectivity, with added structural growth + homeostasis. Configured for attractor dynamics (rather than next-token or retrieval), it should support content-addressable recall of stored patterns with (a) noise robustness comparable to modern Hopfield networks (Ramsauer 2020), (b) capacity that grows with structural plasticity, (c) a real attractor landscape you can visualize.

**Strategic framing:** cleanest research direction of the four. Outputs are either clearly strong (beats modern Hopfield on some axis — capacity, partial-observation completion, noise robustness) or clearly negative (substrate doesn't converge to attractors cleanly). Either result is publishable.

**Target publication:** associative-memory / Hopfield-network workshop or a neuro-inspired ML venue.

**Why it's novel:** modern Hopfield networks are a hot area (2020 breakthrough showed equivalence to attention). Nobody's combined them with structural plasticity + homeostasis in a clean architecture. SOMA substrate has both.

---

## Sub-phase D1: Pattern-storage benchmark + classical baselines

**Goal:** infrastructure. Load three recall benchmarks, measure classical Hopfield, modern Hopfield (Ramsauer), k-NN recall, autoencoder recall.

**Benchmarks:**
- **MNIST denoising:** store N MNIST digits; query with noisy (occluded, gaussian-corrupted) versions; measure per-pixel and per-class recall accuracy.
- **Random binary patterns:** classical Hopfield test. Store N random N-bit patterns; recall from partial presentations. Measure capacity (max N where recall is reliable).
- **Word-association:** cosine-based task. Store (word → definition embedding) pairs; query with typo'd word; measure correct-definition recall.

**Baselines:**
1. **Classical Hopfield** (discrete, sign rule). Implementation: 20 lines of numpy.
2. **Modern Hopfield** (Ramsauer 2020, softmax-based). Implementation: ~50 lines of torch.
3. **k-NN over stored patterns** (no model, just Euclidean). Lower bound.
4. **Autoencoder denoising** (trained on the stored set). Different architectural class; useful reference.

**Files:**
- Create: `research/associative/datasets/` (gitignored data)
- Create: `research/associative/baselines.py`
- Create: `research/associative/harness.py`
- Create: `research/associative/reports/d1_baselines.md`

**Success criterion:** classical Hopfield hits its published 0.14N capacity limit on random binary patterns (within ±5%); modern Hopfield scales to larger N. Off → implementation bug.

**Scope:** ~4-5 days.

---

## Sub-phase D2: SOMA in "attractor mode"

**Goal:** configure SOMA as a recurrent attractor network — disable growth + critical-period schedule + next-token machinery, retain Hebbian + homeostasis + graph connectivity. Store patterns by running the graph forward on them; recall by feeding partial input and iterating until convergence.

**Experimental questions:**
- Does the substrate converge? (Or does homeostasis fight attractor formation?)
- What's the natural capacity at fixed node count?
- Does pattern interference look like classical Hopfield or something different?

**Files:**
- Create: `src/soma/research/attractor_mode.py` — wrapper that configures the graph for attractor use
- Create: `research/associative/soma_attractor.py`
- Create: `tests/test_research/test_attractor_mode.py`

**Success criterion (D2 gate):** SOMA in attractor mode recalls ≥50% of stored patterns correctly on the random-binary benchmark at N=10 patterns, 50 nodes. This is a "does it work at all?" threshold. If it fails, D3 is failure-analysis.

**Scope:** ~1 week.

---

## Sub-phase D3: Growing / pruning the attractor

**Goal:** what SOMA has that classical + modern Hopfield don't — *structural plasticity*. Test whether adding capacity (neurogenesis) when the network saturates extends the capacity limit past the classical bound.

**Experimental method:**
- Start with 50 nodes, classical capacity 0.14 × 50 = 7 patterns.
- Add patterns one at a time. When recall accuracy on the stored set drops below threshold, trigger neurogenesis.
- Measure: capacity as a function of N_nodes. Does it stay at ~0.14 × N_nodes, or does it degrade faster / slower?
- Ablation: prune low-contributing nodes; does capacity stay stable while node count stays bounded?

**Files:**
- Create: `research/associative/reports/d3_growing_capacity.md`

**Success criterion:** capacity scales super-linearly with growth-triggering saturation events (i.e., "smart" growth yields more capacity per node than random growth). Or: bounded-node-count pruning preserves capacity within 10% of never-pruned baseline.

**Scope:** ~1-1.5 weeks.

---

## Sub-phase D4: Theoretical framing + comparison to modern Hopfield

**Goal:** is SOMA's substrate formally equivalent to some known attractor architecture? Write the math. Compare energy landscapes, fixed-point analysis, noise robustness curves.

**Files:**
- Create: `research/associative/reports/d4_theory.md`

**Scope:** ~1 week. Math-heavy. The paper's theoretical spine if we publish.

---

## Sub-phase D5: Writeup

**Goal:** paper or blog post. If D3 beats the Hopfield capacity bound via structural plasticity, that's a clean result with a clear figure. If not, "here's what a plastic-graph-as-Hopfield-generalization looks like, here's its capacity profile" is still publishable as a negative-or-characterization result.

**Files:**
- Create: `docs/papers/2026-xx-associative-plastic-graph.md`

**Scope:** ~1-2 weeks.

---

## Total estimated scope

4-6 weeks. Purest-research direction; cleanest publishability. No product dependency. Can run in parallel with C (they touch different subsystems — C is retrieval, D is attractor dynamics).

**Adjacency:** no conflict with product path, production-readiness phases, or the other three research directions.
