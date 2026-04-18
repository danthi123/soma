# Developmental SOMA — Known Issues & Next Steps

**Date:** 2026-04-18 (updated)  
**Status:** Confidence-gated hybrid validated (+3 over VecDB), adaptive gating, contrastive fine-tuning

---

## Current Performance

| Test | Dev F1 | Blank F1 | Improvement |
|------|--------|----------|-------------|
| Synthetic (300 inputs) | 0.190 | 0.017 | 11x |
| LoCoMo single conv (419 turns) | 0.027 | 0.003 | 9x |
| LoCoMo all 10 convs (5882 turns) | 0.022 | 0.017 | 24:14 wins |
| LoCoMo retrieval-only (no LLM, T=1) | recall F1=0.053 | — | 43/100 hits |
| LoCoMo retrieval-only (T=5, anti-Hebbian) | recall F1=0.047 | — | 44/100 hits |
| LoCoMo graph fingerprint (T=5, min3) | recall F1=0.050 | — | 47/100 hits |
| LoCoMo graph topology (T=5, min3) | recall F1=0.049 | — | 45/100 hits |
| LoCoMo token overlap (baseline) | recall F1=0.068 | — | **53/100** hits |
| Topology wins vs fingerprint | — | — | 20/100 queries |
| Pre-trained graph (20K turns) | recall F1=0.051/0.052 | — | 44/46 hits (fp/topo) |
| Combined fp+topo (any weight) | recall F1=~0.048 | — | 44/100 (signals too correlated) |
| **SOMA + pretrained encoder** | — | — | — |
| Vector DB (all-MiniLM-L6-v2) | — | — | **63/100** hits |
| SOMA + pretrained (full replace) | — | — | 49/100 (-14 vs VecDB) |
| **Confidence-gated hybrid** | — | — | **66/100** (+3 vs VecDB) |
| Query analysis: SOMA wins | — | — | 9 queries |
| Query analysis: VecDB wins | — | — | 6 queries |
| Query analysis: ties | — | — | 85 (graph used in 22) |

**Key insight (updated):** SOMA destroys embedding signal when used as a
REPLACEMENT (-14 hits) but ADDS value as a confidence-gated COMPLEMENT (+3
hits). The graph only intervenes when it has strong structural signal (confidence
gap >= 0.05); otherwise pure embedding similarity is used. SOMA wins on cross-
entity associations, multi-entity recall, and distributed temporal events. VecDB
wins on single-turn factual recall and counting queries.

**Architecture:** Embedding recall (top-20) → fingerprint confidence gate →
selective graph reranking with weight 0.2. Implemented as `retrieve_hybrid()`
on PredictiveSOMA.

## Known Issues

### 1. BPE Encoder Bottleneck
The untrained BPE tokenizer can't bridge semantic gaps:
- "What is my cat's name?" doesn't match "Nebula" (no character overlap)
- "What do I like to cook?" partially matches "cooking" via shared chars
- 128-dim embeddings from random initialization have limited semantic capacity

**Impact:** Limits absolute F1. Relative improvement (dev vs blank) is genuine.

**Options:**
- Train encoder embeddings through SOMA's loss (needs gradient flow redesign)
- Use a small pretrained encoder (violates "no pretrained" principle)
- Accept the limitation — the architecture works, encoder quality is a knob

### 2. Competitive Learning Non-Functional
All nodes receive the same input from the single sensor node, so one node
always dominates. Input diversification works around this.

**Root cause:** Fully-connected sensor→associator topology.
**Fix:** Multiple sensor projections (different random linear transforms
per edge) so each associator genuinely sees different input features.
This is an edge-level change, not a node-level one.

### 3. Graph Densification vs Differentiation
More edges = more signal mixing = less differentiation. Input diversification
mitigates this, but the fundamental tension remains.

**Fix:** Anti-Hebbian learning on losing nodes, topographic organization,
or making synaptogenesis connectivity-aware (don't wire already-similar nodes).

### 4. Retrieval Oscillation
F1 scores oscillate ±0.05 between checkpoints due to the stochastic
nature of which hash buckets get populated for each query.

**Fix:** Larger fingerprint dim (512 or 1024), or average over multiple
retrieval passes.

## Recent Changes (2026-04-18, Phase 3-6)

### Phase 3: Confidence-Gated Hybrid Retrieval (BREAKTHROUGH)
Embeddings for candidate recall (top-20), then SOMA fingerprint similarity
for selective reranking. Gate fires when confidence (top-1 minus top-2 fp
sim) exceeds 0.05. Result: **66/100 hits vs VecDB's 63/100 (+3)**.
Gate fires on 32/100 queries; wins 9, loses 6, ties 17.

Query analysis reveals SOMA wins on:
- Cross-entity: "What volunteering have John and Maria both done?"
- Multi-entity recall: "Which of Deborah's family have passed away?"
- Distributed temporal: "When did John join the support group?"
- Coping/inference: "What helped Deborah find peace?"

SOMA loses on single-turn factual recall and counting queries — the graph's
structural reranking pushes wrong results up for queries where the answer
is in a single well-embedded passage.

### Phase 4: Encoder Fine-Tuning via SOMA Loss (Failed)
Attempted to fine-tune sentence-transformers encoder through SOMA's loss.
Result: 0 gradient updates — SOMA internally detaches tensors before
computing loss, so `loss.backward()` never propagates to the encoder.

### Phase 4b: Contrastive Encoder Fine-Tuning (FAILED)
Designed `compute_contrastive_loss()` that uses graph topology as
supervision: memories sharing active nodes should have similar embeddings.
Provides proper gradient flow to the encoder. Result: **catastrophic**.
VecDB dropped 63→50 (-13), gated dropped 65→51 (-14). Only 316 FT
updates (all in Conv 1-3) were enough to destroy the pretrained encoder.
Gate usage dropped 32→11 (graph also confused by degraded embeddings).

Root cause: Graph co-activation topology is driven by frozen random
projections, not semantic relationships. Training the encoder to match
arbitrary activation patterns causes catastrophic forgetting. The
contrastive loss optimizes for the wrong objective.

### Phase 5: Semantic Lens (Negative)
Learned projection from embeddings to graph fingerprint space. Lens alone
hurts, combined matches baseline. No improvement over raw fingerprints.

### Phase 6: Adaptive Gating (Negative Result)
Tested scaling rerank_weight by absolute fingerprint quality (fp_sorted[0])
to prevent confidently-wrong reranking. Result: same 66 hits but 2 fewer
wins (10 vs 12). Hypothesis was wrong — fingerprint quality is already
high when the gate fires (fp_best=0.7+). Reducing the effective weight
just weakens correct reranking decisions.

Also tested: combined confidence (gap * absolute, 65 hits), higher
threshold (0.10, 65 hits), higher weight (0.3, 65 hits). All worse.

**Conclusion:** Fixed `gate=0.05, w=0.2` is optimal. The 4 VecDB losses
are queries intrinsically unsuited to graph reranking — this is the
price of the 12 graph wins. Net effect: +3 hits (+8 net when counting
ties that shift).

### Previous Changes (2026-04-18)

#### Pre-training Experiment (Negative Result)
Pre-trained SOMA's graph on 20K diverse turns from LongMemEval-S (2000
sessions), cleared memory stores, then developed on LoCoMo. Result:
fingerprint 44 hits (vs 47 baseline), topology 46 hits (vs 45 baseline).
Pre-training doesn't help because the bottleneck is random BPE embeddings,
not lack of training data. The graph can't learn meaningful specialization
when all inputs look similar in embedding space. Graph grew 14→20→21
nodes, 149→326 edges, costing 47 minutes + 2x slower LoCoMo development.

### Fingerprint Collision Fix (SHA256)
Replaced Python `hash()` with `hashlib.sha256()` for fingerprint position
assignment. Old scheme: some node pairs had 15/16 overlapping positions
(due to stride-37 congruence). New scheme: max pairwise overlap is 3/16.
Also increased values per node from 16 to 32 for better activation capture.
Vectorized the inner loop with `scatter_add_` for performance.

### Weight-Based Diversification (Reverted)
Tested adding node weight response factor to diversification gain to
create feedback between competitive learning and winner selection.
Small-scale test showed winner overlap between topics dropped from
2/3 to 1/3. But LoCoMo evaluation showed -4 hits for fingerprint and
-8 hits for topology vs baseline. The `activation_ema` normalization
introduced too much noise. Root cause: frozen random projections
create a FIXED input→winner mapping — competitive learning on
`linear1.weight` doesn't feed back into diversification gain. This is
a fundamental architectural issue (P1 edge-level diversification would
fix it). Reverted to frozen-projection-only diversification.

### Combined Retrieval (Negative Result)
Tested merging fingerprint + topology scores at various weights
(1.0/0.7/0.5/0.3/0.0). All produced ~44 hits — the signals are too
correlated (both depend on which 3 nodes win lateral inhibition).

## Previous Changes (2026-04-17)

### Bug Fix: retrieve_by_graph Missing Diversification
**Critical bug**: `retrieve_by_graph()` applied neither `_diversify_activations()`
nor `_apply_lateral_inhibition()` before fingerprinting, causing all query
fingerprints to be nearly identical. This made retrieval effectively random.
Fixed: query path now applies the same pipeline as storage.

### Encoder Training Wired (P0 DONE)
TextEncoder now trains from SOMA's backward pass:
- `encode_text()` returns grad-enabled tensors when `train_encoder=True`
- SOMA.step() → update_step() → loss.backward() propagates to encoder params
- InteractionLoop applies encoder gradients after each step with Adam lr=0.001
- Result: 2-3x accuracy improvement on topic retrieval diagnostic (12% → 25-38%)

Temporal contrastive loss was attempted but hurts at small corpus scale
(conflicting gradients). May help at LoCoMo scale (5882 turns).

### Neurogenesis Growth Control
With encoder training enabled, prediction error stays elevated (encoder
shifts embeddings each step), causing runaway neurogenesis (42 nodes/766
edges after just 788 turns vs 14/149 without). Fix: feed SOMA's step loss
(more stable) into the neurogenesis trigger instead of prediction error.
Also added lazy projection creation for neurogenesis-born nodes.

### Lateral Inhibition min_active
Added `min_active=3` parameter to `_apply_lateral_inhibition()`. With 14
nodes and 10% keep_ratio, only 1 node survived — too few for discriminative
fingerprints. min_active ensures at least 3 nodes contribute regardless of
graph size.

### Fingerprint Redesign
Changed from full 128-dim scatter (50% collision rate per node) to 16-value
hash projection per node. Each node contributes 16 sampled activation values
at prime-strided positions. Reduces collision saturation while preserving
directional information. Boundary nodes (SENSOR/OUTPUT) excluded.

### Temperature-Scaled Diversification
Input diversification gain function now uses temperature=5.0 for sharper
selection. Increases fingerprint discrimination ratio from 1.07x to 1.18x
on small diagnostics. Gains become bimodal (~0.5 or ~2.0) instead of
clustered around 1.25.

### Anti-Hebbian Learning
Suppressed nodes now adapt AWAY from the current input (anti-Hebbian
synaptic depression), making them less responsive to patterns they "lost"
on. This drives genuine node specialization without requiring sparse
initial connectivity.

### Topology-Based Retrieval
New `retrieve_by_topology()` uses SOMA's learned graph structure directly:
finds which nodes the query activates, looks up which stored memories
activated the same nodes. Wins on 20/100 LoCoMo queries where fingerprint
comparison fails — showing SOMA's structural learning adds value for
cross-domain associations.

### Node Memory Index
Each memory stores which nodes were active (inverted index: node → memory
steps). Enables fast topology-based retrieval without re-running graph
execution per stored memory.

## Completed Work (previously listed as next steps)

### Consolidation Validation — DONE (commit `c553a66`)
Develop 300 steps → consolidate → QA vs no-consolidation test shipped.
Result: **+45% QA improvement with consolidation**. The "sleep" cycle
reorganizes the graph in a way that measurably helps retrieval.

### Multi-Session Development — DONE (commit `6a0a822`)
Save/load across 3 sessions (cooking+family → travel+music → cross-
session recall). Graph grew 275% across sessions and cross-session
recall works. Developmental state is persistent.

### Edge-Level Input Diversification (naive) — REVERTED (commit `4d044f6`)
Tried moving random projections into `Edge.transmit()` directly. Linear
projections preserve inner products, so this doesn't create genuine
input differentiation — it's equivalent to the current post-hoc scheme
in terms of what each associator "sees." A **nonlinear per-edge
variant** (e.g. edge-specific nonlinear activations or learned
projections) remains untried and is the real open version of this
idea.

## Next Steps (Priority Order)

### P1: Fingerprint Vocabulary Scaling — DISPROVEN (2026-04-18)
Hypothesis was: hybrid retrieval plateaus near ~66/100 because
fingerprint vocabulary is too small (~C(14,3)=364 patterns for 5882
turns). Scaling associators should expand vocabulary and lift hits.

`research/developmental/associator_count_sweep.py` swept
`initial_associator_count` ∈ {8, 32, 64}:

| n_init | n_final | Hits | Used | W/L | Net | Time |
|--------|---------|------|------|------|-----|------|
| 8      | 14      | 64   | 40   | 11/3 | +8  | 349s |
| 32     | 38      | 61   | 56   | 13/9 | +4  | 1352s |
| 64     | 70      | 63   | 61   | 8/9  | -1  | 2904s |

**n=8 is Pareto optimal.** Scaling hurt: gate usage climbed (40→56→61)
exactly as predicted, but confidence fired on *mismatches* — losses
tripled from 3 to 9, wiping out the extra wins. At n=64 the graph is
indistinguishable from VecDB (63 vs 63).

Conclusion: the bottleneck is **signal quality, not vocabulary size**.
Fingerprint patterns from random projections are structurally diverse
but semantically arbitrary, consistent with the Phase 4b contrastive-
FT failure finding.

### Phase 10: Multi-Slice Held-Out Validation (2026-04-18)
The +3 hit advantage on LoCoMo was established on a single slice of
100 QA pairs (first 10 per conv). Every phase since tuned against it.
Phase 10 tested whether +3 generalizes to unseen queries.

One dev run (n=8), three disjoint slices evaluated:

| Slice          | N   | Hybrid | VecDB | Delta |
|----------------|-----|--------|-------|-------|
| A [0:10] tuned | 100 | 64     | 63    | +1    |
| B [10:30]      | 200 | 129    | 131   | -2    |
| C [30:50]      | 200 | 129    | 126   | +3    |
| **Combined**   | 500 | 322    | 320   | **+2**|

Wins/losses across 500 queries: 32/34 — near coin flip. The gated-
hybrid is *perturbing* results without adding net signal. The +3 on
slice A was at the high end of the mechanism's real strength, which
is ~+0.4% absolute over 500 queries.

### Phase 11: Shuffle Diagnostic (2026-04-18)
To confirm whether the graph contributes ANY signal vs random noise,
ran gated-hybrid with real fingerprint-to-memory mapping vs 5 random
shuffles on all three slices.

| Slice     | Real Δ | Shuffled Δs         | Shuf mean | Shuf range |
|-----------|--------|---------------------|-----------|------------|
| A [0:10]  | +1     | [-1,-1,-1,-3,-1]    | -1.4      | [-3, -1]   |
| B [10:30] | -2     | [-1,-2,-4,+4,+1]    | -0.4      | [-4, +4]   |
| C [30:50] | +3     | [+1,+1,-4,0,-1]     | -0.6      | [-4, +1]   |

Per-seed totals shuffled = [-1, -2, -9, +1, -1], mean -2.4, max +1.
**Real total +2 exceeds all five shuffled totals.** The graph
contributes real signal — about 4 hits/500 queries (~0.8% absolute)
above random fingerprint assignment. Consistent with earlier Phase 1.3
finding that graph wins concentrate on relational/cross-entity
queries, not uniformly.

Takeaway: the fingerprint mechanism works as designed, but the
retrievable signal magnitude is too small to ship as a general
retrieval enhancement. Per-query attribution could identify the
subset where it consistently helps (tighter gating).

### Phase 12: Per-Query Attribution (2026-04-18)
Classified all 500 queries into 6 categories (cross_entity, temporal,
counting, factual_self, causal, preference, other) via keyword rules.
Per-category win/loss rates:

| Category      | N   | W/L/T     | Delta |
|---------------|-----|-----------|-------|
| other         | 222 | 10/7/205  | +5    |
| temporal      | 182 | 15/17/150 | +1    |
| counting      |  36 | 2/1/33    | 0     |
| cross_entity  |  31 | 3/1/27    | 0     |
| preference    |  23 | 0/0/23    | 0     |
| causal        |   6 | 0/0/6     | 0     |

No clean category signal. cross_entity and counting have 2-3:1 win
ratios as expected, but n is tiny (31, 36) and absolute delta is
near zero. The biggest delta (+5) is in "other," a 44% catch-all with
no semantic pattern. Keyword-gated hybrid won't lift hits meaningfully.

### Phase 13: Topology Signal in Gated Hybrid (2026-04-18)
Swapped fingerprint cosine similarity for node-topology Jaccard
similarity as the gate signal, holding everything else constant.

| Slice | Fingerprint (W/L/Δ) | Topology (W/L/Δ) |
|-------|---------------------|-------------------|
| A     | 8/4/+2              | 6/2/+1            |
| B     | 15/14/-1            | 5/10/-1           |
| C     | 11/11/+2            | 11/9/+2           |

Functionally equivalent: fingerprint 323 hits, topology 322 hits over
500 queries. Both plateau around +2/500 delta. Topology is more
conservative (fires 135 vs 192 times) but hits land in the same
place. **No complementarity** — both signals derive from the same 3
lateral-inhibition-winner nodes, so they're structurally the same
information presented differently.

Takeaway: the ceiling is **architectural, not mechanism-specific**.
Any retrieval signal derived from "which nodes fire together" caps
around +2/500 at this graph scale on LoCoMo.

### Phase 14: LongMemEval Cross-Benchmark Validation (2026-04-18)
Ran the gated hybrid on LongMemEval oracle (100 items, merged
haystack corpus of 3094 turns):

| Metric     | Value                |
|------------|----------------------|
| VecDB hits | 41/100 (41.0%)       |
| Hybrid hits| 40/100 (40.0%)       |
| Delta      | **-1**               |
| Wins/Losses| 3/6 (net -3)         |
| Gate used  | 36/100 (36%)         |

By question type:
- temporal-reasoning (n=60): -1 delta, 1W/5L
- multi-session (n=40): 0 delta, 2W/1L

**Ceiling confirmed cross-benchmark.** The gated hybrid slightly
hurts on LongMemEval (especially temporal-reasoning, where the
graph's winners mismatch actual temporal semantics). The retrieval-
signal ceiling is not LoCoMo-specific. The mechanism has reached
its performance floor on two independent canonical benchmarks.

**Implication:** no further tactical experimentation on the
fingerprint hybrid retrieval approach is warranted. Strategic
decision required — see
[2026-04-18-developmental-findings-and-direction.md](2026-04-18-developmental-findings-and-direction.md)
for current options and recommendation.

### Phase 15: Sequence-prediction env + structural plasticity ablation (2026-04-18)

Shifted to the adaptive-learning track per "D then A" direction.

- **v0** (4-regime, 2000-step schedule): SOMA beats online MLP 3-33x. Ablation: `no_growth` wins 3/4 regimes at small scale (commit c19a0b9).
- **v0.5** (8-regime capacity schedule, 4000-step): `no_growth` still wins 7/8 under default interval-based neurogenesis; `full` grew to 49 nodes / 980 edges yet was 5-10x worse (commit 0390590, paper §4.7 Table 9).
- **v0.5 + PE-gated neurogenesis** (opt-in `neurogenesis_mode="pe_gated"` w/ cooldown=200): fires 66% fewer events, graph is 46% smaller, MSE unchanged — rules out trigger *timing* as the root cause (commit bb9637e, paper §4.7 follow-up).

**Diagnosis:** the failure is new-node *integration*, not trigger
timing or event count. Next probe is the `neurogenesis_init_weight_scale` sweep over {0.01 (legacy), 0.001, 0.0001, 0.0, no_growth} on the same 8-regime schedule (commit cb41b40 adds the knob). Results (in flight) will tell us whether making new edges quieter lets Hebbian updates integrate new nodes without disturbance, or whether the mere presence of new nodes degrades the circuit (scale=0 would falsify).

If init-scale fails too, next candidates are (a) gain-ramped new nodes that start near-inert and mature via homeostasis, (b) non-disruptive neighbor selection (connect to least-active instead of most-active), (c) a structure-control experiment that isolates which substrate component (wave execution, residual MLPs, homeostatic gain, depth, parameter count) actually drives SOMA's 3-40x win over online MLP.

### P2: Nonlinear Edge-Level Diversification (future)
The `4d044f6` revert showed linear per-edge projections don't
differentiate. Nonlinear variants (per-edge activation functions,
per-edge small MLPs, or gated projections) could create genuine input
differentiation without the "linear preserves inner products"
problem. Blocked on P1 outcome — if scaling the graph alone cracks
the plateau, this becomes lower priority.

### P3: Multi-Benchmark Validation
LoCoMo's 100 QA pairs have been our only retrieval benchmark for
Phase 3-8. Before more architectural work, validate the +3 gated-
hybrid result on LongMemEval or a different LoCoMo split to confirm
it's not overfit to this specific slice.

### P4: Separate Project Setup
The developmental module is getting large enough to warrant its own
repo. Keep the SOMA core as a dependency, move developmental/ to a
new project. Admin task, not research.

### P5: Inhibitory Node Type
Add explicit inhibitory nodes to the graph that learn to suppress
specific associators. Would replace the post-hoc lateral inhibition
with learned, input-dependent suppression. Larger architectural
change — park until P1/P2/P3 are exhausted.
