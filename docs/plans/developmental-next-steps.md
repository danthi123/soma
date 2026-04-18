# Developmental SOMA — Known Issues & Next Steps

**Date:** 2026-04-18 (updated)  
**Status:** PoC validated, pre-training tested, fingerprint collision fix, weight-based diversification

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

**Key insight:** Token overlap (53 hits) is the "dumb ceiling" — what character
matching gives you for free. Graph-driven retrieval (47 hits) is SOMA's genuine
contribution. The graph currently underperforms token overlap because random BPE
embeddings limit it to character-level patterns. The graph's unique value is in
cross-domain associations (topology wins 20/100), which token overlap cannot do.
Gap to close: 6 hits (47 → 53). min_active=3 is optimal; min_active=4 hurts
(-6 hits). Temperature=5.0 diversification helps (1.18x vs 1.07x discrimination).

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

## Recent Changes (2026-04-18)

### Pre-training Experiment (Negative Result)
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

## Next Steps (Priority Order)

### P1: Edge-Level Input Diversification
Move the random projections from PredictiveSOMA's post-hoc modulation into
the Edge.transmit() method. Each sensor→associator edge would apply its own
random linear transform. This is architecturally cleaner and would enable
competitive learning to work properly.

### P2: Consolidation Validation
We haven't verified that consolidation ("sleep") cycles actually improve
retrieval. Run a test: develop for 300 steps, consolidate, test QA.
Compare to 300 steps without consolidation.

### P3: Multi-Session Development
Test development across multiple separate sessions with save/load:
- Session 1: Talk about cooking and family (save)
- Session 2: Load, talk about travel and music (save)
- Session 3: Load, test cross-session recall

### P4: Separate Project Setup
The developmental module is getting large enough to warrant its own repo.
Keep the SOMA core as a dependency, move developmental/ to a new project.

### P5: Inhibitory Node Type
Add explicit inhibitory nodes to the graph that learn to suppress
specific associators. This would replace the post-hoc lateral inhibition
with learned, input-dependent suppression.
