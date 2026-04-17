# Developmental SOMA — Known Issues & Next Steps

**Date:** 2026-04-17  
**Status:** PoC validated, architecture sound, retrieval working

---

## Current Performance

| Test | Dev F1 | Blank F1 | Improvement |
|------|--------|----------|-------------|
| Synthetic (300 inputs) | 0.190 | 0.017 | 11x |
| LoCoMo single conv (419 turns) | 0.027 | 0.003 | 9x |
| LoCoMo all 10 convs (5882 turns) | 0.022 | 0.017 | 24:14 wins |

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

## Next Steps (Priority Order)

### P0: Trainable Encoder (Biggest F1 Impact)
Wire the TextEncoder into the computation graph so SOMA's self-supervised
loss trains the embeddings. This would make "cooking" and "travel"
embeddings diverge over training, dramatically improving retrieval.

Approach: Make encode_text() keep gradients, add encoder params to SOMA's
optimizer, compute encoder loss alongside prediction loss.

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
