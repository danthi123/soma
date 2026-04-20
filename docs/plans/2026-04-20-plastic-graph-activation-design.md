# Plastic graph activation — design for a test that could validate "learns from use"

**Status:** Design only. Implementation deferred until QA comparison lands.

**Goal:** Find a realistic memory workload where SOMA's plastic graph
substrate (synaptogenesis, pruning, myelination) actually fires AND
provides measurable value over a non-plastic baseline.

## Problem statement

SOMA ships synaptogenesis + neurogenesis + pruning + myelination as
`src/soma/growth/`. The positioning claims "plastic graph substrate
(in-place)" as a differentiator, but with an asterisk: under the
memory-only workload the growth knobs don't fire.

This is a real gap. The "learns from use" story is the biggest claim
in positioning that lacks hard numbers. If it's true, show it. If not,
remove it or document the gap loudly.

## What the literature suggests

From `project_env_findings.md`:
- On v0.5 prediction tasks: `synap_local` (positional locality filter,
  max_distance=0.5) multi-seed validated as POSITIVE.
- On LoCoMo retrieval: same mechanism is NULL.

The difference: v0.5 prediction is a *task* with gradients flowing
through the graph; retrieval reads from stored tensors and reranks.

Hypothesis: the plastic graph helps when the graph is on the CRITICAL
PATH of computation, not when it's a sidecar re-ranker.

## Proposed test: streaming-topic retrieval with consolidation

### Setup

1. Build 10 "topics." Each topic has 100 semantically-related facts.
   (e.g., Topic A = "my Python coding preferences,"
    Topic B = "my running shoe preferences," ...)
2. 1000 facts total. Pre-embed with sbert.
3. Two systems:
   - **SOMA plastic**: MemoryLayer with `memory_layer()` preset
     (synap_local + consolidation ON), interval=25, rate=1.0
   - **SOMA frozen**: MemoryLayer with growth DISABLED — cosine/hybrid
     only, no consolidation.
   - **Chroma**: pure vector store, no learning.
4. LLM (qwen3.5:4b-q8_0 via Ollama) asks questions. Temperature=0.

### Protocol

**Phase 1: Ingest baseline (same for all 3 systems)**
- Ingest all 1000 facts in random order.

**Phase 2: Focused session on Topic A**
- 50 queries on Topic A (all asking about different aspects of
  Topic A facts).
- Interleaved with 50 queries across Topics B-J (sprinkled evenly).

**Phase 3: Post-session evaluation**
- Fresh held-out 50 queries on Topic A.
- Fresh held-out 50 queries on Topics B-J (5 each).
- Measure R@5 on both held-out sets.

### Metrics

For each system:
- Initial R@5 on Topic A (before focused session)
- Final R@5 on Topic A (after focused session)
- Final R@5 on Topics B-J (control — should be similar across systems)
- Graph structural metrics: node count, edge count, avg degree
  (SOMA only)
- Consolidation hit count

### Success criteria

**SOMA plastic wins** if: Final_R@5(A) - Initial_R@5(A) > 0.05 AND
greater than SOMA frozen and chroma (which should be flat).

**Null result** if: SOMA plastic flat (same as frozen), confirming the
plastic substrate doesn't activate on retrieval workloads.

**SOMA plastic hurts** if: Final_R@5(A) drops after consolidation.
Document as "premature pruning risk" and define guardrails.

### Why this might work (beyond previous negatives)

Previous retrieval tests measured static R@5 after one-shot ingest.
This test measures CHANGE in R@5 over a session. The hypothesis: the
graph needs repeated queries on related facts to co-activate and
strengthen. A single-turn retrieval doesn't give the Hebbian updates
time to take effect.

### Why it might not work

The graph re-rank is already off-by-default (`rerank_weight=0.0`) because
fingerprint signal adds noise. Even with consolidation hits, the graph
structure doesn't affect retrieval ranking — it just tracks co-activation.

To genuinely help retrieval, the graph would need to either:
(a) re-enable rerank with a smarter formulation
(b) influence the stored text's embedding (e.g., via spreading activation)

Neither is currently wired.

## Alternative: query-expansion with graph

A different role for the graph:
1. Cosine-retrieve top-5 (normal)
2. For each hit, pull graph neighbors (via `mem.related()`)
3. Cosine-re-rank the expanded candidate pool
4. Return top-5 from expanded pool

This uses the graph for query expansion rather than ranking. If the
graph has learned clusters, expansion pulls in related facts that
pure cosine missed.

Simpler to wire (no plasticity needed — just use existing graph).

### Success criteria
- R@5 with graph expansion > R@5 without (on same items)
- Effect grows as the graph gets more connected (after consolidation)

## Decision point

Before implementing, wait for the QA comparison result:
- If QA comparison lands strong (SOMA hybrid beats chroma on F1):
  the retrieval story is already proven. Plastic graph is a bonus
  test, worth doing but not urgent.
- If QA comparison lands null/weak: plastic graph becomes the main
  differentiation path. Urgent to validate or invalidate.

## Scope estimate

- Implementation: 4-6 hours (new topic-based synthetic benchmark
  + protocol runner + plasticity tracker)
- Runtime: 2-3 hours (1000 facts × 3 systems × 150 queries each)
- Analysis + findings doc: 2 hours

Total: ~1 working day, blocked on QA result.
