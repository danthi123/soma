# Graph-Retrieval Benchmark Design

## Gap

Current benchmarks (LongMemEval, agentic tasks) test the MemoryLayer API,
which uses embedding similarity for retrieval. The SOMA graph (nodes, edges,
Hebbian learning, structural plasticity) is not exercised in the retrieval
path. We need benchmarks that specifically test what the graph adds.

## What the Graph Should Provide

1. **Associative retrieval** — finding related memories that aren't
   semantically similar but are contextually linked (e.g., "what did
   I discuss right after the meeting about X?")

2. **Hebbian strengthening** — frequently co-accessed memories should
   surface faster/higher in retrieval priority

3. **Temporal coherence** — the graph encodes temporal ordering; queries
   like "what happened before/after X?" should benefit from graph edges

4. **Cross-session linking** — connecting information across separate
   conversations that share entities/topics

5. **Forgetting resistance** — consolidated memories should survive
   even when newer memories dominate embedding similarity

## Proposed Test Scenarios

### Test 1: Associative Hop

**Setup**: Store 100 conversation turns. Some turns mention entity A,
others mention entity B, and a few mention both A and B together.

**Query**: "What do I know about entity A?"

**Graph advantage**: Graph edges between co-mentioned entities should
surface B-related memories that pure embedding similarity would miss.

**Metric**: Recall of indirectly-related memories (B mentions found
when querying A, vs embedding-only baseline).

### Test 2: Temporal Ordering

**Setup**: Store a sequence of events with dates. Events have causal
chains (A caused B, B led to C).

**Queries**: "What happened before X?", "What was the consequence of Y?"

**Graph advantage**: Graph edges encode temporal adjacency. Embedding
similarity treats all events equally regardless of order.

**Metric**: Accuracy of temporal ordering in retrieved context.

### Test 3: Hebbian Prioritization

**Setup**: Store 200 memories. Repeatedly query a subset of 20 (simulating
frequent access). Then query something related to both frequent and
infrequent memories.

**Graph advantage**: Hebbian learning should strengthen frequently-accessed
pathways, prioritizing those memories in retrieval.

**Metric**: Rank of frequently-accessed memories vs infrequent ones
in retrieval results for ambiguous queries.

### Test 4: Consolidation Survival

**Setup**: Store 50 memories (phase 1). Run consolidation. Store 200
more memories (phase 2). Query for phase-1 information.

**Graph advantage**: Consolidated memories should be protected from
interference by newer memories. Without consolidation, recency bias
would bury phase-1 memories.

**Metric**: Recall of phase-1 memories after phase-2 flooding, with
and without consolidation.

### Test 5: Cross-Session Entity Linking

**Setup**: Across 10 separate "conversations," mention a project
evolving over time. Each session adds new details. Some sessions
share entities, others introduce new ones.

**Query**: "Summarize what I know about project X"

**Graph advantage**: Graph edges connecting sessions that share
entities should enable comprehensive retrieval across sessions.

**Metric**: Coverage — fraction of relevant sessions represented in
retrieved context.

## Implementation Approach

Each test should:
1. Have a **graph-enabled** path (full SOMA with Hebbian + consolidation)
2. Have a **vector-only** path (MemoryLayer with embedding similarity only)
3. Use synthetic data (deterministic, reproducible)
4. Measure retrieval quality directly (no LLM needed)
5. Run in <1 min per test

## Priority

Tests 1 (associative hop) and 4 (consolidation survival) are most
aligned with SOMA's unique architecture and most likely to show a
graph advantage. Start there.

## Prerequisite

The MemoryLayer API currently uses embedding similarity for retrieval.
To test graph-based retrieval, we need either:
- A `retrieve_via_graph()` method that traverses SOMA edges
- Or integration of graph proximity into the retrieval scoring

This may require a small API addition to MemoryLayer.
