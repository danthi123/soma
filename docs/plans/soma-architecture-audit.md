# SOMA Architecture Audit Plan

## Motivation

B3 CL experiments revealed that SOMA's graph contributes nothing to
classification features — a plain linear projection matches all node
counts (ACC=0.74 for all of: 0, 1, 2, 4, 8, 16 integrators). This
raises the question: what does each SOMA component actually contribute,
and where?

This audit systematically tests every configurable axis of the SOMA
architecture to determine:
1. **Minimum viable config** — the simplest setup that achieves baseline performance
2. **Scaling curve** — what improves as we add complexity beyond minimum
3. **Dead weight** — components that don't move any metric for any use case
4. **Sweet spots** — optimal configurations per use case

## Use Cases to Test Against

Each component is tested across all relevant use cases, not just one:

| Use Case | Benchmark | What It Tests |
|---|---|---|
| Memory retrieval | LongMemEval (oracle, 50+ items) | Can SOMA retrieve the right context? |
| Associative recall | Custom: associative hop test | Does the graph find indirectly-related memories? |
| Temporal reasoning | LongMemEval temporal subset | Does graph structure help with ordering? |
| Consolidation value | Custom: flood-then-recall test | Do consolidated memories survive interference? |
| Agentic tool use | Existing 6-task benchmark | Does memory quality affect agent performance? |
| CL classification | Permuted-MNIST (B3 harness) | Feature quality for downstream tasks |

---

## Phase 1: Graph Architecture (Est. 2-3 days)

### 1.1 Node Count Scaling

**Already partially done.** Classification shows flat scaling.
Now test retrieval:

- Config axis: `initial_integrator_count` ∈ {0, 1, 2, 4, 8, 16, 32}
- `initial_associator_count` = 2× integrators
- Metric: LongMemEval F1/EM, associative hop recall
- Hypothesis: retrieval quality scales with node count (unlike classification)

### 1.2 Graph Dimensions

Each dimension could be load-bearing or dead weight:

- `sensor_output_dim` ∈ {16, 32, 64, 128, 256}
  - Controls the projection from input to graph. Higher = richer features but slower.
- `associator_hidden_dim` ∈ {32, 64, 128, 256}
- `integrator_hidden_dim` ∈ {64, 128, 256, 512}
- `position_dim` ∈ {0, 4, 8, 16, 32}
  - Does positional encoding in the graph matter at all?

Test each in isolation, holding others at default. Measure:
- Retrieval quality (LongMemEval F1)
- Graph execution speed (samples/sec)
- VRAM usage

### 1.3 graph_rerank_alpha

**Critical test.** This parameter controls whether the graph
actually participates in retrieval. Currently defaults to 0.0 (off).

- `graph_rerank_alpha` ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
- Must have SOMA attached + consolidation run for this to activate
- Metric: LongMemEval F1, associative hop recall
- Compare: alpha=0 (pure cosine) vs alpha>0 (graph-blended)
- **This is the single most important test in the audit** — it answers
  whether the graph helps retrieval at all

### 1.4 Edge Topology

- `max_edges_per_node` ∈ {5, 10, 20, 50}
- `locality_scale` ∈ {0.5, 1.0, 2.0, 4.0}
- `position_jitter` ∈ {0, 0.05, 0.1, 0.2}

These control graph connectivity. Denser graphs = more associative
paths but slower execution.

---

## Phase 2: Learning Rules (Est. 2 days)

### 2.1 Hebbian Learning

- `hebbian_lr` ∈ {0, 0.00001, 0.0001, 0.001, 0.01}
- 0 = no Hebbian learning (graph weights stay at init)
- Metric: does Hebbian learning improve retrieval over time?
- Test: store 100 memories, query 20 of them repeatedly, measure
  whether retrieval priority shifts toward frequently-accessed ones

### 2.2 Base Learning Rate

- `base_lr` ∈ {0.0001, 0.001, 0.01}
- `youth_lr_multiplier` ∈ {1.0, 2.0, 3.0, 5.0}
- Does the "young nodes learn faster" mechanism help?

### 2.3 Edge Weight Dynamics

- `edge_weight_decay` ∈ {1.0 (none), 0.9999, 0.999, 0.99}
- `max_edge_weight` ∈ {1.0, 2.0, 5.0, 10.0}
- Does weight decay help prevent runaway edge weights?
- Or is it just adding entropy?

### 2.4 Gradient Clipping

- `grad_clip_max_norm` ∈ {0.5, 1.0, 2.0, inf}
- Training stability vs learning speed

---

## Phase 3: Growth & Plasticity (Est. 2 days)

### 3.1 Synaptogenesis (New Edges)

- `synaptogenesis_interval` ∈ {0 (off), 50, 100, 500, 1000}
- `synaptogenesis_rate` ∈ {0.001, 0.01, 0.05}
- Does creating new edges improve retrieval?
- Or does it just add noise to graph traversal?

### 3.2 Neurogenesis (New Nodes)

- `neurogenesis_interval` ∈ {0 (off), 100, 500, 1000, 5000}
- `neurogenesis_threshold` ∈ {0.5, 0.8, 1.2, 2.0}
- Does growing new nodes for novel information help?
- At what point does graph bloat hurt performance?

### 3.3 Pruning (Remove Unused)

- `pruning_interval` ∈ {0 (off), 500, 1000, 5000}
- `edge_strength_threshold` ∈ {0.0001, 0.001, 0.01}
- `inactivity_threshold` ∈ {1000, 5000, 10000}
- Does pruning improve retrieval (less noise) or hurt it (lost paths)?

### 3.4 Myelination (Strengthen Strong Paths)

- `myelination_strength_threshold` ∈ {0.1, 0.3, 0.5, 0.8}
- `myelination_age_threshold` ∈ {1000, 5000, 10000}
- Does myelination create useful "fast lanes" in retrieval?

### 3.5 Interaction: Growth + Pruning + Myelination

- Test compound configurations:
  - All off (static graph)
  - Growth only (no pruning)
  - Growth + pruning (balanced)
  - Full pipeline (growth + pruning + myelination)
- Looking for: which combination gives best long-term retrieval?

---

## Phase 4: Memory Systems (Est. 1-2 days)

### 4.1 Working Memory

- `wm_slots` ∈ {0 (off), 8, 16, 32, 64}
- `wm_dim` ∈ {64, 128, 256}
- `wm_decay_rate` ∈ {0.8, 0.9, 0.95, 0.99}
- Does working memory help with multi-turn conversations?
- Test with LongMemEval multi-session questions

### 4.2 Episodic Memory

- `episodic_capacity` ∈ {100, 1000, 10000, 50000}
- At what point does capacity stop mattering?
- How does retrieval scale with stored items?

### 4.3 Consolidation Cycle

- `consolidation_interval` ∈ {0 (off), 100, 500, 1000, 5000}
- `consolidation_replay_steps` ∈ {10, 50, 100, 500}
- `consolidation_error_threshold` ∈ {0.1, 0.3, 0.5, 1.0}
- **Key question**: does consolidation actually protect old memories?
- Test: store phase-1 memories, consolidate, flood with phase-2,
  measure phase-1 recall

---

## Phase 5: Metacognition (Est. 1 day)

### 5.1 Curiosity Module

- `num_curiosity_domains` ∈ {0 (off), 4, 8, 16}
- Does curiosity-driven exploration improve memory coverage?
- Or is it only relevant for the developmental AI framing?

### 5.2 Homeostasis

- `default_target_activation` ∈ {0.3, 0.5, 0.7}
- `gain_min/gain_max` range variations
- Does homeostatic regulation improve graph stability?

---

## Phase 6: Execution & Performance (Est. 1 day)

### 6.1 Batched vs Sequential Executor

- `use_batched_executor` True vs False
- Speed comparison across graph sizes
- Verify numerical equivalence still holds

### 6.2 Throughput Scaling

- Measure samples/sec vs node count (already partially done)
- Identify the knee where throughput becomes impractical
- Profile: what's the bottleneck? (graph execution, embedding, retrieval?)

---

## Implementation Strategy

### Principle: One Axis at a Time

Each test varies ONE parameter while holding all others at default.
Only after individual effects are understood do we test interactions.

### Test Harness

Build a generic parameter sweep runner:
```python
def sweep(param_name, values, benchmark, base_config=SOMAConfig()):
    results = {}
    for val in values:
        cfg = replace(base_config, **{param_name: val})
        score = benchmark(cfg)
        results[val] = score
    return results
```

### Metrics per Benchmark

| Benchmark | Primary Metric | Secondary |
|---|---|---|
| LongMemEval | F1 | EM, ROUGE-L, per-type |
| Associative hop | Indirect recall | Hop depth |
| Consolidation survival | Phase-1 recall after flood | Recall decay curve |
| CL classification | ACC, BWT | FWT |
| Agentic tasks | Task completion % | Tool efficiency |
| Throughput | Samples/sec | VRAM usage |

### Reporting

Each phase produces:
1. A sweep table (parameter value → metric)
2. A scaling curve plot
3. A recommendation: default value, dead weight flag, scaling benefit

### Priority Order

1. **Phase 1.3** (graph_rerank_alpha) — answers the biggest open question
2. **Phase 4.3** (consolidation) — tests the "memory protection" claim
3. **Phase 3** (growth/plasticity) — tests structural adaptation value
4. **Phase 1.1-1.2** (graph dimensions) — establishes minimum viable config
5. **Phase 2** (learning rules) — fine-tuning
6. **Phase 5** (metacognition) — likely lowest priority for memory-layer use case

### Estimated Timeline

With automated sweeps and the caching infrastructure we built:
- Phase 1: 2-3 days (graph_rerank_alpha tests need Ollama)
- Phase 2: 2 days
- Phase 3: 2 days (compound interactions take longer)
- Phase 4: 1-2 days
- Phase 5-6: 1-2 days

**Total: ~10-12 days** of focused experiment work.

---

## Expected Outcomes

### Best Case
Several components show clear scaling benefits beyond minimum config:
- graph_rerank_alpha improves retrieval F1 by 5-10%
- Consolidation protects old memories measurably
- Growth/pruning adapts graph to usage patterns
- Paper can claim "SOMA's architecture provides X, Y, Z advantages"

### Worst Case
Only the MemoryLayer API matters; the graph is pure overhead:
- graph_rerank_alpha = 0 is optimal
- Consolidation doesn't help
- Growth adds noise
- SOMA simplifies to "vector store + smart packing"
- Still a viable product but the brain-inspired architecture is dead weight

### Most Likely
Mixed results:
- Some components help for some use cases (e.g., consolidation helps
  long-term retention, graph helps associative recall)
- Other components are dead weight (e.g., curiosity, myelination)
- Optimal production config is simpler than current defaults
- Paper focuses on the components that work, honest about the rest
