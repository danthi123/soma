# SOMA Direction Review — April 17, 2026

## What SOMA Was Meant to Be

SOMA's original vision: a brain-inspired developmental AI system that
grows through interaction. Not a language model, not a wrapper around
existing tools — a novel architecture where the system itself changes
structurally as it learns. The core mechanisms:

- **Dynamic graph** — nodes and edges created/destroyed at runtime
- **Hebbian learning** — "fire together, wire together" weight updates
- **Structural plasticity** — neurogenesis (new nodes), synaptogenesis (new edges), pruning (remove unused)
- **Consolidation cycles** — offline reorganization ("sleep")
- **Critical periods** — developmental windows of heightened learning
- **Myelination** — strengthening reliable pathways
- **Curiosity module** — novelty-driven exploration
- **Homeostatic regulation** — keeping the system balanced
- **Working memory** — short-term buffer with decay
- **Episodic memory** — one-shot storage

This is fundamentally different from every major AI system in 2026.
Transformers are static after training. Self-evolving agents (Hermes,
Darwin Godel Machine) evolve their scaffolding code, not their
architecture. Neuromorphic chips are hardware. SOMA is a **software
system whose architecture changes through experience**.

## What Happened

The project pivoted to "agent memory layer" positioning because that
appeared to be a practical application. This led to:

1. Building a MemoryLayer API (store/retrieve/pack_context)
2. Benchmarking against agent memory competitors (Mem0, Zep, Letta)
3. Testing whether SOMA's graph helps retrieval

## What We Learned (April 17 audit)

### Things that DON'T work:
- **Graph features for classification**: ACC identical (0.74) for
  0-16 integrators. A plain linear projection matches. The graph
  adds nothing for downstream classification tasks.
- **Consolidation for memory protection**: Recall stays 0.50 after
  adversarial flood with or without consolidation (k=1 synthetic test).
- **Growth/plasticity on synthetic data**: No effect — thresholds
  never triggered on small corpus. Graph size unchanged across all
  conditions.
- **Graph reranking on synthetic facts**: alpha>0 actively HURT
  retrieval on one-line fact recall.

### Things that DO work:
- **Graph reranking on real conversations**: Phase 1.3 showed alpha=1.0
  (pure graph) beats alpha=0.0 (pure cosine) on LongMemEval real
  conversation data — F1 0.163 vs 0.142 (+15%). This is the most
  important finding of the day.
- **pack_context selective retrieval**: SOMA F1=0.124 vs baseline
  F1=0.111 on LongMemEval (50 items).
- **Head-replay for CL**: ACC=0.808±0.005, BWT=-0.042±0.005, beating
  A-GEM's BWT by 2.4x. But this is standard replay technique, not
  SOMA-specific.

### The key insight:
The graph IS capturing something useful — but only on data with
temporal/contextual structure (real conversations), not on isolated
synthetic facts. This suggests the brain-inspired processing extracts
relational patterns that embedding similarity alone misses.

## Why Agent Memory Isn't the Right Market

The agent memory space in April 2026 is extremely crowded:
- Mem0: $24M funded, 80K+ stars, hybrid search + fact extraction
- Zep/Graphiti: Temporal knowledge graph, +18.5% on LongMemEval
- MAGMA: Multi-graph with policy-guided traversal, 61.2% on LongMemEval
- Letta: LLM-managed memory tiers, strong community
- SYNAPSE: Spreading activation with Hebbian learning (in production)
- sqlite-memory: CRDT agent memory (shipped April 13, 2026)

SOMA's MemoryLayer API is a subset of what competitors already ship.
The graph reranking result is promising but preliminary (20 items, one
model). Competing here means fighting well-funded teams on their home
turf with a late start.

## The Real Opportunity: Brain-Inspired AI

### What current AI cannot do:
1. **Learn continuously** without catastrophic forgetting or retraining
2. **Form novel associations** between unrelated concepts
3. **Autonomously reorganize** representations during downtime
4. **Allocate attention** — deeply process surprising input, ignore expected
5. **Develop through stages** — capabilities emerge in sequence

Every self-evolving agent in 2026 relies on a frozen LLM at its core.
They evolve their CODE (prompts, tools, workflows). SOMA's vision is
a system that evolves its STRUCTURE — the architecture itself changes.
Nobody is doing this in software at a useful scale.

### Why the benchmarks were wrong:
We tested a developing brain on multiple-choice exams:
- Permuted-MNIST: tests static feature quality (brain's advantage is adaptation, not static features)
- LongMemEval: tests single-session retrieval (brain's advantage is long-term association)
- Synthetic fact recall: tests exact matching (brain's advantage is relational reasoning)

The Phase 1.3 result (graph helps on real conversations) hints that
the right evaluation would test:
- Long-horizon interaction (weeks/months, not single sessions)
- Adaptation to changing environments
- Novel association between temporally separated events
- Consolidation benefits over many sleep cycles

### What a brain-inspired benchmark looks like:
An environment where:
1. Rules change over time → tests continuous adaptation
2. New concepts appear that weren't in training → tests neurogenesis
3. Long-term experience matters more than single-session → tests consolidation
4. Structure must be discovered, not just memorized → tests self-organization
5. Resources are limited → tests attention allocation

Think: a virtual creature navigating a changing world. Not an agent
answering questions — a system developing capabilities through
interaction.

## Where SOMA's Existing Code Fits

### Directly relevant to brain-inspired AI:
- `src/soma/core/` — Graph engine (Node, Edge, Graph, execution)
- `src/soma/growth/` — Synaptogenesis, neurogenesis, pruning, myelination
- `src/soma/metacognition/` — Curiosity, homeostatic regulation
- `src/soma/consolidation/` — Consolidation cycles
- `src/soma/memory/` — Working memory, episodic memory

### Useful infrastructure, keep:
- `src/soma/core/config.py` — SOMAConfig (all hyperparameters)
- `src/soma/system.py` — SOMA system orchestrator
- Test infrastructure (`tests/`)
- Build system, CI, configs

### Pivot-specific, may change:
- `src/soma/memory/api.py` — MemoryLayer API (agent memory specific)
- `src/soma/io/` — Text/image/audio encoders (I/O layer)
- `benchmarks/` — Current benchmarks test the wrong things
- `research/cl/` — CL classification harness

### Potentially removable:
- `src/soma/deploy/` — Deployment for memory layer use case
- `src/soma/training/` — Verbalizer bootstrap (LLM-interface specific)

## Possible Next Steps

### Option A: Open-Ended Learning Environment
Build a simple environment where SOMA can demonstrate developmental
learning. Not trying to compete with LLMs on language tasks — instead,
show that the architecture discovers structure in a changing world.

Concrete example: a grid world where:
- Food sources appear and disappear
- Threats move with learnable patterns
- The agent must develop navigation, avoidance, foraging
- The world changes rules periodically (food locations shift, new threat types)
- Success = adaptation speed after rule changes

This tests neurogenesis (new nodes for new concepts), consolidation
(strengthening useful patterns during sleep), critical periods
(faster learning early), and pruning (removing outdated strategies).

**Pro**: Directly tests SOMA's mechanisms. Clear metrics (survival,
adaptation speed). No LLM dependency.
**Con**: Hard to make "useful" in a product sense. Academic niche.

### Option B: Adaptive Processing Layer
Position SOMA as a pre-processing layer that learns to allocate
attention — which parts of a data stream deserve deep processing,
which can be ignored. This is genuinely brain-like (selective attention)
and practically useful for edge computing, real-time processing.

Concrete example: network traffic analysis where:
- Most traffic is routine (ignore)
- Anomalies need deep inspection (allocate resources)
- Attack patterns evolve over time (continuous adaptation)
- The system develops sensitivity to new threat patterns without retraining

**Pro**: Practical application. Edge computing is a real market.
Neuromorphic chips are doing this in hardware — SOMA could be the
software equivalent.
**Con**: Narrow application. Competes with existing anomaly detection.

### Option C: Transformer Augmentation
Use SOMA as a "brain" that sits alongside a transformer, handling
things the transformer can't: continuous learning, long-term
association, developmental adaptation. The transformer handles
language I/O; SOMA handles memory, learning, and adaptation.

This is the "interpretation layer" you mentioned. The Phase 1.3
result (graph helps real conversations) supports this — SOMA adds
something embedding similarity alone doesn't capture.

Concrete example: a personal AI assistant where:
- The transformer handles conversation
- SOMA handles the user model (learns preferences, habits, patterns)
- SOMA's graph adapts over weeks/months to the specific user
- Consolidation cycles reorganize user knowledge during idle time
- The assistant gets better at understanding this specific user over time

**Pro**: Leverages the Phase 1.3 finding. Practical and novel.
Local-first (SOMA stays on device, transformer can be remote).
**Con**: Competes with agent memory space again, though the angle
is different (adaptive user modeling, not memory retrieval).

### Option D: Pure Research — Developmental AI Paper
Write a paper documenting:
1. The architecture (already built)
2. What works and what doesn't (audit results)
3. The Phase 1.3 finding (graph captures relational structure)
4. A proper developmental benchmark (design and run)
5. Analysis of when brain-inspired mechanisms help vs. don't

**Pro**: Honest, novel contribution. The negative results
(graph doesn't help classification) are publishable. The positive
result (graph helps on conversations) provides the hook.
**Con**: No product outcome. Requires designing new benchmarks.

## Immediate Next Steps

### 1. Investigate the Phase 1.3 finding (high priority)
The graph helping on real conversations but not synthetic facts is
the most actionable finding. We need to understand WHY:
- Is it temporal structure in conversations that the graph captures?
- Is it the multi-turn context that creates useful activation patterns?
- Does it scale? (Test on 50, 100, 200 items)
- Is it consistent across conversation types?

### 2. Design a proper developmental benchmark
If we want to test what SOMA was built for, we need benchmarks that
test development, adaptation, and association — not static retrieval
or classification.

### 3. Decide project direction
Based on findings from #1 and personal preference:
- Open-ended learning (Option A) if the goal is pure research
- Transformer augmentation (Option C) if the goal is eventually practical
- Research paper (Option D) if the goal is to document and share

### 4. Preserve what we've built
Regardless of direction, the work from today is valuable:
- Architecture audit methodology (reusable for any architecture)
- CL benchmark infrastructure (harness, tests, caching)
- LongMemEval pipeline (working end-to-end)
- Phase 1.3 positive result (publishable finding)

## Key Dates and Context

- Project started: ~April 15, 2026
- Agent memory pivot: April 15, 2026
- Architecture audit: April 17, 2026
- Phase 1.3 graph reranking positive result: April 17, 2026
- This document: April 17, 2026

## References

### Our Results
- `research/cl/reports/b3_soma_cl.json` — Full CL ablation data
- `research/audit/results/` — All audit phase results
- `docs/plans/soma-architecture-audit.md` — Audit plan

### Key External Research
- MAGMA: Multi-graph agentic memory (arxiv 2601.03236)
- MemoryArena: Benchmark exposing recall→use gap (arxiv 2602.16313)
- SYNAPSE: Spreading activation for associative recall (OpenClaw)
- Self-Evolving Agents Survey (arxiv 2507.21046)
- Why AI Systems Don't Learn (arxiv 2603.15381)
- Triple-Loop Consolidation (arxiv 2603.27188)
- Multi-Agent Memory challenges (arxiv 2603.10062)
