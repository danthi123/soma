# SOMA Autonomous Build Design

**Date:** 2026-04-12
**Status:** Approved (v2 — revised after gap review)
**Approach:** Scheduled task chain with checkpoint protocol

## Overview

Autonomous build-out of all 6 SOMA stages using a scheduled task that spawns Claude Code sessions every 45 minutes. Each session builds one work unit (implementation + its tests), commits, updates checkpoint, pushes to Gitea, and exits. Full wiki integration at stage milestones.

## Design Principles

- **Tests bundled with implementation** — each unit writes code AND its tests. No separate "test-only" units. Bugs are caught immediately, not deferred.
- **Dependencies respected** — growth basics (synaptogenesis, pruning) live in Stage 1 because Stage 1's validation criteria require them. Myelination waits for Stage 4.
- **Consolidation is layered** — Stage 2 builds replay-only consolidation. Stage 4 enhances it with structural maintenance (pruning + myelination during sleep).
- **Lock file prevents overlap** — a `.soma-build.lock` file prevents concurrent sessions from causing git conflicts.
- **Environment bootstraps itself** — first session installs dependencies if missing.

## Work Unit Breakdown (31 units)

### Stage 1: Core Graph Engine (9 units)

1. `core/config.py` — SOMAConfig dataclass + YAML loader + tests
2. `core/ring_buffer.py` — RingBuffer for activation history + tests
3. `core/node.py` — Node dataclass, NodeType enum, forward pass, MLP init + tests
4. `core/edge.py` — Edge dataclass, transmit, projection creation + tests
5. `core/graph.py` — Graph container (add/remove node/edge, adjacency, queries, serialization) + tests
6. `core/execution.py` — topological_sort, cycle detection (back-edge flagging), execute_graph, previous_activations buffer + tests
7. `core/learning.py` — update_step (backprop + Hebbian + homeostatic gain + maturity advancement) + tests
8. `growth/synaptogenesis.py` + `growth/pruning.py` — co-activation connection formation + weak edge/orphan removal + tests
9. Integration test: synthetic task (XOR/parity/sequence), verify graph grows, prunes, and is stable over 10K steps

### Stage 2: Memory Systems (5 units)

10. `memory/working_memory.py` — WorkingMemory (attention read, gated write, decay) as nn.Module + tests
11. `memory/episodic_memory.py` — EpisodicMemory (encode, retrieve, sample_for_replay) as nn.Module + tests
12. `consolidation/cycle.py` — consolidation_cycle (replay only — no structural maintenance yet) + tests
13. Integration test: memorization task (immediate recall via WM, delayed recall via episodic, post-consolidation recall via parametric)

### Stage 3: Text I/O (4 units)

14. `io/text_encoder.py` — TextEncoder (BPE tokenizer via `tokenizers` library, embeddings, positional encoding) + tests
15. `io/text_decoder.py` — TextDecoder (logits to tokens, autoregressive generation helper) + tests
16. `io/dataset_feeders.py` — TextDatasetFeeder (chunk prediction from text data) + tests
17. Integration test: expose to text data, verify loss decreases over 5K steps, inspect graph for text-processing subgraph emergence

### Stage 4: Meta-Cognitive Module (6 units)

18. `metacognition/curiosity.py` — CuriosityModule (domain tracking, learning progress estimation) + tests
19. `metacognition/homeostasis.py` — HomeostaticRegulator (loss spike detection, growth rate control, global LR multiplier) + tests
20. `metacognition/development.py` — DevelopmentSchedule + CriticalPeriod (plasticity multipliers by node type and step) + tests
21. `growth/neurogenesis.py` — new node creation when errors persist + tests
22. `growth/myelination.py` — detect_linear_chains, compose_nodes, circuit consolidation + tests
23. Enhance `consolidation/cycle.py` to call pruning + myelination during sleep. Update integration test: verify curiosity correlates with learning, homeostasis prevents divergence, myelination consolidates circuits

### Stage 5: Multimodal (3 units)

24. `io/image_encoder.py` — ImageEncoder (patch-based Conv2d projection) + tests
25. `io/multimodal_curriculum.py` — MultimodalCurriculum (staged modality weights by step) + tests
26. Integration test: cross-modal association (text + image pairs, verify cross-modal connections form)

Note: AudioEncoder is explicitly out of scope for v1. Can be added later following the same ImageEncoder pattern.

### Stage 6: Interactive Environment (4 units)

27. `soma/system.py` — SOMA main class (step loop, seed graph init, save/load state, _create_experience_vector) + tests
28. `scripts/train.py` — training script (CLI args, dataset loading, training loop, checkpointing, logging) + `scripts/visualize.py` (graph topology visualization via networkx/matplotlib)
29. CLI interface — `scripts/interactive.py` for real-time text interaction with SOMA, session management (start/stop/resume)
30. Full system integration test: end-to-end from seed graph through text exposure, verify all subsystems work together. Create SOMA wiki entity, MOC, and final source page.

## Checkpoint Protocol

Checkpoint file: `docs/progress/CHECKPOINT.md`

Each session:
1. Check for `.soma-build.lock` — if exists and < 60 min old, exit immediately
2. Create `.soma-build.lock` with timestamp
3. Check if `pytest` is importable — if not, run `pip install -e ".[dev]"`
4. Read checkpoint -> determine next work unit
5. Read CLAUDE.md + relevant whitepaper sections
6. Build implementation + tests for the unit
7. Run `pytest tests/ -v --tb=short` — fix failures (up to 3 retries)
8. Run `ruff check src/ tests/ --fix && ruff format src/ tests/`
9. Run `mypy src/soma/` — fix type errors
10. Commit + push to Gitea (token auth fallback)
11. Update checkpoint (mark unit complete, advance to next)
12. If stage completed: push wiki updates via Gitea API
13. Remove `.soma-build.lock`
14. If blocked: set status to `blocked` with error details, remove lock, stop

## Scheduled Task

- Runs every 45 minutes (allows headroom for larger units)
- Each session handles exactly one work unit
- ~31 sessions total = ~23 hours of autonomous work
- Lock file prevents overlapping sessions

## Wiki Integration

| Event | Wiki Action |
|---|---|
| First unit starts | Create `entities/soma.md` with initial project info |
| Stage completed | Update `entities/soma` + create `sources/soma-stage-N-completion` |
| Surprising design decision | Create `atoms/soma-{decision-name}` |
| Cross-project pattern found | Update relevant molecule |
| All stages done | Create `moc/soma-developmental-ai`, update `index.md`, `registry.md` |

## Error Recovery

- **Test failure:** Fix in same session, retry up to 3 times, then mark blocked
- **Lint/type failure:** Fix in same session (auto-fixable via ruff --fix)
- **Blocked status:** Next session reads blocker, attempts different approach or simplified implementation
- **Gitea push failure:** Retry with inline token auth, then mark blocked
- **Lock file stale (> 60 min):** Assume previous session crashed, delete lock and proceed
- **Missing dependency:** Run pip install, then retry

## Key Design Decisions

- Whitepaper (docs/whitepaper.md) is the authoritative spec
- Full autonomy — all implementation decisions made by the builder
- One unit per session keeps context clean and commits granular
- Tests are co-located with implementation to catch bugs immediately
- Growth basics in Stage 1, advanced growth in Stage 4
- Consolidation is layered: replay-only in Stage 2, structural maintenance added in Stage 4
- AudioEncoder deferred to post-v1
- 45-minute intervals with lock file prevent overlap
