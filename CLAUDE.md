# SOMA — Self-Organizing Memory Architecture

## What This Is

A brain-inspired developmental AI system built on PyTorch. NOT a language model — a dynamic processing system that grows through interaction via structural plasticity, complementary memory systems, and intrinsic motivation. Target: single NVIDIA RTX 3090 (24GB VRAM).

## Architecture

```
src/soma/
├── core/          # Node, Edge, Graph, execution engine, config
├── memory/        # WorkingMemory, EpisodicMemory (parametric = graph weights)
├── growth/        # Synaptogenesis, neurogenesis, pruning, myelination, critical periods
├── metacognition/ # CuriosityModule, HomeostaticRegulator, DevelopmentSchedule
├── io/            # TextEncoder/Decoder, ImageEncoder, AudioEncoder, dataset feeders
├── consolidation/ # Consolidation cycle (artificial sleep), replay
tests/             # pytest, mirrors src/ structure
configs/           # YAML configs (default.yaml)
scripts/           # Training, visualization, analysis scripts
docs/              # Whitepaper, design docs
```

## Development Stages (Implementation Order)

1. **Core Graph Engine** — Node, Edge, Graph, execute_graph, topological sort, cycle handling
2. **Memory Systems** — WorkingMemory, EpisodicMemory, consolidation cycle
3. **Text I/O** — BPE tokenizer (8K vocab), TextEncoder/Decoder, dataset feeders
4. **Meta-Cognitive Module** — Curiosity, homeostasis, critical periods, myelination
5. **Multimodal** — ImageEncoder, AudioEncoder, MultimodalCurriculum
6. **Interactive Environment** — CLI/web interface, dashboard, session management

## Key Design Principles

- Graph is dynamic — nodes/edges created and destroyed at runtime
- Learning never stops — no train/deploy split
- Three memory tiers: working (32 slots, decay), episodic (10K, one-shot), parametric (graph weights)
- Waves not layers — topological propagation, not fixed layer ordering
- Cycles handled via previous-timestep activations (implicit recurrence)
- Hebbian + backprop learning rule
- All node types are small MLPs with residual connections and homeostatic gain

## Commands

```bash
# Install
pip install -e ".[dev]"

# Test
pytest tests/ -v

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/soma/
```

## Conventions

- Python 3.11+, PyTorch
- Type hints everywhere (strict mypy)
- Ruff for formatting and linting (100 char line length)
- Dataclasses for config/data; `nn.Module` for anything with learnable parameters
- Node, Edge, WorkingMemory, EpisodicMemory are `nn.Module` subclasses (they hold learnable tensors)
- SOMAConfig is a dataclass (no learnable params)
- Tests mirror src/ structure: `tests/test_core/`, `tests/test_memory/`, etc.
- All tensor operations must be GPU-compatible (use `device` parameter)
- Config via SOMAConfig dataclass (configs/ for YAML overrides)
- `global_step` is passed as a parameter (never a global variable)
- Whitepaper constants (e.g., `BASE_LR`) map to SOMAConfig attributes (e.g., `config.base_lr`)
- Utility functions (`generate_uuid`, `create_projection_if_needed`) live in `core/utils.py`

## VRAM Budget

At max scale (50K nodes, 500K edges): ~6-8 GB of 24 GB. Substantial headroom.

## Critical Invariants

- SENSOR and OUTPUT nodes are never pruned
- Edges younger than `pruning_grace_period` steps are never pruned
- Node gain is clamped to [0.1, 10.0]
- Edge weights are clamped to [-5.0, 5.0]
- Max one consolidation cycle per `consolidation_interval` steps
- Graph must remain executable (no orphaned subgraphs blocking I/O path)

## Execution Paths

Two `execute_graph` implementations exist:

- **Sequential (`execute_graph`)**: one node at a time; reference behavior.
- **Batched (`execute_graph_batched`)**: same-shape nodes in each wave share one stacked matmul. 3-5x faster on CUDA, identical outputs (atol=1e-5).

`SOMAConfig.use_batched_executor=True` (default) selects the batched path.
Flip to `False` to fall back to sequential if a bug ever surfaces — no
weight or data migration needed.
