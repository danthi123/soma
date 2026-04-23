# SOMA — Self-Organizing Memory Architecture

## Current Positioning (2026-04-15)

SOMA is being repositioned as a **local-first, learning agent-memory
layer** — a drop-in replacement for vector-DB-plus-RAG where the
store is a plastic graph that grows and prunes with use. The
research-grade "brain-inspired developmental AI" framing (below) is
the engineering substrate, not the product story.

- Product-facing pitch: `docs/positioning.md`
- Pivot decision + roadmap: `docs/plans/2026-04-15-memory-layer-pivot.md`
- Historical hybrid-brain framing: `docs/progress/HYBRID_PIVOT.md`

Anything under `src/soma/core/`, `src/soma/memory/`,
`src/soma/growth/`, `src/soma/io/`, `src/soma/deploy/` is load-bearing
for the memory layer and must stay working. The verbalizer +
bootstrap trainer stay in-tree but are off the critical path.

## What This Is

A brain-inspired developmental AI system built on PyTorch. NOT a language model — a dynamic processing system that grows through interaction via structural plasticity, complementary memory systems, and intrinsic motivation. Target: single NVIDIA RTX 3090 (24GB VRAM).

## Architecture

```
src/soma/
# --- Memory-layer product (what `pip install soma-memory` delivers) ---
├── memory/          # MemoryLayer (public API), ConversationalMemory,
│                    # backends/ (InProc + Qdrant + LanceDB + Chroma + pgvector)
├── serve.py         # FastAPI app: /store /retrieve /forget /auth/* /snapshot ...
├── cli.py           # `soma` entry point (version / index / chat / stats /
│                    # search / serve / auth)
├── _cli_commands/   # Packaged helpers for CLI subcommands (private API)
├── auth.py          # JWT issue / verify (HS256 + RS256)
├── auth_revocation.py  # BlocklistBackend: null / file / Redis
├── llm/             # Pluggable LLM backends (Ollama / OpenAI / Anthropic / vLLM / HF)
├── integrations/    # LangChain + LlamaIndex adapters
├── storage/         # ObjectStore protocol (LocalFS / S3 / GCS)
├── schemas/         # Typed-schema system (31 built-in)
├── metrics.py       # Prometheus counters/gauges/histograms
├── rate_limit.py    # Per-bundle token-bucket limiter
├── forget_audit.py  # GDPR forget audit trail
├── bundle.py        # Bundle-metadata loader
├── log.py           # JSON logging config
# --- Research substrate (brain-inspired dev AI; not on the product path) ---
├── core/            # Node, Edge, Graph, execution engine, config
├── growth/          # Synaptogenesis, neurogenesis, pruning, myelination
├── metacognition/   # Curiosity, HomeostaticRegulator, DevelopmentSchedule
├── io/              # TextEncoder/Decoder, ImageEncoder, AudioEncoder
├── consolidation/   # Consolidation cycle (artificial sleep), replay
├── developmental/   # Developmental trackers, PE, interactions
├── environments/    # Training environments
├── session/         # ChatSession (for substrate interactive mode)
├── training/        # Training loops + trainers
├── research/        # Research experiment harnesses
├── deploy/          # Chat-head factory + inference helpers
├── ui/              # Dear PyGUI dashboards
└── system.py        # Legacy `SOMA` top-level system (substrate brain)
tests/               # pytest, mirrors src/ structure
configs/             # YAML configs (default.yaml)
scripts/             # Training, visualization, analysis scripts (dev-time);
                     # thin wrappers over src/soma/_cli_commands/ for demo_*
docs/                # Whitepaper, positioning, cookbook, quickstart, rest-api
deploy/              # Helm chart, Grafana dashboards, docker-compose
clients/typescript/  # @pypi-named TS client (schema.d.ts regen'd from live app)
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
