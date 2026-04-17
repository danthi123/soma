# Phase 45–47: Agentic Workflow Benchmark Suite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Quantify "standalone LLM" vs "LLM + SOMA" on real agentic
tasks across model tiers. The headline claim: SOMA makes a 4B local
model more effective than a bare 9B model for stateful multi-step
tasks. If that holds, it's the strongest possible product pitch.

**Depends on:** Phase 43 (built-in agent.* schemas + context packer).

**Model configs (April 2026 SOTA — verified via web search):**

Test across a range of VRAM utilizations, not just Q4 everything.
Max total VRAM: 22 GB (2 GB headroom on the 24 GB 3090).

| Config | Model | Quant | ~Model VRAM | +SOMA (~3GB) | Total |
|---|---|---|---|---|---|
| SOTA-max | Qwen3.5-27B | Q4_K_M | ~16GB | 3GB | **~19GB** |
| SOTA-alt | GLM-4.7-Flash | Q4_K_M | ~TBD | 3GB | ~TBD |
| SOTA-high | Qwen3.5-9B | Q8_0 | ~9GB | 3GB | ~12GB |
| SOTA-mid | Qwen3.5-9B | Q4_K_M | ~6GB | 3GB | ~9GB |
| Reasoning | Phi-4 Reasoning 14B | Q4_K_M | ~9GB | 3GB | ~12GB |
| Small-high | Qwen3.5-4B | Q8_0 | ~4GB | 3GB | ~7GB |
| Small-low | Qwen3.5-4B | Q4_K_M | ~3GB | 3GB | ~6GB |
| Ultralight | Qwen3-0.6B | Q8_0 | ~0.7GB | 3GB | ~3.7GB |

SOMA footprint includes sbert (all-MiniLM-L6-v2 ~0.5 GB) + PyTorch
+ MemoryLayer working set. All served via Ollama on the same RTX 3090.

**Key comparisons enabled by the range:**
- Does Qwen3.5-4B Q8 + SOMA beat Qwen3.5-27B Q4 alone? (The
  extreme pitch: 4B with memory outperforms 27B without.)
- Does Qwen3.5-4B Q8 + SOMA beat Qwen3.5-9B Q4 alone? (More
  realistic crossover question.)
- Does quantization quality matter more or less when SOMA
  compensates for context-window limitations?
- At what VRAM point does "more model" stop beating "smaller model +
  SOMA"? (The crossover curve.)

---

## Architecture

```
benchmarks/agentic/
├── harness.py         # generic agent-loop runner (model-agnostic)
├── tasks/             # one Python file per task (defines scenario + ground truth)
│   ├── fact_recall.py
│   ├── tool_learning.py
│   ├── context_overflow.py
│   ├── session_resume.py
│   └── multi_step_plan.py
├── agents/
│   ├── baseline.py    # standalone LLM agent (context-window only)
│   └── soma_agent.py  # LLM + SOMA agent (typed schemas + context packer)
├── models.py          # Ollama model registry (name → config)
├── metrics.py         # scoring: completion, steps, accuracy, latency
├── run_agentic.py     # orchestrator: model × agent × task matrix
└── reports/
    └── agentic_benchmark.md
```

**Two agent implementations, same task interface:**

1. **BaselineAgent** — pure LLM. Maintains a growing conversation
   list. When context exceeds the model's window, truncates from the
   front (realistic — this is what most agent frameworks do). No
   persistent memory. Cross-session = clear slate.

2. **SomaAgent** — LLM + SOMA. Every tool call result, observation,
   decision, and task-state update is `store_typed()` via the Phase
   43 agent.* schemas. Before each LLM call, `pack_context()` builds
   a bounded prompt from recency + relevant facts + task state +
   tool-call history. Cross-session = `MemoryLayer.save/load`.

Both agents talk to the LLM via Ollama's `/api/chat` endpoint with
tool definitions. The harness controls the task, provides tools, and
scores the outcome.

---

## Task suite (5 tasks, each designed to stress a SOMA-advantaged dimension)

### Task 1: Deep fact recall

Scenario: the agent has a 100-turn conversation with a "user." Turns
1-10 establish 5 facts (name, city, favorite food, job, pet). Turns
11-95 are filler (unrelated chitchat). Turn 96-100 ask questions
about the original 5 facts.

**Metrics:** fact-recall accuracy (0-100%), steps to answer.
**SOMA advantage:** retrieves stored facts; baseline must hold them
in a shrinking context window.

### Task 2: Tool-call learning

Scenario: the agent has access to a `search_database` tool with a
quirky API — queries must be lowercase, max 3 words, and results are
paginated (returns `has_more=true` with a `cursor`). The agent must
search for 10 items, handling pagination + format rules.

**Metrics:** tool-call success rate, total calls (fewer = better),
format-error count.
**SOMA advantage:** stores past tool calls with success/failure;
retrieves successful patterns before each new call.

### Task 3: Context-window overflow

Scenario: the agent processes 50 "documents" (each ~500 tokens), must
extract a key fact from each, then answer 10 synthesis questions that
require combining facts from multiple documents. Total raw context:
~25K tokens (exceeds 4B model's effective window at Q4).

**Metrics:** synthesis accuracy, extracted-fact accuracy.
**SOMA advantage:** stores each extracted fact via typed schema;
`pack_context()` retrieves relevant facts per synthesis question
instead of stuffing 25K tokens into a 4K effective window.

### Task 4: Session resume

Scenario: the agent starts a 10-step task (e.g., setting up a
project: create repo, add files, configure CI, write README, etc.).
After step 5, the "session ends" (context cleared, simulating a
restart). The agent is told "continue from where you left off."

**Metrics:** correct-continuation rate (does it pick up at step 6?),
redundant-step count (does it redo steps 1-5?).
**SOMA advantage:** task state persisted via `agent.task_state`;
baseline = total amnesia.

### Task 5: Multi-step planning with dependencies

Scenario: the agent must fix a bug across 5 files. Each file has a
clue pointing to the next. The fix requires changes in a specific
order (file C depends on file B's fix). The agent has `read_file`,
`write_file`, and `run_tests` tools.

**Metrics:** task completion (did the bug get fixed?), steps to
completion, backtrack count (undoing wrong changes).
**SOMA advantage:** stores observations + decisions per file;
retrieves relevant context when working on dependent files.

---

## Evaluation matrix

```
8 model configs × 2 agent types × 5 tasks × 3 seeds = 240 runs
```

Each run is scored on:
- **Completion rate** (0 or 1, did the task succeed?)
- **Steps to completion** (fewer = more efficient)
- **Accuracy** (task-specific, 0-100%)
- **Tool-call error rate** (% of tool calls that failed to parse/execute)
- **Wall-clock time** (seconds)
- **Peak VRAM** (MB, from `nvidia-smi`)

Aggregate: per-task comparison table + per-model-tier comparison.

---

## Sub-phases

### Phase 45: Harness + task suite + baseline agent

**Scope:** build the benchmark infrastructure, define all 5 tasks,
implement `BaselineAgent`, run the baseline numbers.

**Files:**
- Create: all files under `benchmarks/agentic/` except `soma_agent.py`
- Create: `tests/test_agentic/` — unit tests for tasks, harness, metrics

**Commit pattern:** 3 commits (harness+metrics, tasks, baseline agent + numbers).

**Prereq:** Ollama running with at least one model pulled.

### Phase 46: SOMA agent + typed-schema integration

**Scope:** implement `SomaAgent` using Phase 43's agent.* schemas +
`pack_context()`. Run the full matrix.

**Files:**
- Create: `benchmarks/agentic/agents/soma_agent.py`
- Extend: `benchmarks/agentic/run_agentic.py` (add SOMA runs)
- Create: `benchmarks/agentic/reports/agentic_benchmark.md`

**Commit pattern:** 2 commits (soma_agent, full-matrix results).

### Phase 47: Analysis + paper-ready writeup

**Scope:** statistical analysis, visualization, paper section draft.
Answer the headline question: does SOMA make a 4B model more
effective than a bare 9B model?

**Files:**
- Create: `benchmarks/agentic/reports/agentic_analysis.md`
- Create: `benchmarks/agentic/reports/figures/` (comparison charts)
- Extend: `benchmarks/reports/paper-draft.md` (new section)
- Extend: `docs/positioning.md` (headline claim update if warranted)

**Commit pattern:** 2 commits (analysis, paper integration).

---

## Total scope

Phase 45: ~1 week (harness + 5 tasks + baseline)
Phase 46: ~3-4 days (SOMA agent + full matrix)
Phase 47: ~3-4 days (analysis + paper)

Total: ~2-3 weeks.

---

## Gotchas

- **Ollama model pull:** ensure the model is downloaded before the
  benchmark starts. Qwen3.5-9B Q4 is ~5.5 GB; Qwen3.5-4B is ~2.8 GB.
  Add a preflight check.
- **Tool-call format varies by model.** Ollama's `/api/chat` with
  `tools=` is the standard path. Some models need `format: json` for
  structured output. The harness should detect and handle per-model
  quirks.
- **Thinking/reasoning mode must be disabled for tool calling.**
  Models like Qwen3.5 and SmolLM3 support `/think` and `/no_think`
  modes. Thinking tokens can interfere with structured JSON tool-call
  output. The model registry should carry a `disable_thinking: bool`
  flag per model (default True for tool-calling tasks). Pass
  `"think": false` in the Ollama options or use the `/no_think`
  system prompt prefix where the model supports it. Optionally run
  a secondary comparison WITH thinking enabled to measure the
  tradeoff (reasoning quality vs tool-call reliability).
- **Seed control.** Use fixed seeds for reproducibility. Each of the
  3 seeds should produce meaningfully different task instances (not
  just different random numbers for the same scenario).
- **Filler turns in Task 1** must be semantically unrelated to the
  facts — otherwise sbert retrieval gets free signal from embedding
  proximity.
- **Session resume (Task 4)** requires clearing the LLM's
  conversation history without clearing SOMA's MemoryLayer. The
  harness must support this split.
- **VRAM measurement:** `nvidia-smi` snapshot at 1-second intervals
  during each run; report peak.
- **Fair comparison:** both agents get the same system prompt, same
  tool definitions, same task description. The only difference is
  memory management (context-window vs SOMA-backed).
