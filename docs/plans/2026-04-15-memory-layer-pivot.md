# Memory-Layer Pivot — 2026-04-15

**Status:** Decision recorded. Execution tracked via tasks #139–#142.
**Supersedes (positioning only):** "SOMA as portable hybrid brain" from
`docs/progress/HYBRID_PIVOT.md`. The engineering in HYBRID_PIVOT.md
remains valid and is still the substrate this pivot builds on.

---

## TL;DR

Three honest training regimes on the same architecture converged to
essentially the same held-out LM loss. The verbalizer projector, not
SOMA, is doing the useful work under the current bootstrap. Therefore:

- **Stop positioning SOMA as "the brain that makes a small LLM smart."**
- **Start positioning SOMA as a local-first, learning agent-memory
  layer** — a drop-in replacement for vector-DB-plus-RAG that grows
  structure over time.
- The verbalizer stays, but is reframed: it no longer translates "all of
  SOMA" into a soft prompt. It translates **retrieved context** (WM +
  episodic + graph-local neighbourhood) into a soft prompt when a
  dialogue ergonomics story is needed.

## The Evidence That Forced This

All three regimes trained the same verbalizer projector against the
same frozen Qwen2.5-3B-Instruct (int4), batch=8, 10K steps, cosine LR,
`soma_max_tokens=16`, same wikitext-103 training split + wikitext-2
ablation split:

| Regime                          | Held-out loss floor | Δ(zero-SOMA) | Δ(shuffle) |
|---------------------------------|---------------------|--------------|------------|
| Frozen random SOMA              | ~3.12 nats          | +0.008       | +0.002     |
| Unsupervised Hebbian pre-train  | ~3.15 nats          | similar      | similar    |
| Joint SOMA + verbalizer         | ~3.12 nats          | similar      | similar    |

The ablation measures what happens when we replace real SOMA state with
zeros / a shuffled batch. If SOMA were content-bearing, real should
beat zero *and* shuffle by a visible margin. The delta is within the
eval's noise band across all three regimes.

**Interpretation:** the projector learns a corpus prior (visible in
samples: trained outputs carry wikitext-103's `@-@`, `= = Plot =`,
`$ 10 @.@ 5 million` tokenizer signature), and that prior is what
drops loss from 4.2 → 3.1. SOMA's state, whatever it encodes, is
washed out by the projector's MLP or lives in directions the LLM
doesn't condition on.

Raw data:
- `reports/bootstrap-2026-04-15-accel-wt103.md` (frozen regime)
- `artifacts/bootstrap-2026-04-15-joint-wt103/` (joint regime)
- `artifacts/bootstrap-2026-04-15-qwen3b-wikitext2/` (Hebbian pre-train)

## Why The Positioning Shift

"Small LLM + SOMA beats bigger LLM alone" was never supported by the
numbers; it was aspirational. And even if it were, the consumer-JARVIS
story was going to require a level of raw language capability that a
3B LLM doesn't give, *regardless* of what SOMA contributes. Shipping
this as "the portable brain that makes small models smart" puts us in
head-to-head comparison with frontier models — a comparison we lose
on every axis users actually benchmark.

What the architecture *does* demonstrably provide:

1. **Persistent, learning memory that survives across sessions** — the
   growth journal, WM, episodic store, and graph weights are real and
   work today.
2. **Structural plasticity as a memory operation** — new
   associations form edges; unused ones prune. This is genuinely
   different from cosine-similarity-over-a-vector-store.
3. **Local-first, user-owned** — the entire brain is a directory on
   the user's disk. No cloud dependency, no tenant isolation problem,
   no vendor lock-in.
4. **LLM-agnostic** — swap the transformer freely; the memory travels
   with the user. (This was the only part of the "portable brain"
   story that actually held up empirically.)

Those four capabilities are exactly the capabilities the agent-memory
space (Mem0 / Zep / Letta / MemGPT) sells. The difference is that
those systems bolt memory onto an LLM via prompting; SOMA has an
adaptive graph that actually restructures with use. That's a defensible
product story and the engineering to prove it is already built.

## What Stays / What Changes

### Stays (no code changes)
- Entire `src/soma/core/` — Node, Edge, Graph, execution paths.
- `src/soma/memory/` — WorkingMemory, EpisodicMemory, consolidation.
- `src/soma/growth/` — plasticity, pruning, myelination, critical
  periods.
- `src/soma/io/` — TextEncoder/Decoder, tokenizer, BPE.
- `src/soma/deploy/` — tier detection, int4/int8, bundle save/load.
- `src/soma/io/verbalizer.py` — the `SomaVerbalizer` module stays.
- `src/soma/session/chat_session.py` — ChatSession stays.
- The entire test suite.
- The whitepaper as a research document.

### Changes (new code)
- **`src/soma/memory/api.py`** (new): public `MemoryLayer` class.
  `store(text, metadata=None)`, `retrieve(query, k=5)`,
  `get_recent(n)`, `related(node_id, depth=2)`, `forget(node_id)`,
  `save(path)`, `load(path)`. Wraps SOMA's internals behind a surface
  that reads like a vector DB but hides the graph/growth behaviour as
  quality-of-service rather than API complexity.
- **`scripts/demo_chat_persistent.py`** (new): end-to-end demo. Fire
  up a chat with an LLM of choice, chat for N turns, quit, re-open
  bundle, verify the model "remembers" prior turns via MemoryLayer.
- **`reports/memory-layer-vs-rag-benchmark.md`** (later, Stage 3):
  side-by-side quality on a recall-and-reason benchmark vs. Chroma+RAG
  and vs. Mem0.

### Changes (framing only, no code)
- `docs/whitepaper.md` gets a preface noting its status as a research
  document; the product doc is `docs/positioning.md`.
- `CLAUDE.md` gains a "Current positioning" section at the top
  pointing at this file and `docs/positioning.md`.
- `README.md` rewrite deferred until Stage 3 — no point burning
  marketing copy on an API that doesn't exist yet.

### Parked (may come back)
- The bootstrap trainer and all its speedup work is parked as
  `src/soma/training/verbalizer_bootstrap.py`. If Stage 2's retrieval
  + verbalizer path turns out to beat raw-context injection, we'll
  re-train the projector on retrieved-context windows rather than
  full corpus windows.
- The `current.pt` at 2.2M steps is archived to
  `checkpoints/archive/current-2026-04-15-pre-pivot/` along with its
  sidecars. It's structurally degraded (0 integrators pre-860cc82,
  97% weak edges) and not worth resurrecting against the new API.
- Joint SOMA + verbalizer training stays in the trainer but is not on
  the critical path. The experiment was informative; the mechanic is
  there if we want it.

## What This Doesn't Mean

- **SOMA is not dead.** The graph, WM, episodic store, growth, and
  consolidation are the *product*. This is a framing change, not a
  rewrite.
- **The verbalizer is not dead.** It becomes an optional ergonomic
  layer on top of MemoryLayer, for cases where the operator wants
  retrieved memory injected as a soft prompt instead of templated
  context. Same code, narrower input.
- **Not abandoning long-term vision.** The "user-owned brain that
  survives LLM upgrades" still holds — just via the memory layer, not
  via a learned hybrid cognition pipeline.

## Execution Roadmap

| Stage | Task | Scope | ETA |
|-------|------|-------|-----|
| 1 | #139 | Archive + pivot docs (this commit) | same-day |
| 2 | #140 | MemoryLayer API + persistent-chat demo | 1-2 days |
| 3 | #141 | Packaging (`pip install soma-memory`), LangChain/LlamaIndex connectors, benchmark vs Chroma+RAG + Mem0 | 3-5 days |
| 4 | #142 | Research side-bets (graph plasticity vs static RAG, WM vs recency windows), verbalizer retrain on retrieved context, Docker/deploy story | 1-2 weeks |

Each stage produces a shippable, testable artifact before the next
begins. Stage 2 is the go/no-go: if MemoryLayer over today's SOMA
doesn't outperform a basic Chroma+RAG baseline on at least one
dimension (recall quality, latency, or disk footprint), the pivot
itself needs reconsideration.

## Provenance

- Accelerated wt103 run: `reports/bootstrap-2026-04-15-accel-wt103.md`
- Joint-training baseline: commit `5be7a34` (joint tied with frozen)
- Integrator-count fix: commit `860cc82`
- Device-mismatch fix in build_encoders: commit `a26d54e` (bundled
  with joint training landing)
- Three-regime ablation data: `artifacts/bootstrap-2026-04-1*/`
- Pre-pivot current.pt:
  `checkpoints/archive/current-2026-04-15-pre-pivot/`

## Decision author

Autonomous pivot with operator-confirmed direction ("I agree with your
proposed positioning for what's best considering SOMA's current
state"). Recorded here as the canonical decision; future work starts
from this doc, not from HYBRID_PIVOT.md's closing question list.
