# Path B deferred follow-ups — retrieval experiments we chose not to run (yet)

**Date:** 2026-04-20.
**Status:** Deferred by operator decision (opted to pursue Path A
artificial-life research + product polish instead). Kept here as
self-contained "pick up cold" plans so we can revisit without
rebuilding context.
**Related:**
- `research/developmental/results/path_b_closure_summary.md` (why Path B closed)
- `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md` (original design)
- `docs/milestones/2026-04-20-hybrid-retrieval-validated.md` (M1 baseline)

## Why these are deferred, not abandoned

Path B closed Phase 1 (sim CA3 baseline) and Phase 3 (LongMemEval
sparse retrieval) as NO-GO. The sparse-codes primitive in
`src/soma/memory/sparse_codes.py` works on synthetic data
(separation ratio 5.66) but adds no signal over hybrid retrieval on
LongMemEval (rank-1 drops 0.790 → 0.660 at N=100).

The two follow-ups below are orthogonal enough to the closed
direction that they might surface independent signal. Neither is
expected to be a product game-changer — they're each small
experiments that either give a clean negative ("yep, hybrid still
wins") or surface a niche win worth keeping.

---

## Deferred follow-up 2 — Extended industry-benchmark coverage

### What we'd test

Run SOMA (M1 hybrid retrieval, α=0.30) against two more industry
benchmarks to confirm or challenge the "LongMemEval ceiling is
saturated for this class of technique" narrative:

| Benchmark | Size | What it tests | Status in-tree |
|---|---|---|---|
| **AMA-Bench** | ~10K QA, 10 personas | Long-horizon persona consistency | Not integrated |
| **Letta memory benchmark** | ~500 items | Multi-turn dialogue recall | Not integrated |
| **LongMemEval medium** | ~2000 items | Superset of `small`; broader question-type mix | Loader exists, no runs yet |
| **LongMemEval oracle mode** | N/A | Upper-bound retrieval (gold ids fed in directly) | Not implemented |

### Why this matters

- **If SOMA lift holds on AMA-Bench + Letta**: M1 narrative strengthens
  — we can claim "beats hybrid+RAG on three independent benchmarks,
  under two LLMs, under two embedders, under two judges." That's a
  much harder-to-dismiss claim for product positioning.
- **If SOMA lift collapses on a new benchmark**: we've found a
  concrete retrieval gap to investigate. That becomes a focused
  research direction, not a speculative one.
- **LongMemEval oracle mode** tells us the gap between
  perfect-retrieval F1 and hybrid's F1. If oracle is only ~5 points
  above hybrid, we know the retrieval ceiling is near. If it's 30
  points above, there's real retrieval headroom left that just
  isn't accessible via cosine+BM25.

### Entry points / files

- `benchmarks/industry/longmemeval/` — existing LongMemEval harness,
  use it as the reference implementation for adding AMA-Bench /
  Letta. Key pattern: `run_qa_compare.py` (end-to-end with LLM),
  `rank_probe.py` (retrieval-only), `data_loader.py` (dataset parser).
- `benchmarks/industry/locomo/` — LoCoMo harness, similar structure.
- `benchmarks/industry/` — add new subdirectory per benchmark.
- `benchmarks/reports/paper-draft.md` — the M1 paper draft; a new
  benchmark validation would update this.

### Concrete plan (2-3 days)

1. **AMA-Bench integration** (day 1):
   - Download dataset (HuggingFace: `AMA-Bench`)
   - Write `benchmarks/industry/ama_bench/data_loader.py` mirroring
     `longmemeval/data_loader.py`
   - Adapt `longmemeval/run_qa_compare.py` → `ama_bench/run_qa_compare.py`
   - Run with `soma_hybrid` + `chroma_cosine` at N=500, α=0.30
   - Gate: SOMA hit@5 ≥ chroma +0.02 on at least one persona
2. **Letta integration** (day 2):
   - Same template
   - Letta's benchmark is multi-turn dialogues, so session_text
     formatting differs; need to adapt `_session_text` helper
3. **LongMemEval oracle mode** (day 3):
   - Add `--oracle-retrieve` flag to `run_qa_compare.py` that
     injects gold session IDs directly into the LLM prompt, skipping
     retrieval
   - Gives us the retrieval ceiling in F1 terms
   - If oracle ≈ hybrid, retrieval is saturated (informs future
     research scope)

### What to read when resuming cold

- `research/developmental/results/longmemeval_full_evidence_roundup.md`
  — authoritative M1 results, cross-LLM, cross-embedder, cross-judge
- `benchmarks/industry/longmemeval/run_qa_compare.py` — reference
  impl for benchmark integration
- `docs/milestones/2026-04-20-hybrid-retrieval-validated.md` — M1
  claim, alpha=0.30 confirmed optimal

### Time / cost estimate

- 2-3 days wall clock for implementation
- ~1 hour compute per 500-item run (uses Unraid claude-runner, no
  API charge) × 4 variants × 3 benchmarks = ~12 hours compute
- Requires: Ollama + qwen3.5:4b (have) or Claude runner (have)

### What would change our positioning

- **GO (all benchmarks show SOMA lift)**: `docs/positioning.md`
  headline gets tougher: "beats hybrid+RAG on N benchmarks"
- **Mixed**: find the question types or personas where SOMA
  regresses; investigate those as targeted research
- **NO-GO (LongMemEval was an outlier)**: walk back the positioning
  to "hybrid retrieval is a reasonable default" rather than
  "measured advantage" — honesty demands it

---

## Deferred follow-up 3 — Primitive 4 (engram tagging) on a custom temporal benchmark

### What we'd test

The original Path B design called for Primitive 4 (engram tagging —
recency + frequency weighting per stored memory) as the last of
three primitive tests. Phase 3 NO-GO on sparse codes gated Primitives
3 and 4 out by plan, but engram is cleanly orthogonal to the
sparse-code direction that failed. It deserves a standalone test on
a benchmark where recency actually matters.

### Why LongMemEval can't test it directly

LongMemEval's haystack sessions don't have a well-defined "ingestion
order" that maps to user experience. The benchmark treats all
sessions as a static bag. Engram weighting (γ_freq · log(1+count) +
γ_rec · exp(-(now - last)/τ)) needs a temporal sequence where some
memories have been "recently accessed" and others haven't.

### Custom benchmark sketch

Transform LongMemEval into a temporal variant:

1. Select 100 multi-session LongMemEval items (those with ≥ 3
   answer_session_ids)
2. For each item, treat the answer sessions as a TEMPORAL SEQUENCE:
   session 1 stored at t=0, session 2 at t=1, etc.
3. Insert the query at t=N (after all sessions stored)
4. Score retrieval with engram weighting at retrieve time

Variants to compare:

| Variant | Scoring |
|---|---|
| `hybrid_only` | existing 0.7·cos + 0.3·bm25 |
| `hybrid_plus_engram` | hybrid + 0.1·log(1+access_count) + 0.1·exp(-Δ/τ) |
| `engram_only` | engram component alone (sanity) |

Expected win: `hybrid_plus_engram` ≥ `hybrid_only` on multi-session
queries where the gold session was stored "recently" or "frequently
referenced."

### Concrete plan (2-3 days)

1. **Custom benchmark builder** (day 1):
   - `benchmarks/industry/longmemeval/build_temporal_variant.py` —
     reformats a LongMemEval slice into a temporal sequence
   - Outputs a jsonl with per-item `{ingest_order, query_time, ...}`
2. **Engram index implementation** (day 1-2):
   - `src/soma/memory/engram.py` — per-memory `access_count`,
     `last_access_step`, `encoding_step`; scoring function with
     γ_freq, γ_rec, τ parameters
   - TDD tests in `tests/test_memory/test_engram.py`
3. **Probe** (day 2-3):
   - `benchmarks/industry/longmemeval/run_engram_retrieval.py` —
     three-variant probe mirroring `run_sparse_retrieval.py`
   - Parameter sweep over γ_freq ∈ {0.0, 0.1, 0.3}, γ_rec ∈ {0.0, 0.1,
     0.3}, τ ∈ {3, 10, 30} — 3×3×3 = 27 configs
   - At N=100 → 27 × 100 × 3 = 8100 retrievals total, ~30 min
     wall clock on GPU

### Why this might actually work (unlike Path B)

- Engram scoring is **additive to existing hybrid scores**, not a
  replacement. Doesn't depend on embedding structure that could
  mismatch.
- It's a well-studied mechanism in production: Mem0, Zep, and
  Letta all implement some recency/frequency weighting.
- The specific signal it captures (recently-accessed memories should
  score higher) is orthogonal to both semantic similarity (cosine)
  and lexical overlap (BM25). Additional information → probably
  helps, at worst neutral.

### Why it might still fail

- LongMemEval's question types may not have a meaningful temporal
  signal — queries might be about long-past interactions as often as
  recent ones.
- The benchmark as transformed may still be too synthetic to expose
  the recency signal naturally present in real user memory.
- Tuning γ_freq / γ_rec / τ for generalization is hard — what works
  at N=100 may not transfer to N=500 or to real usage patterns.

### Entry points / files

- `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md`
  § Phase 5 — original engram design with formula
- `benchmarks/industry/longmemeval/run_sparse_retrieval.py` — template
  for the three-variant probe architecture
- `src/soma/memory/sparse_codes.py` — template for the primitive
  module structure (dataclass + function set + TDD)
- `tests/test_memory/test_sparse_codes.py` — template for the test
  architecture

### What to read when resuming cold

- `research/developmental/results/path_b_closure_summary.md` — closure
  context + why engram was gated out (and what's different about it)
- `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md` §
  Phase 5 — scoring formula, parameter ranges
- Competitor implementations for calibration:
  - Mem0 uses frequency boost
  - Zep uses recency decay
  - Letta uses ingest-order weighting

### Time / cost estimate

- 2-3 days wall clock
- ~30 min compute for the full 27-config sweep
- No API charges (all local)

### What would change our positioning

- **GO**: `src/soma/memory/engram.py` ships as opt-in. Positioning
  adds "per-memory recency/frequency weighting" to the feature list.
  If the lift is on multi-session specifically, positioning calls out
  that category's advantage.
- **NO-GO**: engram code ships (useful for niche applications) but
  not in default config, not in positioning.

---

## How to pick up either of these later

Minimal context needed to restart:

1. Read `research/developmental/results/path_b_closure_summary.md`
   — gets you up to speed on the "Path B closed" story
2. Read this doc (you're here)
3. Pick one of the two follow-ups and follow its "Entry points /
   files" section — everything should be resumable without this
   conversation's context

Default resumption order: **follow-up 2 (benchmarks)** first, because
its outcome affects whether follow-up 3 is worth doing. If extended
benchmarks show SOMA's lift is robust, engram becomes a niche
enhancement. If they show retrieval gaps, engram's temporal signal
might be one of the gaps.

---

## What's IN the current scope (option 1, not deferred)

Reminder of what we ARE working on now so future-us doesn't confuse
the two:

1. **Path A artificial-life research (background)** — continue the
   biophysical simulator exploration explicitly as non-product
   research. See `docs/plans/2026-04-20-path-a-biophysical-representation-layer-design.md` § Phase 6+.
2. **Product polish (foreground)** — CLI, deploy packaging,
   onboarding docs, dashboard. M1 gives us a shippable story; make
   it usable.

Neither of those blocks a return to follow-ups 2 or 3 — both can
resume whenever we decide the time is right.
