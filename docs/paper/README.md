# Paper: Structural Without Semantic

**Working title:** Structural Without Semantic: An Empirical Ceiling for Brain-Inspired Graph Memory in Retrieval-Augmented Generation

**Status:** Draft in progress (started 2026-04-18)

**Target venue:** TMLR (first choice — accepts negative empirical results,
no strict page limit) or arXiv preprint pending venue choice.

**Estimated length:** 8-12 pages (ML workshop-to-short-paper range).

## Files

- `outline.md` — Section structure with per-section content plan.
- `draft.md` — Active paper draft (Markdown, convert to LaTeX at submission).
- `figures/` — Plots and diagrams (to be created).
- `data/` — Condensed result tables sourced from `research/developmental/results/`.

## Core thesis

SOMA — a brain-inspired memory system with neurogenesis, Hebbian
learning, consolidation, and structural plasticity — was evaluated as a
retrieval signal on top of a pretrained embedding baseline. Through 14
diagnostic experiments on LoCoMo (5882 turns, 500+ held-out queries)
and LongMemEval (3094 turns, 100 queries), we find:

1. **The graph adds a real but architecturally-bounded retrieval signal
   (~0.8% absolute over random fingerprint assignment).** Shuffle
   diagnostics confirm the signal is non-random. Per-query attribution
   shows no clean semantic subset where the mechanism reliably wins.

2. **The ceiling is cross-benchmark.** +2/500 on LoCoMo held-out
   (coin-flip 32W/34L); -1/100 on LongMemEval (3W/6L). Both signals
   (fingerprint cosine, topology Jaccard) plateau identically because
   they derive from the same 3 lateral-inhibition-winner nodes.

3. **Root cause: structural without semantic.** Random input
   projections create fingerprints that are structurally diverse
   (~8000 possible patterns) but semantically arbitrary. Fine-tuning
   the encoder against graph topology causes catastrophic forgetting
   (-13 hits) because the target topology is not correlated with
   semantic structure.

4. **Components that do work on SOMA's native tasks.** Consolidation
   (+45% QA on synthetic). Multi-session development (+275% graph
   growth across 3 sessions, cross-session recall preserved). These
   results support pivoting SOMA toward developmental-AI tasks where
   its mechanisms are load-bearing, rather than retrieval.

## Why this matters

Brain-inspired architectures are being proposed as retrieval
enhancements across the literature (MAGMA, SYNAPSE, Graphiti,
Mem0). This paper provides the first rigorous diagnostic showing
that "which nodes fire together" cannot by construction substitute
for semantic similarity at scale — and that the retrieval-augmentation
framing is itself a mismatch for what structural plasticity provides.
