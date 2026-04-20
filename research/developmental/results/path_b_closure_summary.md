# Path B closure — biology-inspired retrieval primitives do not lift LongMemEval

**Date:** 2026-04-20.
**Status:** Path B closed as a coherent negative result after three
phase gates, two of which are decisive NO-GO.
**Design:** `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md`.
**M1 baseline (unaffected):** `docs/milestones/2026-04-20-hybrid-retrieval-validated.md`.

## One-line summary

We tested whether DG-like sparse codes + pattern separation primitives
(drawn from hippocampal biology) improve retrieval over the shipping
hybrid (cosine + BM25). **They do not.** The M1 hybrid-retrieval
configuration remains the best retrieval setup in SOMA on LongMemEval
and LoCoMo.

## Three-phase evidence trail

### Phase 1 — sim CA3 baseline: NO-GO

**Question:** does the neural-simulator's `HIPPOCAMPUS_CA3_RECURRENT`
preset produce sparse separated codes for controlled concept stimuli?

**Result:** No. Separation ratio **1.00** across all sparsity views
(1%, 2%, 4%) at n=500 trials. Within-concept Jaccard 0.22 at primary
view — barely above chance.

**Root cause:** the sim's CA3 preset has no DG analog. Raw stimulus
patterns go straight into CA3's recurrent layer with their overlap
intact. Sim can't serve as calibration target.

**Findings:** `sim_ca3_baseline_findings.md`.
**Cost:** ~1 day including 465s of sim time + harness development.

### Phase 2 — numpy k-WTA + pattern separation: GREEN in isolation

**Question:** do the numpy primitives (SparseCode, kwta,
pattern_separate, code_similarity) produce well-separated sparse
codes on synthetic embedding data?

**Result:** Yes. Synthetic calibration (50 concepts × 10 trials,
embed_dim=1024, overlap=0.1) at k=32 / dim=4096 produces separation
ratio **5.66** with sparsity 0.78% — well above the 1.5× gate.

**Findings:** `sparse_codes_synthetic_sweep.json`.
**Test coverage:** 14 TDD tests in `tests/test_memory/test_sparse_codes.py`.
**Cost:** ~half a day (tests + implementation + calibration).

### Phase 3 — LongMemEval sparse retrieval: NO-GO

**Question:** does adding a sparse-overlap score to SOMA's hybrid
retrieval lift rank-1 on LongMemEval?

**Result:** No — it hurts substantially. At N=100:

| variant | hit@5 | rank-1 |
|---|---:|---:|
| hybrid_only (shipping) | 0.960 | **0.790** |
| hybrid_plus_sparse | 0.910 | 0.660 |
| sparse_only | 0.560 | 0.340 |

Paired: sparse loses 27×, wins 5×, ties 68×. Net −22 on rank-1.

**Root cause:** random-projection k-WTA is not locality-sensitive.
Two semantically close sbert embeddings (close in cosine) produce
projected vectors that differ slightly, but the top-k boundary is
sharp — small differences flip which 32 of 4096 dims are "top." So
similar inputs get disjoint sparse codes, the overlap score is
mostly noise, and BM25 already covers the "shared distinctive tokens"
structure the sparse code tries to capture.

**Findings:** `longmemeval_sparse_retrieval_findings.md`.
**Cost:** ~1 day including N=100 run + analysis.

## Why this doesn't indict the primitives themselves

The numpy primitives work on synthetic data (Phase 2, ratio 5.66).
They just don't help on LongMemEval because:

1. **Dense cosine + BM25 already capture what sparse codes aim for.**
   BM25 catches shared distinctive tokens. Cosine captures overall
   semantic similarity. There isn't an orthogonal signal left for
   random-projection sparse codes to contribute.

2. **Random projections destroy the property you want from a sparse
   code**: similar inputs → similar active-index sets. Top-k is
   discontinuous at the boundary; small perturbations shuffle indices.
   Proper locality-sensitive hashing (sign-random-projection, learned
   projections) might behave differently; random-normal top-k does not.

3. **The benchmark is saturated for this class of technique.**
   Phase 3 joins earlier null results from Direction 4a (LLM-distilled
   projections), Direction 4b (spatial distillation), and plastic
   graph activation — five consecutive attempts to route any signal
   into retrieval beyond hybrid, all null. The honest interpretation
   is: hybrid retrieval has saturated the LongMemEval rank-1 ceiling
   given sbert embeddings.

## What we did NOT test (gated out by Phase 3 NO-GO)

Per the design doc's risk-mitigation plan, three primitives were
sequenced such that Phase 3 failure gates out the subsequent two:

- **Primitive 3: modern Hopfield attractor retrieval** — uses the same
  sparse codes as query/store encoding, so inherits Phase 3's
  boundary-sensitivity issue. Unlikely to surface new signal.
- **Primitive 4: engram recency/frequency tagging** — orthogonal to
  sparse codes (no encoder component), so could in principle add
  signal. But on LongMemEval, items are not queried in ingestion
  order, so the recency signal is absent from the benchmark. Engram
  would need a different evaluation setup (simulated temporal usage)
  to expose its value.

Both remain open as specific-use-case optimizations if a future
product scenario exposes their strengths. They are not load-bearing
for the "do bio primitives help text retrieval" question — that
question is closed.

## Consequence for Path A

Path A (use the full neural-simulator as SOMA's representation/retrieval
substrate) was gated on Path B Phase 3 answering "do biological
primitives have retrieval headroom at all." Phase 3 says: **not on
LongMemEval, not with the primitive we tested.**

Path A's retrieval-focused phases 1–4 are therefore deprioritized as a
**product** path. They remain viable as a **research** direction if
the goal is artificial-life or biophysically-grounded cognition — that
is Path A's Phase 6 open-ended scope and is valuable research even
without a direct retrieval win.

The updated narrative:

- Path A as "SOMA's biophysical storage layer" → deferred /
  unlikely to ship.
- Path A as "explore artificial life using a validated neural
  simulator and SOMA's memory patterns" → still open as Daniel's
  research interest, orthogonal to product roadmap.

## Consequence for product positioning

No changes to M1 messaging. Positioning still leads with hybrid
retrieval (+22.8% F1 cross-validated three ways). Plastic graph and
biological primitives are both honestly framed as research-only, with
no differentiator claim in product copy.

One addition to `docs/positioning.md` under "What we've ruled out":
another row for "sparse bio-codes" alongside the existing "plastic
graph activation" and "Direction 4a/4b" rows.

## Files committed in this track

**Code (shipped, stays in-tree):**
- `src/soma/memory/sparse_codes.py` — SparseCode, kwta,
  pattern_separate, code_similarity (14 TDD tests GREEN)
- `research/developmental/experiments/sim_ca3_measurement.py` — sim
  driver harness
- `benchmarks/sparse_codes_calibration.py` — numpy calibration sweep
- `benchmarks/industry/longmemeval/run_sparse_retrieval.py` —
  three-variant retrieval probe
- `scripts/analysis/compare_sparse_hybrid.py` — paired analysis

**Findings:**
- `research/developmental/results/sim_ca3_baseline_findings.md`
  (Phase 1 NO-GO)
- `research/developmental/results/longmemeval_sparse_retrieval_findings.md`
  (Phase 3 NO-GO)
- `research/developmental/results/path_b_closure_summary.md` (this doc)

**Results artifacts:**
- `research/developmental/results/sim_ca3_baseline.json`
- `research/developmental/results/sparse_codes_synthetic_sweep.json`
- `benchmarks/industry/longmemeval/results/sparse_retrieval_*_n100_sparse.jsonl`

## Recommendations for next direction

Presented to operator for decision. Four candidates, ranked by
assessed ROI:

1. **Path A artificial-life research (Phase 6+ of Path A design)** —
   continue the biophysical-simulator direction explicitly as a
   research track, not a product path. Keeps the interesting research
   alive without blocking product on nulls. Low short-term risk.

2. **Other industry benchmarks** — AMA-Bench, Letta, or stronger
   LongMemEval variants where hybrid might not be saturated. If these
   show SOMA's lift holding, M1 narrative strengthens. If they expose
   new retrieval regressions, we have concrete new problems to solve.

3. **Product-track work** — dashboard / CLI polish, deploy packaging,
   onboarding docs. M1 gives us a shippable story; time to make it
   usable. Lowest scientific risk, highest near-term product value.

4. **Primitive 4 (engram tagging) on a custom temporal benchmark** —
   build a LongMemEval-derivative with simulated temporal usage
   history, test whether recency/frequency weighting lifts multi-
   session queries. Small chance of GO, moderate effort, cleanly
   orthogonal to the failed sparse-code direction.

My recommendation without further input: **option 1 + option 3 in
parallel**. Path A artificial-life as background research; product
polish as foreground work. That matches the user's stated interest
("extremely interested in path A") with the practical shipping
posture of M1.
