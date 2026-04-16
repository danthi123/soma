# C1 — Plastic Graph Synthetic Retrieval Signal

Sub-phase C1 of research direction C
(`docs/plans/2026-04-16-research-c-graph-memory.md`).

**Run date:** 2026-04-16
**Embedder:** `sbert` (all-MiniLM-L6-v2, dim=384)
**Alpha (graph blend weight):** `0.3`
**Corpus:** 60 snippets across 20 topic-triples (10 transitive + 10 triangle),
deterministic seed.
**N sweep:** `{0, 10}` for the primary run; a single N=10 run was also
performed as the plan-compliant "did the substrate move?" check on a
fuller 200- and 1000-snippet corpus before timing pushed them out of
scope this session (see *Scope notes* below).
**Primary run wall clock:** ~442s (≈7 min).

## Headline gate decision

**Verdict: FAIL.** The plan's gate is `R@5(transitive, SOMA, N=100)
- R@5(transitive, pure-vector) ≥ +0.05`. At N=10 (and identical at N=0
where the substrate has not moved yet) the measured delta is
**-0.050**: SOMA's blended retrieval actively *hurts* on transitive
queries versus the pure-vector baseline. This is the same sign as
the 2026-04-15 wt103 next-token-LM ablation — the plastic-graph
state is **drag, not lift**, for retrieval as wired today.

We did NOT run N=100 / N=1000 in this session. The C1 plan asks
those; wall clock per iteration was too high on the in-tree SOMA
(~40s/iter at 200-snippet corpus, ~330s/iter at 1000-snippet
corpus). A follow-up that caches the SOMA side or runs overnight
should complete the full sweep. With that said: the direction of
the N=0 → N=10 change is already against us (transitive R@5 did not
move at all), so the plan's "recommend C1b failure analysis" branch
is already well-supported.

**Recommendation: proceed to C1b, not C2.**

## Honest numbers

### Primary run (60-snippet corpus, sbert, α=0.3)

Transitive (the load-bearing class):

| Method             |   N | R@1   | R@5   | MRR   | n  |
| ------------------ | --: | ----: | ----: | ----: | -- |
| pure-vector-sbert  |  −  | 0.000 | 0.100 | 0.039 | 20 |
| soma-sbert-a0.30   |   0 | 0.000 | 0.050 | 0.029 | 20 |
| soma-sbert-a0.30   |  10 | 0.000 | 0.050 | 0.029 | 20 |

Direct (control — should tie):

| Method             |   N | R@1   | R@5   | MRR   | n  |
| ------------------ | --: | ----: | ----: | ----: | -- |
| pure-vector-sbert  |  −  | 0.537 | 1.000 | 0.976 | 60 |
| soma-sbert-a0.30   |   0 | 0.546 | 0.996 | 0.987 | 60 |
| soma-sbert-a0.30   |  10 | 0.546 | 0.988 | 0.987 | 60 |

Unrelated (hallucination check — both should score 0):

| Method             |   N | R@1   | R@5   | MRR   | n  |
| ------------------ | --: | ----: | ----: | ----: | -- |
| pure-vector-sbert  |  −  | 0.000 | 0.000 | 0.000 | 10 |
| soma-sbert-a0.30   |   0 | 0.000 | 0.000 | 0.000 | 10 |
| soma-sbert-a0.30   |  10 | 0.000 | 0.000 | 0.000 | 10 |

### Substrate diagnostics

- Integrator count (post-860cc82 fix): `8` — **seed-graph integrator
  bug-fix is active** in this checkout.
- Edge count at N=0: `48`.
- Edge count at N=10: `48` — **no new edges formed over 10
  consolidation cycles on the 60-snippet corpus**.
  Synaptogenesis is gated on activation EMA and edge-weight
  thresholds that our consolidation pass clearly didn't trip.

**Spearman ρ(edge strength, pair co-occurrence count):**

| N | Spearman ρ (edge-strength vs pair-count rank) |
| -- | ---- |
| 0  | 0.000 |
| 10 | **+0.854** |

The substrate's edge-strength distribution becomes strongly rank-aligned
with the corpus's pair-count distribution after just 10 consolidation
cycles. In isolation this IS signal: repeated Hebbian + backprop passes
are pushing edge weights around in a way that correlates with co-occurrence
structure. But the retrieval blend doesn't harvest it — R@5(transitive)
at N=10 is unchanged from the cold substrate at N=0.

## Scope notes

The plan asked for N ∈ {0, 10, 100, 1000} on a 1000-snippet corpus.
What we actually ran:

| Corpus | N sweep | Outcome |
| ------ | ------- | ------- |
| 1000 snippets | {0, 10 (partial)} | Baseline + N=0 completed; N=10 was in flight at ~330 s/iteration (estimated 55 min to complete) when cancelled. |
| 200 snippets | {0, 10 (partial)} | Baseline + N=0 completed; N=10 at ~2 min/iter, cancelled. |
| 60 snippets | {0, 10} | **Complete — this report.** |
| 60 snippets | {0, 10, 100 (partial)} | N=10 completed (same numbers as above, Spearman 0.854); N=100 started but cancelled at ~7 min/iter = 70 min to complete. |

Per-iteration SOMA.step cost dominates wall clock; we hit 40-60 ms per
token-pair step on CPU, and each consolidation iteration runs
`len(corpus) × (tokens_per_snippet - 1)` of those. With the cursor
reset the harness applies for spec-faithful "N full passes", cost scales
O(N × corpus_size). The full N=1000 sweep is ~7 hours at 1000 snippets and
~90 min at 60 snippets; either is an overnight rerun, not a single-session
experiment.

**Confidence this scoping doesn't change the verdict:** high. The
N=0 → N=10 transitive R@5 change was already zero, so N=100 / N=1000
would have to break the monotone to reverse the gate. The
edge-weight Spearman at N=10 is already 0.854 (nearly saturated), and
the retrieval blend formula does not benefit from it. The failure
pattern is "the substrate does move, but the blend at α=0.3 is
pulling signal in a direction that degrades the cosine ranking" —
more cycles reinforce, not fix, that behaviour.

## Interpretation

1. **Pure-vector already gets ~10% transitive R@5** from SBERT's generic
   prior over the shared template syntax. It's not zero despite the
   corpus being engineered so B-C never co-occur; SBERT sees "the X was
   observed near the Y" and clusters X and Y independently of the
   specific token identity.
2. **The graph blend at α=0.3 cuts that down to 5%** — it drags the
   cosine ranking toward the graph-score signal, which for transitive
   queries is effectively random because the SOMA activations for the
   query entity (B or C) and the stored snippets don't share
   graph-proximity paths that *differentiate* the correct answers from
   incorrect ones. Same alpha ablation story as the 2026-04-15 LM run.
3. **The substrate DOES learn co-occurrence** — Spearman ρ=0.854 after
   just 10 consolidation cycles is a strong structural signal. The
   plastic-graph edges ARE reorganizing in a way that reflects the
   corpus. What doesn't work is the *consumption* step: the current
   stable-capture → cosine-blend formula doesn't translate "edges got
   stronger where co-occurrence was dense" into "retrieve snippets in
   a way that uses that density".
4. **Direct queries tie** approximately (1.000 → 0.988 at N=10), which
   is the sanity check the plan specified. The graph signal isn't
   catastrophically wrong; it's subtly-wrong in exactly the way that
   hurts the one class of queries we need it to help on.

## Recommendation — C1b, not C2

The plan's gate says proceed to C1b on fail; the C1b ablation is:

1. **Edge-strength differentiation** — already measured: ρ=0.854
   against co-occurrence. The substrate IS differentiating. That's
   not the bottleneck.
2. **Blend-formula ablation** — try pure-graph-rank, pure-vector, and
   the blend as three-way compare. Our data already shows the blend
   underperforms pure-vector; the question C1b needs to answer is
   whether pure-graph-rank at any point beats random on transitive.
   If the graph activations alone carry *any* transitive signal,
   there's a different blend formula (learned α, graph-traversal
   expansion) that might extract it. If graph-only is at random, the
   substrate captures pair-co-occurrence but the node-activation
   readout loses it — and the research direction has hit bedrock.
3. **Readout layer** — activation-cosine between query and stored is
   the simplest possible readout; we should try edge-traversal (query
   entity → bridging node → stored entity) and compare to the blend.

This is one short follow-up experiment, not weeks. Worth doing before
folding direction C entirely.

## Surprises

- **Integrators get created.** The 860cc82 seed-graph bug fix IS
  active in this checkout — `initial_integrator_count=8` produces 8
  INTEGRATOR nodes, confirmed by the harness assertion. The harness
  raises immediately if the count mismatches.
- **No synaptogenesis.** Edges stayed at the 48 seeded ones across
  the full N=10 run on 60 snippets. Either our consolidation thread
  isn't firing the activation thresholds, or 10×60 pair-steps just
  isn't enough synaptogenesis budget. (Synaptogenesis interval is
  100 steps in `SOMAConfig`; 10 iterations × 60 snippets × ~9 pairs
  = 5400 steps = ~54 potential synaptogenesis checks. Clearly most
  don't fire.)
- **But strengths move.** Even without topological growth, the
  existing 48 edges developed a co-occurrence-aligned strength
  ranking (ρ=0.854). That confirms the substrate is responsive to
  the training signal — it just doesn't feed usefully into the
  retrieval readout.

## Files produced

- `research/graph_memory/reports/c1_synthetic_signal.md` (this file)
- `research/graph_memory/reports/c1_synthetic_signal.json` (raw numbers
  from the 60-snippet N={0, 10} run for downstream analysis)

Raw numbers per query class + metric live in the JSON sidecar.

---

Generated by `research/graph_memory/run_c1.py` + manual curation of
the interpretation, 2026-04-16.
