# Path B: Bio-Validated Algorithmic Primitives for SOMA Retrieval

**Date:** 2026-04-20.
**Status:** Design, ready for implementation.
**Scope:** ~2-3 weeks. Uses the `neural-simulator` project as a
measurement tool to characterize what real biology does, then ports
clean numpy approximations of validated primitives into SOMA's
retrieval path. Targets concrete retrieval wins on LongMemEval /
LoCoMo with runtime costs compatible with shipping.

## Motivation

Three graph-retrieval directions (plastic graph activation,
Direction 4a projection distillation, Direction 4b spatial
distillation) produced null results. The pattern across failures
isn't random — it's **signal attenuation through too many layers**
and **dense-random primitives where biology uses sparse-learned
ones**.

Real brains store semantic memories using primitives SOMA doesn't
have:

1. **Sparse distributed representations** — each concept activates
   ~2% of neurons; overlap encodes similarity directly.
2. **Pattern separation** (DG → CA3) — similar inputs are
   orthogonalized before storage to prevent interference.
3. **Pattern completion** (CA3 recurrent) — partial cues settle
   onto full stored memories via attractor dynamics.
4. **Engram tagging** — recently-used memories are easier to recall;
   unused memories decay unless reinforced.

These are algorithmic-level primitives that numpy can implement
cleanly. The question is: **do they help text retrieval on top of
what hybrid (BM25+cosine) already captures?**

Path B answers that question empirically, using `sim` to calibrate
the primitives against biological ground truth and SOMA's
benchmark suite to measure product impact.

## Hypothesis

At least one of the four primitives, implemented as a numpy
transform layered on top of SOMA's existing dense embeddings,
produces measurable retrieval improvement on a question-type where
hybrid has a known ceiling. Specific candidates:

1. **Sparse codes at ingest** help single-session-preference
   (where gold is a paraphrased sentence and dense cosine smears
   across all paraphrases).
2. **Attractor retrieval** helps queries with significant
   lexical/semantic drift from stored text.
3. **Engram recency/frequency weighting** helps multi-session
   queries where the relevant fact was stated recently or has been
   referenced before.

If zero of the three produce a lift, we've validated that dense
embeddings + hybrid retrieval have saturated the retrieval ceiling
on these benchmarks — a valuable negative result that closes a
long-running question.

## Relationship to Path A

Path A (see `2026-04-20-path-a-biophysical-representation-layer-design.md`)
proposes using the full biophysical simulator as SOMA's
representation layer. Path B is a strict precursor: if Path B's
measurements show sparse codes don't help on LongMemEval, Path A's
retrieval-focused phases are unlikely to succeed either. Path B is
designed to answer "is the primitive useful" in 2-3 weeks at low
cost; Path A then makes an informed bet about whether the months-
long biophysical substrate build is worth it.

## Architecture

```
text
 │
 │  (1) dense embed (existing)
 ▼
embedding [1024d dense]
 │
 │  (2) sparse code transform (NEW, Path B)
 ▼
sparse code [N-dim, k-hot]
 │
 │  (3) pattern-separation transform (NEW, Path B)
 ▼
separated code [N-dim, k-hot, decorrelated from similar inputs]
 │
 │  (4) optional: Hopfield/attractor completion (NEW, Path B)
 ▼
completed code [N-dim, k-hot]
 │
 │  STORE: persist alongside existing dense embedding + BM25 index
 │  RETRIEVE: existing hybrid PLUS sparse-overlap as a third score
 ▼
top-k results
```

Four primitives, each numpy-only, each tested independently against
hybrid baseline. Critical design principle: **each primitive
stacks onto the existing hybrid retrieval without replacing it**.
A/B tests measure the incremental lift of adding that primitive.

### Primitive 1: k-Winner-Take-All sparse codes

Transform dense embedding to sparse k-hot code:

```python
def kwta(dense: np.ndarray, k: int, dim: int) -> SparseCode:
    # Project dense -> dim with a learned or random projection
    projected = projection_matrix @ dense  # dim-vector
    # Top-k active
    active_idx = np.argpartition(-projected, k)[:k]
    return SparseCode(active=active_idx, k=k, dim=dim)
```

Parameters tested:
- `dim`: 1024, 4096, 16384 (capacity vs cost)
- `k`: sparsity target 1%, 2%, 5% of dim
- Projection type: random-normal, random-orthogonal, learned
  (backprop against in-domain teacher)

### Primitive 2: Pattern separation at ingest

Inputs arriving at the same ingest pass go through a separation
step that amplifies their differences:

```python
def pattern_separate(sparse_codes: list[SparseCode],
                     strength: float) -> list[SparseCode]:
    # Measure pairwise overlap
    # For each pair with overlap > threshold, subtract shared
    # active dimensions from whichever has lower activation
    # magnitude on those dims
```

This is a biology-inspired analog of DG's orthogonalization. Key
parameter: `strength` (how aggressively to separate), tuned against
the measured `sim`-CA3 separation ratio.

### Primitive 3: Hopfield-modern attractor retrieval

At retrieve time, project query into sparse code space, then use
modern Hopfield update (Ramsauer et al. 2020) to settle onto the
nearest stored attractor:

```python
def hopfield_retrieve(query_code: SparseCode,
                      stored: list[SparseCode],
                      beta: float) -> list[tuple[SparseCode, float]]:
    # One-step retrieval: q' = X @ softmax(beta * X.T @ q)
    # where X is stored codes stacked, q is query code
    # Returns sorted similarity scores
```

Tested variants:
- 1-step (no iteration)
- iterated to convergence
- `beta` (inverse temperature) scanned

### Primitive 4: Engram tagging (recency/frequency)

Per stored memory, track:

- `access_count: int` — number of times retrieved
- `last_access_step: int` — step at most recent retrieval
- `encoding_step: int` — step at ingest

At retrieve time, score with:

```
score_final = score_hybrid + γ_freq · log(1 + access_count)
             + γ_rec · exp(-(now - last_access_step) / τ)
```

Parameters: `γ_freq`, `γ_rec`, `τ` (recency timescale, in units of
ingest events — roughly analogous to biological days).

## Phases

### Phase 1: `sim` measurement harness (week 1, days 1-3)

**Goal:** empirically measure CA3 sparse-code properties on
controlled stimulus sets so we can calibrate numpy primitives to
match.

Deliverables:
- `research/developmental/experiments/sim_ca3_measurement.py` —
  script that:
  1. Configures `sim` with `HIPPOCAMPUS_CA3_RECURRENT` preset,
     5K neurons, Izhikevich model (fast).
  2. Generates 50 "concept" stimulus patterns (distinct random
     population codes, ~100 active neurons each).
  3. Injects each concept 10 times, 50ms per injection, 200ms
     rest between.
  4. At each injection, records readout population spike counts
     over a 200ms window after stimulus onset.
  5. Computes:
     - Per-concept sparse-code stability (Jaccard within trials)
     - Between-concept separation (Jaccard between concept pairs)
     - Sparsity (k/N)
     - Noise floor (spontaneous readout activity)

Outputs: `research/developmental/results/sim_ca3_baseline.json`
with empirical separation/stability/sparsity numbers that target
our numpy primitives.

GO/NO-GO gate: `sim` produces (a) stable within-concept codes
(within-Jaccard > 0.5), (b) clear between-concept separation
(between-Jaccard < within-Jaccard by factor >1.5). If neither
holds, sim is not doing pattern separation on these stimuli and
Path B's premise is broken; we revisit.

### Phase 2: Numpy k-WTA + pattern separation (week 1 days 4-5, week 2 days 1-2)

**Goal:** a clean numpy implementation of Primitives 1 + 2,
producing sparse codes with sparsity and separation matching the
sim measurements from Phase 1.

Deliverables:
- `src/soma/memory/sparse_codes.py` — `SparseCode` dataclass,
  `kwta()`, `pattern_separate()`, `code_similarity()` functions
- `tests/test_memory/test_sparse_codes.py` — TDD tests:
  - `test_kwta_produces_k_active_dims`
  - `test_kwta_deterministic_given_seed`
  - `test_pattern_separate_reduces_overlap_above_threshold`
  - `test_code_similarity_jaccard_bounded_0_1`
  - `test_pattern_separate_monotone_in_strength`
- Calibration script: `benchmarks/sparse_codes_calibration.py` —
  runs the same 50-concept stimulus from Phase 1 through the numpy
  transforms and reports within/between Jaccard. Target: match sim's
  numbers to within 20% relative error.

GO/NO-GO gate: calibration reports within-Jaccard and separation
ratio that match sim's phase 1 numbers within 20%. If the numpy
version produces wildly different sparsity or separation, the
primitive isn't a faithful approximation and we iterate on the
projection type or separation mechanism.

### Phase 3: Retrieval A/B on LongMemEval N=100 (week 2 days 3-5)

**Goal:** measure whether adding sparse-overlap as a retrieval
signal to the existing hybrid path produces a rank-1 lift.

Variants tested at α=0.30 hybrid default:

| Variant | Retrieval scoring |
|---|---|
| `hybrid_only` (baseline) | `score = 0.7·cosine + 0.3·bm25` (current SOMA default) |
| `hybrid_plus_sparse` | `score = 0.5·cosine + 0.2·bm25 + 0.3·sparse_overlap` |
| `sparse_only` | `score = sparse_overlap` (sanity check) |

Deliverables:
- `benchmarks/industry/longmemeval/run_sparse_retrieval.py` —
  retrieval-only probe mirroring `rank_probe.py`, outputs per-item
  `{gold_rank_hybrid, gold_rank_with_sparse, ...}`
- `scripts/analysis/compare_sparse_hybrid.py` — paired analysis
- Results jsonl + summary findings doc

GO/NO-GO gate: `hybrid_plus_sparse` R@5 ≥ `hybrid_only` R@5 on N=100
with at least one question-type showing rank-1 lift ≥ 5pp. If this
fails, sparse codes aren't adding retrieval signal over hybrid; we
pivot to Primitive 3 or 4 testing before declaring failure.

### Phase 4: Attractor retrieval (Primitive 3) (week 2 days 6-7, week 3 days 1-2)

Only proceed if Phase 3's gate fails OR if Phase 3 passes and we
want to stack primitives. Otherwise skip to Phase 5.

**Goal:** test whether Hopfield-modern attractor retrieval handles
paraphrased / noisy queries better than flat cosine.

Deliverables:
- `src/soma/memory/attractor.py` — modern Hopfield retrieval
  (1-step and iterated)
- `tests/test_memory/test_attractor.py` — TDD tests including
  noisy-query pattern completion
- Retrieval A/B on N=100 LongMemEval, adding `hybrid_plus_attractor`
  and `attractor_only` variants.

GO/NO-GO gate: `hybrid_plus_attractor` R@5 > `hybrid_only` on
single-session-preference (the category most affected by paraphrase).
If this fails and Phase 3 also failed, we've tested the two main
candidates and need to reassess.

### Phase 5: Engram tagging (Primitive 4) (week 3 days 3-4)

**Goal:** measure whether recency/frequency weighting adds signal
on multi-session queries.

Deliverables:
- `src/soma/memory/engram.py` — per-memory recency/frequency
  tracking, scoring function with γ_freq / γ_rec parameters
- Modified retrieval path in `memory/api.py` to include engram
  score term (opt-in flag)
- `benchmarks/industry/longmemeval/run_engram_retrieval.py` —
  probe with simulated ingest history (items arrive in order,
  queried in order, so recency signal exists)

GO/NO-GO gate: `hybrid_plus_engram` R@5 ≥ `hybrid_only` on
multi-session specifically.

### Phase 6: Stack + final A/B (week 3 day 5)

If one or more primitives passed their gates, test stacked variant:

| Variant | Components |
|---|---|
| `hybrid_only` | baseline |
| `winning_stack` | hybrid + all primitives that passed |

Deliverables:
- Full LongMemEval N=500 run with the winning configuration
- Updated `positioning.md` with the new retrieval line (if any
  primitive passed)
- Full findings doc

GO/NO-GO gate: `winning_stack` R@5 ≥ `hybrid_only` R@5 + 0.02 on
N=500. If true, ship the primitive as an opt-in flag. If not, document
the null result and close.

## Implementation details

### Text-to-sparse pipeline

```python
# In MemoryLayer.store:
dense_emb = embed_fn(text)                        # 1024d
sparse_code = kwta(dense_emb, k=32, dim=4096)     # 4096d, k=32 active
if separation_enabled:
    sparse_code = pattern_separate([sparse_code], self._recent_codes)
self._store_entry.sparse_code = sparse_code
self._sparse_index.add(entry_id, sparse_code)

# In MemoryLayer.retrieve:
query_dense = embed_fn(query)
query_sparse = kwta(query_dense, k=32, dim=4096)

# Existing hybrid scores
cosine_scores = self._dense_index.cosine(query_dense, k=candidates)
bm25_scores = self._bm25_index.search(query, k=candidates)

# New sparse score
sparse_scores = self._sparse_index.overlap(query_sparse, k=candidates)

# Combine (weights configurable)
combined = w_cos * cosine_scores + w_bm25 * bm25_scores
         + w_sparse * sparse_scores
return top_k(combined, k)
```

### Sparse index storage

`SparseCode` is a sorted int32 array of active indices. Indexing:

- **Brute force** for N < 10K: numpy broadcasting over all stored
  codes. O(N·k) per query.
- **MinHash-LSH** for N ≥ 10K: each code hashed into `n_hashes`
  buckets; query compares against same-bucket candidates only.
  Sub-linear query time at the cost of approximate recall.

Phase 2 delivers brute force. MinHash-LSH is deferred until the
primitive has a proven retrieval win at small N.

### Integration with existing save/load

`MemoryLayer.save()` / `load()` must persist sparse codes. Simplest:
serialize the `int32` active-index arrays alongside existing tensors
in `memory_embeddings.pt`. Load reads them back into in-memory
`SparseCode` objects.

Backward compatibility: existing saved brains load with no sparse
codes; retrieval falls back to hybrid-only. Optional reindex step
regenerates sparse codes from stored embeddings.

## Evaluation criteria

For every primitive tested, the decision criterion is:

**Include it in the shipping stack if**: (a) it delivers ≥ 0.02 R@5
lift on at least one question-type, (b) end-to-end retrieve latency
stays ≤ 20ms per query at N=500, (c) no other question-type
regresses by more than 0.01 R@5.

Reject if: no category gains or any category regresses by > 0.02.

## Open questions

1. **Projection matrix choice for k-WTA**: random-normal is simplest
   but random-orthogonal has theoretical advantages (JL-lemma). A
   learned projection could be better but introduces training
   dependency. Phase 2 evaluates all three.

2. **Sparse code dimensionality**: 4096 gives capacity ~C(4096, 32)
   = 10^77 distinct codes, vastly more than we need. Smaller dim
   (512) would be cheaper but more collision-prone. Phase 2 scans
   {512, 1024, 4096, 16384}.

3. **Online vs. offline pattern separation**: if items arrive in
   sequence, does separation need to run per-item (expensive) or
   can a batch separation at save-time work? Phase 2 measures cost.

4. **Sim calibration vs. actual benchmark performance**: matching
   sim's sparsity/separation numbers doesn't guarantee retrieval
   wins on LongMemEval. If Phase 2 calibrates to sim but Phase 3
   fails, we've learned the primitive is biologically faithful but
   not product-useful. Still a valid result.

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| All three primitives fail to beat hybrid | Medium | High | Valid negative result; closes question, informs Path A gate |
| Sparse overhead dominates retrieve latency | Medium | Moderate | Strict latency gate; LSH indexing if brute force slow |
| Sim calibration is impossible (Phase 1 no separation) | Low | Moderate | Try cortex preset; skip calibration and go straight to Phase 3 A/B |
| Implementation takes longer than 2-3 weeks | Medium | Low | Hard phase gates; skip Phases 4-5 if Phase 3 passes strongly |

## What this delivers regardless of outcome

Even if all gates fail:

- Empirical `sim`-calibrated measurements of what CA3-like networks
  actually do on controlled stimuli (novel contribution even outside
  SOMA).
- Clean numpy implementations of four biologically-inspired
  primitives as a reusable library (`soma.memory.sparse_codes`,
  `soma.memory.attractor`, `soma.memory.engram`).
- A definitive measurement of whether dense embeddings have a
  headroom ceiling on LongMemEval/LoCoMo that biological primitives
  can break. If no: Path A probably doesn't help either.

## Files

New code:
- `src/soma/memory/sparse_codes.py`
- `src/soma/memory/attractor.py`
- `src/soma/memory/engram.py`
- `src/soma/memory/sparse_index.py`

New tests:
- `tests/test_memory/test_sparse_codes.py`
- `tests/test_memory/test_attractor.py`
- `tests/test_memory/test_engram.py`

New experiments:
- `research/developmental/experiments/sim_ca3_measurement.py`
- `benchmarks/industry/longmemeval/run_sparse_retrieval.py`
- `benchmarks/industry/longmemeval/run_engram_retrieval.py`
- `benchmarks/sparse_codes_calibration.py`

New findings:
- `research/developmental/results/sim_ca3_baseline_findings.md`
- `research/developmental/results/sparse_codes_calibration_findings.md`
- `research/developmental/results/longmemeval_sparse_retrieval_findings.md`
- (and one per primitive that gets tested)

Minor modifications:
- `src/soma/memory/api.py` — add opt-in sparse index + retrieval
  path (feature-flag, default off)
- `src/soma/core/config.py` — new fields `sparse_codes_enabled`,
  `sparse_dim`, `sparse_k`, `engram_enabled`, etc.

## Estimated timeline

- Phase 1 (sim measurement): 3 days
- Phase 2 (numpy sparse + separation): 4 days
- Phase 3 (LongMemEval A/B): 3 days
- Phase 4 (attractor retrieval, conditional): 4 days
- Phase 5 (engram tagging, conditional): 2 days
- Phase 6 (stacked final A/B): 1 day

**Total: 2-3 weeks wall clock**, with early-exit if Phase 3 is
decisive either way.
