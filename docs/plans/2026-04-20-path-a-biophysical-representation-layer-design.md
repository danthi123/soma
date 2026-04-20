# Path A: Biophysical Simulator as SOMA Representation Layer

**Date:** 2026-04-20.
**Status:** Design. Implementation pending Path B validation.
**Scope:** Ambitious, multi-month. Connects SOMA's retrieval/storage
path to the `neural-simulator` project (`E:/Documents/Projects/sim/`)
so memory encoding, storage, and retrieval all happen via a
biophysically-accurate spiking network. The intended outcome is not
merely a faster retriever — it's a substrate that supports artificial
life research (emergent behavior, developmental trajectories, cross-
modal integration, time-dependent dynamics).

## Motivation

Three graph-retrieval directions (plastic graph activation,
Direction 4a input-projection distillation, Direction 4b spatial
distillation) produced null results on real corpora. The pattern
across the failures — signal attenuation through too many layers,
random projections that don't carry semantics, per-node degeneracy
— suggests the issue isn't specific mechanisms but **the attempt to
route a learned-graph signal into retrieval on top of a working
hybrid+embedder path**. The embedder plus BM25 already captures
most of the signal; a secondary scoring channel has little room to
add information without adding noise.

Path A rejects the "secondary channel" framing. Instead, the spiking
network **is** the storage-and-retrieval substrate: text embeddings
become stimulus patterns, the network's self-organized dynamics
form the memory trace, and queries trigger attractor completion.
Success would validate that the right biology, simulated at the
right level of faithfulness, supports semantic memory that hybrid
retrieval only approximates.

Beyond SOMA's memory-layer goals, Path A opens:

- **Artificial life**: a persistent, plastic, developmentally-
  trajectoried network whose behavior is meaningful over sim-weeks
  and sim-months of simulation time.
- **Cross-modal integration**: the same substrate can accept text,
  audio, image, and proprioceptive inputs via different stimulus
  channels without separate model architectures.
- **Temporal cognition**: time dynamics are first-class, not
  approximated. Sequence memory, oscillation-based binding,
  predictive coding all emerge from the substrate.
- **Biological ground truth**: direct comparison to in-vivo
  neuroscience data without a modeling layer in between.

The honest cost: this is orders of magnitude slower than cosine
similarity, requires novel research on text-to-stimulus encoding,
and has significant integration engineering. It will never be the
fastest path to a shipping memory layer. It might be the most
interesting one.

## Relationship to the existing codebases

- **`E:/Documents/Projects/SOMA`**: houses the `MemoryLayer` API
  (`store`, `retrieve`, `consolidate`, `save`, `load`), positioning
  docs, benchmarks, and the current hybrid retrieval implementation.
- **`E:/Documents/Projects/sim`**: GPU-accelerated spiking network
  simulator with biophysically-accurate neuron models (HH,
  Izhikevich, AdEx), STDP, structural plasticity, 17 region presets,
  and an `ExperimentEngine` that already implements the required
  orchestration pattern (stimulus injection + readout + training
  modes including `RESERVOIR_READOUT`).

Path A requires both projects to compose, not merge. A dedicated
package (`soma.research.biophysical`, new) holds the integration
code; `sim` stays an external dependency imported by that package.

## Hypothesis

A sparse-distributed, attractor-based, Hebbian-learned representation
layer built on a biophysically-accurate CA3-like recurrent network
will, for text retrieval on LongMemEval and LoCoMo:

1. **Produce memory codes with sparsity 1–5%** (empirically observed
   CA3 activity) on a 5K–10K neuron readout population.
2. **Exhibit pattern separation**: semantically-similar texts map to
   partially-overlapping but distinguishable sparse codes.
3. **Exhibit pattern completion**: partial or noisy query codes
   settle onto a full stored code via recurrent dynamics within
   200–500ms of simulated time.
4. **Match or exceed hybrid retrieval R@5** on at least one
   question-type where hybrid has a known ceiling (likely
   single-session-preference or temporal-reasoning).

If all four hold, the substrate is validated. If 1–3 hold but 4
fails, we've built an interesting artificial-life substrate that
doesn't beat hybrid on retrieval — still a valuable result.

## Architecture overview

```
text
 │
 │  (1) dense embed (sbert/mxbai)
 ▼
embedding [1024d, dense, real-valued]
 │
 │  (2) text→stimulus encoder (learned or fixed)
 ▼
stimulus pattern [n_input_neurons, time_ms]
 │
 │  (3) inject as current into InputGroup
 ▼
┌───────────────────────────────────────────┐
│       sim network (CA3-style recurrent)   │
│  InputGroup → HiddenGroup (recurrent)     │
│              → OutputGroup (readout)      │
│  STDP on Hidden↔Hidden and Hidden↔Output  │
│  Homeostasis on firing rate               │
│  Structural plasticity for new concepts   │
└───────────────────────────────────────────┘
 │
 │  (4) run dynamics for T_settle ms
 ▼
population activity [n_output_neurons, time]
 │
 │  (5) sparse code readout
 ▼
sparse code [n_output_neurons, binary/k-hot]
 │
 │  STORE: persist code + metadata in SOMA-compatible container
 │  RETRIEVE: compute overlap with stored codes, return top-k
 ▼
results
```

Five integration components, numbered in the diagram:

### Component 1: Dense embedding (existing, unchanged)

SOMA's `MemoryLayer.embed_fn` continues to produce a 384–1024d
dense embedding per text. We don't replace this; the embedding is
the input signal.

### Component 2: Text-to-stimulus encoder

**This is the central research problem.** How do we map a
real-valued embedding to a time-varying current injection pattern
across a population of neurons?

Three candidate schemes, from most tractable to most biological:

| Scheme | Mechanism | Cost | Biology |
|---|---|---|---|
| **Rate coding** | `current[i] = gain · embedding[i]`; embedding dim = stimulus dim | Simple; no training needed | Weakly biological; real sensory input is spike trains, not DC current |
| **Population rate coding** | Assign each embedding dim to a tuning-curve cluster of ~10 neurons; input dim `e[j]` drives cluster `j` with peak at `e[j]` | Moderate; requires tuning curve design | Middle-ground; matches cortical representations |
| **Phase-locked spike trains** | Convert embedding to Poisson spike trains with rate `λ ∝ embedding[i]` and phase locked to gamma oscillation | Higher; needs phase control | Most biological; matches cortical sensory coding |

Implementation plan: start with rate coding (simplest, validates
plumbing), move to population rate coding (expected production),
defer phase-locked for future work.

### Component 3: Stimulus injection

Uses `sim`'s existing `StimulusManager` with a `CUSTOM_WAVEFORM`
channel per input embedding. The integration adapter constructs a
per-neuron current vector from the embedding and passes it as a
stimulus channel to `ExperimentEngine`.

### Component 4: Network dynamics

The recurrent network does the work. Choice of region preset is the
second major design decision:

| Preset | Use case | Tradeoff |
|---|---|---|
| `HIPPOCAMPUS_CA3_RECURRENT` | Memory storage, pattern separation/completion | Primary choice — purpose-built for this |
| `CORTEX_L23_RS_FS` | General associative memory | Cortical baseline; may over-mix |
| `PREFRONTAL_CORTEX_WM` | Working-memory tasks | Wrong scale; better for active maintenance |

We run with `HIPPOCAMPUS_CA3_RECURRENT` and `HIPPOCAMPUS_CA1_RS_FS`
as the store + readout regions. CA3 does the recurrent pattern
separation/completion; CA1 provides a cleaner readout population
that can be linearly decoded.

STDP is on by default. Homeostasis prevents runaway firing.
Structural plasticity is initially off (adds moving parts; re-
enable in Phase 4+).

### Component 5: Sparse code readout

Settle the network for `T_settle = 500 ms`, then measure the spike
count per output neuron over the last 200 ms. Convert to sparse
binary via k-WTA (keep top 2% = ~100 neurons for a 5000-neuron
readout). This is the **memory code**.

For retrieval, store codes as sparse vectors (indices only).
Similarity is set-intersection (`|a ∩ b| / |a ∪ b|` or straight
dot-product on binary vectors). This is O(k log k) per comparison,
and we can index candidates with MinHash/LSH for sub-linear scan
as the store grows.

### Storage schema

A stored memory is:

```python
@dataclass
class BiophysicalMemory:
    memory_id: str                     # UUID, same as MemoryLayer
    text: str                          # raw text
    metadata: dict[str, Any]
    dense_embedding: np.ndarray        # cached for fallback retrieval
    sparse_code: np.ndarray            # active indices, int32
    sparse_k: int                      # k for the k-WTA
    readout_region: str                # e.g. "CA1_readout"
    sim_config_hash: str               # so replays are comparable
    created_step: int                  # sim step number at creation
```

Persisted as HDF5 (matches sim's existing format) in
`soma.research.biophysical.BiophysicalMemoryStore`.

## Implementation phases

### Phase 1: Plumbing (week 1-2)

**Goal:** A `MemoryLayer.with_biophysical()` constructor that
ingests text, drives a sim network, and produces a sparse code.
Retrieval is cosine on dense embedding (Phase 1 sanity check).

Deliverables:
- `src/soma/research/biophysical/__init__.py` (new package)
- `src/soma/research/biophysical/adapter.py` — `SimAdapter` wraps
  `sim.experiment.ExperimentEngine` with SOMA-friendly methods:
  `encode_text(text) -> sparse_code`, `run_settle(ms)`,
  `readout_sparse(k)`
- `src/soma/research/biophysical/encoder.py` — text-to-stimulus
  schemes (rate, population-rate)
- `src/soma/research/biophysical/memory_layer.py` — `MemoryLayer`
  subclass with `.with_biophysical()` constructor
- `tests/research/biophysical/test_adapter.py` — plumbing tests
  (input text → non-empty sparse code)
- `tests/research/biophysical/test_encoder.py` — encoder unit tests

GO/NO-GO gate: a single test passes that stores 10 texts, reads
them back via dense-cosine, and confirms the sparse codes are
(a) non-empty, (b) sparse (k/N < 5%), (c) deterministic given a
fixed seed.

### Phase 2: Sparse-code retrieval baseline (week 3)

**Goal:** retrieval via sparse-code overlap, compared to cosine.

Deliverables:
- `src/soma/research/biophysical/similarity.py` — sparse overlap
  metric (Jaccard, overlap-count, LSH-accelerated)
- `benchmarks/research/biophysical/run_sparse_retrieval.py` —
  harness running N=50 LongMemEval items
- Per-item comparison jsonl: `{question_id, dense_cosine_rank,
  sparse_overlap_rank, gold_hit_@_5_dense, gold_hit_@_5_sparse}`

GO/NO-GO gate: on N=50, sparse-overlap retrieval achieves at least
80% of cosine's R@5. Lower than cosine is expected at this stage;
we're checking that the representation hasn't collapsed.

### Phase 3: Pattern separation validation (week 4)

**Goal:** empirically demonstrate CA3-like pattern separation on
controlled stimulus sets.

Experiments:
1. Inject 50 distinct "concept" embeddings 10 times each with small
   noise. Measure: within-concept sparse-code stability, between-
   concept separation. Pass if within-concept Jaccard > 0.6 and
   between-concept Jaccard < 0.3.
2. Inject semantically-similar pairs (paraphrases) and semantically-
   distinct pairs. Measure: similar-pair overlap > distinct-pair
   overlap. Pass if ratio > 1.5×.

Deliverables:
- `research/developmental/experiments/biophysical_pattern_sep.py`
- Findings doc with empirical sparsity/separation numbers.

GO/NO-GO gate: both separation metrics pass. If they fail, the
text-to-stimulus encoder is not giving the network meaningful
semantic structure; revise encoder before proceeding.

### Phase 4: Pattern completion + STDP learning (week 5-6)

**Goal:** repeated presentation of a stimulus should form a stable
attractor; partial cues should complete to the full code.

Experiments:
1. Present a stimulus 100 times over 10 sim-seconds with STDP on.
   Measure: stable attractor emerges (last 20 trials' sparse codes
   have Jaccard > 0.8).
2. Present a partial stimulus (50% of input group's current). Let
   the recurrent network settle. Measure: readout Jaccard with the
   trained code > 0.7.
3. Present an adversarial near-attractor. Measure: readout
   distinguishes the target from the distractor.

Deliverables:
- `research/developmental/experiments/biophysical_completion.py`
- STDP parameter scan (learning rate, window width) to characterize
  the attractor-formation envelope.

GO/NO-GO gate: completion at 50% partial cue produces Jaccard > 0.6
with the target. If this fails, the recurrent network is not doing
its job; consider region preset change or training regimen.

### Phase 5: End-to-end QA on LongMemEval (week 7-8)

**Goal:** full-pipeline evaluation on LongMemEval N=100 strict.

Variants tested:
- `biophysical_only`: sparse-code retrieval only
- `biophysical_hybrid`: linear combination of dense-cosine and
  sparse-overlap (mirrors SOMA's BM25+cosine hybrid idea)
- `biophysical_rerank`: dense-cosine top-50, sparse-overlap reranks
  to top-5

Baselines: `chroma_cosine`, `soma_hybrid` (shipping default).

Gates (following 4b's precedent of pre-registered gates):

| Gate | Condition | Verdict path if PASS | if FAIL |
|---|---|---|---|
| Primary | `biophysical_*` R@5 ≥ `soma_hybrid` R@5 on one variant | Phase 6 | — |
| Secondary | `biophysical_only` R@5 ≥ `soma_hybrid` R@5 − 0.05 | Option to iterate | Defer fully biophysical |
| Tertiary | `biophysical_hybrid` R@5 ≥ `soma_hybrid` R@5 + 0.01 | Ship hybrid | — |
| Cost | mean retrieve ≤ 5s per item | Productionize | Keep research-only |

### Phase 6: Cross-modal + artificial-life (open-ended)

Once retrieval works, the substrate opens onto:

- **Audio**: `OLFACTORY_BULB` preset + spike-train encoding of
  audio spectrograms. Cross-modal binding via shared hidden region.
- **Image**: `CORTEX_L23_RS_FS` as a visual cortex analog with
  retinotopic stimulus mapping.
- **Developmental trajectories**: run the network continuously for
  sim-weeks, measuring how representations mature over experience.
- **Sleep-replay consolidation**: alternate wake (new input) and
  sleep (no input, STDP on, measured replay) phases and observe
  offline consolidation effects.

These phases are unscoped; they wait on Phase 5 passing.

## Open research questions

1. **Text-to-stimulus encoding**: which scheme (rate, population
   rate, phase-locked) gives the best separation? No known
   literature answer. Resolution: Phase 3 comparison.

2. **Network size vs. expressivity**: 5K–10K neurons likely bound
   the concept capacity. Can we show meaningful retrieval at 5K, or
   does it need 50K? Resolution: Phase 2–3 scaling study.

3. **Settle time vs. latency**: `T_settle = 500 ms` is a guess. At
   dt=0.5ms Izhikevich = 1000 steps = ~50ms wall-clock on a 3090.
   For 500-item retrieval, that's 7 minutes per eval — workable.
   HH dt=0.05ms = 10× slower. Resolution: run with Izhikevich first,
   verify HH only if primary gate passes.

4. **STDP window calibration**: the Bi & Poo STDP window (tau=20ms)
   may not match the timescales of our stimulus bursts (<50ms).
   Resolution: Phase 4 STDP scan.

5. **Distinguishing signal from substrate noise**: `sim` has OU
   noise by default. If the sparse code is dominated by noise, the
   retrieval signal disappears. Resolution: Phase 3 within-concept
   stability metric.

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Text→stimulus encoder never produces separable codes | Medium | Fatal | Start with Phase 3 earliest, iterate on encoder before building downstream |
| Retrieval is 1000× slower than hybrid | High | Moderate | Acceptable for research; document latency trade as core positioning |
| CA3 preset doesn't do pattern separation for our stimulus statistics | Medium | High | Swap to cortex preset; tune STDP timescale |
| Attractor forms but retrieval decoding can't identify it | Low | Moderate | Add a learned linear readout per stored code (reservoir pattern) |
| Integration engineering dominates research | High | Moderate | Ruthless phase gating; stop at any failed GO/NO-GO |

## Relationship to Path B

Path B (see `2026-04-20-path-b-bio-validated-primitives-design.md`)
uses `sim` as a **measurement tool**: run sim to learn what real
biology does, port clean numpy primitives into SOMA. That's fast,
cheap, and answers a concrete product question: do sparse codes
help retrieval today?

Path A uses `sim` as **the substrate itself**: the network is SOMA,
with an adapter around it. That's slow, expensive, and answers a
research question: can a biophysically-accurate network be a
memory layer at all?

**Phase 1 of Path A (plumbing) depends on Path B's measurements.**
If Path B shows sparse codes + pattern separation don't help on
LongMemEval, Path A's Phase 5 gates are unlikely to pass either
(because we'll have evidence the representation primitive isn't the
ceiling). In that case Path A becomes artificial-life-first: we
build it anyway because the emergent-behavior value stands on its
own, and we stop claiming retrieval wins.

## What Path A unlocks that no other path does

A working Path A is not merely a memory layer. It's a persistent,
plastic, biophysically-accurate substrate you can run continuously.
The same codebase supports:

- Artificial-life experiments: long-horizon behavior, developmental
  stages, emergent specialization, predator-prey style dynamics
  between subpopulations.
- Neuroscience hypothesis testing: "does oscillation X improve
  memory retrieval Y?" becomes directly testable.
- Cross-modal agents: text + image + action all driving the same
  substrate, no separate encoders-to-stitch.
- Interpretability: every memory is a real attractor you can probe,
  perturb, and visualize in 3D (via sim's existing visualization).
- Biological realism as a marketing axis: "memory that actually
  works the way brains do, not just takes the name."

The operator's stated interest — "artificial life" and "exploring
not just AI but artificial life" — is exclusively served by Path A.
Path B produces a better product; Path A produces a research
platform.

## Files

All new code and docs land under:
- `src/soma/research/biophysical/` — integration package
- `benchmarks/research/biophysical/` — evaluation harnesses
- `tests/research/biophysical/` — tests
- `research/developmental/experiments/biophysical_*.py` — Phase
  experiments
- `research/developmental/results/biophysical_*_findings.md` —
  findings docs (one per phase)

No modifications to `src/soma/memory/api.py` (the existing
`MemoryLayer`) unless Phase 5 passes and we decide to merge
biophysical retrieval into the shipping path.

## Estimated timeline

- Phase 1 (plumbing): 1-2 weeks
- Phase 2 (sparse retrieval baseline): 1 week
- Phase 3 (pattern separation validation): 1 week
- Phase 4 (completion + STDP learning): 2 weeks
- Phase 5 (end-to-end QA): 2 weeks

**Total: ~7-8 weeks to a GO/NO-GO on biophysical retrieval.**

Phase 6 (cross-modal + artificial-life) is open-ended and gated on
Phase 5.

## Precondition

**Path A should not start before Path B Phase 3 completes.** Path B's
numpy-sparse-code measurements either validate the primitive (in
which case Path A's Phase 3 gate is pre-answered and we build with
confidence) or invalidate it (in which case Path A pivots to
artificial-life-first framing and the retrieval Phase 5 becomes
best-effort rather than decision-point).
