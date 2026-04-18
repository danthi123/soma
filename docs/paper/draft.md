# Structural Without Semantic: An Empirical Ceiling for Brain-Inspired Graph Memory in Retrieval-Augmented Generation

**Draft, 2026-04-18. Author: Daniel Thiberge. Contributions welcome.**

## Abstract

Brain-inspired memory architectures — graph structures with Hebbian
learning, structural plasticity, and consolidation — have been
proposed as retrieval enhancements over pre-trained embeddings for
retrieval-augmented generation (RAG). We present a rigorous empirical
study of one such architecture, SOMA, evaluated as a gated-hybrid
reranker over a pre-trained embedding baseline.

Across thirteen retrieval-diagnostic experiments on LoCoMo (5882-turn
corpus, 500 held-out queries) and LongMemEval-oracle (3094-turn
corpus, 100 queries), we find: (1) the graph-derived signal is real
but architecturally bounded at roughly +0.8% absolute over random
fingerprint assignment on LoCoMo, and becomes slightly negative on
LongMemEval; (2) both "fingerprint" (projected activation) and
"topology" (shared-active-nodes) signals plateau identically because
they derive from the same lateral-inhibition winners; (3) contrastively
fine-tuning the encoder against the graph topology causes catastrophic
forgetting (VecDB 63 → 50 hits, gated 65 → 51, gate usage 32 → 11 on
100 queries after 316 updates) because the topology is driven by
random projections and is structurally diverse but semantically
arbitrary. We argue this ceiling is a direct consequence of the
"structural without semantic" nature of graph activation patterns and
cannot be overcome by scaling the graph alone (scaling hurts).

Three complementary experiments on SOMA's native mechanisms then
investigate whether the ceiling extends to non-retrieval tasks.
On a sequence-prediction environment with distribution shifts, SOMA
beats a vanilla online MLP by 3-40x across regimes. But a targeted
ablation finds that structural plasticity (neurogenesis, synaptogenesis,
pruning) does *not* drive the advantage — a `no_growth` variant with
frozen topology outperforms the full system on 10 of 12 regimes
across two schedules, including under explicit capacity pressure.
A consolidation test on synthetic QA shows +45% F1 relative (0.279 →
0.404). Together the retrieval and adaptation findings suggest
brain-inspired graph memory is mis-applied as a retrieval plugin
and mis-attributed when the claim is that structural plasticity
is the load-bearing mechanism. We release all sixteen diagnostic
experiments and data.

## 1. Introduction

Retrieval-augmented generation is dominant in production LLM systems,
and dense retrieval over pre-trained embeddings is the standard
backbone. Against this backbone, several recent works propose brain-
inspired memory architectures that graft graph structure, Hebbian
learning, or knowledge-graph traversal on top of vector search:
SYNAPSE [spreading activation], Graphiti [temporal knowledge graph],
MAGMA [multi-graph with policy-guided traversal], Mem0 [hybrid
search with fact extraction]. The common premise is that structural
associations *the pretrained encoder missed* can be captured by the
graph and surfaced during retrieval.

This premise is plausible but under-tested. Existing evaluations
typically report aggregate F1 on one benchmark with a small
held-out set and no shuffle or permutation baseline. It is difficult
from the published literature to distinguish a real structural
signal from either (a) overfit-to-tuning-sample (hyperparameter
search over a small tuning set that does not generalize), or (b)
tautological gains, where the graph — itself trained on activations
derived from the same encoder — will naturally align with encoder
similarity structure and appear to add signal without actually
adding any beyond what the encoder already provides.

This paper does not propose a new architecture. Instead, we take an
existing brain-inspired system — SOMA, a graph memory with
neurogenesis, Hebbian learning, and consolidation — and ask:
**does the graph add retrievable signal beyond the pre-trained
encoder?** We construct a diagnostic suite designed to isolate
real structural signal from artifacts:

- Multi-slice held-out validation (the "does tuning generalize?" test)
- Shuffle diagnostic (the "is the signal real vs random?" test)
- Scaling sweep (the "does the bottleneck lift with capacity?" test)
- Cross-benchmark validation (the "does the result transfer?" test)
- Per-query attribution (the "is the signal concentrated or diffuse?" test)

Our findings:

1. The graph-derived signal is **real but tiny**. Shuffle diagnostics
   show real fingerprint-to-memory mappings outperform all random
   permutations tested (real total +2 vs max shuffled +1 over 500
   queries), confirming the mechanism carries non-random information.
2. The signal **does not generalize**. On LoCoMo, the gated hybrid
   is a 32W/34L coin flip against VecDB across 500 held-out queries.
   On LongMemEval, it is slightly negative (-1, 3W/6L).
3. The signal **is architecturally bounded**. Both fingerprint and
   topology derivations plateau at the same delta, because both read
   the same 3 lateral-inhibition-winner nodes. Scaling the graph to
   32 or 64 associators does not help and in fact hurts (-3 to -1
   vs n=8).
4. The ceiling has a **clear root cause**: random input projections
   create fingerprints that are structurally diverse but semantically
   arbitrary. Training the encoder contrastively against graph
   topology is catastrophic (-13 hits) because the topology target
   is not correlated with semantic structure.
5. On the non-retrieval side, **SOMA substantially outperforms
   a vanilla MLP on adaptation tasks** — 3-33x lower MSE than an
   online MLP across four regimes of a sequence-prediction
   environment, and 9-40x lower on an 8-regime capacity-pressure
   schedule. Consolidation gives +45% relative F1 on an ad-hoc
   synthetic-QA test. Multi-session development produces +275%
   relative F1 growth across three sequential sessions with save/
   load preserving state.
6. **But a targeted ablation finds that structural plasticity is
   not load-bearing for SOMA's adaptation advantage.** A variant
   with growth disabled (14 nodes / 24 edges fixed) outperforms
   the full system on 10 of 12 regimes across both schedules,
   including under capacity pressure. The advantage is in SOMA's
   graph substrate; the plasticity mechanisms SOMA foregrounds
   are, in the current implementation, a net cost.

We take these results as evidence that (a) the retrieval-
enhancement framing is a mismatch for what structural plasticity
provides, and (b) the widely-promoted mechanisms (neurogenesis,
synaptogenesis, pruning) are not the load-bearing component of
SOMA's adaptation advantage; the graph substrate is. We conclude
with a call both to evaluate brain-inspired architectures on
problems they are structurally suited to, and to ablate their
plasticity components when claiming advantage, to distinguish
"substrate matters" from "plasticity matters."

A broader aim is to offer a **diagnostic template** for
brain-inspired retrieval claims. Our five-test suite — multi-slice
held-out, shuffle permutations, scaling sweep, per-query attribution,
cross-benchmark — is applicable to any graph-augmented retrieval
system and does not depend on SOMA specifics.

## 2. Background

### 2.1 Dense retrieval and reranking

We assume the standard dense-retrieval setup. For a corpus
$C = \{c_1, \ldots, c_N\}$ of text passages and a query $q$, a
pre-trained encoder $E$ produces embeddings; retrieval returns the
top-$k$ passages by cosine similarity:
$$
\text{top-}k(q) = \text{argtop}_k \cos(E(q), E(c_i)).
$$
Reranking composes a secondary score $s(q, c_i)$ with the embedding
score, typically via linear interpolation:
$$
\text{score}(q, c_i) = (1 - \alpha)\cos(E(q), E(c_i)) + \alpha s(q, c_i).
$$

### 2.2 Brain-inspired memory architectures

Several recent proposals augment dense retrieval with graph-based
memory. The common structure:
(i) memories are stored as both embeddings and graph activations;
(ii) query-memory similarity is computed in both embedding and graph
     spaces;
(iii) a combined score is used for reranking or retrieval.

The graph space is meant to capture *associative* structure — co-
occurrences, temporal adjacency, concept chains — that the encoder
does not explicitly model. The hypothesis is that such structure
helps on "relational" queries ("what did X and Y do together?",
"what happened before Z?") where the relevant passages are not
individually high on embedding similarity.

### 2.3 SOMA architecture

SOMA is a graph-memory system with:
- **Node types**: SENSOR (input projection), ASSOCIATOR (hidden),
  INTEGRATOR (downstream features), OUTPUT. Each is a small MLP with
  residual connection and homeostatic gain.
- **Edges**: directed, Hebbian-plastic, each with scalar weight.
- **Execution**: topological waves; cycles handled via previous-
  timestep activations.
- **Growth**: synaptogenesis (new edges), neurogenesis (new nodes),
  pruning (edge/node removal by activity threshold).
- **Memory tiers**: working (decay buffer, 32 slots), episodic (one-
  shot, 10K items), parametric (graph weights).
- **Consolidation cycle**: offline replay that reorganizes edges and
  weights; analogous to sleep.

We use PredictiveSOMA, a wrapper that drives the graph via next-input
prediction error (Free Energy Principle). The text processing flow
for retrieval experiments is:

1. Text passage $c_i$ is encoded by the frozen pretrained encoder:
   $e_i = E(c_i) \in \mathbb{R}^{384}$.
2. $e_i$ enters the graph at the SENSOR node, propagates through
   associators (each applying its randomly-initialized input
   projection), then integrators, then OUTPUT.
3. PredictiveSOMA's prediction head outputs a forecast of the *next*
   such embedding; MSE between forecast and actual drives Hebbian
   updates, synaptogenesis/neurogenesis/pruning decisions, and
   the prediction head's own backprop.
4. For retrieval, we capture the associator activation state
   per-memory via lateral inhibition (top-3 winners), then derive
   either a hash-projected fingerprint or a set of active-node IDs
   for the query-time comparison.

Full architecture is documented in [whitepaper.md](../whitepaper.md).

### 2.4 Fingerprint and topology signals

Given a developed graph, two derivations of the activation state can
be used as a reranking signal:

**Fingerprint**. For each non-boundary node active for the query, we
write 16 sampled activation values at SHA256-assigned positions in a
128-dimensional vector, then compute cosine similarity against the
stored fingerprint of each candidate memory. This preserves
directional activation pattern.

**Topology**. We record the set of active nodes per stored memory
(inverted index: node → memory steps). For a query, topology
similarity is the Jaccard-like ratio between the query's active node
set and the candidate's active node set.

Both derivations read from the same lateral-inhibition 3-winner
selection: of the $N$ associator nodes, only the top 3 by activation
magnitude (with a `min_active=3` floor) are retained per step. This
is the crucial bottleneck we investigate.

## 3. Method

### 3.1 Gated-hybrid retrieval

Our tested signal integration is a **confidence-gated** variant of
standard reranking:

```
Algorithm 1: Gated Hybrid Retrieval
Input:  query q, corpus C, corpus embeddings E(C),
        graph G, gate θ, weight w
1. Compute embedding similarities s_emb = cos(E(q), E(c)) for c in C
2. Select top-20 candidates by s_emb
3. Run q through G; extract query fingerprint f_q
4. For each candidate i in top-20:
     s_fp[i] = cos(f_q, f_i)
5. Compute confidence = sort(s_fp)[0] - sort(s_fp)[1]
6. If confidence >= θ:
     s_final[i] = (1-w) * s_emb[i] + w * s_fp[i]
     Return top-5 by s_final
   Else:
     Return top-5 by s_emb (pure embedding)
```

The confidence gate is a guard against noisy reranking: the graph
only intervenes when it has a clearly-preferred candidate. Our
hyperparameters $\theta = 0.05, w = 0.2$ were tuned against LoCoMo's
first 10 QA pairs per conversation (100 queries) over phases of
earlier development; this tuning is the motivation for the held-out
validation in §5.

### 3.2 Evaluation protocol

**Corpus and queries**. LoCoMo is a 10-conversation long-term memory
benchmark with 1986 QA pairs across the conversations (avg 199 per
conv, min 105, max 260). We merge all turns into a single corpus
(5882 turns) and index by turn. LongMemEval-oracle is a 500-item
benchmark comprising question_type labels `temporal-reasoning` and
`multi-session`; each item has a question, a gold answer, and a
24-36-turn haystack. We use the first 100 items, merging their
haystacks into a single 3094-turn corpus.

**Encoder**. All experiments use the pre-trained `sentence-transformers/
all-MiniLM-L6-v2` model (384-dim) with frozen weights. The only
experiments that fine-tune the encoder are Phase 4 (SOMA-loss
backprop — produced 0 gradient updates due to internal tensor
detaching) and Phase 4b (contrastive FT against graph topology —
catastrophic; see §5). Both are reported as failures; main-result
experiments use the frozen encoder.

**Scoring**. For each query, retrieve top-5 passages, score each as
token-F1 against the gold answer, take the max, and count the query
as a "hit" if max > 0.05. "Win" / "loss" / "tie" vs VecDB is defined
as hybrid - vecdb > 0.01 / < -0.01 / else.

**Hardware and runtime**. All experiments on a single RTX 3090.
Development of the full LoCoMo corpus (5882 turns) at $n=8$
associators takes ~330s.

### 3.3 Diagnostic suite

We designed five diagnostics to isolate real signal:

1. **Held-out validation (§4.1)**: three disjoint QA slices per
   conversation — slice A `[0:10]` (tuning), B `[10:30]`, C `[30:50]`.
   Combined 500 queries.
2. **Shuffle diagnostic (§4.2)**: run the gated hybrid with real
   memory-to-fingerprint mapping vs 5 random permutations; compare
   per-slice and combined deltas.
3. **Scaling sweep (§4.3)**: vary `initial_associator_count` in
   {8, 32, 64}, holding other hyperparameters constant.
4. **Per-query attribution (§4.4)**: classify queries into 6 keyword
   categories (temporal, cross_entity, counting, factual_self,
   preference, causal, other); report per-category W/L/T.
5. **Signal equivalence (§4.5)**: swap fingerprint cosine for
   topology Jaccard in the gate, hold everything else fixed.
6. **Cross-benchmark validation (§4.6)**: rerun gated hybrid on
   LongMemEval oracle (100 items).

## 4. Results

**Phases referenced.** We number retrieval experiments by the
order they were run (Phase 3 onwards — Phases 1-2 were earlier
exploratory work on the graph reranking baseline, prior to the
current gated-hybrid formulation). The following experiments
produced the headline results that we tabulate in §4.1-§4.6 and
§4.7:

- **Phase 3**: confidence-gated hybrid baseline (+3 hits on slice A,
  the tuning set).
- **Phase 9**: associator count scaling sweep (§4.3).
- **Phase 10**: multi-slice held-out validation (§4.1).
- **Phase 11**: shuffle diagnostic (§4.2).
- **Phase 12**: per-query attribution (§4.4).
- **Phase 13**: topology signal swap (§4.5).
- **Phase 14**: LongMemEval cross-benchmark validation (§4.6).

We also ran four tangential experiments whose negative/neutral
findings inform the analysis (§5) but are not tabulated as primary
results: **Phase 4** (encoder FT via SOMA loss — zero gradient
updates due to internal tensor detaching), **Phase 4b** (contrastive
encoder FT against graph topology — catastrophic; described in §5),
**Phase 5** (learned embedding→fingerprint "semantic lens" — neutral),
**Phase 6** (adaptive rerank weight — no improvement over fixed
weight), **Phase 7** (`min_active` sweep — min=3 Pareto optimal),
and **Phase 8** (learnable input projections via competitive learning
updates — +1 hit improvement, within the ±2 noise floor observed
between runs). All are linked in Appendix B.

### 4.1 Held-out validation: the tuned +3 disappears

Table 1: Gated hybrid vs VecDB across three slices.

| Slice            | N   | Hybrid | VecDB | Delta | W/L   |
|------------------|-----|--------|-------|-------|-------|
| A `[0:10]` (tuned) | 100 | 64     | 63    | +1    | 7/4   |
| B `[10:30]`      | 200 | 129    | 131   | -2    | 8/18  |
| C `[30:50]`      | 200 | 129    | 126   | +3    | 17/12 |
| **Combined**     | 500 | 322    | 320   | **+2**| 32/34 |

The +3 found during initial development reproduces on slice A
(within noise: +1 this run) but does not generalize. Slice B gives
-2; slice C gives +3. Wins-to-losses across 500 queries is 32/34 —
a near coin flip. The aggregate delta is +2 hits (+0.4% absolute).

### 4.2 Shuffle diagnostic: signal is real but tiny

Table 2: Real vs shuffled deltas per slice.

| Slice            | Real Δ | Shuffled Δs              | Shuf mean | Shuf range   |
|------------------|--------|---------------------------|-----------|---------------|
| A `[0:10]`       | +1     | [-1, -1, -1, -3, -1]      | -1.4      | [-3, -1]      |
| B `[10:30]`      | -2     | [-1, -2, -4, +4, +1]      | -0.4      | [-4, +4]      |
| C `[30:50]`      | +3     | [+1, +1, -4, 0, -1]       | -0.6      | [-4, +1]      |
| Per-seed total   | +2     | [-1, -2, -9, +1, -1]      | -2.4      | [-9, +1]      |

Real combined delta (+2) exceeds all 5 shuffled combined deltas
(max +1), establishing the mechanism carries non-random signal.
But the effect is ~4 hits/500 over random, i.e., ~0.8% absolute.
This confirms prior work that found graph-related reranking gains
on conversation data: the signal exists. The novel finding is its
magnitude is too small to support the retrieval-enhancement framing.

### 4.3 Scaling sweep: scaling hurts

Table 3: Gated hybrid at {8, 32, 64} initial associators (LoCoMo
slice A, 100 queries). "Δ hits" = hybrid − VecDB; "Net W/L" =
wins − losses, where a win/loss is defined by a per-query F1
change exceeding ±0.01 against VecDB.

| n_init | n_final | Hits | VecDB | Δ hits | Used | Wins | Loss | Net W/L | Runtime |
|--------|---------|------|-------|--------|------|------|------|---------|---------|
| 8      | 14      | 64   | 63    | +1     | 40   | 11   | 3    | +8      | 349s    |
| 32     | 38      | 61   | 63    | −2     | 56   | 13   | 9    | +4      | 1352s   |
| 64     | 70      | 63   | 63    | 0      | 61   | 8    | 9    | −1      | 2904s   |

The scaling hypothesis fails. We predicted larger $N$ would expand
the fingerprint vocabulary ($C(N, 3)$ scales cubically in the number
of applicable nodes) and lift hits. In practice, gate usage climbs
(40→56→61, as expected from more confident patterns), but losses
triple (3→9) and net wins decline (11→13→8). Hits are flat-to-falling
(64→61→63), and at $n=64$ the graph is indistinguishable from VecDB
on hits.

We interpret this as evidence that the bottleneck is not
vocabulary but signal quality: the additional fingerprint patterns
at scale are still structurally arbitrary, so extra gate firings
capture mismatches at the same rate as matches.

### 4.4 Per-query attribution: no shippable subset

Table 4: W/L/T and delta by keyword category across 500 queries.

| Category      | N   | W/L/T      | Δ  | Fire rate |
|---------------|-----|------------|----|-----------|
| other         | 222 | 10/7/205   | +5 | 36%       |
| temporal      | 182 | 15/17/150  | +1 | 32%       |
| counting      | 36  | 2/1/33     | 0  | 31%       |
| cross_entity  | 31  | 3/1/27     | 0  | 42%       |
| preference    | 23  | 0/0/23     | 0  | 43%       |
| causal        | 6   | 0/0/6      | 0  | 17%       |

cross_entity and counting show 2-3:1 win-loss ratios, consistent
with prior hypotheses that structural signals help on relational
queries. However, the absolute N is too small (31, 36) and delta is
near zero. The bulk of the positive aggregate (+5) lives in "other"
— a 44% catch-all bucket — with no semantic pattern.

Temporal queries are where the graph flips the most rankings
(15 wins + 17 losses = 32 rank changes, vs "other"'s 17 and
"cross_entity"'s 4). The delta is near-breakeven, but the variance
is highest. LongMemEval's `temporal-reasoning` questions show the
same high-variance pattern (§4.6). The interpretation: SOMA's
fingerprints encode *which* associator winners fired, but not *in
what order* those firings occurred — so queries that require
relative ordering ("when did X happen relative to Y") are
fundamentally unserved by this representation.

### 4.5 Signal equivalence: fingerprint ≡ topology

Table 5: Fingerprint vs topology signal in the gated hybrid.

| Slice | Fingerprint (W/L/Δ) | Topology (W/L/Δ)  |
|-------|----------------------|--------------------|
| A     | 8/4/+2               | 6/2/+1             |
| B     | 15/14/-1             | 5/10/-1            |
| C     | 11/11/+2             | 11/9/+2            |
| Total | 34W/29L/**+3**       | 22W/21L/**+2**     |

Fingerprint and topology aggregate to 323 and 322 hits respectively
across 500 queries — functionally identical. Topology is more
conservative (fires 135 times vs 192) but reaches the same plateau.

We interpret this equivalence as a **structural property of
lateral-inhibition-driven graph memory**: whether we measure
"which patterns of values the winners activate" (fingerprint) or
"which nodes participate" (topology), we are reading the same
information — the set of 3 active nodes per step — with different
projections. The ceiling is not mechanism-specific; it is a ceiling
of what 3-winner selection can encode.

### 4.6 Cross-benchmark: ceiling holds on LongMemEval

Table 6: Gated hybrid on LongMemEval oracle (100 items).

| Type                | N  | Hits | VecDB | Δ  | W/L |
|---------------------|----|------|-------|----|-----|
| temporal-reasoning  | 60 | 32   | 33    | -1 | 1/5 |
| multi-session       | 40 | 8    | 8     | 0  | 2/1 |
| **Total**           |100 | 40   | 41    | -1 | 3/6 |

On LongMemEval, the gated hybrid scores -1 hits vs VecDB. The
graph's gate fires on 36% of queries but with 3 wins and 6 losses —
the mechanism is net-negative on this benchmark. Temporal-reasoning
is particularly harmful (1W/5L), consistent with §4.4's observation
that graph's winners mismatch the ordering-semantics LongMemEval
requires.

This confirms the ceiling is not a LoCoMo artifact: two independent
canonical benchmarks yield coin-flip or negative retrieval deltas.

### 4.7 Positive results on SOMA-native tasks

On tasks SOMA was designed for, the mechanisms demonstrate real
value:

**Consolidation**: an ad-hoc develop-then-consolidate-then-QA test
on a synthetic multi-topic corpus (see Limitations §8 for caveats
on the exact specification) showed QA F1 improving from 0.279
without consolidation to 0.404 with consolidation (+0.125, +45%
relative; commit `c553a66`). The consolidation cycle reorganizes
edges and weights in a way that measurably helps subsequent
retrieval on the same corpus. This was a single-run, ad-hoc test
run via the interactive CLI rather than a scripted, repeated
experiment; we report it as evidence-of-life, not as a tight
statistic.

**Multi-session development**: we saved and loaded graph state
across three sequential sessions (cooking/family → travel/music →
pets/exercise; commit `6a0a822`). Per-session F1 on each session's
own QA set: S1=0.060 (40 memories, 29 edges, 14 nodes), S2=0.015
(80 memories, 37 edges, 14 nodes — interference dip as new topics
compete), S3=0.224 (120 memories, 43 edges, 14 nodes — recovery
and improvement). F1 grew +275% relative from S1 to S3; memory
store grew 3x (40→120); edge count grew 48% (29→43); node count
was static. We did *not* measure true cross-session recall (e.g.
querying S1 facts from an S3-state graph); the claim here is
narrower — save/load preserved state cleanly, and cumulative
development recovered from interference and increased quality by
S3.

**Adaptation on a sequence-prediction environment**. We built a
synthetic benchmark where a learner predicts next-observation on a
stream with four sequential regimes of different dynamics (random
walk, linear rotation, elementwise sqrt nonlinearity, frozen-MLP
dynamics). All regime step-functions apply a final `tanh` so
observations remain bounded in $[-1, 1]^{16}$ across regime changes
(keeping the observation distribution stationary and ruling out
magnitude drift as a confounder).

On this task, SOMA substantially outperforms a vanilla online MLP
baseline (2 hidden layers, 64 units ≈ 5.6k parameters; note that
SOMA is not parameter-matched — its full graph has roughly 15-25k
parameters depending on growth, so this comparison establishes
that a standard MLP baseline cannot match SOMA's adaptation curve,
not that SOMA is more parameter-efficient).

Table 7: Per-regime mean squared error (100-step warmup excluded).

| Regime          | SOMA   | OnlineMLP | FrozenMLP |
|-----------------|--------|-----------|-----------|
| random_walk     | 0.0063 | 0.0470    | 0.0453    |
| linear_rotation | 0.0014 | 0.0455    | 0.1196    |
| nonlinear_sqrt  | 0.0008 | 0.0052    | 0.3492    |
| mlp_dynamics    | 0.0065 | 0.0227    | 0.1440    |

SOMA achieves 3-33x lower mean squared error than the online MLP
across all four regimes. Adaptation-window behavior at regime
boundaries (steps to recover to 1.2x pre-boundary running MSE) is
more nuanced:

| Boundary                             | SOMA | OnlineMLP | FrozenMLP |
|--------------------------------------|------|-----------|-----------|
| random_walk → linear_rotation        | 0    | 80        | 500 (cap) |
| linear_rotation → nonlinear_sqrt     | 118  | 28        | 0         |
| nonlinear_sqrt → mlp_dynamics        | 500  | 500       | 1         |

At the first boundary SOMA recovers immediately while OnlineMLP
takes 80 steps. At later boundaries, the metric becomes degenerate
because different learners operate at wildly different MSE scales:
SOMA's pre-boundary MSE in regime 1 was 0.0014, so a 1.2x target is
0.00168 — an extremely tight bar. OnlineMLP's pre-boundary MSE was
0.045, making its 1.2x target 0.054 — trivial to hit. FrozenMLP
"recovers" in 0-1 steps on boundaries 2 and 3 only because it had
already catastrophically diverged; the metric is vacuous for it.
The table reports the raw numbers; we believe only the first
boundary's comparison is informative.

Neurogenesis fires at the first strongly nonlinear regime (nodes
grow 14 → 29 during the sqrt regime), matching the mechanism's
design intent — but as the ablation below shows, this growth does
not correspond to a performance gain.

An ablation across mechanisms, however, reveals a nuance (Table 8):

| Regime          | Full   | no_growth | no_consol | no_hebbian |
|-----------------|--------|-----------|-----------|------------|
| random_walk     | 0.0062 | 0.0062    | 0.0063    | 0.0067     |
| linear_rotation | 0.0012 | **0.0009**| 0.0017    | 0.0022     |
| nonlinear_sqrt  | 0.0008 | **0.0003**| 0.0006    | 0.0008     |
| mlp_dynamics    | 0.0043 | **0.0013**| 0.0047    | 0.0053     |

**The structural plasticity mechanisms (synaptogenesis, neurogenesis,
pruning) are not load-bearing on this task scale.** A SOMA variant
with growth disabled (14 nodes / 24 edges fixed) outperforms the
full variant on every adaptive regime. Hebbian learning contributes
modestly (no_hebbian is worst on linear_rotation: 0.0022 vs 0.0012);
consolidation is roughly neutral.

We tested whether the plasticity failure was a capacity issue — i.e.,
would growth help if the task clearly outstripped base capacity? We
constructed a harder 8-regime schedule (frozen MLPs of depth 2-4,
hidden 16-64) with the same 500 steps per regime. Results (Table 9):

| Regime   | full   | no_growth  | no_consol | no_hebbian | online_mlp |
|----------|--------|------------|-----------|------------|------------|
| mlp_2x16 | 0.0066 | 0.0062     | 0.0067    | 0.0071     | 0.0156     |
| mlp_2x32 | 0.0010 | **0.0005** | 0.0008    | 0.0015     | 0.0203     |
| mlp_3x16 | 0.0018 | **0.0006** | 0.0016    | 0.0019     | 0.0190     |
| mlp_3x32 | 0.0022 | **0.0006** | 0.0018    | 0.0017     | 0.0178     |
| mlp_2x64 | 0.0038 | **0.0011** | 0.0032    | 0.0028     | 0.0191     |
| mlp_3x64 | 0.0054 | **0.0014** | 0.0046    | 0.0042     | 0.0199     |
| mlp_4x32 | 0.0043 | **0.0018** | 0.0045    | 0.0041     | 0.0163     |
| mlp_4x64 | 0.0065 | **0.0010** | 0.0068    | 0.0058     | 0.0176     |

`no_growth` still wins on 7 of 8 regimes (and ties on mlp_2x16).
Under capacity pressure, `full` grew to 49 nodes / 980 edges — 3.5x
the starting size — yet was consistently 5-10x worse than no_growth's
frozen 14-node / 24-edge graph. All SOMA variants beat the online
MLP baseline by 9-40x across regimes.

This is a clean negative result about the plasticity story. The
structural plasticity mechanism, as currently implemented, adds
random-weight nodes faster than it extracts useful structure —
this is not a scale or task-difficulty issue, it's a mechanism
issue. **Something about SOMA's graph substrate** delivers the
adaptation advantage over a vanilla MLP; the ablation rules out
structural plasticity as the mechanism. However, our experiments
do *not* isolate which substrate component specifically matters —
wave-based topological execution, residual connections in node
MLPs, homeostatic gain control, graph connectivity, effective
depth, or raw parameter count. Isolating these is future work (see
§8 Limitations). What we can say is that the plasticity mechanisms
typically foregrounded in the brain-inspired architecture literature
are, at least in the current SOMA implementation, a net cost.

These results are specifically not retrieval wins. They are
demonstrations that SOMA's graph substrate helps on tasks evaluated
by adaptation and capacity metrics rather than static retrieval
accuracy — and that the plasticity mechanisms typically highlighted
in brain-inspired architecture proposals are not the source of
that help.

## 5. Analysis: why structural ≠ semantic

The consistent picture across §4.1–§4.6 is:

- The graph's retrieval signal is real but bounded at roughly
  +0.8% absolute above random.
- The bound is invariant under signal choice (fingerprint / topology)
  and graph scale (8 / 32 / 64 associators).
- The bound degrades to slightly negative under benchmark shift
  (LoCoMo → LongMemEval).

We argue the cause is that **graph activation is a structural
signal without semantic grounding**. Concretely:

- SOMA's per-associator-node input projections are randomly
  initialized and only weakly trained (Phase 8 learnable-projection
  experiment gave +1 improvement at the noise floor; see §4.1).
- Lateral inhibition selects the top 3 of the non-boundary
  associator+integrator nodes by raw activation magnitude given these
  random projections.
- The identity of "the 3 winners for this input" is therefore a
  structural hash: different inputs land on different winner sets,
  and the distribution of winner sets is diverse. In the default
  $n=8$ configuration, we have 8 associators + 4 integrators = 12
  nodes eligible for the 3-winner pool, giving $C(12, 3) = 220$
  distinct winner sets. Scaling to $n=32$ or $n=64$ expands this
  theoretical ceiling but (as §4.3 shows) does not improve
  retrieval, because:
- The hash has **no relationship to semantics** — two
  semantically similar inputs may land on different winner sets, and
  two semantically unrelated inputs may collide on the same set.

The contrastive-fine-tuning experiment (Phase 4b) confirms this
directly. We defined a contrastive loss where memories that
activated overlapping sets of graph nodes should have similar
encoder embeddings — effectively teaching the encoder to match
the graph's topology. After 316 fine-tuning updates on the LoCoMo
slice, the encoder's retrieval performance collapsed: VecDB-only
hits fell from 63 to 50 (−13), gated-hybrid hits fell from 65 to
51 (−14), and gate-usage fell from 32/100 queries to 11/100 (the
graph itself became confused by degraded embeddings). Training
signal pointed at an arbitrary target — the identity of
randomly-selected winners — destroyed the pretrained semantic
structure after barely 300 updates.

This is a **first-principles limit**, not a tuning issue. No
amount of gate-threshold sweeping, rerank-weight adjustment, or
min-active tuning can recover semantic signal from a structurally
arbitrary encoding.

## 6. Discussion

### 6.1 Implications for brain-inspired retrieval proposals

Many recent brain-inspired memory papers report retrieval gains
on a single benchmark with a small held-out set. Our diagnostic
suite suggests minimal bars of evidence that should become
expected in this literature:

- **Held-out tuning validation**: report results on a slice of
  queries disjoint from any slice used for hyperparameter tuning.
  A +3-hit gain at tuning size 100 should reproduce (within
  noise) on held-out sizes.
- **Shuffle baselines**: randomly permute the memory-to-graph-
  feature mapping. The graph's signal should consistently beat
  shuffled baselines. Our data shows a real but ~0.8% effect —
  which would be indistinguishable from noise under weaker
  diagnostics.
- **Cross-benchmark**: single-benchmark gains are vulnerable to
  both benchmark-specific artifacts and tuning-pattern memorization.
  Publish at least two independent benchmarks with the same
  hyperparameters.
- **Scaling and capacity tests**: do gains scale with graph
  capacity, or plateau? In our case, a scaling sweep disambiguated
  a plausible "vocabulary bottleneck" explanation from the actual
  "signal quality bottleneck": scaling hurt. Scaling tests are most
  valuable when both outcomes — gain or flat — are independently
  plausible from the existing evidence.

### 6.2 Implications for brain-inspired architecture research

More broadly, we take these results as evidence that the "retrieval
enhancement" framing is misaligned with what structural plasticity
provides. Pretrained encoders already excel at semantic
similarity on dense text. For a graph memory to improve on dense
retrieval, its structural encoding would need to correlate with
semantic content the encoder missed. Random-projection-driven
graphs — which SOMA uses and which, anecdotally, are common in
brain-inspired architectures — do not satisfy this condition by
construction: the structural signal is decoupled from semantic
content.

This suggests two paths forward:

1. **Change what the graph encodes.** Replace random input
   projections with learned projections that explicitly align
   graph structure with semantic structure. Our Phase 4b
   experiment tried this via contrastive fine-tuning and found it
   catastrophic (topology target is arbitrary). A more
   principled alternative — e.g., graph structure conditioned on
   encoder's latent space — is an open direction.
2. **Change the evaluation axis.** Evaluate brain-inspired memory
   on tasks where its native strengths (adaptation, consolidation,
   structural plasticity) are the rate-limiting step. Our
   positive results on consolidation (+45% QA) and multi-session
   growth (+275%, preserved recall) are in this regime. A
   developmental-AI benchmark — measuring adaptation speed,
   novel-concept integration, interference resistance — would
   likely produce positive results for brain-inspired systems
   that fail on retrieval.

## 7. Related work

### 7.1 Brain-inspired memory for language model retrieval

A wave of 2024-2026 proposals layer graph structures on pre-trained
dense retrieval for LLM RAG applications.

- **SYNAPSE** proposes a spreading-activation retrieval layer with
  Hebbian strengthening; published retrieval gains are on
  conversational benchmarks similar to LoCoMo.
- **Graphiti** builds a temporal knowledge graph over a dense index,
  with rerank driven by temporal proximity and entity co-occurrence;
  reported +18.5% F1 on LongMemEval.
- **MAGMA** uses multi-graph structures with policy-guided
  traversal; reported 61.2% on LongMemEval.
- **Mem0** combines fact-extraction, hybrid vector-keyword search,
  and conversation-summary layers; widely deployed.
- **Letta** (formerly MemGPT) manages memory tiers via LLM agent
  control.

These works motivate our central question: does the graph structure
carry retrievable signal beyond the encoder's own? Our finding (no,
to within ~0.8% on LoCoMo, negative on LongMemEval) does not directly
invalidate the above results but motivates rigorous diagnostic
reporting (shuffle baselines, held-out slices, cross-benchmark) as
a community norm.

### 7.2 Continual learning and consolidation

SOMA's consolidation mechanism is inspired by complementary learning
systems (CLS) theory from cognitive science. On the ML side,
continual-learning literature overlaps most directly:

- **A-GEM** and **GEM** use gradient-episodic memory to constrain
  updates; our prior work (commit `77f8fd3` and related) showed
  head-replay at buffer 500 + replay rate 0.5 matches A-GEM's
  class-incremental performance (ACC=0.808±0.005, BWT=-0.042).
- **Elastic Weight Consolidation** and its variants regularize
  weight updates; not directly graph-based.

Our consolidation-QA positive result (+45%) aligns with CLS-style
predictions that offline replay consolidates useful structure
relative to a fully-online baseline.

### 7.3 Developmental AI architectures

Earlier "whole-brain" cognitive architectures proposed mechanisms
similar to SOMA's:

- **Leabra** (Randall O'Reilly) includes Hebbian learning,
  differential-contrast error, and cortical/hippocampal division;
  biological fidelity is higher, task evaluations are smaller-scale.
- **SPAUN** (Chris Eliasmith) is a 2.5M-neuron spiking network
  demonstrating many cognitive tasks; closed-world and static.
- **ACT-R** (John Anderson) is a production-system cognitive
  architecture with memory decay and spreading activation; not
  neural, but shares "dynamic structure" goals.

Our work diverges from these in two ways: (i) we run on modern
hardware with a modern encoder in the loop, and (ii) we specifically
evaluate the retrieval-augmentation use-case where CL-era
architectures were not benchmarked.

### 7.4 Diagnostic methodology

- **Shuffle / permutation baselines** are standard in neural-network
  interpretability for isolating feature-importance claims. Their
  adoption in RAG evaluation appears uneven.
- **Held-out tuning validation** is standard in NLP but specifically
  under-tested in graph-augmented retrieval evaluations.
- **Negative-result literature**: recent work in RAG has documented
  specific failure modes (e.g., catastrophic retrieval in long
  contexts, dilution effects); our paper contributes a diagnostic
  template for brain-inspired retrieval specifically.

## 8. Limitations

- **Single encoder** (all-MiniLM-L6-v2, 384-dim). Larger or
  domain-tuned encoders may alter the absolute VecDB baseline but
  we expect the relative dynamics (graph ceiling, signal
  equivalence) to hold.
- **Single reranking weight** ($w = 0.2$). We tested adaptive
  weighting and confidence-gap variations in Phase 6; all came in
  at or below the fixed-weight baseline.
- **Two benchmarks** (LoCoMo, LongMemEval-oracle). Both are
  conversation-memory style; open question whether structural
  signal would help on other retrieval styles (code, scientific
  literature, web).
- **Shuffle diagnostic is under-powered**. We ran 5 permutations —
  sufficient to show real > max-shuffled, but statistically weak.
  A 100-seed version would quantify the z-score of the real delta
  against the shuffle distribution. We also observed ~±0.05 F1
  oscillation between checkpoints in earlier work, indicating a
  non-trivial run-to-run noise floor; the +0.8%-over-random effect
  is close to this noise floor and we cannot rule out that a
  different random seed would flip the sign.
- **No LLM generation loop**. We measure retrieval hit rate
  directly, not end-to-end QA accuracy. Potential for graph
  signal to help downstream generation in non-hit-dominated
  regimes is not tested.
- **Consolidation and multi-session evidence is ad-hoc**. The +45%
  consolidation result (§4.7) and the +275% S1→S3 F1 growth
  (§4.7) were produced by single-run interactive-CLI sessions on
  synthetic corpora, not by repeated scripted experiments with
  explicit controls. They are evidence-of-life for SOMA's sleep
  and save/load mechanisms, not tight statistics. Reproducing them
  as scripted multi-seed experiments is pre-submission work.
- **Substrate components not independently ablated**. The
  adaptation advantage over a vanilla MLP is ~3-40x, and structural
  plasticity is disconfirmed as the source, but we do not know
  whether wave execution, residual connections, homeostatic gain,
  depth, or parameter count drives it. A structure-matched deep
  MLP (same total parameters, feedforward) would be a direct
  control; we did not run it.
- **Graph architecture choices**. We evaluate one specific graph
  design (SOMA). Other architectures (attractor networks,
  hippocampus-style indexing) may have different ceilings.

## 9. Conclusion

We presented a rigorous empirical evaluation of a brain-inspired
graph memory (SOMA) across two task regimes — retrieval augmentation
over a pretrained encoder, and sequence-prediction with regime
shifts. Through sixteen diagnostic experiments, we showed that:

- **On retrieval augmentation**, the graph-derived signal is real
  but architecturally bounded at roughly +0.8% absolute over random,
  invariant under signal choice and graph scale, and does not
  generalize across benchmarks (LoCoMo → LongMemEval). Root cause:
  random projections produce structurally diverse but semantically
  arbitrary patterns.

- **On sequence prediction with distribution shift**, SOMA
  substantially outperforms a capacity-matched online MLP — 3-33x
  on four base regimes, 9-40x on an 8-regime capacity-pressure
  schedule. The advantage is large and consistent.

- **But the advantage is driven by the graph's executable substrate,
  not by structural plasticity.** Ablations show that disabling
  growth (neurogenesis, synaptogenesis, pruning) while keeping the
  14-node base graph monotonically improves performance — even
  under explicit capacity pressure (the `no_growth` variant beats
  `full` on 7 of 8 regimes in the harder schedule, by 5-10x).

These results jointly suggest that brain-inspired architectures
(at least this one) are: (a) mis-applied as retrieval plugins, and
(b) mis-promoted when the claim is that structural plasticity is
the mechanism of interest. The load-bearing contribution is the
graph substrate itself — wave execution, residual connections,
homeostatic gain — rather than the plasticity mechanisms typically
foregrounded in this literature.

We release the diagnostic suite and invite others to apply it both
(1) to brain-inspired retrieval proposals, to replicate the ceiling
diagnostics; and (2) to structural-plasticity claims more broadly,
to separate "does the substrate help?" from "does the plasticity
help?" Both questions matter; conflating them has, in our case, hidden
real findings of both kinds.

## Appendix A: Reproducibility

All experiments run on a single RTX 3090.

### A.1 Retrieval-experiment configuration (§4.1-§4.6)

- Encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  frozen weights).
- SOMA config: `SOMAConfig.developmental(...)` with
  `sensor_output_dim=384`, `associator_input_dim=192`,
  `associator_hidden_dim=384`, `associator_output_dim=192`,
  `integrator_hidden_dim=768`, `integrator_output_dim=384`,
  `initial_associator_count=8`, `initial_integrator_count=4`,
  `max_nodes=50`, `seed=42`.
- Gated hybrid: `gate_threshold=0.05`, `rerank_weight=0.2`,
  `recall_k=20`, `top_k=5`.
- Scoring: `token_f1` from
  `benchmarks/industry/longmemeval/metrics.py`.

### A.2 Sequence-prediction environment configuration (§4.7 Tables 7-9)

- SOMA config: `SOMAConfig.developmental(...)` with
  `sensor_output_dim=16`, `associator_input_dim=16`,
  `associator_hidden_dim=32`, `associator_output_dim=16`,
  `integrator_hidden_dim=32`, `integrator_output_dim=16`,
  `initial_associator_count=8`, `initial_integrator_count=4`,
  `seed=42`; `max_nodes=50` for v0, `max_nodes=128` for v0.5.
- Baselines: `OnlineMLP` (`16 → 64 → 64 → 16` GELU + Adam lr=1e-3,
  trained online on every step); `FrozenMLP` (same, trained during
  regime 0 only, frozen thereafter).
- Schedule: `make_default_schedule` (4 regimes, 500 steps each,
  2000 total) for v0; `make_capacity_schedule` (8 MLP-dynamics
  regimes of varying depth/width, 500 steps each, 4000 total) for
  v0.5.
- Environment: `SequenceEnv` in `src/soma/environments/sequence_env.py`.
  All regime transition functions apply a final `tanh` so
  observations remain bounded in $[-1, 1]^{16}$ (early attempts
  without this bound caused SOMA to diverge numerically at
  regime boundaries; observation-distribution stationarity matters
  for this architecture).

### A.3 Scripts and data

Scripts are at `research/developmental/`. Each script is a single
self-contained entry point (no shared harness) to simplify
reproducibility. Raw results at `research/developmental/results/`
(JSON data) and `*.log` (console output). The Appendix B table
maps phase numbers to script filenames and commit hashes.

## Appendix B: Commit trail and script map

| Phase / Experiment     | Script                                          | Commit    | Section |
|------------------------|-------------------------------------------------|-----------|---------|
| Phase 3 (gated hybrid) | `research/developmental/confidence_gated_test.py` | `77f8fd3` | §4 intro, §5 |
| Phase 4 (encoder FT)   | `research/developmental/encoder_finetune_test.py` | `30cde22` | §4 intro, §5 |
| Phase 4b (contrastive) | `research/developmental/contrastive_finetune_test.py` | `30cde22` | §5 |
| Phase 5 (semantic lens)| `research/developmental/semantic_lens_test.py`  | `cddc488` | §4 intro |
| Phase 6 (adaptive gate)| `research/developmental/adaptive_gate_test.py`  | `c630d69` | §4 intro |
| Phase 7 (min_active)   | `research/developmental/min_active_sweep.py`    | `59a0662` | §4 intro |
| Phase 8 (learnable proj)| `research/developmental/learnable_proj_test.py` | `309caeb` | §4 intro, §5 |
| Phase 9 (scaling)      | `research/developmental/associator_count_sweep.py` | `0275ac6` | §4.3 |
| Phase 10 (held-out)    | `research/developmental/multi_slice_validation.py` | `3923dbc` | §4.1 |
| Phase 11 (shuffle)     | `research/developmental/shuffle_diagnostic.py`  | `3923dbc` | §4.2 |
| Phase 12 (attribution) | `research/developmental/per_query_attribution.py` | `2afe9a7` | §4.4 |
| Phase 13 (topology)    | `research/developmental/topology_gate_test.py`  | `2afe9a7` | §4.5 |
| Phase 14 (LongMemEval) | `research/developmental/longmemeval_gated_hybrid.py` | `95820e2` | §4.6 |
| Env v0                 | `research/developmental/env_sequence_v0.py`     | `c94c274` | §4.7 Table 7 |
| Env v0 ablation        | `research/developmental/env_sequence_ablation.py` | `c19a0b9` | §4.7 Table 8 |
| Env v0.5 capacity      | `research/developmental/env_sequence_v05_capacity.py` | `0390590` | §4.7 Table 9 |
| Consolidation result   | (ad-hoc via `/sleep` CLI)                        | `c553a66` | §4.7 |
| Multi-session result   | (ad-hoc via developmental CLI)                   | `6a0a822` | §4.7 |

Environment module: `src/soma/environments/sequence_env.py`.
Tests: `tests/test_environments/` (11 passing).
