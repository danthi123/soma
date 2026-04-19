# §4.7 Rewrite Candidate — Structural plasticity with locality

**Status:** Draft candidate, 2026-04-19. Based on multi-seed v0.5
experimental cascade (commits 2ba566b → c2a456b).

**What changes from the current §4.7:**
- Old framing: "structural plasticity damages prediction; `no_growth`
  wins on every regime." (Negative result, narrow.)
- New framing: "**random** structural plasticity damages prediction;
  **locality-constrained** plasticity matches or beats `no_growth`
  on every hard regime. Positional locality is a sufficient
  structural prior to turn synaptogenesis from harmful into
  beneficial."

---

## 4.7 Positive and negative results on SOMA-native tasks

[Keep the Consolidation and Multi-session paragraphs from the
existing draft verbatim — those findings are unchanged.]

### Adaptation on a sequence-prediction environment

[Keep Tables 7-9 and the online-MLP comparison unchanged — they
establish SOMA's substrate advantage over a vanilla MLP.]

### The plasticity-failure decomposition (replacing current sub-section)

An ablation across mechanisms originally appeared to isolate a
consistent story: structural plasticity (synaptogenesis + neurogenesis
+ pruning together) is net-negative on the v0.5 capacity schedule,
with `no_growth` winning 7/8 or 8/8 of regimes across every
configuration tested. We reported three follow-up experiments to
rule out mundane explanations:

1. **Trigger timing**: interval-based vs PE-gated firing (66% fewer
   events). No change in MSE.
2. **Initial edge magnitude**: new-edge weight scale swept over
   {0.01, 0.001, 0.0001, 0.0}. No change.
3. **Starting capacity**: frozen topologies at 14/25/49 nodes. More
   capacity → worse MSE (capacity is not the bottleneck).

These isolations pointed at the **admission mechanism itself**, not
parameters of how admissions enter. Synaptogenesis reads activation
correlations from random input projections, then installs edges
based on those arbitrary co-activations. Section 5 developed this
diagnosis further, identifying "structural without semantic" as the
common failure mode linking the §4.7 adaptation result to the §5
retrieval ceiling.

### Follow-up: is the failure about WHICH pairs get admitted?

A multi-seed cascade on the v0.5 8-regime capacity schedule tested
four candidate "correct" plasticity-adjustment mechanisms, each a
different grounded-signal hypothesis. Each was implemented as an
opt-in config flag and validated with TDD before benchmarking. All
results reported here are mean across 3 seeds {0, 1, 42} on the
same 8-regime schedule described in Appendix A.2. A variant is
scored as "wins" on a regime if its MSE beats the baseline by
more than the noise-floor margin of 0.00005.

**Direction 1 — PE-supervised synaptogenesis.** Gate admissions by
a per-pair EMA of prediction-error change on co-activation: admit
only pairs whose co-activation correlates with PE reduction.
Single-seed result (seed=42) showed 7/8 regimes improved over
unfiltered synap. But multi-seed validation destroyed the signal:
seeds 0 and 1 showed only 1/8 improvements each. The per-pair EMA
degenerates to a global PE-trend gate on the 14-associator graph
where every active pair is co-active every step, leading to
seed-dependent acceptance patterns. Mean effect across all seeds and
regimes: +0.0002 MSE (slightly harmful). **NEGATIVE.**

**Direction 2B — Learnable input projections.** Replace frozen
random projections with nn.Parameter values updated via prediction
loss. `full_learnable` was required to beat `full_frozen` on 5+/8
regimes for all three seeds. Observed: 4/8, 4/8, 1/8 across seeds.
On the 4×64 regime specifically, `full_learnable` REGRESSED by
+0.0012 on every seed — projection gradients co-adapt with the
prediction head in a way that interacts badly with aggressive
synaptogenesis. **NEGATIVE.**

**Direction 3 — Neuromodulator-style plasticity broadcast.** Scale
Hebbian and synaptogenesis rates by a scalar gain derived from
recent PE trajectory (high PE → more plasticity, low PE → less).
Per-seed scorecard: 5/8, 3/8, 0/8. A seed=0 single-seed preview
looked promising but did not survive validation. **NEGATIVE.**

These three directions shared a common premise: admission decisions
should be gated by a TEMPORAL signal (PE, PE gradient, PE × coactivation
EMA). The premise is wrong on this substrate, which leads to:

**Direction 5 — Positional-locality filter.** Reject candidate pairs
at admission time if their distance in the node-position space
exceeds a threshold. Motivated by analysis of why `neuro_only` was
the only multi-seed-positive mechanism on v0.5 (see Appendix for the
full causal analysis): neurogenesis places new nodes at the centroid
of active neighbors and wires them only to the 5 nearest existing
neighbors, giving each new edge a coherent spatial prior. The
hypothesis: if we apply the same discipline to synaptogenesis, its
admissions become useful instead of arbitrary.

Implemented as `synaptogenesis_max_distance`, a hard cutoff on the
Euclidean distance between candidate pairs. At cutoff=0.5 (roughly
half the mean pair distance under the default `position_jitter=0.1`),
the filter rejects all long-range candidates at admission time.

| Contrast | seed=0 | seed=1 | seed=42 |
|----------|--------|--------|---------|
| `synap_only_local` beats `synap_only` | **8/8** | **7/8** | **8/8** |
| `full_local` beats `full`             | **7/8** | **7/8** | **8/8** |

This is the **first multi-seed-validated positive plasticity-
adjustment mechanism on v0.5.** Effect sizes on hard regimes are
large: mean MSE reductions of 0.003-0.006 vs unfiltered synap, and
`synap_only_local` ties or beats `neuro_only` (the prior
single-mechanism positive) on most hard regimes.

Per-regime mean MSE (mean across 3 seeds):

| Regime    | no_growth | synap_only | synap_only_local | neuro_only |
|-----------|-----------|------------|------------------|------------|
| mlp_2x16  | 0.0075    | 0.0078     | 0.0076           | 0.0075     |
| mlp_2x32  | 0.0008    | 0.0012     | **0.0008**       | 0.0008     |
| mlp_3x16  | 0.0007    | 0.0015     | **0.0003**       | 0.0006     |
| mlp_3x32  | 0.0006    | 0.0017     | **0.0004**       | 0.0004     |
| mlp_2x64  | 0.0014    | 0.0042     | **0.0007**       | 0.0010     |
| mlp_3x64  | 0.0020    | 0.0062     | **0.0012**       | 0.0011     |
| mlp_4x32  | 0.0018    | 0.0048     | **0.0012**       | 0.0012     |
| mlp_4x64  | 0.0011    | 0.0058     | **0.0004**       | 0.0004     |

The locality-filtered synap uses 2-3× fewer edges than unfiltered
synap but achieves 5-10× lower MSE on the hard regimes. The edges
that survive the filter are useful precisely because they respect
a structural prior (positional neighborhood) rather than an
arbitrary pattern (activation-correlation in random-projection space).

### Ruling out sparsity as a confound

An obvious concern: `synap_only_local` admits fewer edges (32-64
total) than `synap_only` (158). Could the benefit come purely from
fewer edges, not their specific locality? We ran four sparsity-
adjacent controls to isolate the effect:

1. **Per-call admission cap** (cap1, cap2): limit admissions per
   synaptogenesis step to K random picks from the passing pool.
   Result: at matched total admissions (158 edges via delayed fills),
   `synap_only_local` still wins 5-8/8 across seeds.
2. **Rate reduction** (r063, r030): reduce `synaptogenesis_rate`
   from 2.0 to 0.63 and 0.30. Result: total admissions unchanged
   (probability clamps to 1.0 for high-coact pairs at these rates);
   MSE perturbed but still far from locality's profile.
3. **Locality cutoff sweep**: vary the filter aperture across
   {0.25, 0.50, 0.75, 1.00}. Result: a clean **inverted-U** with
   a sharp peak at cutoff=0.5. Too tight (0.25) rejects all pairs
   and reproduces `no_growth`; too loose (0.75, 1.00) admits most
   pairs and reproduces `synap_only`.

The inverted-U is the crucial evidence: if sparsity alone drove the
benefit, tighter cutoffs (fewer edges) should monotonically help.
Instead, cutoff=0.50 beats both tighter and looser values — the
operating point matches the expected pair-distance distribution.
The filter's benefit is specifically spatial; sparsity per se does
not transfer.

### Mechanism probe: specific positions vs coherent-metric-only

A natural question about the locality filter: does the benefit come
from the SPECIFIC initial positions carrying information, or from
ANY coherent distance metric being applied at admission time? We
tested by scrambling positions immediately after node construction,
preserving the distance distribution (same ``randn * 0.1`` jitter)
but breaking any correlation between position and other node
attributes. Per-seed wins for `synap_only_local_scramble` vs
`synap_only_local` (matched rng, only position mapping changed):

| Contrast | seed=0 | seed=1 | seed=42 |
|----------|--------|--------|---------|
| scramble beats unfiltered synap_only | 0/8 | 7/8 | 8/8 |
| scramble matches or beats local      | 0/8 | 0/8 | 5/8 |

On two of three seeds, scramble destroys most of the locality
benefit. The scrambled-position filter still reduces admissions
(coherent-metric effect exists) but the SPECIFIC admitted pairs are
no longer useful. On seed=42 alone, scramble approximately matches
local — an anomaly we flag plainly; it may reflect either seed-
specific position-projection correlation strength or a lucky
scrambled-distance distribution. A larger seed sweep would clarify.

Likely mechanism: at node construction, positions and projection
weights are drawn from the same torch rng state in sequence. Nodes
whose positions are close by Euclidean distance also tend to have
correlated projection responses. The locality filter admits pairs
of response-correlated units — edges between them carry genuine
structure. Scrambling severs this correlation and the filter loses
its signal even though its geometry is unchanged.

This is a testable prediction: explicitly correlating position and
projection init (e.g., deriving positions from PCA of projection
weights) should strengthen the effect; fully decorrelating them
should weaken it. We leave these as follow-ups but note that the
scramble result already constrains the mechanism beyond "any
spatial filter works."

### Interpretation

The three failed directions (PE-gating, learnable projections,
plasticity broadcast) and the one successful direction (positional
locality) differ in which dimension of plasticity they constrain:

- Directions 1, 2B, 3 constrain admissions by a TEMPORAL signal
  (PE, PE × coactivation, PE gradient).
- Direction 5 constrains admissions by a SPATIAL structural prior
  (position-space distance).

The spatial filter works where temporal filters fail because SOMA's
random input projections destroy any temporal correlation structure
that maps cleanly to useful graph topology. Co-activation on random
projections is arbitrary noise, and filtering it by a temporal
signal (themselves derived from the same noisy activations) doesn't
recover the signal. A spatial prior is orthogonal to the projection
noise and provides structure that synaptogenesis can exploit.

Generalizing:

> **Structural plasticity requires structural priors. Pattern-based
> priors (co-activation, PE correlation) are insufficient on
> substrates where patterns are themselves generated by arbitrary
> upstream randomness.**

This is consistent with Section 5's "structural without semantic"
diagnosis of the retrieval ceiling: both failure modes arise from
building mechanisms on arbitrary upstream representations. The
§4.7 fix (positional locality as a structural prior) does not
address the §5 ceiling directly because retrieval scores read
cosine similarity on the random projections themselves. But the
design principle is the same: when upstream is random, downstream
mechanisms must introduce external structure.

### Negative results that stand

The current §4.7's follow-up experiments (timing-isolation,
magnitude-isolation, capacity-isolation) still stand as reported.
Their conclusion — "the failure is in WHICH pairs get admitted, not
WHEN or HOW STRONGLY they enter" — is precisely the question the
locality result answers. The current §4.7's negative framing
("plasticity on SOMA doesn't add useful structure under this
substrate") was correct as of the data we had; the locality result
qualifies it without overturning it. What we now know is that the
failure is about missing a SPATIAL admission prior, and that adding
one recovers (and exceeds) the `no_growth` baseline on every hard
regime.

### Scope: the locality result is substrate-specific

To preempt overclaim: we attempted to replicate the locality benefit
on a semantic-retrieval workload (LoCoMo, 5882 turns / 1982 queries;
also a synthetic 50-fact probe). Both tests showed neutral effect
on retrieval Recall@k when the locality filter was toggled on/off.
On LoCoMo specifically, all three tested SOMA variants (flat cosine,
graph re-rank without locality, graph re-rank with locality) produced
identical Recall@1/5/10 to three decimal places. Two plausible
reasons:

1. **Retrieval reads activations directly, not via edge-propagated
   predictions.** The graph's structural signal is a secondary
   re-rank blended at α into a cosine score that already captures
   most of the retrievable information. Locality-filtered edges
   produce a cleaner graph but not necessarily a cleaner re-rank
   score.
2. **LoCoMo's cosine baseline is weak** (R@5 = 0.238 on sbert
   `all-MiniLM-L6-v2`). Many queries fail before the re-rank step
   can affect them. The graph contribution is effectively noise
   across all variants.

So the locality finding is a positive design principle on SOMA's
NATIVE prediction substrate (where edges carry the computation
directly). It does not, on current evidence, transfer to retrieval
workloads where SOMA is a secondary signal. We report this limit
plainly rather than framing the v0.5 result as a retrieval
improvement.

---

## Notes for the revision

- Four existing §4.7 paragraphs can be kept verbatim (consolidation,
  multi-session, Tables 7-9, online-MLP comparison).
- The PE-gated-neurogenesis follow-up (existing §4.7) becomes
  part of the "Direction N failures" block.
- The scale-sweep follow-up (existing §4.7) becomes part of the
  "magnitude isolation" block.
- The frozen-topology follow-up (existing §4.7) becomes part of
  the "capacity isolation" block.
- NEW content: the multi-seed cascade, the four sparsity controls,
  the locality-cutoff inverted-U, and the interpretation section.
- Section 5 needs a forward-reference to the locality result so
  the retrieval-ceiling discussion can cite the §4.7 design
  principle explicitly.

**Length**: ~1200 new words added, ~400 existing words repositioned.
Net ~+1000 words — substantial but proportional to the positive
result's importance.

**Figures to generate**:
1. Per-seed scorecard bar chart (local vs baselines, 3 seeds)
2. Cutoff inverted-U curve (win-rate vs cutoff value)
3. Edge count vs MSE scatter across all variants (showing locality
   occupies a distinct region — fewer edges, lower MSE)
