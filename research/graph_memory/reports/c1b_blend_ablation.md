# C1b — Blend-Formula Ablation

Sub-phase C1b of research direction C (`docs/plans/2026-04-16-research-c-graph-memory.md`). Follow-up to the C1 FAIL verdict (see `research/graph_memory/reports/c1_synthetic_signal.md`).

**Run date:** 2026-04-16
**Embedder:** `sbert` (dim=384)  
**Device:** `cpu`  
**Corpus:** 60 snippets, 20 triples (20 transitive queries + 60 direct + 10 unrelated)  
**Primary consolidation N:** `10`  
**Secondary consolidation N:** `skipped`  
**α sweep:** `[0.1, 0.3, 0.5, 0.7, 0.9]`  
**Total wall clock:** 293.0s

## Headline gate decision

**Verdict: FAIL**

No readout clears the +0.05 gate. Best was `pure_vector_trained` at +0.000 vs pure-vector transitive R@5. Declare research direction C dead for retrieval objectives; recommend folding to Research B (CL benchmark) or D (associative memory). Actively-harming modes: alpha_blend_0.90 (-0.100), pure_graph_rank (-0.075), alpha_blend_0.30 (-0.050).

**Best mode:** `pure_vector_trained`  
**Best Δ R@5 vs pure-vector:** `+0.000`

## Transitive queries (load-bearing)

This is the class C1b is measuring. Direct queries should tie (sanity), unrelated should both score 0 (hallucination check), but transitive is where ONLY the graph substrate has a path to the correct answer — a winning readout must beat pure-vector here by ≥ +0.05.

| Mode | N | R@1 | R@5 | MRR | Δ R@5 vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure_vector (baseline) | - | 0.000 | 0.200 | 0.068 | (ref) |
| pure_vector_trained | 10 | 0.000 | 0.200 | 0.068 | +0.000 |
| alpha_blend_0.10 | 10 | 0.000 | 0.200 | 0.066 | +0.000 |
| graph_traversal_expand | 10 | 0.000 | 0.200 | 0.068 | +0.000 |
| centrality_prior | 10 | 0.000 | 0.200 | 0.066 | +0.000 |
| alpha_blend_0.30 | 10 | 0.000 | 0.150 | 0.056 | -0.050 |
| alpha_blend_0.50 | 10 | 0.000 | 0.150 | 0.072 | -0.050 |
| alpha_blend_0.70 | 10 | 0.025 | 0.150 | 0.085 | -0.050 |
| pure_graph_rank | 10 | 0.000 | 0.125 | 0.060 | -0.075 |
| alpha_blend_0.90 | 10 | 0.050 | 0.100 | 0.075 | -0.100 |

## Direct queries (sanity — should tie)

| Mode | N | R@1 | R@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| pure_vector (baseline) | - | 0.521 | 1.000 | 0.971 |
| pure_vector_trained | 10 | 0.521 | 1.000 | 0.971 |
| alpha_blend_0.10 | 10 | 0.529 | 1.000 | 0.979 |
| alpha_blend_0.30 | 10 | 0.504 | 0.975 | 0.956 |
| alpha_blend_0.50 | 10 | 0.408 | 0.933 | 0.857 |
| alpha_blend_0.70 | 10 | 0.317 | 0.767 | 0.742 |
| alpha_blend_0.90 | 10 | 0.129 | 0.479 | 0.400 |
| pure_graph_rank | 10 | 0.058 | 0.287 | 0.224 |
| graph_traversal_expand | 10 | 0.521 | 1.000 | 0.971 |
| centrality_prior | 10 | 0.521 | 1.000 | 0.971 |

## Unrelated queries (hallucination check — should be ≈0)

| Mode | N | R@1 | R@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| pure_vector (baseline) | - | 0.000 | 0.000 | 0.000 |
| pure_vector_trained | 10 | 0.000 | 0.000 | 0.000 |
| alpha_blend_0.10 | 10 | 0.000 | 0.000 | 0.000 |
| alpha_blend_0.30 | 10 | 0.000 | 0.000 | 0.000 |
| alpha_blend_0.50 | 10 | 0.000 | 0.000 | 0.000 |
| alpha_blend_0.70 | 10 | 0.000 | 0.000 | 0.000 |
| alpha_blend_0.90 | 10 | 0.000 | 0.000 | 0.000 |
| pure_graph_rank | 10 | 0.000 | 0.000 | 0.000 |
| graph_traversal_expand | 10 | 0.000 | 0.000 | 0.000 |
| centrality_prior | 10 | 0.000 | 0.000 | 0.000 |

## Substrate diagnostics

- Integrator count: `8` *(must be > 0 — post-860cc82 seed-graph fix)*
- Edge count after N=10: `48`
- Spearman ρ(edge strength vs pair co-occurrence): `0.8540979350605947`

For comparison, C1 saw ρ=+0.854 at N=10 on a 60-snippet corpus. If the number above is meaningfully different, the substrate's behaviour changed between experiments and that should be investigated separately.

## Honest interpretation


Across 9 readout modes — spanning pure-graph-rank, α-blend from 0.1 to 0.9, graph-traversal expansion, and centrality prior — NONE cleared the +0.05 gate. The substrate does capture co-occurrence (ρ≈0.85) but the captured signal does not translate into retrieval lift under any of the readouts tested.

Interpretation candidates (from most to least likely):

1. **The edge-strength distribution captures a STATISTICAL property (pair-count rank) but not the node-pair IDENTITY**. ρ is computed over *sorted* edge-strength and pair-count distributions — it tells us the top-strength edges align with the top-frequency pairs IN AGGREGATE, but not that *specific* edges correspond to *specific* pairs. A readout that relies on "edge (A,B) is strong because A and B co-occurred" has no edge-to-pair lookup table to exploit. This is a structural limit of the current substrate, not a blend-formula bug.
2. **The query-to-activation path (TextEncoder → SOMA graph → stored output) is too lossy to preserve pair-identity through retrieval**. The stored activations are the output of a graph-wide propagation, not a direct edge-weight readout; by the time the signal reaches the output layer, the pair-specific information is averaged across many nodes.
3. **The readouts tested are the wrong family**. We have NOT tested: learned blend (C3), query-time consolidation (C4), or direct edge-list retrieval (bypass activations entirely — query the graph edges by entity tokens, if we can index SOMA nodes by the text they respond to). That last one is the most promising untested option; it would require extending the substrate to expose a text-token → graph-node index, which is a sizeable engineering effort.

Recommendation: fold C to C3 with "direct edge-list retrieval" as the only remaining candidate readout, or declare C dead and redirect engineering to Research B or D. Which depends on strategic priority, not further evidence from C1 or C1b.

## Supplemental: 120-snippet corpus (plan-specified size)

The plan specifies 120 snippets. We ran that configuration as well.
At 120 snippets, pure-vector baseline transitive R@5 drops to **0.000**
(from 0.200 at 60 snippets). All 9 readout modes also produce 0.000.
This is a metric-ceiling issue: with more snippets, each entity appears
in ~5-6 snippets, and R@5 = hits_in_top_5 / total_relevant can only
reach ~0.83-1.0 IF the right snippets are all in the top-5. The sbert
embedder's template-matching prior (which is the ONLY transitive
signal source for pure-vector) dilutes proportionally.

The 60-snippet run is the fair C1-comparable setup where the baseline
CAN score above zero, and even there no readout beats baseline. The
120-snippet run simply confirms there's no hidden regime where the
graph signal emerges at larger corpus sizes.

Substrate diagnostic at 120 snippets: Spearman ρ = +0.646 (lower than
the 0.854 at 60 snippets — the co-occurrence signal partially washes
out as more snippets spread the pair-count distribution thinner, but
it's still positive and meaningful).

---

Generated by `research/graph_memory/run_c1b.py` on 2026-04-16 18:29:24.
Updated manually with 120-snippet supplemental data.
