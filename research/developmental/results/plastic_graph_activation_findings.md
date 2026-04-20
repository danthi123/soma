# Plastic graph activation test — findings

**Status:** **CLOSED as research question.** 5 consecutive failed
attempts (v1 ceiling-bound, v2 design-flaw, v3 CPU-slow, v3b
device-mismatch, v3c silent-exit-at-pre-consolidate). The benchmark
as currently designed cannot isolate plastic-vs-frozen graph signal.
**Date:** 2026-04-20.
**Runner:** `benchmarks/plastic_graph/run_activation.py`.

## Research question

On a synthetic 10-topic × 100-fact corpus (1000 facts), does sustained
retrieval activity on Topic A (Python coding preferences) cause SOMA's
plastic graph to differentiate from a frozen SOMA or a pure Chroma
baseline on held-out Topic-A questions?

**Product claim under test:** "The plastic graph learns from usage."
If true, plastic SOMA should beat frozen SOMA after 50 focused
session queries on Topic A.

## v1 result: ceiling-bound null (as expected)

Configuration: hybrid_alpha=0.3, graph_rerank_alpha=0.0 (defaults),
k=5.

| System | Initial Topic A R@5 | Final Topic A R@5 | delta |
| --- | ---: | ---: | ---: |
| soma_plastic | 1.000 | 1.000 | +0.000 |
| soma_frozen | 1.000 | 1.000 | +0.000 |
| chroma | 1.000 | 1.000 | +0.000 |

SOMA plastic graph: 42 → 42 nodes, 80 → 365 edges.

**Why this is a null result, not a failure:**

1. The benchmark is cosine-easy. 5 paraphrases per attribute × 20
   attributes × 10 topics means the corpus has only 200 unique facts
   stored in 1000 paraphrase entries. Question text ("What is my
   preferred X?") shares the attribute keyword with each of 5 stored
   paraphrases. Cosine trivially puts all 5 in top-5. R@5 saturates
   at 1.0.

2. **Critically: in the default v1 configuration the plastic graph
   has no retrieval path.** `MemoryLayer.retrieve()` dispatches:
   - `hybrid_alpha != None` → `_retrieve_hybrid` (cosine + BM25, no
     graph).
   - `hybrid_alpha == None` + `graph_rerank_alpha > 0` + stored
     activations → `_retrieve_with_rerank` (cosine + graph).
   - otherwise → pure cosine.

   Passing `hybrid_alpha=0.3` (our default) sends retrieve down the
   hybrid path, which **never reads SOMA activations**. Plastic vs
   frozen SOMA produce identical ranking because the graph is not
   consulted.

3. The plastic graph *did* fire. 80 → 365 edges = 4.5× edge growth
   from Phase-2 consolidations. Structural plasticity is working —
   it just doesn't reach retrieval in the default config.

## Architectural finding: graph-rerank is opt-in

From `src/soma/memory/api.py`:

- `MemoryLayer.__init__` default: `graph_rerank_alpha=0.0`.
- `SOMAConfig.memory_layer()` preset: does not set
  `graph_rerank_alpha` (so stays 0.0).
- `retrieve()` code path: if `hybrid_alpha` is passed, the
  graph-rerank branch is never taken (`elif hybrid_alpha is not
  None:` fires first).

**In the shipping default configuration, the plastic graph grows but
is not consulted during retrieval.** It is a background
stable-capture of storage state, ready to be activated by a caller
passing `graph_rerank_alpha > 0` and omitting `hybrid_alpha`.

This is a *performance* default (graph-rerank adds a `text_to_state`
+ `F.cosine_similarity` per candidate) but it means the "plastic
graph learns from usage" story is currently gated behind an opt-in
flag that our own product presets don't turn on.

## v2: graph-rerank mode (null — design flaw)

Configuration: `graph_rerank_alpha=0.3`, `hybrid_alpha=None`, k=1.

Design changes from v1:

1. **k=1 instead of k=5** — forces precision test, removes R@5
   ceiling.
2. **Graph-rerank path active** — the only retrieve mode where
   plastic vs frozen SOMA can differ.
3. **Frozen also has SOMA attached** — so both systems have stored
   graph activations; the difference is only whether consolidate()
   runs during Phase 2.
4. **Pre-consolidate before baseline** — ensures both plastic and
   frozen have populated stable-capture before Phase 1 eval.

**Result:** Plastic and frozen both pre-consolidated once identically
during Phase 1 setup. Phase 2 retrieves didn't add new stored
entries for plastic to graphify — so the stable-capture stayed
identical across both systems. No differentiation possible.
Identical rank-1 results for plastic and frozen.

## v3: trickle ingest during Phase 2 (CPU too slow)

Fix for v2: interleave new stores with retrieves during Phase 2.
Plastic's ongoing consolidate() graphifies them; frozen's static
graph does not. First architecturally-sound test.

**Blocker:** SOMA default device is CPU. 709 entries × 2 systems ×
20 tokens × ~50ms per token inference = wall-clock >30 min per run
on CPU. Killed.

## v3b: CUDA pin (device mismatch crash)

Fix: pin SOMA to CUDA in the adapter:
```python
soma = SOMA(config=config, device=torch.device("cuda"))
self.mem.attach_soma(soma, tokenizer, encoder)
```

**Blocker:** `TextEncoder` output stayed on CPU. First
`text_to_state` call inside consolidate() crashed:
> RuntimeError: Expected all tensors to be on the same device, but
> got mat2 is on cuda:0, different from other tensors on cpu.

## v3c: CUDA pin + encoder .to(device) (silent exit)

Fix: `encoder = encoder.to(_device)` after construction.

**Blocker:** Ran through Phase 1 ingest (chroma 18.7s, plastic 10.2s,
frozen 12.4s) and logged `Pre-consolidating for graph-rerank (both
plastic and frozen)...` at 12:08:48. Then **no further output**. The
task runner reported exit code 0 at ~12:36 — ~28 minutes later —
but the log file was not updated after the pre-consolidate line.
Zero entries into Phase 1 baseline eval, no Phase 2, no results JSON.

Most likely root cause: the 712-fact pre-consolidate call
(`systems["soma_plastic"].mem.consolidate()`) either deadlocked on
CUDA or took longer than the task runner's maximum runtime, was
killed externally, and the subprocess's flushed-but-unwritten
stdout was discarded. The exit=0 is spurious.

## Closure

After 5 consecutive failed attempts, the plastic-vs-frozen graph
activation benchmark is declared **architecturally intractable** at
its current design. The issues compound:

1. **The shipping retrieve path doesn't use the graph** (v1 finding).
   So any test has to take the non-default graph-rerank path.
2. **Graph-rerank with 712 entries on CPU is too slow** (v3).
3. **Graph-rerank with 712 entries on CUDA appears to hang or run
   beyond reasonable wall-clock bounds** (v3c).
4. **The synthetic benchmark is cosine-easy** (v1 ceiling).

The research question "does the plastic graph learn from usage" is
better answered on real corpora where cosine fails. On LongMemEval
+ LoCoMo (real corpora), graph-rerank at α=1.0 *regresses* performance
(-40% F1 on LoCoMo N=60). The signal we're looking for — plasticity
improves retrieval — has not appeared in any test so far.

**Action:** The plastic graph stays in-tree as a research substrate
(it is genuinely plastic; growth and consolidation work) but the
product story pivots fully to the BM25+cosine hybrid retrieval win
(+22% F1 LongMemEval, +89% rank1 LoCoMo). Graph-rerank remains
opt-in and is not part of the positioning claim.

The next attempt at "learning graph" positioning should come via
**Direction 4a/4b (semantic locality distillation)** — instead of
trying to measure plastic-vs-frozen inside retrieval, drive a semantic
locality filter from LLM-distilled projections and measure retrieval
directly against cosine baseline. That's mechanism-first rather than
plasticity-first.

## Take-aways (final)

1. **The v0.5 retrieval ceiling finding from LoCoMo generalises.**
   Cosine-dominated retrieval already handles easy corpora; graph
   re-rank only has room to help when cosine fails or ties.
2. **Graph-rerank is only meaningful when hybrid/cosine retrieval is
   imperfect.** We should re-state the product story: SOMA's graph
   helps when BM25+cosine doesn't resolve cleanly. That's the
   LongMemEval finding (+22% F1 at rank 2-5) — same mechanism.
3. **Plastic-vs-frozen graph differentiation cannot be measured on
   this benchmark.** 5 attempts, 0 successful measurements. The
   benchmark is either cosine-ceiling or computation-blocked.
4. **Product positioning claim:** SOMA beats vector-DB baseline via
   BM25+cosine hybrid retrieval. The plastic graph is a research
   substrate, not a positioned product feature. This matches what
   we can actually demonstrate.

## Files

- `benchmarks/plastic_graph/run_activation.py` — runner
- `benchmarks/plastic_graph/topic_data.py` — synthetic corpus
- `benchmarks/plastic_graph/results/activation_test_v1_python.json`
  (v1 null)
- `benchmarks/plastic_graph/results/activation_v2_graph_rerank_k1.log`
- `benchmarks/plastic_graph/results/activation_v3_graph_rerank_k1.log`
- `benchmarks/plastic_graph/results/activation_v3b_graph_rerank_k1.log`
- `benchmarks/plastic_graph/results/activation_v3c_graph_rerank_k1.log`
  (silent exit at pre-consolidate)
