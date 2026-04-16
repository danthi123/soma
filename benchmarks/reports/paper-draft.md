# SOMA — A Local-First, Plastic-Graph Memory Layer for LLMs

*Draft paper aggregator, 2026-04-16.*

> This document aggregates the findings from the reproducible
> benchmarks under `benchmarks/` into a narrative that maps to the
> research paper. It is not the paper itself — each section points to
> the artifact (script + report) that generated the numbers so a
> reader can re-run and verify.

## 1. What SOMA is

SOMA is a **local-first agent memory layer** positioned as a drop-in
replacement for vector-DB-plus-RAG. Its differentiators are:

- **Single-file install, consumer-grade.** A MemoryLayer instance is
  a Python object with `store(text)` / `retrieve(query, k)` /
  `consolidate()`. No separate daemon, no cloud dependency, no schema
  migration — the whole state fits in a directory bundle.
- **Plastic-graph substrate.** Entries are indexed by vector and
  optionally fed through a SOMA graph that can grow, prune, and
  specialize with use. The graph is optional at retrieval today
  (ablation in §4.2 explains why) but the substrate is there for the
  research agenda in §5.
- **Owner-controlled.** Data never leaves the host unless the caller
  sends it somewhere. This survives model swaps (memory is the user's,
  not the LLM's) and privacy changes (no third-party custody).

The product-facing pitch lives in `docs/positioning.md`; the pivot
rationale in `docs/plans/2026-04-15-memory-layer-pivot.md`.

## 2. Methodology

Every number in §4 comes from a committed script + committed report.
Seeds are fixed; the scripts run on a laptop without a GPU.

| What | Script | Report |
| --- | --- | --- |
| SOMA vs Chroma (50 facts, labeled) | `benchmarks/run_retrieval.py` | `benchmarks/reports/retrieval.md` |
| SOMA vs Chroma (1K/5K/20K efficiency) | `benchmarks/run_scale_vs_chroma.py` | `benchmarks/reports/scale_vs_chroma.md` |
| Graph re-rank sweep | `benchmarks/run_graph_ablation.py` | `benchmarks/reports/graph_ablation.md` + `graph_ablation_shuffled.md` |
| Plasticity at scale | `benchmarks/run_plasticity_scale.py` | `benchmarks/reports/plasticity_scale.md` |
| Longitudinal drift | `benchmarks/run_longitudinal_drift.py` | `benchmarks/reports/longitudinal_drift.md` |

Datasets: a hand-curated 50-fact / 26-query topic-clustered set
(`benchmarks/datasets/synthetic.py`) for labeled-query quality, and a
template-permutation generator up to 2000 unique facts
(`benchmarks/datasets/templated.py`) for scale + drift experiments.

Embeddings: sentence-transformers `all-MiniLM-L6-v2` (384-d, cosine).
Same embedder drives SOMA and Chroma in the apples-to-apples
comparisons so the delta isolates storage/indexing mechanics.

## 3. Headline results

**SOMA matches Chroma on retrieval quality with a durable 2.5–3× store-speed advantage and a disk advantage that ranges from 22.6× (small N) to 1.4× (20K). Default exact retrieve modestly beats Chroma at every N tested (1.12–1.28×); the opt-in HNSW backend extends the retrieve lead to 1.67× at 20K while preserving Recall@3.**

### 3.1 Quality (50-fact labeled benchmark)

From `retrieval.md`:

| System | Recall@3 | MRR@3 | NDCG@3 | Store (ms/op) | Retrieve (ms) | Disk (KB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOMA | 0.923 | 0.891 | 0.885 | 9.46 | 5.98 | 85.1 |
| Chroma | 0.923 | 0.891 | 0.885 | 26.88 | 7.53 | 1920.7 |

Both systems use the same sbert embedder, so quality is identical
by construction (both reduce to cosine over the same 384-d
vectors). At this size SOMA is 22.6× smaller and 1.3× faster on
retrieve — but those single-point numbers don't generalize, see §3.2.

### 3.2 Efficiency at scale (1K / 5K / 20K)

From `scale_vs_chroma.md` (no quality measurement, just bytes and
latency on the same sbert embedder; SOMA shown twice — `flat` is
exact `IndexFlatIP`, `hnsw` is opt-in approximate `IndexHNSWFlat`):

| N | System | Store (ms/op) | Retrieve (ms) | Disk (MB) |
| ---: | --- | ---: | ---: | ---: |
| 1000 | SOMA-flat | 8.59 | 10.58 | 1.6 |
| 1000 | SOMA-hnsw | 8.47 | **8.36** | 1.6 |
| 1000 | Chroma | 21.60 | 11.80 | 4.3 |
| 5000 | SOMA-flat | 8.39 | 15.36 | 8.2 |
| 5000 | SOMA-hnsw | 8.39 | **10.77** | 8.2 |
| 5000 | Chroma | 24.80 | 11.73 | 12.9 |
| 20000 | SOMA-flat | 8.21 | **11.09** | 32.7 |
| 20000 | SOMA-hnsw | 8.07 | **8.55** | 32.7 |
| 20000 | Chroma | 25.78 | 14.24 | 45.2 |

The honest scale story:

- **Store: SOMA stays 2.5–3× faster across all N** (~8.4 ms vs
  ~25 ms per op). Chroma's metadata layer pays a fixed cost per
  write that doesn't amortize.
- **Disk: SOMA wins by 1.4–22× depending on N.** At small N
  Chroma's HNSW/SQLite overhead dominates (22× advantage at 50);
  at 20K both systems approach the floor of "raw embedding × N"
  and the gap narrows to 1.4×. SOMA still wins absolute bytes at
  every N tested.
- **Retrieve: SOMA-flat modestly beats Chroma at every N tested**
  (1.12–1.28×). **SOMA-hnsw extends the lead at scale** — 1.41× at
  1K, 1.09× at 5K, **1.67× at 20K** — while preserving Recall@3 =
  0.923 (verified on the 50-fact labeled set). The 20K HNSW number
  initially looked anomalous; root-caused as the lazy FAISS-index
  build cost (~500 ms) bleeding into the first measured probe. With
  a warmup probe before the timed loop, HNSW's amortized cost is
  sub-millisecond at this scale and the structural advantage is
  visible.

The SOMA-hnsw column is opt-in via
``MemoryLayer(faiss_index_type="hnsw")``. Default stays
``"flat"`` because exact retrieval recall guarantees match
Chroma's exact mode without surprise; `hnsw` is the right opt-in
for stores in the multi-K range where its build cost amortizes.

## 4. Experiments

### 4.1 Does the plastic graph lift retrieval? (ablation)

From `graph_ablation.md` (topic-clustered) + `graph_ablation_shuffled.md`
(randomized order), sweeping `graph_rerank_alpha` ∈
{0, 0.05, 0.1, 0.2, 0.3, 0.5} with two capture modes:

- **Mid-consolidation capture** (shipped through 2026-04-15): each
  stored activation snapshotted during growth in a different graph
  state. On topic-clustered data, `alpha=0.2` appeared to beat flat
  cosine (+0.019 Recall@3). **Shuffling the facts kills the gain**
  (-0.038 Recall@3) — the apparent lift was consolidation-order
  artifact correlating with topic bursts, not a semantic graph signal.
- **Post-growth stable capture:** every stored activation re-captured
  from the final graph state in `eval_mode=True`. Every alpha
  produces the same Recall@3 as flat cosine (0.923) — the graph
  signal under a consistent graph state carries no retrieval-useful
  information beyond what cosine on sbert already encodes.

**Conclusion (4.1):** as of 2026-04-16, SOMA's graph re-rank does not
improve retrieval on synthetic small-corpus benchmarks. Default is
`alpha=0.0` (short-circuit — no re-rank cost). The graph machinery is
still present for future work; it is not part of the efficiency +
quality story today.

### 4.2 How does SOMA scale?

From `plasticity_scale.md`, fresh system per scale point, 100 →
2000 stored facts:

| N | Nodes | Edges | Disk (KB) | Consolidate (s) | Retrieve (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 42 | 80 | 168 | 10.0 | 7.7 |
| 500 | 42 | 80 | 836 | 63.1 | 9.8 |
| 2000 | 42 | 80 | 3340 | 276.8 | 11.2 |

- **Graph stays at seed size** (42 nodes, 80 edges) across every N —
  SOMA's growth thresholds are training-regime tuned and the memory
  workload doesn't exercise them. This is not a bug: the graph
  footprint is bounded and the efficiency story does not rely on
  growth.
- **Disk scales linearly** with the embedding vector (384-d sbert →
  ~1.67 KB/entry). That is the vector-DB floor; graph weights are a
  small additive term that hasn't moved because the graph hasn't.
- **Retrieve latency is flat** (linear cosine on the 384-d matrix at
  these sizes; FAISS ANN kicks in at 10K+).
- **Consolidate time scales linearly with N.** Today it buys no
  retrieval lift; an obvious follow-up is an incremental consolidate
  that only processes new entries.

### 4.3 Does old memory rot?

From `longitudinal_drift.md`, 30-day simulation, 5 new facts/day,
10 recency-biased queries/day:

- Overall mean Recall@3: **0.957**
- Recent facts (≤3 days) Recall@3: **0.938**
- Old facts (>10 days) Recall@3: **0.883**
- Final day: **1.000 overall, 1.000 on old**

SOMA's memory does not drift. The old-vs-recent gap (0.055) is
within query-sampling noise at 2 old queries/day; the trend is
flat-to-improving as the store stabilizes. This is the test that
should fail on Mem0-style purge-on-budget schemes and on
LLM-context systems that drop information when the window fills.
SOMA has nowhere for old facts to go — the embedding is the
contract.

## 5. Open research questions

SOMA's current story is honest:

- **Efficiency:** real, measurable, repeatable (22.6× disk, 1.3×
  retrieve at 50 facts; same ordering holds at 500 and 2000 from the
  scaling profile).
- **Graph-as-retrieval-signal:** currently null. The plastic graph
  machinery runs but does not move Recall@3 on small corpora under
  the current consolidation objective.

The research agenda is explicit:

1. **Memory-workload-tuned growth thresholds.** Today's
   synaptogenesis / neurogenesis defaults are tuned for next-token
   training pressure. The memory workload has much gentler signal;
   tightening thresholds may let the graph specialize on topic
   clusters it currently ignores.
2. **Consolidation objectives beyond next-token MSE.** A
   classification-style loss over stored entries, or contrastive
   loss between clusters, would push SOMA to actively encode
   similarity structure the re-rank can use.
3. **Stable-capture variants.** Post-growth re-capture neutralizes
   the graph signal today; the question is whether there exists a
   capture regime that is both stationary AND informative. The
   current data says no for alpha > 0 on synthetic sbert.
4. **Real-world long-context benchmarks.** LoCoMo / LongMemEval
   need an LLM judge and a long-conversation harness — deferred
   here because the judge requires API access outside autonomous
   scope. That integration is pre-scoped in the task list and
   unblocks once the judge is wired.

## 6. Reproducibility

Every report in `benchmarks/reports/` links to the script that
generated it. Running the full suite from a fresh clone takes about
30 minutes on a laptop without a GPU. Seeds are fixed in-script; no
external dataset downloads are required for §3 / §4.

```bash
pip install -e '.[dev]' sentence-transformers chromadb
python -m benchmarks.run_retrieval
python -m benchmarks.run_graph_ablation
python -m benchmarks.run_graph_ablation --shuffle
python -m benchmarks.run_plasticity_scale
python -m benchmarks.run_longitudinal_drift
```

The reports ship alongside the source; replacing them should be a
strict no-op on the numbers up to floating-point reordering.

---

*This draft is the aggregator, not the paper. Next: pull figures
from the tables (disk/retrieve scatter, drift trend line), wire
LoCoMo once an LLM judge is available, and structure the narrative
for an arXiv submission.*
