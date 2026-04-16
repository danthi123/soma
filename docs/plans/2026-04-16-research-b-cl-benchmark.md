# Research Direction B: Continual-Learning Benchmark Runner

> **For Claude:** research-phase plan. Sub-phases are experiments with pass/fail criteria, not feature builds.

**Hypothesis:** SOMA's whitepaper has claimed since Day 1 that homeostasis + critical periods + structural plasticity prevent catastrophic forgetting. The project has never tested this head-to-head against established continual-learning baselines. This direction settles the question decisively — either SOMA is competitive on standard CL benchmarks (publishable with a clean narrative) or it isn't (failure mode is informative; we know exactly which architectural claims don't carry their weight).

**Strategic framing:** the one non-negotiable piece of due-diligence before any SOMA paper ships a CL claim. Either SOMA is in the CL landscape or it isn't, and "we've always said it is, we just haven't tested it" is not a defensible position after two years of investment.

**Why it matters:** the paper-draft.md already positions SOMA as a CL system. If the CL community asks "where's your Split-CIFAR number?" and we have to say "we don't have one," the whole paper's credibility suffers. Running the benchmarks is the cheapest insurance against that. Prerequisite: the seed-graph integrator bug fix (`860cc82`) must be in every eval — we've never trained a SOMA with its intended architecture before that commit.

**Target publication:** results table suitable for the main paper, or a standalone workshop paper at NeurIPS CL / ICML CL / CoLLAs.

---

## Sub-phase B1: Benchmark harness + metrics

**Goal:** infrastructure. Load three standard CL benchmarks + implement standard metrics (ACC, BWT, FWT, plasticity/stability ratio).

**Datasets (in order of standard adoption):**
- **Permuted-MNIST** — 10 tasks, each a different permutation. Classic "does the model forget?" test.
- **Split-CIFAR-10** — 5 tasks, 2 classes each. Harder: visual features.
- **CORe50** — real-world object recognition, 50 classes × 11 sessions. Closer to deployment.

**Metrics (Lopez-Paz & Ranzato 2017):**
- **ACC (average accuracy):** mean over tasks at the end of training
- **BWT (backward transfer):** how much training on task k+1 hurts task k
- **FWT (forward transfer):** how much prior tasks help task k+1
- Plus: plasticity/stability ratio (single-task accuracy / retention), wall-clock training time, parameter count.

**Files:**
- Create: `research/cl/datasets/` (gitignored data, loaders in-tree)
- Create: `research/cl/harness.py` (common train/eval loop, metric recording)
- Create: `research/cl/metrics.py`
- Create: `tests/test_research/test_cl_metrics.py` (metric-math sanity)

**Success criterion:** vanilla MLP baseline reproduces published numbers within ±2% ACC on Permuted-MNIST (well-known, heavily reproduced). Off by more → data loader or metric is wrong.

**Scope:** ~3-4 days.

---

## Sub-phase B2: Baselines — the CL-research menu

**Goal:** implement the standard comparators. This is the bulk of the "establish ourselves in the literature" work.

**Baselines to implement:**
1. **Naive** (no CL defense — sequentially train on all tasks, pure catastrophic forgetting floor).
2. **Joint** (all tasks trained together — upper bound; "if we had full access" ceiling).
3. **EWC** (Elastic Weight Consolidation, Kirkpatrick 2017) — Fisher-information-weighted regularizer.
4. **PackNet** (Mallya & Lazebnik 2018) — pruning + parameter isolation per task.
5. **A-GEM** (Chaudhry 2019) — gradient-projection memory.
6. **Progressive Networks** (Rusu 2016) — one column per task, lateral connections.

**Pick:** implement EWC and A-GEM (replay-family) as minimum; PackNet and Progressive if time. These are the "benchmarks you must beat to be taken seriously" set.

**Files:**
- Create: `research/cl/baselines/naive.py`, `joint.py`, `ewc.py`, `agem.py`
- Create: `tests/test_research/test_cl_baselines.py` (sanity: naive forgets, joint doesn't)

**Success criterion:** each baseline reproduces published ordering (Naive < EWC < Joint on Permuted-MNIST). Exact numbers will differ from published work — reproduce the *ordering*, not the values.

**Scope:** ~1-1.5 weeks.

---

## Sub-phase B3: SOMA adapter for classification CL

**Goal:** wire SOMA into the harness. Class prediction requires a classifier head, not a language head.

**Approach:**
- Feed raw input (flattened MNIST pixel vector, CIFAR patch embedding) into SOMA as a single observation per step.
- Read SOMA's output activations as a fixed-dim feature vector.
- Train a linear classifier on top. Only the linear head is optimized by task-label loss — SOMA learns unsupervised / Hebbian during the stream.
- Between tasks: consolidation cycle (artificial sleep).

**Ablations built in:**
- SOMA-frozen (no plasticity) vs SOMA-plastic
- With vs without inter-task consolidation
- With vs without critical-period schedule

**Files:**
- Create: `research/cl/soma_cl.py`

**Success criterion:** on Permuted-MNIST, SOMA-plastic beats SOMA-frozen on BWT (less forgetting). This is the minimum "plasticity helps" signal. If it fails, the architectural claim is falsified and B4 becomes a post-mortem instead of a comparison.

**Scope:** ~1 week.

---

## Sub-phase B4: The matrix

**Goal:** run everything × everything. Three datasets × six+ methods × three seeds = 54+ runs. Build a results table.

**Deliverables:**
- `research/cl/reports/b4_results.md` — headline table (method × dataset → ACC/BWT/FWT)
- `research/cl/reports/figs/` — per-dataset forgetting curves
- `research/cl/golden/b4.json` — seed-averaged numbers for paper-draft import

**Success criterion:** honest result regardless of direction. If SOMA lands in the top-3 on at least one benchmark, paper-worthy. If it's always in the bottom half, failure-mode analysis in B5 becomes the contribution ("our architectural claims miss X").

**Scope:** ~1 week (mostly wall-clock on the 3090).

---

## Sub-phase B5: Writeup (or post-mortem)

**Goal:** either a results paper or a clean "what we learned" post-mortem.

**If wins:** draft `docs/papers/2026-xx-cl-structural-plasticity.md`. Figures, tables, related-work, limitations.

**If loses:** draft `docs/postmortems/2026-xx-cl-benchmark-lessons.md`. Which architectural claims held / failed. Which hyperparameters are load-bearing. What a v2 substrate would change.

**Scope:** ~1-2 weeks.

---

## Total estimated scope

5-7 weeks. Lowest-novelty, highest-credibility of the four directions. The one we most regret skipping if someone asks "did you try it on the standard CL suite?"

**Adjacency:** independent of product path. Benefits from the seed-graph bug fix being current. Can run parallel to any product phase.
