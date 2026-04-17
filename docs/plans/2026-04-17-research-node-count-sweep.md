# Research Ablation: Node-Count Sweep

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Systematically test whether SOMA's initial graph capacity
(node count at construction time) affects performance on CL and
associative-memory tasks. All prior experiments used
`initial_integrator_count=8` (14 total nodes). This ablation sweeps
{8, 16, 32, 64, 128} integrators while holding everything else
constant.

**Hypothesis:** a larger initial graph MAY:
- (a) reduce catastrophic forgetting (more capacity = more room to
  separate task-specific representations)
- (b) improve per-bit accuracy in attractor mode (more nodes =
  richer fixed-point landscape)
- (c) slow down training / increase VRAM / add overhead without
  benefit (the null hypothesis: 8 nodes is already sufficient)

D3 showed that GROWING from 14 to 128 nodes dynamically hurt
(interference). But starting WITH more nodes is architecturally
different — the seed graph's topology is more structured than
randomly-added growth nodes.

**Depends on:** B3 (CL adapter baseline) for CL sweep; D2/D3
(attractor mode) for associative sweep.

---

## Sub-phase 1: CL node-count sweep

**Run B3's SOMA-plastic configuration on Permuted-MNIST with
integrator counts {8, 16, 32, 64, 128}.** Everything else fixed
(same LR, epochs, head architecture, consolidation protocol).

**Metrics:** ACC, BWT per node count. Also: wall-clock per task
(does 128 integrators make training unacceptably slow?), peak VRAM.

**Files:**
- Create: `research/cl/run_node_sweep.py` — orchestrator that loops
  over integrator counts, calls the B3 SOMA adapter for each
- Create: `research/cl/reports/node_sweep_cl.md`
- Create: `research/cl/reports/node_sweep_cl.json`

**Success criterion:** if any node count beats the 8-integrator
baseline by ≥5 points ACC or ≥0.05 BWT, that's a meaningful finding
worth incorporating into the default config. If all are within noise,
the null hypothesis holds and 8 integrators is the right default.

---

## Sub-phase 2: Associative memory node-count sweep

**Run D2's attractor-mode protocol on random-binary (D=50) with
integrator counts {8, 16, 32, 64, 128}.** Sweep N (stored patterns)
from 1 to 3×D as in D1/D2.

**Metrics:** per-bit accuracy, nearest-pattern recall rate, and
capacity cliff point (if any) per node count.

**Files:**
- Create: `research/associative/run_node_sweep.py`
- Create: `research/associative/reports/node_sweep_assoc.md`
- Create: `research/associative/reports/node_sweep_assoc.json`

**Success criterion:** if per-bit accuracy improves meaningfully with
more nodes (e.g., 0.90 → 0.95+ at N=10, enabling sign-exact recall),
that directly informs D5's writeup. If not, the 14-node architecture
is already at its representational ceiling for this task.

---

## Sub-phase 3: Scaling characteristics report

**Combine both sweeps into a single analysis:**
- Node count vs accuracy (CL + associative on one plot description)
- Node count vs wall-clock (training cost)
- Node count vs peak VRAM
- Recommendation: optimal node count for each use case, or "8 is fine"

**Files:**
- Create: `research/reports/node_count_scaling.md`

---

## Implementation notes

- **Device:** cuda for CL (batch processing benefits from GPU); cpu
  for D=50 associative (too small for GPU benefit per D2 finding).
- **Associator count:** scale proportionally with integrators. E.g.,
  if integrators=32, set associators=64 (2× ratio matches the
  default 8:16). This keeps the graph structure proportional.
- **Config:** only change `initial_integrator_count` and
  `initial_associator_count`. All other SOMAConfig params stay at
  the values used in B3 / D2.
- **Seeds:** 3 seeds per config for statistical stability.

## Commit pattern

1. Commit 1: CL node sweep + results
2. Commit 2: Associative node sweep + results
3. Commit 3: Combined scaling report

Three commits. Co-Authored-By trailer as usual.

## Total scope

~2-3 days. Most of the time is wall-clock for the 5×10-task CL sweep
(5 node counts × 10 tasks × 5 epochs × 3 seeds = 750 training runs,
but each is fast on GPU).
