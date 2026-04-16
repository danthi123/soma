# Research Direction A: Online Anomaly / Novelty Detection via Structural Plasticity

> **For Claude:** research-phase plan. Sub-phases are experiments with pass/fail criteria, not feature builds.

**Hypothesis:** SOMA's substrate already computes the right signal for online anomaly detection — consolidation error + curiosity spike + structural-growth event — it just expresses those internally to drive learning. Rewired as an output surface, the plastic graph should be competitive with LSTM-autoencoder / Isolation-Forest on streaming anomaly benchmarks while also giving interpretability (a new node = a new concept) and online continual learning (no retrain window).

**Strategic framing:** this is the cleanest way to use the substrate for what it's actually good at, after the 2026-04-15 ablation showed SOMA-content contributes noise to next-token prediction. Next-token is the wrong objective; "did something unexpected happen" is the right one.

**Why it's novel:** plenty of unsupervised anomaly detectors exist, but none combine (a) streaming online learning without retrain windows, (b) structural growth that surfaces new concepts as explicit graph nodes, (c) homeostatic stability so the detector doesn't drift toward "everything is normal" or "everything is anomalous."

**Target publication:** a workshop paper at a CL / anomaly-detection venue (NeurIPS CL workshop, ICML CL workshop, KDD anomaly track) or an engineering blog post with open-source release.

---

## Sub-phase A1: Streaming anomaly harness + baselines

**Goal:** infrastructure — can we score something? Load three public benchmarks (NAB, KDD99 subset, SWaT industrial), define ground truth, run classical baselines, record AUC / F1 / latency-per-event.

**Experimental method:**
- Datasets: Numenta Anomaly Benchmark (57 real-world time-series with labels), KDD99 (network intrusion classics — cleaned subset, not the full 4GB), SWaT (Secure Water Treatment process data with attacks).
- Baselines: `sklearn.ensemble.IsolationForest`, `sklearn.svm.OneClassSVM`, LSTM-autoencoder (simple 2-layer, reconstruct current window).
- Metrics: ROC-AUC, PR-AUC, F1 at operating-point, latency per event (ms).

**Files:**
- Create: `research/anomaly/harness.py`
- Create: `research/anomaly/datasets/` (loaders, gitignored data dir)
- Create: `research/anomaly/baselines.py`
- Create: `research/anomaly/reports/a1_baselines.md`

**Success criterion:** baseline numbers land within published ranges (NAB IF score ≈ 0.55; LSTM-AE on KDD99 AUC ≈ 0.95). If off, data loader is wrong — fix before proceeding.

**Scope:** ~3-4 days.

---

## Sub-phase A2: Expose surprise signals from SOMA

**Goal:** add a "surprise score" output from the substrate. Combines consolidation error, curiosity delta, structural-growth event, and homeostatic-gain deviation into a single scalar per event.

**Experimental method:**
- Identify the internal signals already computed per step in `src/soma/metacognition/*` and `src/soma/consolidation/*`.
- Wrap them in a `SomaSurpriseExtractor(soma)` that exposes `score(event) -> float`.
- Calibrate: on in-distribution data, surprise should average near zero with small variance. Document the calibration baseline.

**Files:**
- Create: `src/soma/research/surprise.py` (new research-scope module)
- Create: `tests/test_research/test_surprise.py`
- Extend: `research/anomaly/harness.py` to accept a SOMA-backed detector

**Success criterion:** surprise score tracks consolidation-error within r=0.9 on held-out in-distribution data (sanity check that the signal is well-defined), and spikes measurably on injected synthetic anomalies (e.g., uniform-noise bursts).

**Scope:** ~3 days.

---

## Sub-phase A3: SOMA anomaly detector wrapper + first comparison

**Goal:** end-to-end pipeline. Stream events into SOMA, extract surprise, threshold → binary anomaly decision. Compare to baselines on all three datasets.

**Files:**
- Create: `research/anomaly/soma_detector.py`
- Create: `research/anomaly/reports/a3_soma_vs_baselines.md`

**Success criterion (preliminary):** SOMA within 10% of best baseline AUC on at least 2/3 datasets. If SOMA is best on any dataset, run-against-more-datasets in A4. If >10% worse on all three, stop and diagnose in an A3b failure-analysis subphase before continuing.

**Scope:** ~1 week. This is the "does the hypothesis hold?" decision point.

---

## Sub-phase A4: Ablation + interpretability

**Goal:** if A3 clears the bar, characterize what SOMA gives that baselines don't. Run ablations:
- Frozen-graph SOMA (no growth, no plasticity) — does structural plasticity actually help?
- No consolidation — does replay matter?
- No homeostasis — drift over time?

Plus qualitative: pick 3 real anomalies SOMA caught; show the new-node / edge-growth event that fired. Does the graph state make the anomaly interpretable to a human?

**Files:**
- Create: `research/anomaly/reports/a4_ablations_interpretability.md`
- Create: `research/anomaly/viz/` (graph-state snapshots around anomaly events)

**Scope:** ~1 week.

---

## Sub-phase A5: Writeup

**Goal:** draft paper (or long-form blog post) with figures, tables, reproducibility instructions. Land in `docs/papers/2026-xx-anomaly-via-structural-plasticity.md`.

**Success criterion:** a co-author (you) can read it end-to-end and either (a) agree with the claims or (b) spot the gap cleanly.

**Scope:** ~1-2 weeks.

---

## Total estimated scope

4-6 weeks of sustained work. Biggest risk is A3's "hypothesis holds" gate — if SOMA doesn't clear 10%-of-best-baseline on at least 2/3 datasets, the thesis is weaker than claimed and we either re-scope to a specific anomaly sub-domain where SOMA is strong, or pivot to a different research direction.

**Adjacency to other work:** doesn't touch the memory-layer product path. Pure research track. Can run in parallel with any product phase.
