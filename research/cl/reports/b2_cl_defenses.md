# B2: CL Defense Baselines -- EWC + A-GEM

**Date:** 2026-04-16
**Hardware:** NVIDIA RTX 3090 (24 GB), CUDA
**Wall-clock:** 385s (full run including lambda sweep)

## Implementations

### EWC (Elastic Weight Consolidation)
- Online variant: running Fisher accumulator (sum of per-task diagonals)
- Fisher normalized by task count in penalty computation
- Fisher diagonal clamped at 100.0 to prevent numerical blow-up
- Empirical Fisher (model's own predictions, mean-reduced per batch)
- Default lambda: 400.0

### A-GEM (Averaged Gradient Episodic Memory)
- 256 samples stored per completed task (random subset)
- Reference gradient computed from random 128-sample batch of buffer
- Gradient projection: `g_proj = g - (g.g_ref / g_ref.g_ref) * g_ref` when `g.g_ref < 0`
- Buffer stored on CPU; moved to device for reference gradient computation

## Results

### Permuted-MNIST (10 tasks, 5 epochs/task, batch_size=128)

| Method            | ACC    | BWT     | FWT    |
|-------------------|--------|---------|--------|
| Naive (no defense)| 0.7019 | -0.2340 | 0.0236 |
| EWC (lam=400)     | 0.7136 | -0.2199 | 0.0155 |
| A-GEM (buf=256)   | 0.8233 | -0.0987 | 0.0121 |

### Split-CIFAR-10 (5 tasks, 2 classes each, 5 epochs/task)

| Method            | ACC    | BWT     | FWT    |
|-------------------|--------|---------|--------|
| Naive (no defense)| 0.1400 | -0.6617 | -0.1000|
| EWC (lam=400)     | 0.1545 | -0.6253 | -0.1000|
| A-GEM (buf=256)   | 0.1573 | -0.6915 | -0.1000|

### EWC Lambda Sweep (Permuted-MNIST)

| Lambda | ACC    | BWT     | FWT    |
|--------|--------|---------|--------|
| 100    | 0.7001 | -0.2349 | 0.0198 |
| 400    | 0.7107 | -0.2233 | 0.0207 |
| 1000   | 0.7164 | -0.2159 | 0.0186 |

## Analysis

### Ordering: Naive < EWC < A-GEM (Permuted-MNIST)
The expected literature ordering holds. A-GEM is the clear winner with
12 percentage points higher ACC than naive and BWT reduced from -0.23
to -0.10. EWC provides modest improvement (+1.2 pp ACC).

### EWC underperformance vs published numbers
Published EWC on Permuted-MNIST typically achieves 85-90% ACC. Our
online EWC with task-normalized Fisher is more conservative. The
Fisher normalization by task count (dividing by N_consolidated) and
the clamp at 100.0 prevent numerical blow-up but reduce the penalty
strength. Without these safeguards, EWC produces NaN/Inf on CIFAR
due to Fisher explosion in the high-dimensional input space (3072-d).

Higher lambda helps marginally (1000 -> ACC 0.7164) but does not
close the gap to published numbers. A separate per-task Fisher
(classic EWC rather than online) or a different Fisher estimation
strategy (true Fisher with per-sample gradients rather than empirical
batch-mean) could improve results.

### Split-CIFAR-10: catastrophic forgetting across all methods
All methods show near-complete forgetting (0.000 accuracy on old
tasks). This is expected for a shared-head 2-layer MLP on Split-CIFAR:
- The model has a single 10-way output layer shared across all tasks
- Each task only trains on 2 classes, leaving other output weights stale
- When the model switches tasks, the gradient immediately overwrites
  task-specific features in the 512-dim hidden layer
- EWC and A-GEM provide marginal improvement (~1-2 pp) but cannot
  overcome the fundamental capacity limitation of a small MLP

This result is consistent with published benchmarks where Split-CIFAR
requires either task-specific heads or more sophisticated architectures.

### A-GEM on Permuted-MNIST: strong showing
ACC=0.8233 with BWT=-0.0987 falls within the published range (80-85%).
The gradient projection effectively prevents interference: the model
can learn new tasks while approximately preserving performance on the
memory buffer. The 256 samples/task buffer is small but sufficient for
the low-dimensional Permuted-MNIST setting.

### Numerical issues
- EWC Fisher explosion on CIFAR required two fixes:
  1. Switch from `reduction="sum"` to `reduction="mean"` in Fisher loss
  2. Clamp Fisher diagonal at 100.0 and normalize by task count
- No issues with A-GEM gradient projection (the `ref_sq < 1e-12`
  guard prevents division by zero)

## Hyperparameters

| Parameter         | Value  | Notes                              |
|-------------------|--------|------------------------------------|
| EWC lambda        | 400.0  | Swept {100, 400, 1000}; 1000 best  |
| EWC fisher_clamp  | 100.0  | Prevents numerical blow-up         |
| A-GEM buffer/task | 256    | Random subset per completed task   |
| A-GEM ref_batch   | 128    | Batch for reference gradient       |
| Learning rate     | 0.01   | SGD, no momentum                   |
| Hidden dim (MNIST)| 256    | 784 -> 256 -> 10                   |
| Hidden dim (CIFAR)| 512    | 3072 -> 512 -> 10                  |

## Files

- `research/cl/baselines/ewc.py` -- EWC implementation
- `research/cl/baselines/agem.py` -- A-GEM implementation
- `research/cl/run_b2.py` -- orchestrator
- `research/cl/harness.py` -- extended with `train_one_epoch` and `on_task_end` hooks
- `tests/test_research/test_cl_ewc.py` -- 8 tests
- `tests/test_research/test_cl_agem.py` -- 8 tests
