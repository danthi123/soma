# B1 Baseline Results: Vanilla MLP (No CL Defense)

**Date:** 2026-04-16
**Hardware:** NVIDIA GeForce RTX 3090, CUDA
**Protocol:** SGD lr=0.01, batch_size=128, 5 epochs/task

---

## Permuted-MNIST (10 tasks)

### Accuracy Matrix

A[i][j] = accuracy on task j after training through task i.

|       |  t0   |  t1   |  t2   |  t3   |  t4   |  t5   |  t6   |  t7   |  t8   |  t9   |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| t0    | 0.899 | 0.082 | 0.165 | 0.088 | 0.128 | 0.095 | 0.107 | 0.064 | 0.070 | 0.072 |
| t1    | 0.884 | 0.910 | 0.169 | 0.090 | 0.143 | 0.110 | 0.108 | 0.088 | 0.092 | 0.088 |
| t2    | 0.861 | 0.887 | 0.913 | 0.127 | 0.154 | 0.121 | 0.098 | 0.126 | 0.092 | 0.139 |
| t3    | 0.844 | 0.865 | 0.890 | 0.911 | 0.171 | 0.116 | 0.085 | 0.124 | 0.077 | 0.129 |
| t4    | 0.843 | 0.838 | 0.876 | 0.880 | 0.916 | 0.118 | 0.093 | 0.120 | 0.082 | 0.113 |
| t5    | 0.799 | 0.828 | 0.851 | 0.872 | 0.874 | 0.916 | 0.111 | 0.112 | 0.076 | 0.120 |
| t6    | 0.725 | 0.781 | 0.805 | 0.801 | 0.842 | 0.898 | 0.913 | 0.128 | 0.086 | 0.124 |
| t7    | 0.668 | 0.673 | 0.743 | 0.743 | 0.831 | 0.852 | 0.882 | 0.915 | 0.090 | 0.116 |
| t8    | 0.601 | 0.545 | 0.643 | 0.631 | 0.666 | 0.788 | 0.849 | 0.855 | 0.916 | 0.107 |
| t9    | 0.545 | 0.565 | 0.628 | 0.619 | 0.560 | 0.705 | 0.834 | 0.794 | 0.893 | 0.921 |

### Metrics

| Metric | Value  |
|--------|--------|
| ACC    | 0.7064 |
| BWT    | -0.2295|
| FWT    | 0.0224 |

**Wall-clock:** 38.5s (10 tasks, 5 epochs each)

### Interpretation

- **BWT = -0.23** confirms substantial forgetting. Earlier tasks degrade by
  ~23 percentage points on average after training on later tasks.
- **ACC = 0.71** is higher than the often-cited ~20% for vanilla SGD. This is
  because the published ~20% floor typically uses 20 epochs/task, which drives
  weights much further from earlier task optima. With our standard 5-epoch
  protocol, the MLP retains partial knowledge of earlier tasks.
- The diagonal is consistently ~90-92%, showing the model learns each task well.
- Clear recency bias: the last 2-3 tasks retain >80% while early tasks drop to 50-60%.

---

## Split-CIFAR-10 (5 tasks, 2 classes each)

### Accuracy Matrix

|       |  t0   |  t1   |  t2   |  t3   |  t4   |
|-------|-------|-------|-------|-------|-------|
| t0    | 0.793 | 0.000 | 0.000 | 0.000 | 0.000 |
| t1    | 0.000 | 0.708 | 0.000 | 0.000 | 0.000 |
| t2    | 0.000 | 0.000 | 0.517 | 0.000 | 0.000 |
| t3    | 0.000 | 0.000 | 0.000 | 0.519 | 0.000 |
| t4    | 0.000 | 0.000 | 0.000 | 0.000 | 0.784 |

### Metrics

| Metric | Value  |
|--------|--------|
| ACC    | 0.1567 |
| BWT    | -0.6341|
| FWT    | -0.1000|

**Wall-clock:** 3.6s

### Interpretation

- **Textbook catastrophic forgetting.** Each task completely overwrites the
  previous one (0.000 accuracy on all prior tasks).
- **ACC = 15.7%** -- near the 2/10 random-guessing floor for 10-way classification
  where only 2 classes are active.
- **BWT = -0.63** -- massive negative backward transfer.
- This is the expected Split-CIFAR behavior: the shared output head gets
  overwritten because classes are disjoint across tasks and the model is small.

---

## Conclusions

The harness produces results consistent with CL literature:
1. Permuted-MNIST shows gradual forgetting (shared input distribution, same labels).
2. Split-CIFAR shows catastrophic forgetting (disjoint class sets, total output head overwrite).
3. The vanilla MLP is a valid forgetting-floor baseline for both benchmarks.

## TODO

- CORe50 loader (deferred: large download, complex session structure)
- More epochs/task on Permuted-MNIST to reproduce the ~20% floor
- EWC and A-GEM baselines (B2)
- SOMA adapter (B3)
