# B3: SOMA as CL Feature Extractor -- Preliminary Results

## Architecture

SOMA is wired as a feature extractor with a trainable linear classification
head.  The pipeline per sample:

1. **Input projection**: `nn.Linear(784, 256)` maps raw pixels to sensor dim.
2. **SOMA forward**: projected vector fed to SOMA graph; self-supervised
   MSE reconstruction target drives Hebbian learning internally.
3. **Feature concat**: `[projected (grad-carrying), soma_output (detached)]`
   concatenated to form a 512-dim feature vector.
4. **Linear head**: `nn.Linear(512, 10)` produces class logits; trained
   by cross-entropy via SGD (lr=0.1).

SOMA processes one sample at a time (~66 samples/sec on CUDA RTX 3090).
Training uses subsampled data (1000 train / 500 test per task) to keep
wall-clock manageable.

## Ablations

| Ablation | Hebbian | Consolidation | Critical Periods |
|---|---|---|---|
| soma-plastic | ON | ON | ON |
| soma-frozen | OFF | OFF | OFF |
| soma-no-consolidation | ON | OFF | ON |
| soma-no-critical-periods | ON | ON | OFF |

## Partial Results (soma-plastic, 3/10 tasks)

After training on the first 3 tasks of 10-task Permuted-MNIST:

| After task | t0 acc | t1 acc | t2 acc |
|---|---|---|---|
| Task 0 | **0.712** | 0.038 | 0.110 |
| Task 1 | 0.684 | **0.712** | 0.116 |
| Task 2 | 0.690 | 0.752 | **0.790** |

Observations from partial data:
- **Classification works**: 71-79% single-task accuracy (vs 10% random).
- **Mild forgetting**: t0 drops from 71.2% to 69.0% after 2 more tasks (BWT ~ -0.02 per task).
- **Forward transfer**: t1 accuracy improved from 71.2% to 75.2% after learning t2, suggesting
  the input projection generalizes across permutations.

## Comparison with B2 Baselines (reference, full 60K train)

| Method | ACC | BWT |
|---|---|---|
| Naive | 0.70 | -0.23 |
| EWC (lam=1000) | 0.72 | -0.22 |
| A-GEM (buf=256) | **0.82** | **-0.10** |
| SOMA-plastic (partial, 3/10 tasks) | TBD | TBD |

**Note**: B2 baselines used full 60K training set; SOMA uses 1K subsampled
due to per-sample processing throughput (~66 samples/sec on CUDA).
Direct comparison is approximate.

## Throughput

SOMA's per-sample graph execution is the bottleneck:
- Training: ~66 samples/sec on CUDA (RTX 3090)
- Each task: ~1000 train * 3 epochs = 3000 train samples + 10 * 500 eval = 5000 eval = 8000 total
- Per ablation (10 tasks): ~80K samples = ~20 min (estimated, slower with graph growth)
- Full run (4 ablations): ~2-4 hours

## Status

Full run in progress (background PID 28416 on CUDA). Results will be
written to `b3_soma_cl.json` on completion.

## API Shape Notes

SOMA's `step()` accepts `dict[str, Tensor]` inputs keyed by modality.
The sensor node expects input matching `sensor_output_dim` (not
`text_embed_dim`), requiring a projection layer. The `set_input`
check enforces `data.shape[-1] == output_dim`. Self-supervised
targets must be skipped during `torch.no_grad()` eval (SOMA's
internal `loss.backward()` fails without grad context).
