# First Real-Corpus Training Run — 2026-04-12

Closes DEFERRED #4. Establishes an initial baseline so the UI has real
data to show during early development.

## Setup

- **Corpus:** TinyShakespeare (`data/tinyshakespeare.txt`, 1,115,394 bytes).
  Source: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`.
- **Config:** `configs/default.yaml` unchanged.
- **Tokenizer:** 512-vocab BPE trained on the full corpus.
- **Steps:** 500 (full-sequence mode, one SOMA.step per token).
- **Device:** CPU.
- **Seed:** 0.

Command:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --corpus data/tinyshakespeare.txt \
    --steps 500 \
    --log-every 50 \
    --checkpoint-dir checkpoints/tinyshakespeare \
    --metrics-file checkpoints/tinyshakespeare/metrics.jsonl \
    --tokenizer-vocab-size 512 \
    --device cpu \
    --seed 0
```

## Results

| Metric | Start (step 50) | End (step 500) |
|---|---|---|
| window_mean_loss | 0.01852 | 0.01817 |
| num_nodes | 34 | 34 |
| num_edges | 64 | 73 |
| curiosity (last) | 0.0 (warmup) | 0.0029 |
| lr_multiplier | 0.63 (spike) | 1.00 |

- Final checkpoint: `checkpoints/tinyshakespeare/soma_final.pt`.
- Per-interval metrics: `checkpoints/tinyshakespeare/metrics.jsonl`.
- Human-readable summary: `checkpoints/tinyshakespeare/summary.txt`.
- networkx-style graph JSON: `checkpoints/tinyshakespeare/graph.json`.

## Observations

- **Loss is almost flat.** 500 steps with the default `base_lr=0.001` and a
  vocabulary the encoder has never seen is not enough to see clear
  learning. The existing slow-integration stability test uses 10K steps
  with synthetic targets and sees real drop. For TinyShakespeare the
  system needs either more steps, a higher learning rate, or both.
  Tracked informally — not a blocker for UI development.
- **Synaptogenesis triggered.** Edges grew from 64 to 73 over 500 steps,
  so the growth engine is firing as expected.
- **Homeostasis dampened once.** At step 50 the regulator detected a
  >3σ loss spike and pulled `lr_multiplier` down to 0.63; it recovered
  by step 100. This matches the whitepaper-intended behavior.
- **Curiosity started emitting around step 150** — matches the module's
  warmup window (default 10 samples per domain, with 3 domains assigned
  by the classifier).
- **Working memory stayed sparse** (~9% occupancy). Expected at this
  training length.
- **Episodic memory filled linearly** — 500 entries after 500 steps.

## Takeaways for UI Defaults

- The UI's loss plot should autoscale aggressively — early runs may be
  flat over small ranges (e.g., 0.017-0.019).
- Growth-event strip plot should use a log-ish y-axis or just colored
  ticks, since events are sparse (a few per 50 steps).
- Initial `max_render_nodes` of ~100 is plenty given realistic early
  run sizes (<50 nodes).
- Expose `base_lr`, `synaptogenesis_rate`, and `consolidation_interval`
  prominently in the config panel — those are the knobs a user will
  need to tune first when they see loss not moving.
