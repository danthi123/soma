# SOMA — Deferred Items & Known Gaps

**Date:** 2026-04-12
**Status:** 6 of 7 resolved (see per-item status below); item 4 replaced
with a preliminary baseline run in
`docs/progress/FIRST_REAL_CORPUS_RUN.md`.

These are gaps and deviations identified at the end of the autonomous
build. The main suite (446 tests, mypy clean, ruff clean) passes in
spite of them — they represent intentional deferrals from the plan, or
parts of the system that were never exercised end-to-end.

Each item has a matching tracking issue on Gitea
(`git.dant123.com/dant123/soma/issues`, label `deferred-v1`) so follow-up
work can be claimed there:

| # | Gitea issue | Title |
|---|---|---|
| 1 | #1 | Implement AudioEncoder |
| 2 | #2 | Full-sequence training in scripts/train.py |
| 3 | #3 | GPU smoke test on target RTX 3090 |
| 4 | #4 | Train on a real corpus and tune intervals |
| 5 | #5 | Add TextDecoder.tie_weights |
| 6 | #6 | Optional networkx/matplotlib visualization backend |
| 7 | #7 | Warn or require trained checkpoint in interactive.py |

Each entry documents:
- **What**: the gap itself.
- **Why**: whether it's a planned deferral, a simplification, or a genuine miss.
- **Impact**: what won't work (or will work poorly) because of it.
- **Fix sketch**: the shape of the follow-up work.

---

## 1. AudioEncoder not implemented

- **Status**: Deferred (by plan).
- **Why**: `docs/plans/2026-04-12-autonomous-build-design.md` explicitly cuts
  audio from v1: "AudioEncoder is explicitly out of scope for v1. Can be added
  later following the same ImageEncoder pattern."
- **Impact**: The system can't ingest audio. `input_modalities` accepts the
  name `"audio"` but nothing encodes raw audio into sensor-dim vectors.
- **Fix sketch**: Add `src/soma/io/audio_encoder.py` mirroring
  `src/soma/io/image_encoder.py`. A `torchaudio`-based log-mel spectrogram
  into patches, projected to `sensor_output_dim` by a Conv1d or small CNN,
  should suffice. Tests should follow `tests/test_io/test_image_encoder.py`.

## 2. Full-sequence training not wired into `scripts/train.py`

- **Status**: Simplification.
- **Why**: `SOMA.step` processes one token at a time; the script feeds only
  `sample.inputs["text"][0]` and `sample.target[0]` per step to keep the
  per-step semantics intact. See `scripts/train.py` around line 167-174.
- **Impact**: Multi-token context is not trained through the graph — only
  the first input and first target token of each sample drive learning.
  Losses will decrease but the system won't actually learn the rest of the
  context window.
- **Fix sketch**: Wrap the feeder loop so each `Sample` produces
  `chunk_size` stepwise calls: iterate `sample.inputs[m][t]` and
  `sample.target[t]` for `t in range(seq_len)`, stepping SOMA each time.
  Reset working memory per sample if sequences should not bleed.

## 3. No GPU smoke test performed

- **Status**: Gap.
- **Why**: All tests and demos ran on CPU. `--device cuda` is wired through
  `SOMA(..., device=device)`, `TextEncoder(..., device=device)`, etc., but
  the code path was never exercised.
- **Impact**: Driver / CUDA / torch-wheel mismatches on the target 3090 will
  surface on first `cuda` run rather than in tests. Some `register_buffer`
  transfers or `.to(device)` chains may have subtle typing issues under CUDA.
- **Fix sketch**: Add `tests/test_system/test_cuda.py` guarded by
  `pytest.mark.skipif(not torch.cuda.is_available(), ...)` that runs a
  short `SOMA.step` loop on `cuda`. Then run `train.py --device cuda` on a
  small corpus and fix any `Expected all tensors to be on the same device`
  errors that appear.

## 4. No real-corpus training demonstrated

- **Status**: Gap.
- **Why**: Only synthetic or multi-kB fixture corpora were ever fed to the
  system. The 10K-step stability test uses synthetic linear targets.
- **Impact**: Unknown whether the default config's growth / consolidation
  intervals are sensible for real data. Loss scale, curiosity range, and
  VRAM usage at full `max_nodes=50000` are all unmeasured.
- **Fix sketch**: Run `scripts/train.py` against a small public corpus
  (e.g. TinyShakespeare ~1 MB) for 10K+ steps. Watch `metrics.jsonl` for
  loss trend + node/edge growth + homeostasis LR multiplier behavior.
  Tune `base_lr`, `synaptogenesis_rate`, and `consolidation_interval`
  from real measurements.

## 5. `TextDecoder` has no `tie_weights(encoder)` method

- **Status**: Minor gap.
- **Why**: `TextEncoder` has a learnable `nn.Embedding(vocab_size, embed_dim)`;
  `TextDecoder` has a separate learnable `nn.Linear(embed_dim, vocab_size)`.
  They share a tokenizer but **not** embedding geometry. `scripts/train.py`
  originally had `decoder.tie_weights(encoder)` but it was removed once the
  method was found missing.
- **Impact**: Decoder argmax won't necessarily recover token identities the
  encoder was trained on. Expect poor text generation until the decoder
  projection learns the same vocabulary geometry from scratch.
- **Fix sketch**: Add a `tie_weights` method to `TextDecoder` that sets
  `self.output_proj.weight = encoder.embedding.weight` (or a view of it).
  Standard weight-tying pattern. Re-enable the call in `scripts/train.py`.

## 6. `scripts/visualize.py` uses hand-written DOT, not networkx/matplotlib

- **Status**: Plan deviation.
- **Why**: The plan called for "graph topology visualization via
  networkx/matplotlib". I shipped a text summary + manual graphviz DOT
  generator to avoid the networkx/matplotlib dependencies at install time.
- **Impact**: Functionally equivalent for off-line rendering, but callers
  who want an interactive matplotlib plot or networkx-based graph metrics
  (e.g. betweenness centrality) have nothing to build on.
- **Fix sketch**: Add an optional `--format networkx` flag that builds a
  `networkx.DiGraph` from the SOMA graph and either serializes it as
  JSON/GEXF or draws it via `networkx.draw_spring`. Guard the import so
  the base install stays lean.

## 7. `scripts/interactive.py` generation quality is expected-poor

- **Status**: Expected, not a bug.
- **Why**: An untrained SOMA produces near-random tokens because the
  decoder projection is randomly initialized and the graph weights haven't
  been shaped by data. Tests verify the REPL runs end-to-end but don't
  evaluate output quality.
- **Impact**: Real interactive use will look like gibberish until the
  checkpoint has been trained on a substantive corpus.
- **Fix sketch**: Not a code change — just train first (see item 4).
  Optionally: add a `--banner-warning` that makes it obvious when a
  freshly-built SOMA is being chatted with without prior training.

---

## Process Note

These seven items were surfaced in a post-build audit. None of them
invalidate the existing test suite; they describe where the "it runs
end to end" bar is lower than the "it does the whitepaper's intended
job well" bar. Prioritize items 3 and 4 first — they establish real-
world performance baselines and are prerequisites for tuning everything
else.
