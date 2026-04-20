# Milestone: Hybrid Retrieval Validated — 2026-04-20

A pinned reference for the current shipping state of SOMA as of
2026-04-20. Call this "M1: hybrid retrieval validated" — the first
milestone where SOMA has reproducible, cross-validated, cross-
methodology evidence of beating vector-DB+RAG on real benchmarks.
Future work (Path A biophysical substrate, Path B bio-validated
primitives) can reference this as the known-good baseline.

## Headline claim

**SOMA's hybrid retrieval (BM25 + cosine, α=0.30) delivers +22.8% F1
and +15.6% rank-1 over a Chroma-cosine baseline on LongMemEval N=500,
using the same embedder and same LLM under matched context budgets.**

Triple cross-validation of the ~22% F1 lift:

| Metric | Value |
|---|---|
| Token-F1 (qwen4b strict, N=500) | **+22.8%** |
| qwen4b as judge (self-evaluation) | **+22.2%** |
| Claude as judge (independent) | **+23.7%** |

## Validation axes

Every axis reproducible in the repo:

| Axis | Result | Reference |
|---|---|---|
| Cross-LLM (qwen4b → qwen9b) | +22.8% → +22.0% F1 | `longmemeval_qa_compare_qwen9b_strict_findings.md` |
| Cross-LLM (qwen4b → Claude) | +22.8% → +14.7% F1 (harder baseline) | `longmemeval_claude_runner_findings.md` |
| Cross-embedder (SBERT → mxbai) | +15.6% → +45% rank-1 | `longmemeval_mxbai_cross_embedder.md` |
| Cross-benchmark (LongMemEval → LoCoMo) | +15.6% → +88.7% rank-1 | `locomo_rank_probe_findings.md` |
| Cross-judge (qwen-judge → Claude-judge) | +22.2% → +23.7% | `longmemeval_judge_findings_claude.md` |
| α sweep (N=500) | α=0.30 confirmed optimal (0.886 rank-1) | `longmemeval_alpha_sweep_findings.md` |

## Infrastructure delivered

### Claude-as-LLM via Unraid claude-code-runner (new in this session)

SOMA benchmarks can now use Claude as the answerer or judge via the
operator's Claude Max subscription, no per-request API charges.
Transport: SSH + `docker run` + stdin-piped prompt; auth via
long-lived OAuth token at
`/mnt/cache/appdata/claude-runner/auth-token` (synced by the
operator's existing n8n `Claude Runner - Token Health Check`
workflow).

- `benchmarks/industry/llm_backends/claude_runner_client.py` —
  `ClaudeRunnerClient` wrapper + `ClaudeRunnerError`.
- `benchmarks/tests/test_claude_runner_client.py` — 9 passing
  tests (command construction, stdin, timeout, UTF-8 for emoji).
- `--provider claude_runner` added to both
  `benchmarks/industry/longmemeval/run_qa_compare.py` and
  `benchmarks/industry/longmemeval/judge_predictions.py`.

### Benchmark harnesses

- `benchmarks/industry/locomo/rank_probe.py` — LoCoMo turn-level
  retrieval probe (mirror of LongMemEval's `rank_probe.py`)
- `benchmarks/industry/longmemeval/mxbai_rank_probe.py` — mxbai-
  embed-large probe for cross-embedder validation
- `benchmarks/industry/longmemeval/alpha_sweep.py` — α sweep
  for hybrid blend calibration
- `scripts/analysis/compare_claude_vs_qwen4b.py` — cross-LLM
  analysis
- `scripts/analysis/alpha_sweep_summary.py` — α sweep aggregator

## What's ruled out (honest state)

Three serious attempts at routing a learning-graph signal into
retrieval, all null on real corpora:

1. **Plastic graph activation** — 5 measurement attempts on a
   synthetic 10-topic benchmark, all architecturally intractable.
   `plastic_graph_activation_findings.md`.
2. **Direction 4a (LLM-distilled input projections)** — LoCoMo
   R@5 = 0.334 vs chroma 0.349. Null. Root cause: teacher signal
   diluted through too many downstream layers.
3. **Direction 4b (spatial distillation)** — LoCoMo R@5 = 0.335
   vs chroma 0.349; locality-on makes it worse (0.328). Null with
   negative-signal diagnostic. Root cause: random projection
   compresses but doesn't carry semantics.
   `locomo_direction4b_findings.md`.

The plastic graph substrate ships for research use but no longer
appears in product positioning as a differentiator. `positioning.md`
leads with hybrid retrieval; graph is "What we've ruled out."

## Product positioning as of this milestone

See `docs/positioning.md`. Lead claim is hybrid retrieval; plastic
graph is honestly framed as research-only after three nulls. The
comparison table no longer includes a "Learns from use" claim
(since we can't substantiate it with current data). Comparison rows
preserved: local-first / hybrid retrieval built-in / working-memory
/ episodic store / single-directory portability / crash-safe WAL /
multi-user auth / pluggable backends.

## What's planned next

Two design docs ship alongside this milestone:

1. **Path B: Bio-Validated Primitives** —
   `docs/plans/2026-04-20-path-b-bio-validated-primitives-design.md`.
   Uses the `E:/Documents/Projects/sim` neural simulator as a
   measurement tool to calibrate numpy sparse-code, pattern-
   separation, attractor, and engram primitives. 2-3 weeks. Answers
   "do biological primitives help text retrieval."
2. **Path A: Biophysical Representation Layer** —
   `docs/plans/2026-04-20-path-a-biophysical-representation-layer-design.md`.
   Uses the full sim as SOMA's storage/retrieval substrate. 7-8
   weeks per phase 1-5; phase 6 open-ended. Opens onto artificial-
   life research.

Path B starts first. Path A is gated on Path B's Phase 3 outcome
(answering whether biological primitives have retrieval headroom at
all).

## Reference constellation

Current canonical findings docs, in recommended reading order:

1. `research/developmental/results/longmemeval_full_evidence_roundup.md`
   — headline table + all axes consolidated
2. `research/developmental/results/longmemeval_claude_runner_findings.md`
   — Claude-as-answerer N=500
3. `research/developmental/results/longmemeval_judge_findings_claude.md`
   — Claude-as-judge N=500 triple-agreement
4. `research/developmental/results/longmemeval_alpha_sweep_findings.md`
   — α=0.30 validation
5. `research/developmental/results/locomo_rank_probe_findings.md`
   — LoCoMo +89% rank-1
6. `research/developmental/results/locomo_direction4b_findings.md`
   — closure of 4b / end of graph-retrieval experiments

## Reproducing the headline

```bash
# LongMemEval N=500 strict qwen4b (the +22.8% F1 baseline)
python -m benchmarks.industry.longmemeval.run_qa_compare \
    --variant small --limit 500 \
    --modes chroma_cosine soma_hybrid \
    --provider ollama --model qwen3.5:4b-q8_0 \
    --strict-prompt --out-suffix _n500_strict

# Rank probe for direct rank measurement
python -m benchmarks.industry.longmemeval.rank_probe \
    --variant small --limit 500 --top-k 5

# Claude-as-judge on the outputs (uses Unraid runner)
python -m benchmarks.industry.longmemeval.judge_predictions \
    --pred-jsonl .../qa_compare_soma_hybrid_n500_strict.jsonl \
    --judge-provider claude_runner --judge-model claude \
    --variant small --out-suffix _judged_claude

# α=0.30 validation
python -m benchmarks.industry.longmemeval.alpha_sweep \
    --variant small --limit 500 \
    --alphas 0.0 0.1 0.2 0.3 0.5 0.7 1.0
python -m scripts.analysis.alpha_sweep_summary
```

## Commit trail for this milestone

All changes in this milestone land in a sequence of commits dated
2026-04-20; this doc's siblings in the commit message body list the
file-level detail.
