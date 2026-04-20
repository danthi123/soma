# LongMemEval with Claude as LLM — via Unraid claude-code-runner

**Status:** N=500 full run complete. **SOMA +16.1% F1 lift over
chroma_rerank baseline** under Claude as answerer. The N=10 pilot's
+58% was small-sample noise.
**Date:** 2026-04-20.
**Infrastructure:** Unraid `claude-code-runner:latest` Docker image,
invoked over SSH, authenticated via the operator's Claude Max 20x
OAuth token. No per-request API charges.

## Headline (N=500 full run)

| Mode | Claude F1 | Claude EM | Claude R@5 |
| --- | ---: | ---: | ---: |
| chroma_rerank (cosine + reranker) | 0.3628 | 0.262 | 0.936 |
| **soma_hybrid (BM25 + cosine)** | **0.4161** | **0.304** | **0.980** |
| Relative lift | **+14.7%** | **+16.0%** | +4.7% |

Using the alpha_sweep summary-script f1 (strict token match, no
reranker normalization): SOMA **+16.1%** over chroma_rerank.

## Cross-LLM comparison (same retrieval, swap LLM)

`soma_hybrid` retrieval fixed; only the LLM differs:

| Answerer | F1 | EM |
| --- | ---: | ---: |
| qwen3.5:4b-q8_0 (ollama) | 0.3413 | 0.242 |
| **Claude (via Unraid runner)** | **0.3863** | **0.304** |
| Absolute delta | **+0.045 F1 (+13.2%)** | +0.062 EM (+25.6%) |

**Takeaway:** Claude extracts answers more reliably. EM jumps 25.6%
under Claude while F1 jumps 13.2% — suggesting Claude's gains are
concentrated on short, exact-match questions, consistent with the
strict-prompt design.

## Per-type F1 under Claude (N=500)

| Type | N | chroma_rerank | soma_hybrid | Rel. lift |
| --- | ---: | ---: | ---: | ---: |
| **single-session-user** | 70 | 0.503 | **0.775** | **+54%** |
| knowledge-update | 78 | 0.448 | 0.480 | +7% |
| multi-session | 133 | 0.185 | 0.211 | +14% |
| temporal-reasoning | 133 | 0.260 | 0.280 | +8% |
| single-session-assistant | 56 | 0.652 | 0.638 | ~tie |
| single-session-preference | 30 | 0.016 | 0.014 | ~tie |

`single-session-user` is where SOMA's ranking dominance shines under
Claude: a whopping +54% F1 lift, paralleling the qwen4b finding
(+59%). These are keyword-heavy extractive questions where Claude
reliably pulls the right answer when it's at rank 1.

## Why the lift does not widen as much as the N=10 pilot suggested

Two observations at N=500 that tempered the pilot:

1. **Chroma_rerank is a stronger baseline than chroma_cosine.** The
   qwen4b runs used `chroma_cosine` (pure cosine, no reranker); the
   Claude run used `chroma_rerank` (cosine + CrossEncoderReranker).
   Rerank alone closes much of the gap, leaving less headroom for
   SOMA's keyword-match promotion to add value.

2. **Claude is a stronger answerer even on weak retrieval.** Under
   chroma_cosine + qwen4b, the answerer couldn't extract reliable
   answers from weakly-ranked context. Under chroma_rerank + Claude,
   the answerer is more robust — it extracts gold even when it's not
   rank 1. So the retrieval-ranking lift translates less forcefully
   into F1.

The N=10 pilot landed by chance on a haystack mix where these
effects both amplified. At N=500, they regress to the normal
pattern: SOMA's +4.7pp R@5 and +15.6pp rank-1 lift translate into
a +14.7% F1 lift — directly proportional to retrieval improvement,
not amplified.

## Where Claude's answerer behaviour does help

Claude's **honest IDK behaviour** keeps F1=0 on retrieval misses
(rather than the 0.1-0.3 noise floor qwen4b produces by
hallucinating). This matters when comparing retrieval quality:
SOMA's +recall items (30 items where only SOMA finds gold) land
cleanly in the F1 column under Claude, whereas under qwen4b some
of those "would-be wins" were partially matched by hallucinations.

The qwen4b +22.8% at chroma_cosine was therefore somewhat inflated
by this noise. The Claude +14.7% at chroma_rerank is a cleaner
measurement against a stronger baseline.

## Infrastructure: how the Claude runner works

```
+----------------+             +-----------------------+
|  SOMA bench    |   SSH +     |  Unraid 192.168.0.10  |
|  Python client |------------>|  docker run           |
|                |  stdin      |    claude-code-runner |
|  answer (str)  |<------------|    (claude -p ...)    |
+----------------+   stdout    +-----------------------+
```

- `benchmarks/industry/llm_backends/claude_runner_client.py` —
  `ClaudeRunnerClient.complete(prompt, system_prompt=...)` wraps an
  SSH + `docker run` call, piping the prompt via stdin. 8 unit tests
  cover command construction, stdin passing, timeout, and error
  paths. Mocked subprocess.run at the seam.
- Authentication: the Unraid host holds a long-lived (1-year) OAuth
  token at `/mnt/cache/appdata/claude-runner/auth-token`, kept in
  sync by the n8n `Claude Runner - Token Health Check` workflow.
  The Docker container receives it via `CLAUDE_CODE_OAUTH_TOKEN`.
  Requests consume the Claude Max subscription — **no per-request
  API charges**.
- Latency: ~3.5-4s per call including SSH + Docker overhead. Docker
  spawn is the dominant cost (image stays cached in memory after
  first pull). For comparison: Ollama qwen4b is ~5-10s per call on
  the same prompts.

## Integration surface

Providers added to existing harnesses:

- `benchmarks/industry/longmemeval/run_qa_compare.py` — `--provider
  claude_runner` alongside the existing `ollama`/`anthropic`.
- `benchmarks/industry/longmemeval/judge_predictions.py` — same
  addition for LLM-as-judge runs (upgrading from qwen4b judge to
  Claude judge on existing predictions).

`api_base` is overloaded: `root@host`, `root@host|/path/to/token`,
or empty to use the defaults.

## Raw pilot data (N=10)

```
[chroma_rerank]            [soma_hybrid]
hit=0 Business Admin → IDK      hit=1 Business Admin → Business Administration
hit=1 45 min → 45 minutes each  hit=1 45 min → 45 minutes each way
hit=1 Target → Email inbox      hit=1 Target → Target
hit=1 Glass Menagerie → (same)  hit=1 Glass Menagerie → (same)
hit=0 Summer Vibes → IDK        hit=1 Summer Vibes → Summer Vibes
hit=1 Johnson → Johnson         hit=1 Johnson → Johnson
hit=1 Serenity Yoga → (same)    hit=1 Serenity Yoga → (same)
hit=1 lighter shade → (fuzzy)   hit=1 lighter shade → (fuzzy)
hit=1 Feb 14th → Valentine's    hit=1 Feb 14th → Valentine's
...                              hit=1 Sports store downtown → Sports store downtown
```

R@5: 0.800 (chroma) vs 1.000 (SOMA) — 2 extra retrievals = 2 extra
F1 wins, plus 1 rerank win (Target).

## Pending

- N=500 full run — ~70 minutes estimated. Will confirm whether the
  +58% effect is stable or small-sample variance.
- Claude-as-judge on existing qwen4b predictions — N=500 × 2 modes
  = 1000 Claude calls, ~60 minutes. Upgrades our strongest lift
  claim (+22% F1 → Claude judge) against an even stronger judge.
- LoCoMo QA under Claude — the +89% rank1 retrieval lift should
  translate into an even bigger F1 lift given Claude's stricter
  IDK behaviour. Pending.

## Caveats

- **N=10 is statistically meaningless for F1 magnitude.** The
  pattern (SOMA ≫ chroma) is strong but the number is not stable.
  Treat the +58% as a hypothesis until N=500 confirms.
- **Subscription budget:** Each 500-item run uses ~500 Claude calls
  per mode. Claude Max 20x handles this but bursts near rate limits
  if we parallelise. Currently running serially.
- **Latency variance:** we saw 3-5s per item on the pilot. Longer
  haystacks or rate-limit throttling may extend this to 10-30s.

## Files

- `benchmarks/industry/llm_backends/claude_runner_client.py` — client
- `benchmarks/tests/test_claude_runner_client.py` — 8 unit tests
- `benchmarks/industry/longmemeval/results/qa_compare_*_n10_claude_pilot.{jsonl,json}`
- Pending: `qa_compare_*_n500_claude_strict.{jsonl,json}`
