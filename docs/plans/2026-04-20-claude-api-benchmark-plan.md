# Claude-API-based benchmark plan

**Status:** Design only. Ready to execute when `ANTHROPIC_API_KEY` is
set in the environment.

**Why:** Two experiments have been blocked by local-LLM constraints:
1. SOMA vs Mem0 head-to-head — Mem0's `infer=True` at ~60-120 min/item
   with local qwen3.5:4b.
2. Fair evaluation of retrieval quality with a stronger reasoner —
   qwen3.5:9b is actually better at finding answers but gets penalized
   by token-F1 for verbosity (see n200_qwen9b findings).

Claude API solves both: fast extraction (~200-500ms/call) and a
reasoner that can follow "answer with just the value" prompts
reliably.

## Experiment 1: Fair SOMA vs Mem0 QA head-to-head

### Setup

Same 100 LongMemEval items for both systems. Same embedder (sbert).
Same Claude model for QA (haiku for speed + cost, or sonnet for
max quality).

- **Arm A: SOMA hybrid + Claude QA**
  - Ingest raw session texts into SOMA MemoryLayer.
  - Retrieve top-5 via `hybrid_alpha=0.3`.
  - Pack into context, send to Claude for answer.

- **Arm B: Mem0 infer=True + Claude QA**
  - Ingest messages via `Memory.add(..., infer=True)` with Claude as
    extraction LLM.
  - Retrieve top-5 extracted facts via `Memory.search()`.
  - Pack into context, send to same Claude model for answer.

- **Arm C: Mem0 infer=False + Claude QA** (fair indexing baseline)
  - Skip LLM extraction, just store + embed + cosine retrieve.
  - Same QA prompt and model.

### Budget estimate

Using Claude haiku 3.5 (~$0.80/M input, ~$4/M output):
- Mem0 extraction: ~50 calls per item × 100 items = 5,000 calls
  × 2K input / 200 output tokens = 10M input + 1M output
  ≈ **$12** for extraction pass.
- QA pass: 100 items × 3 arms × 1 call each = 300 calls
  × 3K input / 100 output ≈ 1M input + 30K output
  ≈ **$2** for QA.
- **Total: ~$15** for the whole head-to-head.

(Using Sonnet 4 would be ~10× more but gives a clearer signal on
whether a stronger reasoner still favors SOMA.)

### What we learn

- **Ingest cost parity**: the ~60 min/item Mem0 penalty is about
  local LLM, not the architecture. With Claude, Mem0 ingest drops
  to ~2-5 min/item. SOMA ingest stays at ~1s/item. A real cost
  comparison: for 1M turns, Mem0-via-Claude costs ~$X of API calls
  vs SOMA's zero.
- **QA quality**: does Mem0's extracted facts deliver better or
  worse QA than SOMA's raw-text hybrid retrieval when both feed the
  same strong reasoner?
- **The interesting hypothesis**: SOMA wins on retrieval-bottlenecked
  questions (+49% F1 pattern we saw). Mem0 might win on multi-session
  questions where pre-extracted facts save the LLM from doing the
  synthesis at query time.

### Implementation

`benchmarks/industry/longmemeval/run_mem0_compare.py` already exists
and supports all three modes. Adding Claude support is ~10 lines:

```python
# In _call_llm or _run_mem0, swap ollama for Anthropic
from anthropic import Anthropic
client = Anthropic()  # reads ANTHROPIC_API_KEY

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_msg}],
)
return resp.content[0].text.strip()
```

For Mem0's extraction LLM:
```python
config = {
    "llm": {
        "provider": "anthropic",
        "config": {"model": "claude-haiku-4-5-20251001"},
    },
    ...
}
```

## Experiment 2: Fair bigger-LLM retrieval-to-QA test

### Setup

Re-run the `run_qa_compare.py` harness with Claude sonnet as the QA
generator and a STRICT answer-format prompt:

```
System: Answer with ONLY the specific fact (1-5 words). No
explanation, no "Based on..." preamble. If not in context, say
exactly "I don't know".
```

With a strong LLM + strict prompting:
- Bigger LLM can't hide behind verbosity.
- F1 becomes a fair metric across model sizes.
- The question: does SOMA's retrieval win still matter when the
  reasoner is not the bottleneck?

### Hypothesis

Two possible outcomes:

**A. SOMA wins more clearly**: retrieval is still the binding
constraint for multi-session questions, and a strong reasoner
AMPLIFIES the value of better retrieval.

**B. SOMA wins less clearly**: strong reasoner compensates for
imperfect retrieval, closing the gap.

The result shapes the positioning: SOMA either "gets more valuable
with better LLMs" (A) or "SOMA's retrieval advantage is bounded by
LLM capability" (B).

### Budget estimate

Using Claude sonnet 4 (~$3/M input, ~$15/M output):
- 100 items × 3 arms (chroma, soma, chroma+rerank) × 1 call
  × 3K input / 50 output tokens ≈ 1M input + 15K output
- **Total: ~$5** for the QA pass.

### What we learn

Definitive answer to "does the +10% F1 overall and +49% single-session
story hold with a strong LLM?" If yes, the positioning gets
strengthened. If no, we need to narrow the claim further.

## Experiment 3: Plastic graph activation test (optional)

Separately, the plastic graph activation test
(`docs/plans/2026-04-20-plastic-graph-activation-design.md`) also
benefits from Claude:
- Agent simulating the user in a streaming conversation is much more
  realistic with Claude than with qwen3.5:4b.
- QA side uses Claude for fair evaluation.

Budget: ~$10-20 depending on session length and number of trials.

## Decision: which to run first?

Ordered by value-per-dollar:

1. **Experiment 2 (fair bigger-LLM retrieval test, ~$5)**: closes the
   question of whether the retrieval-to-QA story holds with a real
   reasoner. Highest signal-to-cost.
2. **Experiment 1 (SOMA vs Mem0 with Claude, ~$15)**: positions us
   vs the market leader with real head-to-head numbers.
3. **Experiment 3 (plastic graph, ~$15)**: speculative; only if the
   retrieval story is solidly locked in.

Total for all three: ~$35. Worth it for the positioning coverage.

## Prerequisites

- [ ] `ANTHROPIC_API_KEY` exported in shell.
- [ ] `pip install anthropic` (already installed, v0.45.2).
- [ ] `run_qa_compare.py` + `run_mem0_compare.py` modified to support
      `--llm-provider anthropic` + `--llm-model MODEL_NAME` flags.
- [ ] Stricter system prompt for Claude runs (answer-with-value-only).
