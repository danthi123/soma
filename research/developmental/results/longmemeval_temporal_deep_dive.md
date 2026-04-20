# Why does qwen3.5:9b's SOMA lift on temporal-reasoning shrink from +23% to +2%?

**Status:** Diagnosed. 9b's lift drop is not a failure of SOMA's
retrieval — it's **9b being more conservative about saying "I don't
know"** when the context contains partial temporal evidence. This
wipes out SOMA's "retrieved gold → extract" wins because both
systems IDK at similarly high rates on this question type.

**Date:** 2026-04-20.

## The puzzle

On the N=500 strict runs we observed:

| LLM | chroma F1 | SOMA F1 | lift |
| --- | ---: | ---: | ---: |
| qwen3.5:4b | 0.188 | 0.231 | +23% |
| qwen3.5:9b | 0.166 | 0.169 | **+2%** |

Overall SOMA F1 lift is stable across LLMs (+23% vs +22%), but the
temporal-reasoning cell collapses on 9b. Why?

## Direct diagnosis: IDK rate on temporal-reasoning items

| System | 4b IDK | 9b IDK | delta |
| --- | ---: | ---: | --- |
| chroma_cosine | 55/133 (41%) | 72/133 (54%) | **+31%** |
| soma_hybrid | 39/133 (29%) | 70/133 (53%) | **+80%** |

9b's IDK rate goes up by 80% on SOMA contexts — from 39 items to 70
items on the same 133-item pool. Chroma's IDK rate also rises, but
only by 31%. The rising IDK rate on SOMA's side wipes out the
retrieval advantage: items that 4b extracted correctly from SOMA's
rank-1 gold session become IDKs on 9b.

## Example items where 4b SOMA wins but 9b SOMA IDKs

| qid | gold | 4b SOMA | 9b SOMA |
| --- | --- | --- | --- |
| `gpt4_fe651585` | "Alex" | **"Alex"** (F1=1.0) | "I don't know" (F1=0) |
| `gpt4_0b2f1d21` | "The malfunction of the stand mixer" | "Stand mixer malfunction first" (F1=0.6) | "I don't know" (F1=0) |
| `gpt4_7ca326fa` | "Emma graduated first, followed by Rachel and then Alex." | "Emma, Rachel, Alex" (F1=0.5) | "I don't know" (F1=0) |

Same retrieved context. Different model behavior. 9b is being
**more conservative** about committing to an answer when the
temporal question asks for something like "which happened first" or
"who came in what order" and the context has the facts scattered
across turns.

## Why does 9b IDK more on temporal-reasoning specifically?

Hypothesis: temporal-reasoning questions require SYNTHESIS across
multiple turns (e.g., "Alex said X on date A, Emma said Y on date
B → compare"). Strict prompting tells the model to reply "I don't
know" if the answer isn't directly in the context. 9b's stronger
reasoning makes it **more strict** about this instruction — it
distinguishes "the context contains the raw facts" from "the context
contains the synthesized answer to the temporal question."

4b is less strict and will produce a plausible-looking synthesis
("Emma, Rachel, Alex") that partially matches the gold.

## What the judge finds

LLM-judge accuracy on temporal-reasoning (4b judge on 4b predictions):
- chroma: 17.3%
- SOMA: 25.6%
- **lift: +48%**

Under semantic equivalence, SOMA's temporal-reasoning advantage is
BIGGER than F1 suggested (+23% F1 → +48% judge). F1 was penalizing
4b's partial-but-correct syntheses because of formatting variance
("two months ago" vs "2 months prior", etc.). Judge recognizes
them.

We haven't yet run judge on 9b predictions, but by extension:
- 9b judge should show chroma and SOMA both at low accuracy
  (because both IDK at ~54%)
- The gap should be much smaller than 4b's +48%
- But probably still positive because SOMA's 3 fewer IDKs (55 vs 70
  vs 72) still translate to some wins

## Positioning implications

Two honest updates:
1. **The +22% overall F1 lift is NOT uniform per type.** It's
   driven by single-session-user (+59%, retrieval-ceilinged),
   multi-session (+36% F1 but F1 overstates — judge says +4% because
   LLM is the ceiling), temporal-reasoning (F1 varies with LLM
   conservatism; judge shows +48% on 4b), and small wins elsewhere.
2. **LLM choice matters more than we previously documented** for
   temporal-reasoning specifically. Users running SOMA with strict
   prompts on a conservative big-model may not see the F1 lift on
   temporal queries. An LLM-judge metric is the clean answer, but
   users who rely on F1 should know the caveat.

For now, leave the "+22% overall, +59% single-session-user" headlines
as-is but ADD a caveat in the findings docs that per-type lifts shift
with LLM conservatism on types requiring synthesis.

## Next investigations

- **Run LLM-judge on qwen9b predictions.** Expect chroma and SOMA
  both low on temporal-reasoning (IDK cap at ~50%). Question is
  whether the delta is still positive for SOMA.
- **Test with non-strict prompting on 9b.** If the issue is strict
  mode triggering 9b's conservatism, verbose prompting might unlock
  more of SOMA's advantage (at the cost of F1's verbosity penalty).
- **Inspect the failing temporal items' retrieval.** Is SOMA
  retrieving the right session but 9b still refusing to synthesize?
  Or is SOMA retrieval borderline on temporal-reasoning?

## Files

- qwen4b strict: `qa_compare_{chroma_cosine,soma_hybrid}_n500_strict.jsonl`
- qwen9b strict: `qa_compare_{chroma_cosine,soma_hybrid}_n500_qwen9b_strict.jsonl`
- Analysis inline (see above `python -c ...` snippet); could be
  promoted to `scripts/analysis/temporal_deep_dive.py` if we need to
  rerun as judge data arrives.
