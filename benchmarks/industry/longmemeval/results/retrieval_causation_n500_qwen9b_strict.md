# Retrieval-to-F1 causal analysis (chroma_cosine vs soma_hybrid)

Suffix: `_n500_qwen9b_strict`  |  N_a=500  N_b=500

## Retrieval x F1 partition

(a_hit, b_hit) -> (0,0) = neither retrieved gold, etc.

- **(0,0) neither retrieved gold**: 4 items | chroma_cosine F1=0.000 vs soma_hybrid F1=0.000 | chroma_cosine wins=0 soma_hybrid wins=0 tied-correct=0 tied-fail=4

- **(0,1) only soma_hybrid retrieved**: 30 items | chroma_cosine F1=0.012 vs soma_hybrid F1=0.396 | chroma_cosine wins=0 soma_hybrid wins=14 tied-correct=0 tied-fail=16

- **(1,0) only chroma_cosine retrieved**: 6 items | chroma_cosine F1=0.000 vs soma_hybrid F1=0.019 | chroma_cosine wins=0 soma_hybrid wins=1 tied-correct=0 tied-fail=5

- **(1,1) both retrieved gold**: 460 items | chroma_cosine F1=0.309 vs soma_hybrid F1=0.351 | chroma_cosine wins=28 soma_hybrid wins=46 tied-correct=116 tied-fail=270


## Per-type F1 win distribution

| Question type | N | chroma_cosine wins | soma_hybrid wins | tied | soma_hybrid win advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi-session | 133 | 6 | 11 | 116 | +5 |
| temporal-reasoning | 133 | 11 | 10 | 112 | -1 |
| knowledge-update | 78 | 7 | 13 | 58 | +6 |
| single-session-user | 70 | 0 | 23 | 47 | +23 |
| single-session-assistant | 56 | 2 | 2 | 52 | +0 |
| single-session-preference | 30 | 2 | 2 | 26 | +0 |

## Retrieval -> F1 causation check

### Does F1 follow retrieval?

- When only **chroma_cosine** retrieves: N=6
  - chroma_cosine F1 wins=0, soma_hybrid F1 wins=1, tied=5 (retrieval advantage translated: 0/6 = 0.0%)
- When only **soma_hybrid** retrieves: N=30
  - chroma_cosine F1 wins=0, soma_hybrid F1 wins=14, tied=16 (retrieval advantage translated: 14/30 = 46.7%)
