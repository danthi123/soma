# Retrieval-to-F1 causal analysis (chroma_cosine vs soma_hybrid)

Suffix: `_n500_strict`  |  N_a=500  N_b=500

## Retrieval x F1 partition

(a_hit, b_hit) -> (0,0) = neither retrieved gold, etc.

- **(0,0) neither retrieved gold**: 4 items | chroma_cosine F1=0.000 vs soma_hybrid F1=0.000 | chroma_cosine wins=0 soma_hybrid wins=0 tied-correct=0 tied-fail=4

- **(0,1) only soma_hybrid retrieved**: 30 items | chroma_cosine F1=0.013 vs soma_hybrid F1=0.396 | chroma_cosine wins=1 soma_hybrid wins=15 tied-correct=0 tied-fail=14

- **(1,0) only chroma_cosine retrieved**: 6 items | chroma_cosine F1=0.000 vs soma_hybrid F1=0.019 | chroma_cosine wins=0 soma_hybrid wins=1 tied-correct=0 tied-fail=5

- **(1,1) both retrieved gold**: 460 items | chroma_cosine F1=0.325 vs soma_hybrid F1=0.374 | chroma_cosine wins=24 soma_hybrid wins=54 tied-correct=119 tied-fail=263


## Per-type F1 win distribution

| Question type | N | chroma_cosine wins | soma_hybrid wins | tied | soma_hybrid win advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi-session | 133 | 6 | 14 | 113 | +8 |
| temporal-reasoning | 133 | 9 | 20 | 104 | +11 |
| knowledge-update | 78 | 7 | 11 | 60 | +4 |
| single-session-user | 70 | 0 | 22 | 48 | +22 |
| single-session-assistant | 56 | 1 | 1 | 54 | +0 |
| single-session-preference | 30 | 2 | 2 | 26 | +0 |

## Retrieval -> F1 causation check

### Does F1 follow retrieval?

- When only **chroma_cosine** retrieves: N=6
  - chroma_cosine F1 wins=0, soma_hybrid F1 wins=1, tied=5 (retrieval advantage translated: 0/6 = 0.0%)
- When only **soma_hybrid** retrieves: N=30
  - chroma_cosine F1 wins=1, soma_hybrid F1 wins=15, tied=14 (retrieval advantage translated: 15/30 = 50.0%)
