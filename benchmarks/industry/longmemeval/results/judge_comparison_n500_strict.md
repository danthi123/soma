# Judge-vs-F1 comparison (chroma_cosine vs soma_hybrid)

Suffix: `_n500_strict_judged_qwen4b`

## Headline

| Mode | F1 | judge_accuracy |
| --- | ---: | ---: |
| chroma_cosine | 0.2994 | 0.3600 |
| soma_hybrid | 0.3677 | 0.4400 |

**SOMA lift: F1 +22.8%  |  judge-accuracy +22.2%**

## Per-type breakdown

| Type | N | A F1 | B F1 | F1 lift | A judge | B judge | judge lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi-session | 133 | 0.113 | 0.154 | +36% | 0.188 | 0.195 | +4% |
| temporal-reasoning | 133 | 0.188 | 0.231 | +23% | 0.173 | 0.256 | +48% |
| knowledge-update | 78 | 0.408 | 0.451 | +10% | 0.487 | 0.538 | +11% |
| single-session-user | 70 | 0.482 | 0.767 | +59% | 0.529 | 0.843 | +59% |
| single-session-assistant | 56 | 0.771 | 0.766 | +-1% | 0.929 | 0.946 | +2% |
| single-session-preference | 30 | 0.031 | 0.031 | +2% | 0.167 | 0.200 | +20% |

## Where judge and F1 disagree

Items where judge says CORRECT but F1 is low (<0.5), or judge says WRONG but F1 is high (>0.5):

| Mode | judge=1 & F1<0.5 (F1 underscored) | judge=0 & F1>0.5 (F1 flattered) |
| --- | ---: | ---: |
| chroma_cosine | 46 | 4 |
| soma_hybrid | 51 | 5 |

High 'judge=1 & F1<0.5' count => the mode is giving semantically-correct but lexically-different answers. High 'judge=0 & F1>0.5' => judge disagrees with F1 on overlap-heavy but wrong-meaning answers (rare).
