# LongMemEval — qwen9b strict predictions, 4b judge (N=500)

Judged 500 paired items. Judge model: qwen3.5:4b-q8_0.

## Headline

| System | Judge-accuracy |
| --- | ---: |
| chroma cosine @ qwen9b | 0.3600 |
| **SOMA hybrid @ qwen9b** | **0.4340** |
| Delta | +0.0740 (+20.56%) |

## Per-question-type accuracy

| Type | N | chroma | SOMA | abs lift | rel lift |
| --- | ---: | ---: | ---: | ---: | ---: |
| knowledge-update | 78 | 0.513 | 0.590 | +0.077 | +15.00% |
| multi-session | 133 | 0.188 | 0.233 | +0.045 | +24.00% |
| single-session-assistant | 56 | 0.929 | 0.964 | +0.036 | +3.85% |
| single-session-preference | 30 | 0.200 | 0.200 | +0.000 | +0.00% |
| single-session-user | 70 | 0.500 | 0.829 | +0.329 | +65.71% |
| temporal-reasoning | 133 | 0.165 | 0.165 | +0.000 | +0.00% |

## Where SOMA and chroma disagree

- SOMA correct, chroma wrong: **53**
- Chroma correct, SOMA wrong: **16**
- Both correct: 164
- Both wrong: 267

Net items SOMA correct where chroma isn't: **+37**
