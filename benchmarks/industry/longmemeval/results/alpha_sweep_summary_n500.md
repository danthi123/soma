# Hybrid alpha sweep — N=500 LongMemEval small-variant
Retrieval-only; ``alpha`` blends BM25 and cosine (``alpha=0.0`` pure BM25, ``1.0`` pure cosine). Gold session rank probe.
| alpha | N | hit@5 | rank=1 frac | mean rank | median rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 500 | 0.9680 | 0.8620 | 1.205 | 1 |
| 0.10 | 500 | 0.9720 | 0.8720 | 1.175 | 1 |
| 0.20 | 500 | 0.9740 | 0.8840 | 1.144 | 1 |
| 0.30 | 500 | 0.9800 | 0.8860 | 1.161 | 1 |
| 0.50 | 500 | 0.9800 | 0.8680 | 1.171 | 1 |
| 0.70 | 20 | 0.8500 | 0.6500 | 1.353 | 1 |

## Rank-1 fraction per question type
| alpha | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.949 | 0.865 | 1.000 | 0.400 | 0.900 | 0.835 |
| 0.10 | 0.962 | 0.880 | 1.000 | 0.433 | 0.900 | 0.842 |
| 0.20 | 0.974 | 0.895 | 1.000 | 0.467 | 0.886 | 0.865 |
| 0.30 | 0.962 | 0.910 | 1.000 | 0.567 | 0.871 | 0.850 |
| 0.50 | 0.897 | 0.917 | 0.982 | 0.667 | 0.814 | 0.827 |
| 0.70 | 0.000 | 0.000 | 0.000 | 0.000 | 0.650 | 0.000 |
