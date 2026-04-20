# LongMemEval QA comparison -- retrieval lift -> QA lift?

Variant: small  |  N=20  |  top-k=5  |  model=qwen3.5:9b-q8_0

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| soma_hybrid | 0.7517 | 0.5000 | 0.7517 | 0.7517 | 1.000 | 3870 | 35 | 1756 |
