# LongMemEval QA comparison -- retrieval lift -> QA lift?

Variant: small  |  N=500  |  top-k=5  |  model=qwen3.5:9b-q8_0

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_cosine | 0.2847 | 0.1860 | 0.2847 | 0.2821 | 0.932 | 3879 | 13 | 1871 |
| soma_hybrid | 0.3472 | 0.2220 | 0.3472 | 0.3457 | 0.980 | 3879 | 45 | 1885 |
