# LongMemEval QA comparison -- retrieval lift -> QA lift?

Variant: small  |  N=500  |  top-k=5  |  model=qwen3.5:4b-q8_0

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_cosine | 0.2994 | 0.1960 | 0.2994 | 0.2954 | 0.932 | 3879 | 11 | 1510 |
| soma_hybrid | 0.3677 | 0.2420 | 0.3677 | 0.3638 | 0.980 | 3879 | 39 | 1460 |
