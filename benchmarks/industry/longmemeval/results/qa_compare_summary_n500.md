# LongMemEval QA comparison -- retrieval lift -> QA lift?

Variant: small  |  N=500  |  top-k=5  |  model=qwen3.5:4b-q8_0

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_cosine | 0.1481 | 0.0080 | 0.1481 | 0.1343 | 0.932 | 3879 | 8 | 1878 |
| soma_hybrid | 0.1635 | 0.0060 | 0.1635 | 0.1496 | 0.980 | 3879 | 27 | 1861 |
