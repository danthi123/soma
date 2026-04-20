# LongMemEval QA comparison -- retrieval lift -> QA lift?

Variant: small  |  N=500  |  top-k=5  |  model=claude

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_rerank | 0.3628 | 0.2620 | 0.3628 | 0.3604 | 0.936 | 3879 | 207 | 4239 |
| soma_hybrid | 0.4161 | 0.3040 | 0.4161 | 0.4141 | 0.980 | 3879 | 31 | 4242 |
