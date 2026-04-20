# LongMemEval QA comparison -- retrieval lift -> QA lift?

Variant: small  |  N=10  |  top-k=5  |  model=claude

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_rerank | 0.4889 | 0.4000 | 0.4889 | 0.4889 | 0.800 | 3869 | 1240 | 3704 |
| soma_hybrid | 0.7746 | 0.6000 | 0.7746 | 0.7746 | 1.000 | 3869 | 62 | 3842 |
