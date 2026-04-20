# LongMemEval QA comparison — retrieval lift → QA lift?

Variant: small  |  N=2  |  top-k=5  |  model=qwen3.5:4b-q8_0

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_rerank | 0.3333 | 0.0000 | 0.3333 | 0.3333 | 0.500 | 60 | 3757 |
| soma_hybrid | 0.3333 | 0.0000 | 0.3333 | 0.3333 | 1.000 | 26 | 1393 |
