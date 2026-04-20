# LongMemEval QA comparison -- retrieval lift -> QA lift?

Variant: small  |  N=100  |  top-k=5  |  model=qwen3.5:4b-q8_0

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_cosine | 0.1677 | 0.0200 | 0.1677 | 0.1588 | 0.850 | 3871 | 8 | 1683 |
| chroma_rerank | 0.1703 | 0.0100 | 0.1703 | 0.1649 | 0.830 | 3871 | 89 | 1772 |
| soma_hybrid | 0.2383 | 0.0200 | 0.2383 | 0.2314 | 0.990 | 3870 | 26 | 1664 |
| full_context | 0.0286 | 0.0000 | 0.0286 | 0.0254 | 0.040 | 2032 | 0 | 899 |
