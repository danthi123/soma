# LongMemEval QA comparison -- retrieval lift -> QA lift?

Variant: small  |  N=200  |  top-k=5  |  model=qwen3.5:9b-q8_0

| Mode | F1 | EM | ROUGE-1 | ROUGE-L | R@K | input tok | retrieval (ms) | LLM (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chroma_cosine | 0.1118 | 0.0000 | 0.1118 | 0.0968 | 0.885 | 3874 | 10 | 103731 |
| soma_hybrid | 0.1402 | 0.0000 | 0.1402 | 0.1248 | 0.975 | 3873 | 38 | 2778 |
