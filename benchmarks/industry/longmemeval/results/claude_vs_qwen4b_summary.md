# LongMemEval QA — Claude vs qwen4b on identical retrieval

## Headline F1

| Mode | qwen4b F1 | Claude F1 | Claude/qwen4b |
| --- | ---: | ---: | ---: |
| chroma_cosine (N_q=500, N_c=0) | 0.2768 | 0.0000 | +-100.0% |
| chroma_rerank (N_q=0, N_c=500) | 0.0000 | 0.3328 | +-100.0% |
| soma_hybrid (N_q=500, N_c=500) | 0.3413 | 0.3863 | +13.2% |

## SOMA lift (soma_hybrid / chroma_rerank)

| LLM | chroma F1 | SOMA F1 | Relative lift |
| --- | ---: | ---: | ---: |
| qwen4b | 0.0000 | 0.3413 | **+0.0%** |
| claude | 0.3328 | 0.3863 | **+16.1%** |

## Per-type F1 (Claude)

| Type | chroma_cosine | chroma_rerank | soma_hybrid |
| --- | ---: | ---: | ---: |
| knowledge-update | — | 0.448 (n=78) | 0.480 (n=78) |
| multi-session | — | 0.185 (n=133) | 0.211 (n=133) |
| single-session-assistant | — | 0.652 (n=56) | 0.638 (n=56) |
| single-session-preference | — | 0.016 (n=30) | 0.014 (n=30) |
| single-session-user | — | 0.503 (n=70) | 0.775 (n=70) |
| temporal-reasoning | — | 0.260 (n=133) | 0.280 (n=133) |
