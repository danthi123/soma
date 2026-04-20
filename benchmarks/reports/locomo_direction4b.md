# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.147 | 0.349 | 0.446 | 52.4s | 10.1ms |
| soma-random | 0.133 | 0.337 | 0.433 | 223.4s | 19.0ms |
| soma-distilled | 0.134 | 0.344 | 0.441 | 238.2s | 19.5ms |
| soma-spatial | 0.134 | 0.335 | 0.437 | 225.0s | 15.9ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.262 | 0.361 | 0.217 | 0.357 | 0.406 |
| soma-random | 0.245 | 0.361 | 0.239 | 0.342 | 0.386 |
| soma-distilled | 0.259 | 0.364 | 0.239 | 0.344 | 0.404 |
| soma-spatial | 0.252 | 0.336 | 0.239 | 0.344 | 0.390 |

## Primary comparison (soma-distilled / soma-spatial variants vs chroma-mxbai on R@5)

| Variant | R@5 | Delta | Verdict |
| --- | :---: | :---: | :---: |
| soma-distilled | 0.344 | -0.005 | NULL |
| soma-spatial | 0.335 | -0.014 | NULL |