# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.162 | 0.321 | 0.417 | 7.0s | 10.2ms |
| soma-random | 0.139 | 0.321 | 0.424 | 26.1s | 15.5ms |
| soma-distilled | 0.162 | 0.321 | 0.407 | 27.9s | 16.4ms |
| soma-spatial | 0.162 | 0.325 | 0.407 | 30.4s | 15.9ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.140 | 0.317 | 0.182 | 0.333 | 0.437 |
| soma-random | 0.140 | 0.349 | 0.273 | 0.325 | 0.408 |
| soma-distilled | 0.140 | 0.333 | 0.273 | 0.325 | 0.423 |
| soma-spatial | 0.140 | 0.333 | 0.273 | 0.325 | 0.437 |

## Primary comparison (soma-distilled / soma-spatial variants vs chroma-mxbai on R@5)

| Variant | R@5 | Delta | Verdict |
| --- | :---: | :---: | :---: |
| soma-distilled | 0.321 | +0.000 | NULL |
| soma-spatial | 0.325 | +0.003 | NULL |