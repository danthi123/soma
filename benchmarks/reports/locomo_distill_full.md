# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.147 | 0.349 | 0.447 | 196.0s | 34.3ms |
| soma-random | 0.130 | 0.343 | 0.441 | 188.0s | 16.0ms |
| soma-distilled | 0.127 | 0.334 | 0.436 | 198.0s | 16.3ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.262 | 0.361 | 0.228 | 0.357 | 0.406 |
| soma-random | 0.248 | 0.368 | 0.239 | 0.352 | 0.390 |
| soma-distilled | 0.245 | 0.340 | 0.239 | 0.344 | 0.388 |

## Primary comparison

**Delta (soma-distilled - chroma-mxbai) on R@5: -0.015** → NULL