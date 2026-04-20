# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-distilled_locality | 0.135 | 0.337 | 0.436 | 196.2s | 16.7ms |
| soma-spatial_locality | 0.132 | 0.328 | 0.436 | 203.0s | 14.2ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-distilled_locality | 0.230 | 0.343 | 0.239 | 0.350 | 0.395 |
| soma-spatial_locality | 0.230 | 0.327 | 0.217 | 0.341 | 0.388 |