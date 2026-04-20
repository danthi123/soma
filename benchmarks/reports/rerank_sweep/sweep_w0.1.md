# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_sweep_w0.1 | 0.145 | 0.347 | 0.445 | 219.5s | 20.5ms |
| soma-distilled_sweep_w0.1 | 0.145 | 0.349 | 0.446 | 196.1s | 15.6ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_sweep_w0.1 | 0.248 | 0.364 | 0.239 | 0.356 | 0.404 |
| soma-distilled_sweep_w0.1 | 0.259 | 0.368 | 0.239 | 0.356 | 0.401 |