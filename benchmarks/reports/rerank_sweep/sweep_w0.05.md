# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_sweep_w0.05 | 0.148 | 0.352 | 0.449 | 197.2s | 16.7ms |
| soma-distilled_sweep_w0.05 | 0.148 | 0.350 | 0.448 | 217.2s | 16.5ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_sweep_w0.05 | 0.262 | 0.371 | 0.239 | 0.360 | 0.404 |
| soma-distilled_sweep_w0.05 | 0.259 | 0.364 | 0.239 | 0.357 | 0.408 |