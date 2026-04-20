# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_sweep_w0.2 | 0.140 | 0.339 | 0.440 | 160.0s | 14.0ms |
| soma-distilled_sweep_w0.2 | 0.138 | 0.341 | 0.442 | 211.1s | 16.0ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_sweep_w0.2 | 0.238 | 0.346 | 0.196 | 0.356 | 0.395 |
| soma-distilled_sweep_w0.2 | 0.252 | 0.352 | 0.239 | 0.350 | 0.392 |