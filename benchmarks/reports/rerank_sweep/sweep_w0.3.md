# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_sweep_w0.3 | 0.132 | 0.334 | 0.435 | 176.1s | 15.1ms |
| soma-distilled_sweep_w0.3 | 0.133 | 0.343 | 0.438 | 179.5s | 13.5ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_sweep_w0.3 | 0.241 | 0.349 | 0.228 | 0.342 | 0.386 |
| soma-distilled_sweep_w0.3 | 0.252 | 0.352 | 0.239 | 0.352 | 0.399 |