# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-distilled_a01_local | 0.127 | 0.341 | 0.440 | 185.2s | 24.4ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-distilled_a01_local | 0.241 | 0.343 | 0.228 | 0.356 | 0.397 |