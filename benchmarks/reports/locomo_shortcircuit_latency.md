# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_shortcircuit | 0.137 | 0.303 | 0.386 | 75.3s | 1.2ms |
| soma-distilled_shortcircuit | 0.137 | 0.303 | 0.386 | 52.1s | 0.7ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| soma-random_shortcircuit | 0.189 | 0.311 | 0.105 | 0.300 | 0.411 |
| soma-distilled_shortcircuit | 0.189 | 0.311 | 0.105 | 0.300 | 0.411 |