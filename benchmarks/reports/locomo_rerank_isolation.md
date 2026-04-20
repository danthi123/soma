# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.147 | 0.349 | 0.447 | 4.3s | 1.9ms |
| soma-random_norerank | 0.148 | 0.350 | 0.447 | 220.3s | 18.9ms |
| soma-distilled_norerank | 0.148 | 0.350 | 0.447 | 215.4s | 17.6ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.262 | 0.361 | 0.228 | 0.357 | 0.406 |
| soma-random_norerank | 0.262 | 0.364 | 0.228 | 0.357 | 0.406 |
| soma-distilled_norerank | 0.262 | 0.364 | 0.228 | 0.357 | 0.406 |

## Primary comparison (soma-distilled / soma-spatial variants vs chroma-mxbai on R@5)

| Variant | R@5 | Delta | Verdict |
| --- | :---: | :---: | :---: |
| soma-distilled_norerank | 0.350 | +0.001 | NULL |