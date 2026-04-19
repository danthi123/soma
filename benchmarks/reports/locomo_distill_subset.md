# LoCoMo distillation ablation

Direction 4a Phase 3: tests whether LLM-distilled projections
give SOMA's graph rerank retrieval advantage over pure embedding
retrieval using the same embedding model.

## Overall Recall@k

| System | R@1 | R@5 | R@10 | store_total | retrieve_avg |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.166 | 0.325 | 0.421 | 6.6s | 38.5ms |
| soma-random | 0.156 | 0.325 | 0.424 | 23.3s | 16.1ms |
| soma-distilled | 0.152 | 0.331 | 0.421 | 24.8s | 15.5ms |

## Per-category R@5

| System | single-hop R@5 | multi-hop R@5 | temporal R@5 | open-domain R@5 | adversarial R@5 |
| --- | :---: | :---: | :---: | :---: | :---: |
| chroma-mxbai | 0.140 | 0.333 | 0.182 | 0.333 | 0.437 |
| soma-random | 0.140 | 0.317 | 0.273 | 0.342 | 0.423 |
| soma-distilled | 0.140 | 0.333 | 0.273 | 0.351 | 0.423 |

## Primary comparison

**Delta (soma-distilled - chroma-mxbai) on R@5: +0.007** → WEAK