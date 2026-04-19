# Direction 4a design: LLM-distilled projections for semantic grounding

**Date:** 2026-04-19
**Author:** autonomous research session
**Parent:** `docs/plans/2026-04-18-grounding-plasticity.md` §Direction 4a
**Status:** design approved (autonomous mode) → implementation

## Motivation

The session cascade on 2026-04-19 (commits `2ba566b` → `70ca36c`)
established that SOMA's structural plasticity works when admission
is gated by a STRUCTURAL prior (positional locality) rather than a
TEMPORAL one (PE, co-activation). On v0.5 prediction this is a clean
positive: multi-seed validated, inverted-U characterized, four
sparsity controls ruled out.

But this does NOT transfer to retrieval — LoCoMo shows locality is
vacuous because the retrieval task reads cosine similarity on random
projections, and random projections are semantically arbitrary. The
§5 retrieval ceiling from the paper persists.

Direction 4a's hypothesis: **replacing random projections with
projections distilled from a pretrained embedding model gives the
graph SEMANTIC structure to operate on. Locality would then mean
"semantic neighborhood" rather than "position-space neighborhood,"
and retrieval should benefit.**

## Success criteria (the critical part)

The user's directive: "show SOMA adds value that standard LLMs
without SOMA don't." This imposes a strict comparison design.

**Primary**: on LoCoMo, `soma-graph-distilled` > `chroma-same-embedder`
(both using the same teacher embeddings). Delta must be > noise
floor (probably 0.01-0.02 on R@5 for the effect to be shippable).

**Secondary**: `soma-graph-distilled+locality` > `soma-graph-distilled`
(the locality principle still contributes after distillation).

**Tertiary**: `soma-graph-distilled` > `soma-graph-random` (distillation
itself helps, independent of locality).

If ALL THREE pass → ship as "SOMA with LLM-distilled graph delivers
retrieval beyond what vector DBs can do with the same embeddings."

If only primary fails but secondary and tertiary pass → the mechanism
is real but doesn't beat well-tuned vector retrieval. Smaller claim.

If primary and tertiary both fail → distillation doesn't help on
retrieval. Honest null. Pivot research focus.

## Teacher model selection

Pulled `mxbai-embed-large` (1024-dim, MTEB-leading, runs via ollama).

Rationale:
- **Strong** enough that the distilled student has signal to learn
- **Local** so reproducible and offline
- **Fast** enough for 5882 LoCoMo turns without blocking
- **Uniform API** via ollama so Chroma baseline uses identical teacher
- Already installed; no cloud deps

Fallback if mxbai underperforms: `nomic-embed-text` (768-dim, already
there) or `snowflake-arctic-embed` (pullable). Cloud escalation only
if local hits ceiling AND we want to stretch the claim.

## Architecture

### Distillation loss

Add to `PredictiveSOMA`'s training step:

```python
# Alongside the existing prediction loss
if config.projection_distillation_target == "llm_embedding":
    teacher_emb = teacher.embed(input_text)  # cached
    student_output = mean(proj @ _last_summary)  # existing
    distill_loss = 1 - cosine_similarity(student_output, teacher_emb)
    total_loss = pred_loss + alpha * distill_loss
```

Key design decisions:
- **Cosine loss** not L2: matches retrieval metric; scale-invariant;
  proven in knowledge-distillation literature.
- **alpha hyperparameter** controls distillation weight vs
  prediction. Start at 1.0 (equal weight); sweep if needed.
- **Teacher caching**: teacher embeddings are deterministic per
  input. Cache to avoid redundant API calls. SHA-256 of input text
  as key.
- **Projections are the student**: `nn.Parameter` (as in Direction
  2B), updated via Adam. Same code path as 2B, just a different
  target.

### Teacher interface

```python
class LLMTeacher:
    def __init__(self, model: str, base_url: str): ...
    def embed(self, text: str) -> torch.Tensor:  # shape (emb_dim,)
        ...
    def embed_batch(self, texts: list[str]) -> torch.Tensor:  # (B, emb_dim)
        ...

class OllamaEmbedder(LLMTeacher):
    # Calls POST /api/embeddings
    ...

class CachedEmbedder(LLMTeacher):
    # Wraps any teacher; caches by hash(text)
    ...
```

Cache persists to disk between runs (`benchmarks/.teacher_cache/<model>/<hash>.pt`)
so repeated benchmarks don't re-embed.

### Config surface

```python
# New SOMAConfig fields
projection_distillation_target: Literal["none", "llm_embedding"] = "none"
projection_distillation_model: str = "mxbai-embed-large"
projection_distillation_base_url: str = "http://localhost:11434"
projection_distillation_weight: float = 1.0  # alpha
```

Default `"none"` preserves existing behavior. Opt-in preserves
reproducibility of prior results.

## Experimental design

### Phase 1: Implementation (TDD)

- Config fields (validate enum, weight >= 0)
- Teacher interface + Ollama adapter + cache
- Distillation loss added to PredictiveSOMA
- Unit tests: teacher returns correct shape; cache hit/miss; loss
  decreases on trivial input; non-distill path unchanged.

Deliverable: all tests pass, synap_local v0.5 results unchanged
(backward compat).

### Phase 2: Synthetic validation

Run on v0.5 capacity schedule with distillation ON. Since v0.5
inputs are synthetic vectors (not text), create a "pseudo-text"
mapping: convert each 16-dim synthetic vector to a stable text
description (e.g., "vector-regime-mlp_2x16-step-500"). Teacher
embeds those descriptions.

Hypothesis: distillation loss should decrease smoothly. Projections
should converge to have higher cosine similarity to teacher vectors
over time.

Checks:
- Does MSE on prediction task stay similar to synap_local (distillation
  shouldn't hurt prediction if done right)?
- Do projection-teacher cosine similarities increase over training?

If Phase 2 shows distillation learns cleanly, move to Phase 3.

### Phase 3: LoCoMo retrieval test (the critical one)

Five systems, all using `mxbai-embed-large` for retrieval/distillation:

1. `chroma-mxbai` — Chroma with mxbai embeddings (the baseline to beat)
2. `soma-flat-mxbai` — SOMA with MemoryLayer using mxbai cosine, no graph
3. `soma-graph-random` — attach_soma, alpha=0.3, locality=0.0, distillation=none
4. `soma-graph-distilled` — attach_soma, alpha=0.3, locality=0.0, distillation=llm_embedding
5. `soma-graph-distilled+local` — attach_soma, alpha=0.3, locality=0.5, distillation=llm_embedding

Metrics: R@1, R@5, R@10 overall + per-category (single-hop, multi-hop,
temporal, open-domain, adversarial).

**Fix the CATEGORY_NAMES iteration bug** from the first LoCoMo run
before running this so per-category numbers actually report.

Primary comparison: (4) or (5) vs (1). Delta > 0.02 on R@5 = ship
material. Delta < 0.005 = null.

### Phase 4: Composition with locality (cutoff sweep on distilled)

If Phase 3 shows distillation helps, run a cutoff sweep on the
distilled variant: {0.0, 0.25, 0.5, 0.75, 1.0}. Hypothesis: the
cutoff sweet spot may shift because "distance in position space"
now correlates with "distance in semantic space" (shared-rng
property). Characterize.

### Phase 5: Temporal / multi-hop stress test

Design or find a test that specifically stresses:
- **Multi-hop queries**: answer requires chaining two memories
- **Temporal patterns**: co-occurring memories over interaction history

LoCoMo has categories for multi-hop and temporal — if the category
breakdown fix works, we can read these directly. Otherwise, small
synthetic probe.

Hypothesis: SOMA-graph-distilled specifically wins on multi-hop /
temporal vs Chroma-mxbai, because the graph encodes structural
information that cosine similarity on single-vector embeddings
can't capture.

This is the "value SOMA adds over standard LLM" demonstration.

## Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Distillation loss dominates prediction loss, wrecks v0.5 | Medium | Alpha tuning; fallback to smaller alpha |
| Teacher embeddings too strong; distilled student is just a copy | Medium | Graph should add structure *on top* of distilled proj, not replace |
| Teacher cache grows unbounded on LoCoMo (5882 items × 1024 floats × 4 bytes ≈ 24 MB per run) | Low | Disk cache; acceptable size |
| Ollama throughput blocks training | Low | Pre-compute + cache all LoCoMo embeddings before first benchmark |
| SOMA's graph re-rank formula doesn't exploit distilled projections meaningfully | Medium | Phase 3 result would surface this; may need to revisit re-rank formula |

## Timeline

- Phase 1 (implementation): ~3-4 hours
- Phase 2 (synthetic validation): ~1-2 hours
- Phase 3 (LoCoMo): ~2-3 hours total (cache pre-compute + 5 runs at ~30 min each)
- Phase 4 (cutoff sweep on distilled): ~2 hours
- Phase 5 (multi-hop / temporal analysis): ~1-2 hours

Total: ~10-15 hours, or 1.5-2 full work days. Timeboxed: if Phase 3
shows null, stop and document.

## Exit criteria

**SHIP condition**: Phase 3 primary + secondary + tertiary all pass.
Write up, update positioning docs, rerun LoCoMo as a promoted
benchmark.

**STOP condition**: Phase 3 primary fails. Document honestly. The
§5 retrieval ceiling doesn't break under this intervention. Pivot
research focus to other angles (e.g., temporal / continual learning
demonstration) or narrow the product positioning.

**ITERATE condition**: Phase 3 partial — distillation helps but
Chroma-mxbai is still equal-or-better. Try alpha sweep, compose
with locality more aggressively, or revisit the re-rank formula.
Timebox: one additional day.

## Open implementation questions (to decide during coding)

1. Should distillation happen during `consolidate()` or during
   `process_input()`? Currently leaning toward `process_input()`
   because that's where the prediction loss already lives, and
   projections should update per-input for stable gradient signal.
2. Should teacher embeddings be computed per-node (each SENSOR node
   gets its own distillation target) or per-input (shared across
   nodes)? Leaning per-input: SOMA's graph expresses diversity
   through node-level noise, but the distillation target should
   be shared semantic ground truth.
3. Should the distillation apply to ALL projections or just SENSOR
   inputs? Direction 2B applied to SENSOR→summary projections.
   Start there; expand later if helpful.
