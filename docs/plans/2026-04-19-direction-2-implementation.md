# Direction 2 Implementation Plan — Learnable Input Projections

**Status**: drafted 2026-04-19 during Direction 3 multi-seed GPU wait.
**Parent plan**: `docs/plans/2026-04-18-grounding-plasticity.md` Direction 2.
**Prerequisite**: `no_diversify` ablation results (runner committed
2026-04-19 as `research/developmental/env_sequence_v05_no_diversify.py`,
launches after Direction 3 multi-seed finishes).

## Current architecture (as of commit 894695d)

`PredictiveSOMA._input_projections` is a `dict[str, torch.Tensor]` —
one frozen `(dim, dim)` random matrix per associator node. These
projections are used by `_diversify_activations(input_tensor)` to
modulate each associator's `last_activation` by a sigmoid-scaled
alignment between the projection and the current input.

**Key observation**: these projections are NOT in the prediction-loss
gradient path. The prediction head takes `current_summary`, which is
extracted from `step_result["outputs"]` (OUTPUT-node activations
BEFORE diversification modifies associator `last_activation`). The
projections are updated ONLY via `_competitive_learning`'s Hebbian
outer-product rule, not via backprop.

So "the projections are frozen" in the backprop sense, but not fully
static — they drift via Hebbian updates every step the winner changes.

## Direction 2 goal

Create a gradient path from the prediction loss to the input
projections. Intent: if the loss is "predict next observation," and
the projections shape what activations feed the prediction, then
backprop will push projections toward inputs that *help* prediction
— i.e., inputs semantically aligned with the task.

## Design variants (simplest → most complete)

### 2A: Parallel learnable-projection predictor baseline (lowest-lift)

Add a second pathway alongside SOMA: `sensor_input -> learnable
projection matrix -> prediction head`. Train it on the same
next-observation loss. Compare its prediction_MSE against SOMA's.

**Hypothesis**: if the simple learnable-projection baseline matches
or beats SOMA on v0.5, SOMA's graph contributes nothing measurable
that learnable projections alone couldn't. That's an important
diagnostic finding independent of Direction 2 per se.

**Implementation**: add ~30 lines to `PredictiveSOMA.__init__` and
`process_input` for a second forward pass. No graph restructure.

**Effort**: 2 hours.

### 2B: Learnable projections feeding prediction head

Restructure so the prediction head operates on a learnable-projection
view of the activations instead of the raw OUTPUT-node sum.

```python
# In __init__:
if config.projection_mode == "learnable":
    self._input_projections = nn.ParameterDict()
    for node in graph.all_nodes():
        if node.node_type == NodeType.ASSOCIATOR:
            p = nn.Parameter(torch.randn(dim, dim) * 0.1)
            self._input_projections[node.id] = p
    # Include in optimizer:
    self._pred_optimizer = torch.optim.Adam(
        list(self.prediction_head.parameters()) +
        list(self._input_projections.parameters()),
        lr=3e-4,
    )

# In process_input — replace _get_activation_summary with
# a projection-based aggregator:
def _get_projected_summary(self, input_tensor):
    summary = torch.zeros(self.config.sensor_output_dim)
    for node in active_associators:
        proj = self._input_projections[node.id]
        view = proj @ input_tensor  # (dim,)
        gain = sigmoid(view.dot(input_tensor) / input_norm)
        summary += gain * node.last_activation.detach()
    return summary
```

Then `predicted = prediction_head(summary)` trains both the head AND
the projections via the prediction loss.

**Gotcha**: `node.last_activation.detach()` keeps us from training
through the graph itself. Only the projections and head train. This
matches Direction 2's intent (train projections, not graph weights).

**Effort**: 1 day. Includes detach audit.

### 2C: Full gradient through graph (largest-lift)

Keep `node.last_activation` non-detached and let gradient flow all
the way back through node MLPs, edges, and projections. This is
what the plan doc originally envisioned. Requires:
- Audit every `.detach()` call in the forward path.
- Decide whether node MLPs and edge weights should ALSO be updated
  by the prediction loss (they already have their own Hebbian and
  SOMA-internal backprop paths — need to disentangle).
- Test that gradient magnitudes don't explode.

**Effort**: 2+ days. Highest risk of breaking existing behavior.

## Recommended sequence

1. **Run the no_diversify ablation first** (already queued).
   - If diversification HELPS: random projections carry signal;
     Direction 2's bar is "beat them by learning." Do 2B.
   - If diversification HURTS: random projections are noise;
     Direction 2's bar is "be meaningfully above no_diversify." Do
     2A as the simplest informative next test.
   - If NEUTRAL: diversification is a wash; do 2A to see if any
     learnable mechanism beats frozen-random.

2. **Do 2A as the always-informative first step.** The
   parallel-baseline comparison answers "does SOMA's graph add
   measurable value on v0.5?" — a fundamental question that
   informs every future design decision.

3. **Do 2B if 2A's learnable-projection baseline is too good.**
   That would mean the projection itself is the source of signal and
   we should integrate it into SOMA, not run it alongside.

4. **Reserve 2C for after 2A/2B show signal.** The detach audit is
   expensive and only worth doing if we know gradient flow matters.

## Config surface (proposed)

```python
# In SOMAConfig (for 2B/2C):
projection_mode: Literal["frozen_random", "learnable"] = "frozen_random"
projection_lr: float = 1e-4
# For 2C only:
projection_backprop_through_graph: bool = False  # gates full gradient flow

# In PredictiveSOMA config (for 2A — doesn't need SOMAConfig changes):
enable_parallel_projection_baseline: bool = False
```

## Validation plan

All experiments: v0.5 capacity schedule, seeds {0, 1, 42}, 4000 steps/variant.

Variants for 2B validation:
- `no_growth_frozen` (baseline)
- `no_growth_learnable` (isolates: does learning help without growth?)
- `full_frozen` (baseline reproduction)
- `full_learnable` (primary claim: does learning help WITH growth?)
- `neuro_only_learnable` (reference)

Success criteria (revised for post-Direction-1 era, requires multi-seed):
- `full_learnable` beats `full_frozen` on 6+/8 regimes for ALL three
  seeds (not just mean across seeds — strict per-seed win).
- `no_growth_learnable` ≥ `no_growth_frozen` (doesn't regress when
  there's nothing for the projections to shape).

## Known risks

1. **Gradient explosion from learnable matmul chain**. Mitigation:
   clip gradients, use lower LR, use orthogonal init with small scale.
2. **Interaction with `_competitive_learning`'s Hebbian projection
   updates** — if both backprop and Hebbian run, they may fight.
   Mitigation: skip Hebbian updates when `projection_mode="learnable"`.
3. **Double-counting in the prediction loss path**. Mitigation: the
   pred loss already updates `prediction_head`; adding projections to
   the same optimizer is fine, but don't accidentally call
   `loss.backward()` twice.

## Files touched (estimated)

- `src/soma/core/config.py` (+15 lines, validation)
- `src/soma/developmental/prediction.py` (+60 lines, restructured
  summary + optimizer plumbing)
- `tests/test_developmental/test_learnable_projections.py` (new,
  ~150 lines)
- `research/developmental/env_sequence_v05_learnable_proj.py` (new,
  ~200 lines)

Total: ~425 net new lines plus ~40 modified. Manageable for ~1 day
of focused work.
