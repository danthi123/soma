# Why `neuro_only` works — mechanism analysis

**Date:** 2026-04-19 (during Direction 2B multi-seed GPU wait)
**Context:** Multi-seed validation revealed `neuro_only` is the sole
mechanism on v0.5 that beats `no_growth` reliably across seeds
(by 0.0005–0.0009 MSE on all 4 high-capacity regimes). Directions
1 and 3 (plasticity gating on synap) both failed multi-seed.
Direction 2B (learnable projections at summary level) in progress.

This note characterizes WHY neuro_only works, based on code reading
of `src/soma/growth/neurogenesis.py`.

## The four-factor story

### 1. Need-based trigger

```
trigger = recent_pe_mean(last 100) / baseline_pe_mean(last 1000)
if trigger > config.neurogenesis_threshold (1.2):  # ~20% PE spike
    spawn new node
```

Neurogenesis fires **only when PE is persistently elevated**
relative to the recent baseline — i.e., when the graph is demonstrably
failing to represent current inputs. Capacity is added in response to
demand, not on a fixed interval.

Compare: synaptogenesis fires on `config.synaptogenesis_interval`
regardless of whether the graph "needs" more edges.

### 2. Spatial locality of new position

```python
def _compute_new_position(graph, config, rng):
    active = graph.most_active_nodes(k=10)
    centroid = mean(positions of active nodes)
    jitter = randn() * config.position_jitter
    return centroid + jitter
```

A new node is spawned **at the centroid of the currently-most-active
nodes, with small jitter**. Critically, it does NOT land in a random
position — it lands near the "hot" region of the position space.

Since positions are initialized as small random vectors at __init__
and then static, the spatial centroid of the most-active nodes
represents the "current representational center" of the graph. New
nodes appear in that neighborhood.

### 3. Local bidirectional wiring

```python
neighbors = graph.get_nearest_nodes(new_position, k=5, exclude=[new_node])
for neighbor in neighbors:
    add_edge(neighbor → new_node, weight ~ 0.01 * randn)
    add_edge(new_node → neighbor, weight ~ 0.01 * randn)
```

The new node gets **bidirectional low-weight edges to its 5
nearest neighbors by position.**

Compare to synaptogenesis, which wires:
- Any pair of nodes whose activations happen to co-exceed threshold
  this step (no positional locality)
- Unidirectional edges only (no feedback cycles)
- Initial weight scale 0.01 * randn (same, but random co-activation
  doesn't filter meaningfully)

### 4. Low initial weight lets Hebbian refine

`neurogenesis_init_weight_scale = 0.01` means new edges start at
±0.01 magnitude. Hebbian updates (`w += lr * source_mag * target_mag`)
accumulate over many co-activations, selecting the edges that are
ACTUALLY useful.

Combined with pruning (disabled in our 4000-step experiments but
enabled in production), edges that don't accumulate strength
get removed. This creates a **Darwinian filter**: only
co-activation-reinforced edges survive long-term.

## Why the interaction of factors matters

Each factor alone is ordinary. The combination is what makes
neuro_only work:

- **Factor 1** ensures capacity is added WHERE IT'S USEFUL
  (during PE spikes, when the graph is failing).
- **Factor 2** ensures the new node lands WHERE ACTIVE CIRCUITRY IS
  (the representational hot zone, not a random place).
- **Factor 3** ensures the new node CONNECTS TO ACTIVE CIRCUITRY
  via spatial proximity (bidirectional), not via random co-activation.
- **Factor 4** ensures the new connections ARE REFINED BY USE via
  Hebbian selection, not frozen at random weights.

Under this framing, neurogenesis is a **conservative, demand-driven
capacity expansion** that respects spatial structure. Synaptogenesis
is an **aggressive, interval-driven random connection creation**.

## Why synaptogenesis fails

Multi-seed has validated that synap admission doesn't help on v0.5.
Plausible causes, ordered by how well they fit the evidence:

1. **No positional locality** (matches "full spaghetti graph" effect)
   — synap admits edges across the entire graph regardless of
   positional distance. Random correlations fossilize as edges.
2. **Unidirectional edges** prevent feedback, so synap-built subgraphs
   can't implement recurrent patterns.
3. **Interval trigger** adds edges even when the graph doesn't need
   more (overparameterization without specialization).
4. **Random init weight without follow-through** — while synap uses
   the same 0.01 scale, the admission process doesn't include a
   selection mechanism for "which ones should survive." Hebbian
   does select, but by then the damage of admitting bad edges is
   done (they dilute useful signals).

Directions 1 and 3 tried to fix (3) and (4) by gating admission on
PE signal, but the gate was either per-pair-degenerate (D1) or
too coarse (D3), and neither addressed the positional-locality
issue (1).

## Practical implications

1. **For v0.5 research**: neuro_only is the benchmark. Any new
   mechanism needs to match OR exceed it to be worth keeping.
2. **For the paper**: the §4.7 plasticity adaptation section should
   reframe around this finding. Not "structural plasticity is
   powerful" — but "capacity-pressure-driven spatial neurogenesis
   is powerful; random co-activation-based synaptogenesis is not."
3. **For Direction 2+ design**: if we want to recover synap's lost
   potential, we need to add positional-locality to its admission
   criterion. Maybe: only admit pairs whose positions are within
   a learned radius, with the radius itself being task-supervised.
   This is a fundamentally different Direction 2C than what I
   implemented (summary-level post-processing).

## Testable predictions

If this analysis is right:

1. **Disabling positional locality in neurogenesis should break
   the mechanism.** Replace `get_nearest_nodes` with random neighbors
   → neuro_only should lose its edge.
2. **Disabling the PE-spike trigger should also break it.** Fire
   neurogenesis on a fixed interval → neuro_only should match synap's
   random admission pattern.
3. **Synap with positional-locality filter should approach neuro_only.**
   This is the real Direction 2/3 probe.

These predictions are falsifiable with ablations on the existing
neurogenesis code — all small changes to `_compute_new_position`
and the neighbor selection. Worth queuing as follow-up probes.

## Files referenced

- `src/soma/growth/neurogenesis.py` (the mechanism).
- `src/soma/core/graph.py::get_nearest_nodes` (the locality helper).
- `src/soma/core/node.py::position` (the position state).
- `src/soma/metacognition/development.py` (maturity progression,
  not analyzed here but relevant for "low init weight + slow Hebbian"
  interpretation).
