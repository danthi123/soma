# Lazy Stable-Capture — Design Exploration

> Tracks task #173. **Not a final design.** Captures the trade-off
> space so the next person picking this up doesn't re-derive it.

## 1. Problem statement

`MemoryLayer.consolidate()` runs in two passes when SOMA is attached
and `graph_rerank_stable_capture=True`:

1. **Growth pass** (incremental, post-#171): processes entries past
   `_consolidation_cursor` in `eval_mode=False` so SOMA can fire
   synaptogenesis / neurogenesis / Hebbian + backprop. Captures each
   entry's output activation as the SOMA state evolves.
2. **Stable-capture pass** (still O(N) per call): re-runs *every*
   stored entry through the post-growth graph in `eval_mode=True`
   so all stored activations live in the same final graph + weight
   state the query will see at retrieval time.

The growth pass updates weights on every `soma.step()` (Hebbian +
backprop are unconditional in `eval_mode=False`). So even when no
new nodes/edges form (the common case under memory workload — see
`benchmarks/reports/plasticity_scale.md`), the *weights* drift. A
stored activation from an earlier consolidate is captured under
weights that no longer match the current ones.

When the query comes in at retrieval, it's computed under the
*current* weights. Comparing query activation (current) vs stored
activations (mixed older versions) is the theoretical correctness
gap stable-capture closes.

**Today this is sunk research overhead.** `graph_rerank_alpha=0.0`
is the default since the alpha sweep + shuffle ablation showed
the graph signal is null on synthetic data. With alpha=0 the
`retrieve()` path short-circuits before computing query activation,
so the staleness of stored activations is irrelevant. Stable-capture
runs but its work is never used.

**It becomes load-bearing** if/when the graph re-rank starts
helping (next obvious milestone: tune SOMA's growth thresholds for
the memory workload + show that recall improves with re-rank
enabled).

## 2. Cost analysis

With `auto_consolidate_every=K` over a session of `N` total stores:

| Component | Per consolidate call | Total session cost |
|---|---|---|
| Growth (post-#171) | O(K) | O(N) |
| Stable-capture (current) | O(N_so_far) | O(N²/K) |

The stable-capture term dominates. At N=10K with K=100, that's
~1M stable steps total over the session — the work that was
amortized across many small batches in the original O(N²) design.

#171 fixed the growth-side O(N²) problem; stable-capture is the
remaining one.

## 3. Design space

### Approach A — Lazy-on-retrieve

Mark `_stable_capture_dirty=True` in `consolidate()` whenever the
growth pass processed anything. In `retrieve()`, before the
re-rank blend, if `alpha > 0 AND dirty`, run a full stable-capture
pass and clear the dirty bit.

```python
# In consolidate()
if processed > 0:
    self._stable_capture_dirty = True
# (no longer call self._recapture_activations_stable here)

# In retrieve(), inside has_graph_signal branch
if has_graph_signal and self._stable_capture_dirty:
    soma_output_dim = int(self._soma.config.sensor_output_dim)
    self._recapture_activations_stable(soma_output_dim)
    self._stable_capture_dirty = False
```

**Pros:**
- Writes are now genuinely incremental: O(K) per consolidate
- Single-shot batch users (store all → consolidate → retrieve many)
  pay O(N) exactly once on first retrieve
- One bit of state, no per-entry bookkeeping
- Total session cost drops from O(N²/K) to O(N) for batch workloads

**Cons:**
- First retrieve after a write batch has unpredictable latency
  (seconds at N=10K+)
- Bad p99 for write/read interleaved loops (chat agents)
- A naive retrieve benchmark under-reports the real cost — needs
  warmup like the scale benchmark does

### Approach B — Per-entry dirty bits

Track `_dirty_activations: list[bool]` parallel to `_soma_activations`.
`consolidate()` marks all entries dirty (because growth-pass weight
updates affect everything). `_retrieve_with_rerank()` re-captures
only the dirty entries it actually touches in its candidate set.

```python
# In consolidate()
if processed > 0:
    # Every prior activation is now stale wrt the new weights.
    self._dirty_activations = [True] * len(self._texts)

# In _retrieve_with_rerank(), inside the candidate loop
for hit in candidates:
    idx = self._ids.index(hit.node_id)
    if self._dirty_activations[idx]:
        self._refresh_activation(idx)  # single-entry recapture
        self._dirty_activations[idx] = False
    stored_act = self._soma_activations[idx]
    # ... existing blend logic
```

**Pros:**
- Optimal: only pay for what's actually queried
- Predictable per-query latency (proportional to candidates touched,
  not store size)
- Works for write-heavy + interleaved retrieve patterns

**Cons:**
- Subtle correctness issue: if growth fired *between* two retrieves,
  the candidates touched in the first retrieve got refreshed under
  the previous weights; the second retrieve sees mixed-version
  activations again. Per-entry tracking only stays consistent if
  consolidate doesn't run between retrieves.
- More state to manage (parallel list + invalidation)
- Refreshing one entry at a time loses the batch-amortization that
  `_recapture_activations_stable` could exploit (token-batch
  encoding, etc.)
- Risks giving stable-capture's API contract without the consistency
  it promises

### Approach C — Skip entirely (status quo when alpha=0)

Document that `graph_rerank_stable_capture` is for research; users
running with `alpha > 0` should explicitly opt in and accept the
cost. The retrieve path's existing alpha=0 short-circuit is the
production-time behavior.

```python
# No code change. Document only.
```

**Pros:**
- Zero new code, zero new bugs
- Honest about the current state (graph signal is null;
  stable-capture is research overhead)

**Cons:**
- Doesn't actually solve the problem when graph re-rank starts
  earning its keep
- Punts the work to the future-self who will be tuning growth
  thresholds and trying to use the re-rank

### Approach D — Per-call opt-out (simplest "actual" fix)

Add `consolidate(skip_stable_capture: bool = False)` so write-heavy
callers can defer stable-capture explicitly until they know they're
about to retrieve.

```python
def consolidate(self, *, skip_stable_capture: bool = False) -> int:
    # ... growth pass ...
    do_stable = (
        self._graph_rerank_stable_capture
        and processed > 0
        and not skip_stable_capture
    )
    if do_stable:
        self._recapture_activations_stable(soma_output_dim)
    if skip_stable_capture and processed > 0:
        self._stable_capture_dirty = True
    return processed

# In retrieve(), if dirty and graph re-rank active, force stable
# capture before blend:
if has_graph_signal and self._stable_capture_dirty:
    soma_output_dim = int(self._soma.config.sensor_output_dim)
    self._recapture_activations_stable(soma_output_dim)
    self._stable_capture_dirty = False
```

**Pros:**
- Caller-controlled: the loop that knows it's bursting writes can
  pass `skip_stable_capture=True` and amortize the cost into the
  next retrieve
- Falls back to current behaviour when caller doesn't pass the flag
- Combines well with auto_consolidate_every — auto-fired
  consolidations could default to skip_stable=True and let the next
  retrieve pay the cost (effectively becomes Approach A for
  auto-fired calls)

**Cons:**
- Two-bit state machine (dirty + the flag) — slightly more
  surface area than A or C
- Still needs the lazy-retrieve hook from A to be useful
- Doesn't help callers that don't think about it

## 4. Recommendation

**Build Approach A.** It's the simplest design that actually moves
the cost off the write path, and the cons (unpredictable first-
retrieve latency) are addressable with a documented `prewarm()`
method that calls `_recapture_activations_stable` explicitly.

Don't build B — the consistency story is too easy to get wrong and
the savings only show up under workloads (interleaved write/retrieve
with re-rank active) that don't exist today.

C is what we have. D is "A with one extra knob"; if A becomes painful
in practice we can add the explicit opt-in later without breaking
anyone.

## 5. Concrete implementation plan for Approach A

```python
# Constructor — one new field, no new params
self._stable_capture_dirty: bool = False

# consolidate() — mark dirty instead of running stable pass
def consolidate(self) -> int:
    if self._soma is None:
        return 0
    self._stores_since_consolidation = 0
    soma_output_dim = int(self._soma.config.sensor_output_dim)

    # ... existing incremental growth pass ...

    # Stable-capture is now lazy. Mark dirty so the next retrieve
    # that needs the graph signal refreshes the captures.
    if processed > 0 and self._graph_rerank_stable_capture:
        self._stable_capture_dirty = True
    return processed

# retrieve() — refresh before blend if dirty
def retrieve(self, query: str, k: int = 5) -> list[MemoryHit]:
    # ... existing setup ...
    if has_graph_signal:
        if self._stable_capture_dirty:
            soma_output_dim = int(self._soma.config.sensor_output_dim)
            self._recapture_activations_stable(soma_output_dim)
            self._stable_capture_dirty = False
        return self._retrieve_with_rerank(query, q_vec, k=k)
    return self._rank(q_vec, k=k, exclude_idx=None)

# Optional: explicit pre-warm for callers that want predictable
# retrieve latency
def prewarm_graph_rerank(self) -> None:
    if (
        self._soma is not None
        and self._graph_rerank_stable_capture
        and self._stable_capture_dirty
    ):
        soma_output_dim = int(self._soma.config.sensor_output_dim)
        self._recapture_activations_stable(soma_output_dim)
        self._stable_capture_dirty = False
```

**Test plan:**
1. `test_consolidate_skips_stable_when_alpha_zero` — verify
   `consolidate()` doesn't fire `_recapture_activations_stable`
   when re-rank is off. Use a counter-mock on the recapture method.
2. `test_first_retrieve_after_consolidate_runs_stable_capture` —
   set alpha>0, call consolidate, retrieve once; the recapture
   counter should advance exactly once.
3. `test_subsequent_retrieves_skip_stable_capture` — second retrieve
   without intervening writes should not re-capture.
4. `test_prewarm_clears_dirty_flag` — explicit prewarm call leaves
   the dirty flag false.

**Estimated effort:** ~1 hour including tests + benchmark
re-validation. Low-risk diff (~30 LoC).

## 6. When to pull this off the shelf

Trigger conditions:
- Graph re-rank starts beating flat cosine on a benchmark we trust
  (not just topic-ordered synthetic; needs shuffled or real data)
- Anyone hits the auto_consolidate_every path with N > 10K and
  notices the write latency
- A user files an issue along the lines of "my consolidate gets
  slower with each store"

Until then, the alpha=0 short-circuit means the cost isn't paid
in the production path and the existing `graph_rerank_stable_capture=True`
default is fine for users who do opt into the re-rank.
