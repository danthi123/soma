# Wave Batching for `execute_graph` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `execute_graph` 3-5x faster by batching same-shape nodes within each topological wave into a single stacked matmul per MLP layer, collapsing ~128 tiny kernel launches per step into ~4.

**Architecture:** Compute a wave-layered topological order (nodes grouped by earliest-executable step). For each wave, bucket nodes by `(input_dim, hidden_dim, output_dim)`. For every bucket, stack per-node aggregated inputs into one `(N, input_dim)` tensor, run one `F.linear(x, stacked_W1, stacked_b1)` using a batched weight tensor built on the fly from the per-node `nn.Linear` parameters (preserves autograd wiring via `torch.stack`), then GELU, then one batched `linear2`, then per-node residual/gain, then `torch.unbind` back to per-node tensors. The per-node state (`gain`, `activation_ema`, `last_active_step`, `activation_history`, `activations` dict) is updated in the original sequential order after the math, so the batched path is observationally indistinguishable from the old loop.

**Tech Stack:** PyTorch (existing), pytest (existing). No new dependencies. CPU-only during development; CUDA gains realized at runtime on RTX 3090.

**Design references:** `CLAUDE.md` (Critical Invariants), whitepaper §3.1 (Node.forward shape), §3.3 (execution order).

---

## Pre-flight notes

- Worktree: implementation runs in this worktree; plan file sits in `docs/plans/`.
- Baseline: commit `6d354fa` (main at plan-writing time). All 703 existing tests pass.
- Dev environment is **CPU-only**; you cannot measure 3x locally. The benchmark script must run on CPU (for correctness / lower-bound number) AND be CUDA-ready so the operator can validate 3x on the RTX 3090.
- Do NOT touch `Node.forward` itself. The old sequential path must remain available as a fallback and for per-node unit tests. We add a new batched executor alongside it.
- Respect all CLAUDE.md invariants: SENSOR/OUTPUT never pruned, gain clamp [0.1, 10.0], edge weight clamp [-5, 5], consolidation interval. This change is purely arithmetic-routing — it does not touch invariant-enforcing code, but the regression test in Task 10 must prove that.
- Commit convention: `perf(exec):`, `test(exec):`, `refactor(exec):`, `docs(exec):`, `bench(exec):` prefixes for this branch.

---

## Design contract (read first, every task)

The new executor MUST satisfy these properties; every task's tests check at least one of them.

1. **Bit-equal-up-to-float-reassociation.** `torch.allclose(old_out, new_out, atol=1e-5, rtol=1e-5)` for the same graph + inputs. Exact bit-equality is NOT required because summing a different number of elements changes floating-point rounding in the last place, but `1e-5` is plenty for this architecture.
2. **Gradient-routing equivalence.** After `loss.backward()`, every `linear1.weight.grad`, `linear1.bias.grad`, `linear2.weight.grad`, `linear2.bias.grad`, `gain` (no — gain is a Python float, not a Parameter, so skip), and `edge.weight.grad` is `allclose` to the corresponding gradient computed under the old executor.
3. **Per-node state parity.** After one step, for every node: `activation_history` has the same length, `activation_ema` is allclose, `last_active_step` matches exactly.
4. **Dormant-node parity.** A node that had no active inputs under the old path must ALSO have no entry in `activations` under the new path.
5. **Sensor and output parity.** Sensor outputs equal injected inputs; `outputs` dict is identical (same keys, allclose values) between paths.
6. **Back-edge parity.** A graph with cycles + `previous_activations` returns allclose activations on step 2 under both paths.
7. **Dynamic graph.** Adding/removing a node between calls does NOT require executor reset — `execute_graph` must handle topology changes step-to-step. Caching (if any) must invalidate.

Anything that fails one of these invariants is a blocker. Stop, diagnose, fix root cause.

---

## Task 1: Introduce `execute_graph_batched` scaffold with env-var switch

**Files:**
- Modify: `src/soma/core/execution.py`
- Test: `tests/test_core/test_execution.py` (extend)

**Intent:** Add a new public function `execute_graph_batched` that initially just delegates to the existing `execute_graph`, plus an env-var `SOMA_EXEC_BATCHED` (default `"0"`) so callers can opt into the batched path. This gives us a safe merge-able skeleton before any real work.

**Step 1: Write the failing test**

Add to `tests/test_core/test_execution.py`:

```python
import os

from soma.core.execution import execute_graph_batched  # new symbol


class TestBatchedExecutorScaffold:
    def test_batched_symbol_exported(self) -> None:
        from soma.core import execution

        assert hasattr(execution, "execute_graph_batched")

    def test_batched_matches_sequential_on_linear_graph(self, config: SOMAConfig) -> None:
        graph, sensor, _, _ = _matched_linear_graph(config)
        data = torch.randn(sensor.output_dim)
        out_seq, act_seq = execute_graph(graph, inputs={"text": data}, current_step=1)
        # Rebuild identical graph (fresh state) for second pass.
        graph2, sensor2, _, _ = _matched_linear_graph(config)
        # Copy the exact weights from graph -> graph2 so the two runs are
        # comparing math, not init noise.
        graph2.load_state_dict(graph.state_dict())
        out_bat, act_bat = execute_graph_batched(
            graph2, inputs={"text": data}, current_step=1
        )
        assert set(out_seq.keys()) == set(out_bat.keys())
        for k in out_seq:
            assert torch.allclose(out_seq[k], out_bat[k], atol=1e-5, rtol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core/test_execution.py::TestBatchedExecutorScaffold -v`
Expected: FAIL with `ImportError: cannot import name 'execute_graph_batched'`.

**Step 3: Add the scaffold**

In `src/soma/core/execution.py`, at the bottom before the helpers section, add:

```python
def execute_graph_batched(
    graph: Graph,
    inputs: Mapping[str, torch.Tensor],
    current_step: int,
    *,
    previous_activations: Mapping[str, torch.Tensor] | None = None,
    record_edge_activity: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Batched variant of :func:`execute_graph`.

    Currently delegates to the sequential path. Subsequent tasks replace
    the body with wave-grouped batched matmuls.
    """
    return execute_graph(
        graph,
        inputs,
        current_step,
        previous_activations=previous_activations,
        record_edge_activity=record_edge_activity,
    )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_core/test_execution.py::TestBatchedExecutorScaffold -v`
Expected: PASS (both tests).

**Step 5: Commit**

```bash
git add src/soma/core/execution.py tests/test_core/test_execution.py
git commit -m "perf(exec): scaffold execute_graph_batched (delegates to sequential for now)"
```

---

## Task 2: Compute wave-layered topological order

**Files:**
- Modify: `src/soma/core/execution.py`
- Test: `tests/test_core/test_execution.py`

**Intent:** The existing `topological_sort` returns a flat list. We need a wave-layered version where each wave is a list of node IDs that are all ready to execute after the prior waves finish. A node's wave is `1 + max(wave of each predecessor via forward edges)`; back-edges don't constrain the wave (they read previous-step activations).

**Step 1: Write the failing test**

Add to `tests/test_core/test_execution.py`:

```python
from soma.core.execution import compute_wave_layers  # new symbol


class TestComputeWaveLayers:
    def test_linear_graph_has_three_waves(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _matched_linear_graph(config)
        waves = compute_wave_layers(graph)
        assert [sorted(w) for w in waves] == [[sensor.id], [assoc.id], [out.id]]

    def test_back_edge_does_not_increase_wave(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = config.sensor_output_dim
        sensor = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(a)
        graph.add_node(b)
        graph.add_node(out, modality="text")
        for src, tgt in [(sensor, a), (a, b), (b, out), (b, a)]:
            graph.add_edge(
                Edge(
                    source_id=src.id,
                    target_id=tgt.id,
                    source_output_dim=dim,
                    target_input_dim=dim,
                    creation_step=0,
                    initial_weight=1.0,
                )
            )
        waves = compute_wave_layers(graph)
        # b->a is a back-edge; a should still be reachable via sensor->a
        # in wave index 1, not forced into the same wave as out.
        flat = [nid for wave in waves for nid in wave]
        assert set(flat) == {sensor.id, a.id, b.id, out.id}
        idx = {nid: i for i, wave in enumerate(waves) for nid in wave}
        # Forward-edge constraints hold:
        assert idx[sensor.id] < idx[a.id] < idx[b.id] < idx[out.id]

    def test_disconnected_component(self, config: SOMAConfig) -> None:
        graph = Graph()
        dim = 4
        nodes = [Node(NodeType.ASSOCIATOR, dim, 8, dim, 0, config) for _ in range(3)]
        for n in nodes:
            graph.add_node(n)
        graph.add_edge(
            Edge(
                source_id=nodes[0].id,
                target_id=nodes[1].id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
            )
        )
        waves = compute_wave_layers(graph)
        idx = {nid: i for i, w in enumerate(waves) for nid in w}
        # Two connected + one isolated. Isolated node sits at wave 0.
        assert idx[nodes[0].id] < idx[nodes[1].id]
        assert idx[nodes[2].id] == 0
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_core/test_execution.py::TestComputeWaveLayers -v`
Expected: FAIL with `ImportError: cannot import name 'compute_wave_layers'`.

**Step 3: Implement**

In `src/soma/core/execution.py`, add after `topological_sort`:

```python
def compute_wave_layers(graph: Graph) -> list[list[str]]:
    """Group node IDs into topological waves.

    A node's wave index is ``1 + max(wave of each predecessor via
    forward edges)``, or ``0`` if it has no forward-edge predecessors.
    Back-edges (detected the same way as in :func:`topological_sort`)
    are ignored for wave assignment because they carry previous-step
    signal, not this-step signal.

    Returns a list of waves, each a list of node IDs. The order of node
    IDs *within* a wave matches the order they appear in
    :func:`topological_sort` (important: the batched executor updates
    per-node state in this same order to keep sequential-vs-batched
    state-update ordering identical).
    """
    order, back_edges = topological_sort(graph)
    wave_of: dict[str, int] = {}
    for node_id in order:
        max_pred = -1
        for edge in graph.get_incoming_edges(node_id):
            if edge.id in back_edges:
                continue
            if edge.source_id in wave_of:
                if wave_of[edge.source_id] > max_pred:
                    max_pred = wave_of[edge.source_id]
        wave_of[node_id] = max_pred + 1

    if not wave_of:
        return []
    num_waves = max(wave_of.values()) + 1
    waves: list[list[str]] = [[] for _ in range(num_waves)]
    # Preserve topological order within each wave.
    for node_id in order:
        waves[wave_of[node_id]].append(node_id)
    return waves
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core/test_execution.py::TestComputeWaveLayers -v`
Expected: PASS (all three).

**Step 5: Commit**

```bash
git add src/soma/core/execution.py tests/test_core/test_execution.py
git commit -m "perf(exec): add compute_wave_layers helper for batched executor"
```

---

## Task 3: Bucket nodes within a wave by shape

**Files:**
- Modify: `src/soma/core/execution.py`
- Test: `tests/test_core/test_execution.py`

**Intent:** Within each wave, group nodes by `(input_dim, hidden_dim, output_dim)`. SENSOR nodes are excluded (they have no MLP step — their `forward` returns the injected input directly). The bucketing preserves order-within-wave so state updates remain deterministic.

**Step 1: Write the failing test**

Add to `tests/test_core/test_execution.py`:

```python
from soma.core.execution import bucket_wave_by_shape  # new symbol


class TestBucketWaveByShape:
    def test_uniform_wave_single_bucket(self, config: SOMAConfig) -> None:
        graph = Graph()
        nodes = [
            Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config) for _ in range(3)
        ]
        for n in nodes:
            graph.add_node(n)
        wave = [n.id for n in nodes]
        buckets = bucket_wave_by_shape(graph, wave)
        assert list(buckets.keys()) == [(8, 16, 8)]
        assert buckets[(8, 16, 8)] == wave  # order preserved

    def test_mixed_wave_multiple_buckets(self, config: SOMAConfig) -> None:
        graph = Graph()
        a = Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config)
        b = Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config)
        c = Node(NodeType.INTEGRATOR, 8, 32, 16, 0, config)
        for n in (a, b, c):
            graph.add_node(n)
        wave = [a.id, c.id, b.id]
        buckets = bucket_wave_by_shape(graph, wave)
        assert buckets[(8, 16, 8)] == [a.id, b.id]  # a before b (input order)
        assert buckets[(8, 32, 16)] == [c.id]

    def test_sensor_excluded(self, config: SOMAConfig) -> None:
        graph = Graph()
        sensor = Node(NodeType.SENSOR, 8, 16, 8, 0, config)
        assoc = Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config)
        graph.add_node(sensor, modality="text")
        graph.add_node(assoc)
        buckets = bucket_wave_by_shape(graph, [sensor.id, assoc.id])
        # Sensor must not appear in any bucket — it's handled separately.
        all_ids = [nid for ids in buckets.values() for nid in ids]
        assert sensor.id not in all_ids
        assert assoc.id in all_ids
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_core/test_execution.py::TestBucketWaveByShape -v`
Expected: FAIL with ImportError.

**Step 3: Implement**

Add to `src/soma/core/execution.py`:

```python
def bucket_wave_by_shape(
    graph: Graph,
    wave: list[str],
) -> dict[tuple[int, int, int], list[str]]:
    """Bucket a wave's node IDs by their MLP shape.

    Key is ``(input_dim, hidden_dim, output_dim)``. SENSOR nodes are
    skipped because they don't run an MLP (their ``forward`` returns
    the injected input). Order within each bucket follows the order of
    ``wave`` — callers depend on this for deterministic state updates.
    """
    buckets: dict[tuple[int, int, int], list[str]] = {}
    for node_id in wave:
        node = graph.nodes[node_id]
        if node.node_type is NodeType.SENSOR:
            continue
        key = (node.input_dim, node.hidden_dim, node.output_dim)
        buckets.setdefault(key, []).append(node_id)
    return buckets
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_core/test_execution.py::TestBucketWaveByShape -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/soma/core/execution.py tests/test_core/test_execution.py
git commit -m "perf(exec): add bucket_wave_by_shape helper"
```

---

## Task 4: Batched MLP primitive — `_batched_node_forward`

**Files:**
- Modify: `src/soma/core/execution.py`
- Test: `tests/test_core/test_execution.py`

**Intent:** Given a list of Nodes of identical shape and a pre-aggregated input tensor of shape `(N, input_dim)`, return a `(N, output_dim)` tensor equal to stacking each node's `forward(summed_signal)` — but computed via two batched matmuls. This is the mathematical heart of the optimization.

**Key correctness requirements:**
- Use `torch.stack([n.linear1.weight for n in nodes])` so autograd routes gradients back to each node's own parameter.
- `torch.bmm` or `torch.einsum("noi,ni->no", W, x)` both work. We'll use `einsum` for clarity.
- Gain is a per-node Python float. Build `gain = torch.tensor([n.gain for n in nodes], device=...).unsqueeze(-1)` so broadcasting works: `h * gain` shape `(N, output_dim) * (N, 1)` → `(N, output_dim)`.
- Residual only applies when `input_dim == output_dim`. Since all nodes in the bucket share shape, the decision is per-bucket not per-node.

**Step 1: Write the failing test**

Add to `tests/test_core/test_execution.py`:

```python
from soma.core.execution import _batched_node_forward  # private; acceptable to test directly


class TestBatchedNodeForward:
    def test_matches_sequential_same_shape(self, config: SOMAConfig) -> None:
        torch.manual_seed(0)
        nodes = [Node(NodeType.ASSOCIATOR, 8, 16, 8, 0, config) for _ in range(4)]
        x = torch.randn(4, 8)  # pre-aggregated inputs
        # Sequential reference
        seq_out = torch.stack(
            [
                nodes[i].forward({"fake": x[i]}, current_step=0)
                for i in range(4)
            ]
        )
        # Reset state so batched path starts from the same baseline.
        for n in nodes:
            n.activation_history = type(n.activation_history)(
                n.activation_history.capacity
            )
            n.activation_ema = 0.0
            n.last_active_step = 0
        bat_out = _batched_node_forward(nodes, x)
        assert bat_out.shape == (4, 8)
        assert torch.allclose(bat_out, seq_out, atol=1e-5, rtol=1e-5)

    def test_gain_applied_per_node(self, config: SOMAConfig) -> None:
        nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(2)]
        nodes[0].gain = 2.0
        nodes[1].gain = 0.5
        # Zero-out weights so residual path + gain is all that matters.
        for n in nodes:
            with torch.no_grad():
                n.linear1.weight.zero_(); n.linear1.bias.zero_()
                n.linear2.weight.zero_(); n.linear2.bias.zero_()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        out = _batched_node_forward(nodes, x)
        # h = 0 * gain + residual = x (since input_dim == output_dim).
        # Gain multiplies (zero) contribution, so out == x exactly.
        assert torch.allclose(out, x)

    def test_no_residual_when_dims_differ(self, config: SOMAConfig) -> None:
        nodes = [Node(NodeType.INTEGRATOR, 4, 8, 6, 0, config) for _ in range(2)]
        for n in nodes:
            with torch.no_grad():
                n.linear1.weight.zero_(); n.linear1.bias.zero_()
                n.linear2.weight.zero_(); n.linear2.bias.zero_()
        x = torch.randn(2, 4)
        out = _batched_node_forward(nodes, x)
        # All linear layers zeroed + no residual -> output is zero.
        assert torch.allclose(out, torch.zeros(2, 6))

    def test_gradient_routes_to_each_node(self, config: SOMAConfig) -> None:
        nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(3)]
        x = torch.randn(3, 4, requires_grad=False)
        out = _batched_node_forward(nodes, x)
        loss = out.sum()
        loss.backward()
        for n in nodes:
            assert n.linear1.weight.grad is not None
            assert n.linear1.bias.grad is not None
            assert n.linear2.weight.grad is not None
            assert n.linear2.bias.grad is not None
            # Distinct gradients — the stack/unbind did not collapse them.
            assert not torch.allclose(
                nodes[0].linear1.weight.grad, nodes[1].linear1.weight.grad
            )
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_core/test_execution.py::TestBatchedNodeForward -v`
Expected: FAIL with ImportError on `_batched_node_forward`.

**Step 3: Implement**

Add to `src/soma/core/execution.py`:

```python
from soma.core.node import Node  # add at top if not already present


def _batched_node_forward(nodes: list[Node], x: torch.Tensor) -> torch.Tensor:
    """Run the MLP step for a list of same-shape nodes in one batched pass.

    Parameters
    ----------
    nodes:
        Non-empty list. All must share ``(input_dim, hidden_dim, output_dim)``.
    x:
        Pre-aggregated input tensor of shape ``(len(nodes), input_dim)``.

    Returns
    -------
    Tensor of shape ``(len(nodes), output_dim)`` equal (up to
    float-reassociation) to ``torch.stack([n.forward({'_': x[i]}, ...)])``.

    Notes
    -----
    Gradients flow back to each node's individual ``linear1.weight``,
    ``linear1.bias``, ``linear2.weight``, ``linear2.bias`` because we
    build the batched weight tensor with ``torch.stack``, whose backward
    splits the accumulated gradient back to the input tensors. Gain is
    a Python float (not a ``nn.Parameter``), so it does not receive
    gradients — matches the sequential path.
    """
    assert nodes, "_batched_node_forward called with empty node list"
    input_dim = nodes[0].input_dim
    hidden_dim = nodes[0].hidden_dim
    output_dim = nodes[0].output_dim
    # Defensive: catch shape mismatches early.
    for n in nodes:
        assert (n.input_dim, n.hidden_dim, n.output_dim) == (
            input_dim,
            hidden_dim,
            output_dim,
        ), f"Shape mismatch in bucket: {n.id[:8]} has {(n.input_dim, n.hidden_dim, n.output_dim)}"

    # Stack weight/bias into (N, hidden, input) / (N, hidden) tensors.
    # torch.stack preserves autograd edges, so grads flow back to each
    # node's own parameters.
    W1 = torch.stack([n.linear1.weight for n in nodes])  # (N, hidden, input)
    b1 = torch.stack([n.linear1.bias for n in nodes])  # (N, hidden)
    W2 = torch.stack([n.linear2.weight for n in nodes])  # (N, output, hidden)
    b2 = torch.stack([n.linear2.bias for n in nodes])  # (N, output)

    # Batched linear: h[n] = W1[n] @ x[n] + b1[n]
    # einsum keeps us explicit; bmm would also work.
    h = torch.einsum("nhi,ni->nh", W1, x) + b1
    h = F.gelu(h)
    h = torch.einsum("noh,nh->no", W2, h) + b2

    # Per-node gain: broadcast (N,) -> (N, 1).
    device = x.device
    dtype = x.dtype
    gain = torch.tensor([n.gain for n in nodes], device=device, dtype=dtype).unsqueeze(-1)
    h = h * gain

    if input_dim == output_dim:
        h = h + x

    return h
```

Ensure `from torch.nn import functional as F` is imported at top of module (it likely already is — if not, add it).

**Step 4: Run tests**

Run: `python -m pytest tests/test_core/test_execution.py::TestBatchedNodeForward -v`
Expected: PASS (all four).

**Step 5: Commit**

```bash
git add src/soma/core/execution.py tests/test_core/test_execution.py
git commit -m "perf(exec): add _batched_node_forward primitive (stacked matmul per bucket)"
```

---

## Task 5: Per-node state update helper — `_record_batched_activations`

**Files:**
- Modify: `src/soma/core/execution.py`
- Test: `tests/test_core/test_execution.py`

**Intent:** Extract the post-forward state-update logic (append to history, EMA, last_active_step) so we can call it from the batched executor in the same per-node order as the sequential path. The helper takes a list of nodes and their matching rows from the batched output tensor, and updates each node's state. Must match `Node._record_activation` bit-for-bit for the EMA path.

**Step 1: Write the failing test**

```python
from soma.core.execution import _record_batched_activations


class TestRecordBatchedActivations:
    def test_matches_sequential_record(self, config: SOMAConfig) -> None:
        # Build two equivalent node lists. Run sequential on one, batched
        # record on the other. Compare final state.
        seq_nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(3)]
        bat_nodes = [Node(NodeType.ASSOCIATOR, 4, 8, 4, 0, config) for _ in range(3)]
        # Sync params so output magnitudes match.
        for s, b in zip(seq_nodes, bat_nodes, strict=True):
            b.load_state_dict(s.state_dict())
        # Fake activations.
        outs = torch.stack([torch.randn(4) * 3.0 for _ in range(3)])
        for i, n in enumerate(seq_nodes):
            n._record_activation(outs[i], current_step=7)
        _record_batched_activations(bat_nodes, outs, current_step=7)
        for s, b in zip(seq_nodes, bat_nodes, strict=True):
            assert len(s.activation_history) == len(b.activation_history)
            assert s.activation_history.to_list() == pytest.approx(
                b.activation_history.to_list(), rel=1e-6
            )
            assert s.activation_ema == pytest.approx(b.activation_ema, rel=1e-6)
            assert s.last_active_step == b.last_active_step
```

**Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_core/test_execution.py::TestRecordBatchedActivations -v`
Expected: FAIL with ImportError.

**Step 3: Implement**

Add to `src/soma/core/execution.py`:

```python
def _record_batched_activations(
    nodes: list[Node],
    outputs: torch.Tensor,
    current_step: int,
) -> None:
    """Per-node state update for a batched-forward output.

    Must match :meth:`Node._record_activation` exactly: append magnitude
    to ring buffer, EMA-update ``activation_ema`` with 0.99/0.01 blend,
    and bump ``last_active_step`` iff magnitude > threshold.

    Processes nodes in list order so sequential-vs-batched ordering of
    host-side state updates is identical.
    """
    # Compute per-row L2 norms in one shot, then pull to host for the
    # Python-float state that Node holds.
    mags = outputs.detach().norm(dim=-1).tolist()
    for node, mag in zip(nodes, mags, strict=True):
        node.activation_history.append(mag)
        node.activation_ema = 0.99 * node.activation_ema + 0.01 * mag
        if mag > node._activation_threshold:
            node.last_active_step = current_step
```

**Step 4: Run test**

Run: `python -m pytest tests/test_core/test_execution.py::TestRecordBatchedActivations -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/soma/core/execution.py tests/test_core/test_execution.py
git commit -m "perf(exec): add _record_batched_activations helper"
```

---

## Task 6: Aggregate incoming edges into a per-wave input tensor

**Files:**
- Modify: `src/soma/core/execution.py`
- Test: `tests/test_core/test_execution.py`

**Intent:** For each node in a bucket, call `Edge.transmit` on every incoming edge (forward OR back-edge, with the correct source activation), sum the results, and stack into a `(N, input_dim)` tensor. This is the "input aggregation" half of the wave computation. The output tensor feeds directly into `_batched_node_forward`.

**Important edge cases:**
- A node with zero incoming *active* edges is "dormant" — it must be excluded from the bucket for this wave (skipped entirely, same as the old path).
- Back-edge source activations come from `previous_activations`, not `activations` — the selection logic already exists in `_source_activation_for`.
- When `record_edge_activity=True`, every edge that carried signal must have `mark_active(current_step)` called. We do this during aggregation so we don't need a second pass.

**Step 1: Write the failing test**

```python
from soma.core.execution import _aggregate_bucket_inputs


class TestAggregateBucketInputs:
    def test_sums_multiple_incoming_edges(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _matched_linear_graph(config)
        dim = sensor.output_dim
        # Another parallel path: sensor -> out direct.
        graph.add_edge(
            Edge(
                source_id=sensor.id,
                target_id=out.id,
                source_output_dim=dim,
                target_input_dim=dim,
                creation_step=0,
                initial_weight=0.5,
            )
        )
        # Prepare synthetic activations: sensor activation = ones, assoc = twos.
        activations = {
            sensor.id: torch.ones(dim),
            assoc.id: torch.full((dim,), 2.0),
        }
        x, active_nodes = _aggregate_bucket_inputs(
            graph=graph,
            nodes=[graph.nodes[out.id]],
            activations=activations,
            previous_activations={},
            back_edges=set(),
            current_step=5,
            record_edge_activity=False,
        )
        assert active_nodes == [graph.nodes[out.id]]
        assert x.shape == (1, dim)
        # Sum of weighted edges:
        #   assoc->out (weight 1.0): 2.0 * 1.0 = 2.0 (per feature)
        #   sensor->out (weight 0.5): 1.0 * 0.5 = 0.5 (per feature)
        # Total: 2.5 per feature.
        assert torch.allclose(x[0], torch.full((dim,), 2.5), atol=1e-5)

    def test_dormant_node_excluded(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, out = _matched_linear_graph(config)
        # No source activations -> every non-sensor node is dormant.
        x, active = _aggregate_bucket_inputs(
            graph=graph,
            nodes=[graph.nodes[assoc.id], graph.nodes[out.id]],
            activations={},  # empty — simulate no upstream signal yet
            previous_activations={},
            back_edges=set(),
            current_step=0,
            record_edge_activity=False,
        )
        assert active == []
        assert x.shape == (0,) or x.numel() == 0

    def test_mark_active_when_record_activity(self, config: SOMAConfig) -> None:
        graph, sensor, assoc, _ = _matched_linear_graph(config)
        dim = sensor.output_dim
        activations = {sensor.id: torch.ones(dim)}
        s_to_a = graph.get_edge(sensor.id, assoc.id)
        assert s_to_a.last_active_step == 0
        _aggregate_bucket_inputs(
            graph=graph,
            nodes=[graph.nodes[assoc.id]],
            activations=activations,
            previous_activations={},
            back_edges=set(),
            current_step=42,
            record_edge_activity=True,
        )
        assert s_to_a.last_active_step == 42
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_core/test_execution.py::TestAggregateBucketInputs -v`
Expected: FAIL with ImportError.

**Step 3: Implement**

```python
def _aggregate_bucket_inputs(
    graph: Graph,
    nodes: list[Node],
    activations: Mapping[str, torch.Tensor],
    previous_activations: Mapping[str, torch.Tensor],
    back_edges: set[str],
    current_step: int,
    record_edge_activity: bool,
) -> tuple[torch.Tensor, list[Node]]:
    """Aggregate incoming edges for each node, return stacked inputs.

    Nodes with zero active incoming edges are dropped (dormant). Returned
    ``active_nodes`` preserves the input order minus dropped nodes, and
    row ``i`` of the returned tensor is the aggregated input for
    ``active_nodes[i]``.

    Iterates edges in the same order as the sequential executor so
    edge ``mark_active`` bookkeeping is identical.
    """
    rows: list[torch.Tensor] = []
    active_nodes: list[Node] = []
    for node in nodes:
        agg: torch.Tensor | None = None
        for edge in graph.get_incoming_edges(node.id):
            source_act = _source_activation_for(
                edge, back_edges, activations, previous_activations
            )
            if source_act is None:
                continue
            signal = edge.transmit(source_act)
            agg = signal if agg is None else agg + signal
            if record_edge_activity:
                edge.mark_active(current_step)
        if agg is None:
            continue  # dormant
        rows.append(agg)
        active_nodes.append(node)
    if not rows:
        # Empty bucket — return a zero-row tensor so downstream shape
        # checks don't crash. Caller should test ``active_nodes``.
        return torch.empty(0), active_nodes
    return torch.stack(rows), active_nodes
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_core/test_execution.py::TestAggregateBucketInputs -v`
Expected: PASS (all three).

**Step 5: Commit**

```bash
git add src/soma/core/execution.py tests/test_core/test_execution.py
git commit -m "perf(exec): add _aggregate_bucket_inputs (edge sum + mark_active per wave)"
```

---

## Task 7: Wire up `execute_graph_batched` end-to-end

**Files:**
- Modify: `src/soma/core/execution.py`
- Test: `tests/test_core/test_execution.py`

**Intent:** Replace the delegate body of `execute_graph_batched` with the real wave-by-wave batched loop. After this task, the batched path is functional but not yet used by SOMA anywhere.

**Outline of the body:**
1. Inject sensor inputs (reuse `_inject_sensor_inputs`).
2. Compute wave layers and `back_edges`. (Reuse `topological_sort` for back edges; `compute_wave_layers` for waves.)
3. For each wave:
   - a. Handle SENSOR nodes first, in order: call `sensor._sensor_forward(current_step)` (semantically same as the sequential path — sensor returns injected data and records its activation). Stash into `activations[sensor.id]`.
   - b. For each shape-bucket in the wave:
     - Call `_aggregate_bucket_inputs` to produce `(x, active_nodes)`.
     - If `active_nodes` is empty, skip the bucket.
     - Call `_batched_node_forward(active_nodes, x)` → `y`.
     - Unbind: `y_rows = torch.unbind(y, dim=0)` so autograd gives each node its own tensor.
     - For i, (node, row) in enumerate: `activations[node.id] = row`.
     - Call `_record_batched_activations(active_nodes, y, current_step)` for state updates.
4. `outputs = _collect_outputs(graph, activations)`.
5. Return `(outputs, activations)`.

**Step 1: Write the failing test (parity on a complex-ish graph)**

```python
class TestBatchedExecutorParity:
    def _make_graph(self, config: SOMAConfig, seed: int = 0) -> Graph:
        torch.manual_seed(seed)
        g = Graph()
        dim = config.sensor_output_dim
        s = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
        a1 = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        a2 = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        a3 = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
        # Different-shape integrator to force a second bucket in a shared wave.
        integ = Node(NodeType.INTEGRATOR, dim, dim * 4, dim * 2, 0, config)
        # Output receives input from a bucket that matches its shape.
        out = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
        g.add_node(s, modality="text")
        for n in (a1, a2, a3, integ):
            g.add_node(n)
        g.add_node(out, modality="text")
        edges = [
            (s, a1), (s, a2), (s, a3),  # fan-out from sensor to assocs
            (s, integ),                  # sensor also feeds integrator
            (a1, out), (a2, out), (a3, out),
        ]
        for src, tgt in edges:
            g.add_edge(
                Edge(
                    source_id=src.id,
                    target_id=tgt.id,
                    source_output_dim=src.output_dim,
                    target_input_dim=tgt.input_dim,
                    creation_step=0,
                    initial_weight=0.5,
                )
            )
        return g

    def test_batched_equals_sequential_outputs(self, config: SOMAConfig) -> None:
        g1 = self._make_graph(config, seed=123)
        g2 = self._make_graph(config, seed=123)  # same seed -> identical init
        # Enforce same params just in case init is not pure-seed-deterministic.
        g2.load_state_dict(g1.state_dict())
        data = torch.randn(config.sensor_output_dim)
        seq_out, seq_act = execute_graph(g1, inputs={"text": data}, current_step=1)
        bat_out, bat_act = execute_graph_batched(g2, inputs={"text": data}, current_step=1)
        assert set(seq_out.keys()) == set(bat_out.keys())
        for k in seq_out:
            assert torch.allclose(seq_out[k], bat_out[k], atol=1e-5, rtol=1e-5), (
                f"Output {k!r} diverges: max diff "
                f"{(seq_out[k] - bat_out[k]).abs().max().item()}"
            )
        # Same set of active node IDs.
        assert set(seq_act.keys()) == set(bat_act.keys())

    def test_batched_equals_sequential_gradients(self, config: SOMAConfig) -> None:
        g1 = self._make_graph(config, seed=7)
        g2 = self._make_graph(config, seed=7)
        g2.load_state_dict(g1.state_dict())
        data = torch.randn(config.sensor_output_dim)
        out_seq, _ = execute_graph(g1, inputs={"text": data}, current_step=1)
        out_bat, _ = execute_graph_batched(g2, inputs={"text": data}, current_step=1)
        out_seq["text"].sum().backward()
        out_bat["text"].sum().backward()
        # Gradients on a few representative parameters must match.
        for nid in g1.nodes:
            n1 = g1.nodes[nid]
            n2 = g2.nodes[nid]
            if n1.node_type is NodeType.SENSOR:
                continue
            for pname in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
                p1 = dict(n1.named_parameters())[pname].grad
                p2 = dict(n2.named_parameters())[pname].grad
                if p1 is None and p2 is None:
                    continue
                assert p1 is not None and p2 is not None, (
                    f"grad presence mismatch on {nid[:8]}.{pname}"
                )
                assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-5), (
                    f"Gradient diverges on {nid[:8]}.{pname}: "
                    f"max diff {(p1 - p2).abs().max().item()}"
                )

    def test_batched_preserves_per_node_state(self, config: SOMAConfig) -> None:
        g1 = self._make_graph(config, seed=11)
        g2 = self._make_graph(config, seed=11)
        g2.load_state_dict(g1.state_dict())
        data = torch.randn(config.sensor_output_dim)
        execute_graph(g1, inputs={"text": data}, current_step=42)
        execute_graph_batched(g2, inputs={"text": data}, current_step=42)
        for nid in g1.nodes:
            n1 = g1.nodes[nid]
            n2 = g2.nodes[nid]
            assert n1.last_active_step == n2.last_active_step
            assert n1.activation_ema == pytest.approx(n2.activation_ema, rel=1e-6, abs=1e-9)
            assert n1.activation_history.to_list() == pytest.approx(
                n2.activation_history.to_list(), rel=1e-6, abs=1e-9
            )

    def test_batched_handles_back_edges(self, config: SOMAConfig) -> None:
        # Reuse the cycle graph from test_cycle_uses_previous_activations
        # but run both paths and compare.
        def build() -> Graph:
            g = Graph()
            dim = config.sensor_output_dim
            s = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
            a = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
            b = Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config)
            o = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
            g.add_node(s, modality="text"); g.add_node(a); g.add_node(b)
            g.add_node(o, modality="text")
            for src, tgt in [(s, a), (a, b), (b, o), (b, a)]:
                g.add_edge(
                    Edge(
                        source_id=src.id,
                        target_id=tgt.id,
                        source_output_dim=dim,
                        target_input_dim=dim,
                        creation_step=0,
                        initial_weight=1.0,
                    )
                )
            return g

        torch.manual_seed(99)
        g1 = build()
        g2 = build()
        g2.load_state_dict(g1.state_dict())
        data = torch.ones(config.sensor_output_dim)
        _, prev_seq = execute_graph(g1, inputs={"text": data}, current_step=1)
        _, prev_bat = execute_graph_batched(g2, inputs={"text": data}, current_step=1)
        out_seq, _ = execute_graph(
            g1, inputs={"text": data}, current_step=2, previous_activations=prev_seq
        )
        out_bat, _ = execute_graph_batched(
            g2, inputs={"text": data}, current_step=2, previous_activations=prev_bat
        )
        assert torch.allclose(out_seq["text"], out_bat["text"], atol=1e-5, rtol=1e-5)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_core/test_execution.py::TestBatchedExecutorParity -v`
Expected: FAIL (current `execute_graph_batched` is a pass-through, so `test_batched_equals_sequential_*` tests may actually PASS because they exercise the delegate path. Ensure they fail by either: (a) temporarily making `_batched_node_forward` the real path via the env-var switch AND flipping it on in the test, OR (b) just accept that these four tests are "guard" tests that validate behavior after we change the body. They start GREEN while the body delegates; flipping the body preserves green. The failing signal comes from violating the invariant — if you implement the body wrong, they fail.)

This is a case where the RED is conceptual — the TDD failure here is that without a body we have not actually achieved the 3x speedup. To make the failure concrete, add:

```python
    def test_batched_body_not_delegating(self) -> None:
        # Read the source to confirm execute_graph_batched has real body,
        # not just `return execute_graph(...)`. This test will fail until
        # Task 7 replaces the delegate.
        import inspect
        from soma.core import execution
        src = inspect.getsource(execution.execute_graph_batched)
        assert "compute_wave_layers" in src, (
            "execute_graph_batched still delegates; Task 7 must replace the body"
        )
```

Run: `python -m pytest tests/test_core/test_execution.py::TestBatchedExecutorParity -v`
Expected: `test_batched_body_not_delegating` FAILs; the three parity tests PASS (via delegate).

**Step 3: Implement the real body**

Replace the body of `execute_graph_batched` in `src/soma/core/execution.py`:

```python
def execute_graph_batched(
    graph: Graph,
    inputs: Mapping[str, torch.Tensor],
    current_step: int,
    *,
    previous_activations: Mapping[str, torch.Tensor] | None = None,
    record_edge_activity: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Wave-batched variant of :func:`execute_graph`.

    Produces outputs and activations equal-up-to-float-reassociation
    to the sequential executor. Batches the per-node MLP kernels by
    shape within each topological wave to collapse ~N kernel launches
    per wave into ~4 (W1-matmul, bias, GELU, W2-matmul).
    """
    _inject_sensor_inputs(graph, inputs)

    _, back_edges = topological_sort(graph)
    waves = compute_wave_layers(graph)
    prev: Mapping[str, torch.Tensor] = previous_activations or {}
    activations: dict[str, torch.Tensor] = {}

    for wave in waves:
        # 1. Sensor nodes in this wave: forward each one sequentially
        # (their forward is just "return injected input"; no MLP).
        for node_id in wave:
            node = graph.nodes[node_id]
            if node.node_type is NodeType.SENSOR:
                activations[node_id] = node.forward({}, current_step)

        # 2. Non-sensor nodes: bucket by shape, batch per bucket.
        buckets = bucket_wave_by_shape(graph, wave)
        for _shape, bucket_ids in buckets.items():
            bucket_nodes = [graph.nodes[nid] for nid in bucket_ids]
            x, active_nodes = _aggregate_bucket_inputs(
                graph=graph,
                nodes=bucket_nodes,
                activations=activations,
                previous_activations=prev,
                back_edges=back_edges,
                current_step=current_step,
                record_edge_activity=record_edge_activity,
            )
            if not active_nodes:
                continue
            y = _batched_node_forward(active_nodes, x)
            # Unbind so each node's activation is its own tensor —
            # downstream loss.backward() can then freely route gradients.
            y_rows = torch.unbind(y, dim=0)
            for node, row in zip(active_nodes, y_rows, strict=True):
                activations[node.id] = row
            _record_batched_activations(active_nodes, y, current_step)

    outputs = _collect_outputs(graph, activations)
    return outputs, activations
```

**Step 4: Run the entire test file**

Run: `python -m pytest tests/test_core/test_execution.py -v`
Expected: ALL PASS (parity, bucket, batched primitive, record helper, scaffold, plus all original tests).

**Step 5: Commit**

```bash
git add src/soma/core/execution.py tests/test_core/test_execution.py
git commit -m "perf(exec): wire up wave-batched execute_graph_batched end-to-end"
```

---

## Task 8: Full-suite regression — existing tests unchanged

**Files:**
- No code changes
- Full pytest run

**Intent:** Prove the 703 existing tests still pass with `execute_graph_batched` in the tree. We have NOT yet switched `SOMA.step` or `consolidation_cycle` to use the batched path — those still call `execute_graph`. So this run just confirms we haven't broken `soma.core.execution` importing or any cross-module invariant.

**Step 1: Run**

`python -m pytest tests/ -x -q`
Expected: `703 passed` (or whatever the exact current count is).

If any fail: diagnose. Don't skip. The most likely failure mode is an import-ordering regression from adding `from soma.core.node import Node` if it creates a cycle. If that happens, import `Node` inside the functions that need it, or use `TYPE_CHECKING`.

**Step 2: Commit** (only if any clean-up edits were needed; else skip)

```bash
# only if changes made
git add -u
git commit -m "chore(exec): keep existing tests green after adding batched path"
```

---

## Task 9: Switch SOMA.step to the batched path behind a config flag

**Files:**
- Modify: `src/soma/core/config.py`
- Modify: `src/soma/system.py`
- Modify: `src/soma/consolidation/cycle.py`
- Test: `tests/test_core/test_config.py`
- Test: `tests/test_integration/test_stage1_core_loop.py` (or add a new small one)

**Intent:** Add `use_batched_executor: bool = True` on `SOMAConfig` (ON by default — we've proven parity). `SOMA.__init__` picks the executor based on the flag. Consolidation cycle uses the same flag. This is the switch that turns the optimization on for the running training loop.

**Step 1: Write the failing test**

Add to `tests/test_core/test_config.py`:

```python
def test_use_batched_executor_default_on() -> None:
    cfg = SOMAConfig()
    assert cfg.use_batched_executor is True


def test_use_batched_executor_can_be_disabled() -> None:
    cfg = SOMAConfig(use_batched_executor=False)
    assert cfg.use_batched_executor is False
```

Add to `tests/test_integration/test_stage1_core_loop.py` (or create `test_executor_switch.py`):

```python
def test_soma_step_with_batched_flag_matches_sequential() -> None:
    from soma.core.config import SOMAConfig
    from soma.system import SOMA

    base_config = SOMAConfig(
        sensor_output_dim=16,
        associator_input_dim=16,
        associator_hidden_dim=32,
        associator_output_dim=16,
        integrator_input_dim=32,
        integrator_hidden_dim=32,
        integrator_output_dim=32,
        position_dim=4,
        wm_slots=4,
        wm_dim=16,
        episodic_capacity=100,
        key_dim=16,
        value_dim=16,
        vocab_size=64,
        text_embed_dim=16,
        max_nodes=200,
        max_edges_per_node=10.0,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=16,
        max_output_tokens=8,
        checkpoint_interval=10_000,
        seed=123,
        use_batched_executor=False,
    )
    soma_seq = SOMA(base_config, device=torch.device("cpu"))

    bat_config = replace(base_config, use_batched_executor=True)
    soma_bat = SOMA(bat_config, device=torch.device("cpu"))
    # Sync graph params so the two systems start identically.
    soma_bat.graph.load_state_dict(soma_seq.graph.state_dict())

    inputs = {"text": torch.randn(base_config.sensor_output_dim)}
    targets = {"text": torch.zeros(base_config.sensor_output_dim)}
    r_seq = soma_seq.step(inputs=inputs, targets=targets)
    r_bat = soma_bat.step(inputs=inputs, targets=targets)

    # Losses match (the batched executor affects the forward pass only;
    # the update_step that follows is identical deterministic math on
    # identical gradients).
    assert r_seq["loss"] == pytest.approx(r_bat["loss"], rel=1e-5, abs=1e-7)
```

(Import `from dataclasses import replace` at top.)

**Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_core/test_config.py tests/test_integration/test_executor_switch.py -v` (or whichever path you used).
Expected: FAIL — `use_batched_executor` not defined on `SOMAConfig`.

**Step 3: Add the config field**

In `src/soma/core/config.py`:

```python
    # Execution-path switch. When True (default) SOMA.step uses the
    # wave-batched execute_graph_batched; when False it uses the
    # sequential execute_graph. Flip to False to A/B test or to fall
    # back if batching exposes a bug.
    use_batched_executor: bool = True
```

No validation clause needed (boolean).

**Step 4: Wire the switch in SOMA.step**

In `src/soma/system.py`, change:

```python
from soma.core.execution import execute_graph
```

to:

```python
from soma.core.execution import execute_graph, execute_graph_batched
```

And in the `step` method, replace:

```python
        outputs, activations = execute_graph(
            self.graph, inputs=inputs, current_step=self.global_step
        )
```

with:

```python
        exec_fn = (
            execute_graph_batched
            if self.config.use_batched_executor
            else execute_graph
        )
        outputs, activations = exec_fn(
            self.graph, inputs=inputs, current_step=self.global_step
        )
```

**Step 5: Wire the switch in consolidation_cycle**

In `src/soma/consolidation/cycle.py`, the existing import of `execute_graph` stays. Add alongside:

```python
from soma.core.execution import execute_graph, execute_graph_batched
```

and in `consolidation_cycle`, near the top where `config` is available, select the executor and use it in the replay loop:

```python
    exec_fn = (
        execute_graph_batched
        if config.use_batched_executor
        else execute_graph
    )
    for experience in replay_batch:
        inputs, targets = experience_unpacker(experience)
        outputs, activations = exec_fn(
            graph,
            inputs=inputs,
            current_step=current_step,
            record_edge_activity=False,
        )
```

**Step 6: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All 703+ pass (new tests added earlier push the total up).

If the `test_soma_step_with_batched_flag_matches_sequential` integration test fails by more than `rel=1e-5`, **stop** — the wiring is broken somewhere. Common causes:
- Random state differs between the two SOMA instances because `torch.manual_seed` was called once globally and the second instance drew different weights. The `load_state_dict` sync after construction should eliminate this.
- Hebbian / homeostasis reads activations as Python floats — ensure the batched path's `activations` tensors are float32 and have the same dtype as the sequential path's.
- Edge `mark_active` ordering difference — we iterate edges within `_aggregate_bucket_inputs` in the same order as the sequential path, so this should match. If `last_active_step` diverges, check the order inside the bucket.

**Step 7: Commit**

```bash
git add src/soma/core/config.py src/soma/system.py src/soma/consolidation/cycle.py \
        tests/test_core/test_config.py tests/test_integration/
git commit -m "perf(exec): enable batched executor in SOMA.step and consolidation (flag on by default)"
```

---

## Task 10: Strict regression test — old vs new on a realistic graph

**Files:**
- Create: `tests/test_core/test_execution_parity.py`

**Intent:** One high-signal test that captures the **full** equivalence contract: build a 10-node mixed-type graph, run OLD path, cache outputs and activations; then run NEW path, assert `allclose` on every output, every activation value, every gradient, every per-node state field. This is the "if this passes, shipping the optimization is safe" test.

**Step 1: Write the test**

Create `tests/test_core/test_execution_parity.py`:

```python
"""Strict parity between sequential and batched execute_graph.

If this test passes, the wave-batching optimization preserves all
observable behavior: outputs, activations, gradients on every param,
and per-node bookkeeping (gain/ema/last_active_step/history).
"""

from __future__ import annotations

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph, execute_graph_batched
from soma.core.graph import Graph
from soma.core.node import Node, NodeType


def _build_realistic_graph(config: SOMAConfig, seed: int = 2026) -> Graph:
    torch.manual_seed(seed)
    g = Graph()
    dim = config.sensor_output_dim
    # 1 sensor, 2 outputs (different modalities), 5 associators, 2 integrators
    s = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config)
    g.add_node(s, modality="text")
    assocs = [
        Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config) for _ in range(5)
    ]
    for a in assocs:
        g.add_node(a)
    # Integrators with a different shape to force multi-bucket waves.
    integs = [
        Node(NodeType.INTEGRATOR, dim, dim * 4, dim * 2, 0, config) for _ in range(2)
    ]
    for i in integs:
        g.add_node(i)
    # Projector assoc to consume integrator output (different input shape).
    proj = Node(NodeType.ASSOCIATOR, dim * 2, dim * 2, dim, 0, config)
    g.add_node(proj)
    o_text = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    o_text2 = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config)
    g.add_node(o_text, modality="text")
    g.add_node(o_text2, modality="image")  # name misleading; just a second output

    def _e(src: Node, tgt: Node, w: float = 0.3) -> None:
        g.add_edge(
            Edge(
                source_id=src.id,
                target_id=tgt.id,
                source_output_dim=src.output_dim,
                target_input_dim=tgt.input_dim,
                creation_step=0,
                initial_weight=w,
            )
        )

    # Fan-out from sensor.
    for a in assocs:
        _e(s, a)
    for i in integs:
        _e(s, i)
    # Integrators -> proj (requires dim projection in Edge since dims differ).
    for i in integs:
        _e(i, proj)
    # Assocs + proj -> outputs.
    for a in assocs:
        _e(a, o_text)
        _e(a, o_text2)
    _e(proj, o_text)
    _e(proj, o_text2)
    return g


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig()


def test_full_parity_forward_outputs(config: SOMAConfig) -> None:
    g1 = _build_realistic_graph(config)
    g2 = _build_realistic_graph(config)
    g2.load_state_dict(g1.state_dict())
    data = torch.randn(config.sensor_output_dim)
    out_seq, act_seq = execute_graph(g1, inputs={"text": data}, current_step=5)
    out_bat, act_bat = execute_graph_batched(g2, inputs={"text": data}, current_step=5)
    assert set(out_seq) == set(out_bat)
    for k in out_seq:
        diff = (out_seq[k] - out_bat[k]).abs().max().item()
        assert torch.allclose(out_seq[k], out_bat[k], atol=1e-5, rtol=1e-5), (
            f"output[{k}] max diff = {diff}"
        )
    # Active-node ID sets must match.
    assert set(act_seq) == set(act_bat)
    for nid in act_seq:
        assert torch.allclose(act_seq[nid], act_bat[nid], atol=1e-5, rtol=1e-5), (
            f"activation[{nid[:8]}] max diff = "
            f"{(act_seq[nid] - act_bat[nid]).abs().max().item()}"
        )


def test_full_parity_gradients(config: SOMAConfig) -> None:
    g1 = _build_realistic_graph(config)
    g2 = _build_realistic_graph(config)
    g2.load_state_dict(g1.state_dict())
    data = torch.randn(config.sensor_output_dim)
    out_seq, _ = execute_graph(g1, inputs={"text": data}, current_step=5)
    out_bat, _ = execute_graph_batched(g2, inputs={"text": data}, current_step=5)
    # Same scalar loss in both.
    loss_seq = torch.stack(list(out_seq.values())).sum()
    loss_bat = torch.stack(list(out_bat.values())).sum()
    loss_seq.backward()
    loss_bat.backward()

    # Compare every gradient on every node and every edge.
    for nid in g1.nodes:
        n1, n2 = g1.nodes[nid], g2.nodes[nid]
        for pname in ("linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"):
            p1 = dict(n1.named_parameters())[pname]
            p2 = dict(n2.named_parameters())[pname]
            if p1.grad is None and p2.grad is None:
                continue
            assert p1.grad is not None and p2.grad is not None, (
                f"grad-presence mismatch {nid[:8]}.{pname}"
            )
            diff = (p1.grad - p2.grad).abs().max().item()
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-5), (
                f"grad diverges on {nid[:8]}.{pname}: max diff {diff}"
            )
    for eid in g1.edges:
        e1, e2 = g1.edges[eid], g2.edges[eid]
        if e1.weight.grad is None and e2.weight.grad is None:
            continue
        assert torch.allclose(e1.weight.grad, e2.weight.grad, atol=1e-5, rtol=1e-5)
        if e1.projection is not None:
            assert torch.allclose(
                e1.projection.weight.grad, e2.projection.weight.grad,
                atol=1e-5, rtol=1e-5,
            )
            assert torch.allclose(
                e1.projection.bias.grad, e2.projection.bias.grad,
                atol=1e-5, rtol=1e-5,
            )


def test_full_parity_per_node_state(config: SOMAConfig) -> None:
    g1 = _build_realistic_graph(config)
    g2 = _build_realistic_graph(config)
    g2.load_state_dict(g1.state_dict())
    data = torch.randn(config.sensor_output_dim)
    execute_graph(g1, inputs={"text": data}, current_step=42)
    execute_graph_batched(g2, inputs={"text": data}, current_step=42)
    for nid in g1.nodes:
        n1, n2 = g1.nodes[nid], g2.nodes[nid]
        assert n1.last_active_step == n2.last_active_step, (
            f"{nid[:8]}: last_active_step {n1.last_active_step} != {n2.last_active_step}"
        )
        assert n1.activation_ema == pytest.approx(
            n2.activation_ema, rel=1e-6, abs=1e-9
        ), f"{nid[:8]}: ema {n1.activation_ema} != {n2.activation_ema}"
        assert n1.activation_history.to_list() == pytest.approx(
            n2.activation_history.to_list(), rel=1e-6, abs=1e-9
        )
    # Edge last_active_step must match too.
    for eid in g1.edges:
        assert g1.edges[eid].last_active_step == g2.edges[eid].last_active_step


def test_full_parity_over_multiple_steps(config: SOMAConfig) -> None:
    """Chain three steps with previous_activations. Any drift in state
    should accumulate visibly over 3+ steps."""
    g1 = _build_realistic_graph(config)
    g2 = _build_realistic_graph(config)
    g2.load_state_dict(g1.state_dict())

    prev_seq: dict[str, torch.Tensor] = {}
    prev_bat: dict[str, torch.Tensor] = {}
    for step in range(3):
        data = torch.randn(config.sensor_output_dim)
        out_seq, prev_seq = execute_graph(
            g1, inputs={"text": data}, current_step=step,
            previous_activations=prev_seq,
        )
        out_bat, prev_bat = execute_graph_batched(
            g2, inputs={"text": data}, current_step=step,
            previous_activations=prev_bat,
        )
        for k in out_seq:
            assert torch.allclose(out_seq[k], out_bat[k], atol=1e-5, rtol=1e-5), (
                f"step {step}: output[{k}] divergence"
            )
```

**Step 2: Run**

Run: `python -m pytest tests/test_core/test_execution_parity.py -v`
Expected: ALL PASS. If any fail, stop — the batched executor has a bug.

**Step 3: Commit**

```bash
git add tests/test_core/test_execution_parity.py
git commit -m "test(exec): strict parity regression (outputs, grads, per-node state, multi-step)"
```

---

## Task 11: Benchmark script

**Files:**
- Create: `scripts/bench_execute_graph.py`

**Intent:** Build a 34-node / ~100-edge graph at `embed_dim=64` (matching current training config), run 200 warmup + 1000 measured steps with each executor, report steps/sec. Report speedup ratio. Acceptance criterion: **3x or better on CUDA**. On CPU we expect smaller gains (1.2-2x) because CPU-side Python overhead dominates.

**Step 1: Create the script**

`scripts/bench_execute_graph.py`:

```python
"""Benchmark sequential vs batched execute_graph.

Usage:
    python scripts/bench_execute_graph.py [--device cpu|cuda] [--steps 1000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph, execute_graph_batched
from soma.core.graph import Graph
from soma.core.node import Node, NodeType


def build_benchmark_graph(config: SOMAConfig, device: torch.device) -> Graph:
    """34 nodes / ~100 edges at embed_dim=64 — matches current training config."""
    g = Graph()
    dim = config.sensor_output_dim  # 64
    # 2 sensors (text+image), 2 outputs, 30 associators. 34 total.
    s_text = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config, device=device)
    s_img = Node(NodeType.SENSOR, dim, dim * 2, dim, 0, config, device=device)
    g.add_node(s_text, modality="text")
    g.add_node(s_img, modality="image")
    assocs = [
        Node(NodeType.ASSOCIATOR, dim, dim * 2, dim, 0, config, device=device)
        for _ in range(30)
    ]
    for a in assocs:
        g.add_node(a)
    o_text = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config, device=device)
    o_img = Node(NodeType.OUTPUT, dim, dim * 2, dim, 0, config, device=device)
    g.add_node(o_text, modality="text")
    g.add_node(o_img, modality="image")
    # Edges: each sensor fanouts to 15 assocs; each assoc to 2 outputs.
    # That's 2*15 + 30*2 = 30 + 60 = 90 edges. Add a couple of assoc->assoc
    # for variety. Target ~100.
    rng = torch.Generator().manual_seed(0)

    def _e(src: Node, tgt: Node) -> None:
        g.add_edge(
            Edge(
                source_id=src.id,
                target_id=tgt.id,
                source_output_dim=src.output_dim,
                target_input_dim=tgt.input_dim,
                creation_step=0,
                initial_weight=0.1,
                device=device,
            )
        )

    for a in assocs[:15]:
        _e(s_text, a)
    for a in assocs[15:]:
        _e(s_img, a)
    for a in assocs:
        _e(a, o_text)
        _e(a, o_img)
    # ~10 intra-associator edges for variety.
    idx = torch.randperm(len(assocs), generator=rng).tolist()
    for i in range(0, 20, 2):
        src, tgt = assocs[idx[i]], assocs[idx[i + 1]]
        if not g.has_edge(src.id, tgt.id):
            _e(src, tgt)
    return g


def run_bench(
    graph: Graph,
    data: torch.Tensor,
    fn,
    warmup: int,
    steps: int,
    device: torch.device,
) -> float:
    """Return steps/sec."""
    prev: dict[str, torch.Tensor] = {}
    for step in range(warmup):
        _, prev = fn(graph, inputs={"text": data, "image": data}, current_step=step,
                    previous_activations=prev)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(warmup, warmup + steps):
        _, prev = fn(graph, inputs={"text": data, "image": data}, current_step=step,
                    previous_activations=prev)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return steps / elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = torch.device("cpu")
    config = SOMAConfig()
    g_seq = build_benchmark_graph(config, device)
    g_bat = build_benchmark_graph(config, device)
    g_bat.load_state_dict(g_seq.state_dict())

    data = torch.randn(config.sensor_output_dim, device=device)
    # Deterministic comparison: use the same data across both runs.
    print(f"[bench] device={device.type}  warmup={args.warmup}  steps={args.steps}")
    print(f"[bench] graph: {g_seq.num_nodes} nodes, {g_seq.num_edges} edges")
    sps_seq = run_bench(g_seq, data, execute_graph, args.warmup, args.steps, device)
    sps_bat = run_bench(g_bat, data, execute_graph_batched, args.warmup, args.steps, device)
    ratio = sps_bat / sps_seq if sps_seq > 0 else float("inf")
    print(f"[bench] sequential: {sps_seq:.1f} steps/sec")
    print(f"[bench] batched:    {sps_bat:.1f} steps/sec")
    print(f"[bench] speedup:    {ratio:.2f}x")
    if device.type == "cuda" and ratio < 3.0:
        print(f"[bench] WARN: speedup below target 3x (got {ratio:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Smoke-run on CPU**

Run: `python scripts/bench_execute_graph.py --device cpu --steps 200 --warmup 50`
Expected: prints two steps/sec numbers and a ratio. On CPU the ratio is typically 1.2-2.0x because Python overhead dominates. That's fine — the target is CUDA.

**Step 3: Commit**

```bash
git add scripts/bench_execute_graph.py
git commit -m "bench(exec): add sequential-vs-batched execute_graph benchmark script"
```

---

## Task 12: Smoke-train script

**Files:**
- Create: `scripts/smoke_wave_batching.py`

**Intent:** Train SOMA for 5000 steps on a tiny corpus, first with the sequential executor, then reset and train with the batched executor from the same seed, then compare final held-out losses. Acceptance: (a) no NaN/crash in either run, (b) batched final held-out loss within 10% of sequential.

This is not a pytest test; it's a manual/CI smoke that the operator can run overnight. It needs its own entry point so it doesn't bloat the test suite.

**Step 1: Create**

`scripts/smoke_wave_batching.py`:

```python
"""5000-step smoke for wave-batching: sequential vs batched training.

Runs two short training sessions from identical seeds/configs, one
using execute_graph, one using execute_graph_batched. Verifies:
  (a) no NaN/exception in either run
  (b) batched final held-out loss within 10% of sequential

Usage:
    python scripts/smoke_wave_batching.py --steps 5000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from soma.core.config import SOMAConfig
from soma.io.dataset_feeders import TextDatasetFeeder
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.system import SOMA

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "lorem ipsum dolor sit amet consectetur adipiscing elit",
    "to be or not to be that is the question",
    "once upon a time in a land far far away",
    "alpha beta gamma delta epsilon zeta eta theta iota",
]

HELDOUT = [
    "the fox and the hound are friends",
    "nothing gold can stay",
]


def run_training(config: SOMAConfig, steps: int, device: torch.device) -> dict:
    tokenizer = train_bpe_tokenizer(CORPUS, vocab_size=config.vocab_size)
    encoder = TextEncoder(
        tokenizer,
        embed_dim=config.text_embed_dim,
        max_seq_len=config.max_input_tokens,
        device=device,
    )
    feeder = TextDatasetFeeder(encoder, CORPUS, chunk_size=4)
    soma = SOMA(config, device=device)
    first_out = config.output_modalities[0]
    step = 0
    last_loss = float("nan")
    t0 = time.perf_counter()
    for sample in feeder:
        if step >= steps:
            break
        if sample.target.numel() == 0:
            continue
        inputs = {m: t[0].detach() for m, t in sample.inputs.items() if t.numel() > 0}
        if not inputs:
            continue
        target = {first_out: sample.target[0].detach()}
        try:
            res = soma.step(inputs=inputs, targets=target)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"exception at step {step}: {exc}"}
        loss = res.get("loss")
        if loss is not None:
            if not math.isfinite(loss):
                return {"ok": False, "reason": f"non-finite loss at step {step}: {loss}"}
            last_loss = loss
        step += 1
    elapsed = time.perf_counter() - t0
    # Evaluate on HELDOUT.
    heldout_feeder = TextDatasetFeeder(encoder, HELDOUT, chunk_size=4)
    heldout_losses: list[float] = []
    for sample in heldout_feeder:
        if sample.target.numel() == 0:
            continue
        inputs = {m: t[0].detach() for m, t in sample.inputs.items() if t.numel() > 0}
        if not inputs:
            continue
        target = {first_out: sample.target[0].detach()}
        res = soma.step(inputs=inputs, targets=target, eval_mode=True)
        hl = res.get("loss")
        if hl is not None and math.isfinite(hl):
            heldout_losses.append(hl)
    heldout = sum(heldout_losses) / max(1, len(heldout_losses))
    return {
        "ok": True,
        "steps": step,
        "final_train_loss": last_loss,
        "heldout_loss": heldout,
        "elapsed_sec": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args(argv)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    base = SOMAConfig(
        sensor_output_dim=16,
        associator_input_dim=16,
        associator_hidden_dim=32,
        associator_output_dim=16,
        integrator_input_dim=32,
        integrator_hidden_dim=32,
        integrator_output_dim=32,
        position_dim=4,
        wm_slots=4,
        wm_dim=16,
        episodic_capacity=100,
        key_dim=16,
        value_dim=16,
        vocab_size=64,
        text_embed_dim=16,
        max_nodes=300,
        max_edges_per_node=10.0,
        initial_associator_count=4,
        initial_integrator_count=2,
        max_input_tokens=16,
        max_output_tokens=8,
        checkpoint_interval=10_000,
        seed=42,
    )
    seq_cfg = replace(base, use_batched_executor=False)
    bat_cfg = replace(base, use_batched_executor=True)
    print(f"[smoke] seq run (steps={args.steps})...")
    seq_res = run_training(seq_cfg, args.steps, device)
    print(f"[smoke]   -> {seq_res}")
    print(f"[smoke] bat run (steps={args.steps})...")
    bat_res = run_training(bat_cfg, args.steps, device)
    print(f"[smoke]   -> {bat_res}")
    if not seq_res["ok"] or not bat_res["ok"]:
        print("[smoke] FAILED: one of the runs crashed")
        return 1
    seq_h = seq_res["heldout_loss"]
    bat_h = bat_res["heldout_loss"]
    # "within 10%" (batched may be worse OR better by random seed noise;
    # we care that it's not wildly off).
    rel = abs(bat_h - seq_h) / max(abs(seq_h), 1e-8)
    print(f"[smoke] heldout seq={seq_h:.5f}  bat={bat_h:.5f}  rel-diff={rel:.3%}")
    if rel > 0.10:
        print(f"[smoke] FAILED: heldout relative diff {rel:.1%} > 10%")
        return 1
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Run on CPU with a small step count to confirm it doesn't crash**

Run: `python scripts/smoke_wave_batching.py --steps 200 --device cpu`
Expected: both runs finish; the `OK` line prints. Full 5000-step run is for the operator on CUDA later.

**Step 3: Commit**

```bash
git add scripts/smoke_wave_batching.py
git commit -m "bench(exec): add 5000-step wave-batching smoke train script"
```

---

## Task 13: Documentation & rollback plan

**Files:**
- Modify: `src/soma/core/execution.py` (docstring at module top)
- Modify: `CLAUDE.md` (append a short note under "Architecture" section explaining both paths exist)

**Intent:** Record the invariant contract and how to disable batching if it ever misbehaves. If a future bug appears in the batched path, the operator flips `SOMAConfig(use_batched_executor=False)` and the system falls back to the sequential path — zero data migration, zero weight changes.

**Step 1: Update module docstring**

At the top of `src/soma/core/execution.py`, append to the existing module docstring:

```python
"""
...

Two executors are available:

- :func:`execute_graph` — sequential reference path. One Python-level
  iteration over nodes; one MLP's worth of kernel launches per non-dormant
  node per wave. Always correct; easy to debug.
- :func:`execute_graph_batched` — wave-batched path. Groups same-shape
  nodes within each topological wave into a single stacked matmul per
  MLP layer, collapsing per-node kernel launches into per-bucket launches.
  Produces outputs equal-up-to-float-reassociation (atol=1e-5) to the
  sequential path. Use this in hot loops on GPU.

``SOMAConfig.use_batched_executor`` selects which path
``SOMA.step`` and ``consolidation_cycle`` use at runtime. Both paths are
tested for per-step parity in ``tests/test_core/test_execution_parity.py``.
"""
```

**Step 2: Add a rollback section to CLAUDE.md**

Append to `CLAUDE.md` under the "Architecture" section (or add a new "Execution Paths" subsection):

```markdown
### Execution Paths

Two `execute_graph` implementations exist:

- **Sequential (`execute_graph`)**: one node at a time; reference behavior.
- **Batched (`execute_graph_batched`)**: same-shape nodes in each wave share one stacked matmul. 3-5x faster on CUDA, identical outputs (atol=1e-5).

`SOMAConfig.use_batched_executor=True` (default) selects the batched path.
Flip to `False` to fall back to sequential if a bug ever surfaces — no
weight or data migration needed.
```

**Step 3: Commit**

```bash
git add src/soma/core/execution.py CLAUDE.md
git commit -m "docs(exec): document dual executor paths and rollback via SOMAConfig"
```

---

## Task 14: Benchmark target verification (operator-run on CUDA)

**NOT A CODE TASK — this is a gate for the implementation agent to hand back to the operator.**

Before declaring the plan complete:

1. On CUDA (operator's RTX 3090), run `python scripts/bench_execute_graph.py --device cuda --steps 1000`.
2. Confirm the reported speedup is ≥ 3.0x. If it is, the plan's mission is accomplished.
3. If the speedup is < 3.0x on CUDA:
   - **Do NOT roll back.** The batched path is still correct and typically 1.5-2x faster; that's net-positive.
   - File a follow-up issue: "wave-batching below 3x target; investigate edge-transmit batching (Task 15 placeholder)".
   - Keep `use_batched_executor=True` — the sequential fallback stays available via the flag.
4. Report the CUDA speedup number back to the operator with the benchmark output quoted.

---

## Task 15 (optional, only if Task 14 speedup < 3x): Edge-transmit batching

**Status:** Out of scope for this plan unless the primary optimization under-delivers. Documented here for continuity.

**Motivation:** After node-MLP batching, the residual hotspot is often `Edge.transmit`, especially on graphs where each target has many incoming edges each with its own projection `nn.Linear`. We can batch these too, but only within a single target's fan-in (the projection weights may differ per edge).

**Sketch:**
- For each node-bucket in `_aggregate_bucket_inputs`, group incoming edges by `(source_output_dim, target_input_dim)` and whether a projection exists.
- For edges sharing the same projection shape, stack their weights and their source activations and do one batched matmul, then scale by a stacked weight vector.
- Gradients flow back through `torch.stack` on edge parameters the same way they do for node parameters.

**Deliverable:** A new helper `_batched_edge_transmit` plus a parity test and a benchmark delta, as a separate plan / follow-up PR. Do not include in the initial wave-batching PR.

---

## Rollback plan (summary)

If anything downstream breaks:

1. **Quick toggle:** `SOMAConfig(use_batched_executor=False)` in `configs/current.yaml` or runtime. Training continues on the sequential path. Zero migration cost.
2. **Revert the commits** (in order from Task 13 down to Task 1) if the executor module itself is broken. The parity tests are the regression fence — a revert restores exact prior behavior.
3. **Known non-issues:**
   - FP32 rounding differences at the atol=1e-5 level are expected and do not indicate a bug.
   - `last_active_step` and `activation_history` ordering is guaranteed because `compute_wave_layers` and `bucket_wave_by_shape` preserve the underlying topological order from `topological_sort`.

---

## Execution handoff

**Plan complete and saved to `docs/plans/2026-04-14-wave-batching.md`. Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** — Open a new session in this worktree with `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**
