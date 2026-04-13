# Training Stability Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the monotonic weight-saturation bug that drove edge weights to ±5.0 clamp on 99.8% of edges by step 10K, and harden the training loop against future numerical instability.

**Architecture:** Three independent fixes layered together. (1) Add multiplicative weight decay to the Hebbian rule so weights don't drift to the clamp; this is the root-cause fix. (2) Add gradient clipping after `loss.backward()` to bound backprop-driven updates. (3) Replace the homeostasis raise-on-non-finite with a graceful skip + escalation counter so the train_service stops dying on transient bad steps.

**Tech Stack:** PyTorch (existing), pytest (existing), `torch.nn.utils.clip_grad_norm_`.

**Why these three together:** Diagnostic in `scripts/diagnose_nan.py` shows the existing Hebbian rule is monotonically positive (only ever adds) so weights drift to the +clamp inevitably. Once saturated, activations explode (we observed 1e15) and loss goes inf/nan. Fix #1 prevents the drift. Fix #2 prevents backprop's gradient-explosion contribution. Fix #3 keeps the service alive while #1 and #2 take effect, and provides a real signal when training is genuinely stuck.

---

## Pre-flight notes

- Branch: working on `main` directly (single-operator project, autonomous loop already running off main).
- All existing checkpoints at step ≥10000 are corrupt (saturated weights). After the code changes, archive them and let train_service start fresh.
- Push will likely fail (no tty in this session); commits land locally for operator visibility.
- Loop is currently halted via `.soma-loop/STOP`. Will be cleared as the last step.

---

## Task 1: Add new SOMAConfig fields

**Files:**
- Modify: `src/soma/core/config.py`
- Test: `tests/test_core/test_config.py` (extend existing)

**Step 1: Write the failing test**

Add to `tests/test_core/test_config.py`:

```python
def test_new_stability_fields_have_sane_defaults() -> None:
    cfg = SOMAConfig()
    assert 0.0 < cfg.edge_weight_decay <= 1.0
    assert cfg.edge_weight_decay > 0.99  # very gentle by default
    assert cfg.grad_clip_max_norm > 0.0
    assert cfg.max_consecutive_skipped_steps >= 1


def test_edge_weight_decay_validated() -> None:
    with pytest.raises(ValueError, match="edge_weight_decay"):
        SOMAConfig(edge_weight_decay=0.0)
    with pytest.raises(ValueError, match="edge_weight_decay"):
        SOMAConfig(edge_weight_decay=1.5)


def test_grad_clip_max_norm_validated() -> None:
    with pytest.raises(ValueError, match="grad_clip_max_norm"):
        SOMAConfig(grad_clip_max_norm=0.0)
    with pytest.raises(ValueError, match="grad_clip_max_norm"):
        SOMAConfig(grad_clip_max_norm=-1.0)


def test_max_consecutive_skipped_steps_validated() -> None:
    with pytest.raises(ValueError, match="max_consecutive_skipped_steps"):
        SOMAConfig(max_consecutive_skipped_steps=0)
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_core/test_config.py -k "stability_fields or edge_weight_decay or grad_clip or max_consecutive" -v`
Expected: FAIL (AttributeError or similar — fields don't exist yet)

**Step 3: Add fields and validation**

In `src/soma/core/config.py`, in the `SOMAConfig` dataclass, add to the Learning section (after `maturity_increment`):

```python
    # Edge weight decay: multiplicative shrink applied to every edge each
    # step. With hebbian_lr=0.0001 the equilibrium weight ≈ s*t for a
    # constantly co-active edge. Default 0.9999 ≈ 1 - hebbian_lr.
    edge_weight_decay: float = 0.9999
    # Gradient L2-norm clip applied after loss.backward() in update_step.
    grad_clip_max_norm: float = 1.0
    # If the loss is non-finite for this many consecutive training steps,
    # SOMA.step raises so train_service's CrashBackoff fires.
    max_consecutive_skipped_steps: int = 50
```

In `_validate()`, append:

```python
        if not 0.0 < self.edge_weight_decay <= 1.0:
            raise ValueError(
                f"SOMAConfig.edge_weight_decay must be in (0, 1], got {self.edge_weight_decay!r}"
            )
        if self.grad_clip_max_norm <= 0.0:
            raise ValueError(
                f"SOMAConfig.grad_clip_max_norm must be positive, got {self.grad_clip_max_norm!r}"
            )
        if self.max_consecutive_skipped_steps < 1:
            raise ValueError(
                f"SOMAConfig.max_consecutive_skipped_steps must be >= 1, "
                f"got {self.max_consecutive_skipped_steps!r}"
            )
```

**Step 4: Verify tests pass**

Run: `pytest tests/test_core/test_config.py -v`
Expected: PASS (all original + 4 new tests)

**Step 5: Commit**

```bash
git add src/soma/core/config.py tests/test_core/test_config.py
git commit -m "feat(config): add edge_weight_decay, grad_clip_max_norm, max_consecutive_skipped_steps"
```

---

## Task 2: Implement edge weight decay (Hebbian counterbalance)

**Files:**
- Modify: `src/soma/core/learning.py:133-162` (the `_apply_hebbian_edge_updates` function)
- Test: `tests/test_core/test_learning.py` (add new test class)

**Background:** The current rule `edge.weight.data.add_(hebbian_lr * source_mag * target_mag)` is one-way upward. We add a multiplicative decay applied to ALL edges (active or not) before the additive Hebbian term. Equilibrium for a constantly co-active edge is roughly `s*t / (1 - decay)`, which with `decay=0.9999` and `hebbian_lr=0.0001` gives weight ≈ `s*t`.

**Step 1: Write failing tests**

Add to `tests/test_core/test_learning.py` (a new class at the bottom):

```python
class TestEdgeWeightDecay:
    """Verify Hebbian counterbalance: weights don't drift to clamp."""

    def _two_node_decay_graph(self, initial_weight: float) -> tuple[Graph, Edge]:
        graph = Graph()
        dim = 4
        cfg = SOMAConfig(
            base_lr=0.001,
            hebbian_lr=0.001,
            edge_weight_decay=0.99,  # aggressive for fast tests
            activation_threshold=0.01,
        )
        sensor = Node(NodeType.SENSOR, dim, dim, dim, 0, cfg)
        out = Node(NodeType.OUTPUT, dim, dim, dim, 0, cfg)
        graph.add_node(sensor, modality="text")
        graph.add_node(out, modality="text")
        edge = Edge(
            source_id=sensor.id,
            target_id=out.id,
            source_output_dim=dim,
            target_input_dim=dim,
            creation_step=0,
            initial_weight=initial_weight,
        )
        graph.add_edge(edge)
        return graph, edge

    def test_inactive_edge_decays_toward_zero(self) -> None:
        """An edge with no co-activation should shrink each step."""
        graph, edge = self._two_node_decay_graph(initial_weight=1.0)
        cfg = SOMAConfig(
            base_lr=0.001, hebbian_lr=0.001, edge_weight_decay=0.99,
            activation_threshold=0.01,
        )
        # Provide empty activations so no edge is "co-active."
        loss = torch.zeros((), requires_grad=True)
        initial = float(edge.weight.detach().item())
        for _ in range(10):
            update_step(graph, loss, activations={}, config=cfg)
        final = float(edge.weight.detach().item())
        assert final < initial, f"weight should decay; initial={initial}, final={final}"
        # 10 * decay 0.99 → 0.904 of initial
        assert abs(final - initial * (0.99 ** 10)) < 1e-4

    def test_co_active_edge_reaches_equilibrium_below_clamp(self) -> None:
        """A constantly co-active edge should converge below the clamp."""
        cfg = SOMAConfig(
            base_lr=0.0,  # disable backprop SGD so we isolate Hebbian + decay
            hebbian_lr=0.01,
            edge_weight_decay=0.99,
            activation_threshold=0.01,
            max_edge_weight=5.0,
        )
        graph, edge = self._two_node_decay_graph(initial_weight=0.01)
        # equilibrium: w*(1-decay) = hebbian_lr * s*t
        #            → w_eq = 0.01 * 1 * 1 / (1 - 0.99) = 1.0
        sensor_id, out_id = list(graph.nodes.keys())
        activations = {
            sensor_id: torch.ones(4),
            out_id: torch.ones(4),
        }
        loss = torch.zeros((), requires_grad=True)
        for _ in range(2000):
            update_step(graph, loss, activations=activations, config=cfg)
        final = float(edge.weight.detach().item())
        assert 0.5 < final < 2.0, f"equilibrium should be ~1.0, got {final}"
        assert final < cfg.max_edge_weight, "must stay below clamp"

    def test_decay_does_not_flip_sign(self) -> None:
        """Negative weights stay negative under decay."""
        cfg = SOMAConfig(
            base_lr=0.0, hebbian_lr=0.0,  # only decay
            edge_weight_decay=0.5,  # extreme
            activation_threshold=0.01,
        )
        graph, edge = self._two_node_decay_graph(initial_weight=-1.0)
        loss = torch.zeros((), requires_grad=True)
        for _ in range(20):
            update_step(graph, loss, activations={}, config=cfg)
        final = float(edge.weight.detach().item())
        assert final < 0.0, f"negative weight stayed negative? got {final}"
        assert final > -1e-3, f"weight should decay close to zero, got {final}"
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_core/test_learning.py::TestEdgeWeightDecay -v`
Expected: FAIL (decay isn't applied yet — `test_inactive_edge_decays_toward_zero` fails because weight stays at 1.0)

**Step 3: Implement decay**

In `src/soma/core/learning.py`, modify `_apply_hebbian_edge_updates` (around line 133):

```python
def _apply_hebbian_edge_updates(
    graph: Graph,
    activations: Mapping[str, torch.Tensor],
    *,
    config: SOMAConfig,
) -> None:
    """Fire-together-wire-together edge reinforcement, clamp, and strength EMA.

    Also applies a multiplicative weight decay to every edge (active or
    not). The decay counterbalances the monotonically-positive Hebbian
    add term so weights don't drift to the clamp boundary over time.
    """
    threshold = config.activation_threshold
    max_w = config.max_edge_weight
    hebbian_lr = config.hebbian_lr
    decay = config.edge_weight_decay

    with torch.no_grad():
        # Pass 1: passive decay on every edge.
        if decay < 1.0:
            for edge in graph.all_edges():
                edge.weight.data.mul_(decay)

        # Pass 2: Hebbian + clamp + strength EMA on co-active edges.
        for edge in graph.all_edges():
            source_act = activations.get(edge.source_id)
            target_act = activations.get(edge.target_id)
            if source_act is None or target_act is None:
                continue  # edge did not participate this step

            source_mag = float(source_act.detach().norm().item())
            target_mag = float(target_act.detach().norm().item())

            if source_mag > threshold and target_mag > threshold:
                edge.increment_coactivation()
                edge.weight.data.add_(hebbian_lr * source_mag * target_mag)

            # Clamp after Hebbian bump so weights never exceed ±max_w.
            _clamp_edge(edge, max_w)

            # Strength EMA (pruning utility).
            edge.update_strength(signal_magnitude=source_mag, decay=0.999)
```

**Step 4: Verify tests pass**

Run: `pytest tests/test_core/test_learning.py -v`
Expected: PASS (existing + 3 new TestEdgeWeightDecay tests)

If `test_co_active_edge_reaches_equilibrium_below_clamp` is finicky about 2000 iterations, it's because the equilibrium is asymptotic — extend to 5000 or relax bounds.

**Step 5: Commit**

```bash
git add src/soma/core/learning.py tests/test_core/test_learning.py
git commit -m "fix(core): add edge weight decay to counterbalance monotonic Hebbian growth"
```

---

## Task 3: Implement gradient clipping in update_step

**Files:**
- Modify: `src/soma/core/learning.py:64-86` (the `update_step` function)
- Test: `tests/test_core/test_learning.py` (add new test class)

**Step 1: Write failing tests**

Add to `tests/test_core/test_learning.py`:

```python
class TestGradientClipping:
    """Verify gradient clipping bounds the update magnitude."""

    def test_huge_gradient_gets_clipped(self, config: SOMAConfig) -> None:
        # Use the existing _three_node_graph helper from test_learning.py.
        graph, sensor, assoc, out = _three_node_graph(config)
        cfg = SOMAConfig(
            base_lr=config.base_lr,
            hebbian_lr=config.hebbian_lr,
            grad_clip_max_norm=0.5,  # tight bound
            activation_threshold=config.activation_threshold,
        )
        # Manually inject huge gradients on assoc node params.
        for param in assoc.parameters():
            param.grad = torch.full_like(param, 100.0)
        loss = torch.zeros((), requires_grad=True)
        before = {id(p): p.detach().clone() for p in assoc.parameters()}
        update_step(
            graph, loss, activations={assoc.id: torch.ones(8)}, config=cfg
        )
        # Without clipping, change would be 100 * base_lr * youth_factor.
        # With clipping, gradient L2 norm <= 0.5, so per-param change is
        # bounded.
        for p in assoc.parameters():
            delta = (p.detach() - before[id(p)]).abs().max().item()
            assert delta < 1.0, f"clipped update should be small, got {delta}"

    def test_clipping_handles_none_grads(self, config: SOMAConfig) -> None:
        """Params with grad=None must not cause clip_grad_norm_ to fail."""
        graph, _, _, _ = _three_node_graph(config)
        loss = torch.zeros((), requires_grad=True)
        # No backward call; all grads are None. Should not raise.
        update_step(graph, loss, activations={}, config=config)
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_core/test_learning.py::TestGradientClipping -v`
Expected: FAIL on `test_huge_gradient_gets_clipped` (delta will be ~30 without clipping)

**Step 3: Add clipping to update_step**

In `src/soma/core/learning.py`, modify `update_step` (around line 64):

```python
    # 1. Backprop through the active subgraph.
    loss.backward()  # type: ignore[no-untyped-call]

    # 1b. Global gradient clipping. Bounds the L2 norm of all node + edge
    # gradients so a runaway batch can't drive weights to the clamp in
    # one step. Param groups with grad=None are filtered automatically.
    all_params = [
        param
        for node in graph.all_nodes()
        for param in node.parameters()
    ] + [
        param for edge in graph.all_edges() for param in edge.parameters()
    ]
    if all_params:
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=config.grad_clip_max_norm)
```

**Step 4: Verify tests pass**

Run: `pytest tests/test_core/test_learning.py -v`
Expected: PASS (all)

**Step 5: Commit**

```bash
git add src/soma/core/learning.py tests/test_core/test_learning.py
git commit -m "fix(core): add gradient clipping in update_step to bound backprop updates"
```

---

## Task 4: Replace homeostasis raise with graceful return (server-side)

**Files:**
- Modify: `src/soma/metacognition/homeostasis.py:73-108` (the `update` method)
- Modify: `tests/test_metacognition/test_homeostasis.py` (update existing test)

**Step 1: Update the existing failing-behavior test**

In `tests/test_metacognition/test_homeostasis.py`, REPLACE `test_nan_loss_rejected` with:

```python
def test_nan_loss_returns_none(self, empty_graph: Graph) -> None:
    """Non-finite loss returns None instead of raising; state unchanged."""
    reg = HomeostaticRegulator()
    initial_ema = reg.loss_ema
    initial_count = reg._update_count
    assert reg.update(empty_graph, current_loss=float("nan")) is None
    assert reg.update(empty_graph, current_loss=float("inf")) is None
    assert reg.update(empty_graph, current_loss=float("-inf")) is None
    assert reg.loss_ema == initial_ema, "EMA must not be corrupted by bad loss"
    assert reg._update_count == initial_count, "step count must not advance"


def test_finite_loss_after_nan_works_normally(self, empty_graph: Graph) -> None:
    reg = HomeostaticRegulator()
    reg.update(empty_graph, current_loss=float("nan"))  # ignored
    mult = reg.update(empty_graph, current_loss=0.5)
    assert mult is not None
    assert 0.0 < mult <= 1.0
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_metacognition/test_homeostasis.py -v`
Expected: FAIL — `test_nan_loss_returns_none` fails because update still raises.

**Step 3: Modify homeostasis.update**

In `src/soma/metacognition/homeostasis.py`, change the signature and replace the raise:

```python
    def update(self, graph: Graph, current_loss: float) -> float | None:
        """Update the internal state for this step; return the LR multiplier.

        ``current_loss`` should be a finite non-negative scalar (MSE,
        cross-entropy, etc.). If a non-finite value is passed, returns
        ``None`` and leaves all state unchanged so SOMA.step can skip
        the learning step gracefully.
        """
        if not math.isfinite(current_loss):
            return None
        # ... existing body unchanged ...
```

(Keep the existing EMA / spike / gating logic verbatim; only the raise is replaced.)

**Step 4: Verify tests pass**

Run: `pytest tests/test_metacognition/test_homeostasis.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/soma/metacognition/homeostasis.py tests/test_metacognition/test_homeostasis.py
git commit -m "refactor(homeostasis): return None on non-finite loss instead of raising"
```

---

## Task 5: SOMA.step skips on None lr_multiplier + escalates after N skips

**Files:**
- Modify: `src/soma/system.py:180-234` (the `step` method) and add a counter attribute
- Test: `tests/test_system/test_soma.py` (add new test class)

**Step 1: Write failing tests**

Add to `tests/test_system/test_soma.py`:

```python
import math
from unittest.mock import patch

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.system import SOMA


class TestNonFiniteLossSkip:
    """SOMA.step gracefully skips when loss is inf/nan; raises after N skips."""

    def _soma_with_target(self, **cfg_overrides) -> tuple[SOMA, dict, dict]:
        cfg = SOMAConfig(seed=42, max_consecutive_skipped_steps=3, **cfg_overrides)
        soma = SOMA(cfg, device="cpu")
        # Fabricate sensor input + target.
        sensor_id = next(n.id for n in soma.graph.all_nodes() if n.node_type.value == "SENSOR")
        output_id = next(n.id for n in soma.graph.all_nodes() if n.node_type.value == "OUTPUT")
        inputs = {sensor_id: torch.randn(soma.config.sensor_output_dim)}
        targets = {output_id: torch.randn(soma.config.sensor_output_dim)}
        return soma, inputs, targets

    def test_inf_loss_returns_skipped_marker(self) -> None:
        soma, inputs, targets = self._soma_with_target()
        with patch.object(SOMA, "_compute_loss",
                          return_value=(torch.tensor(float("inf")), float("inf"))):
            result = soma.step(inputs=inputs, targets=targets)
        assert result.get("skipped") is True
        assert soma.global_step == 1, "global_step must still advance"

    def test_skipped_step_does_not_corrupt_homeostasis(self) -> None:
        soma, inputs, targets = self._soma_with_target()
        initial_ema = soma.homeostasis.loss_ema
        with patch.object(SOMA, "_compute_loss",
                          return_value=(torch.tensor(float("nan")), float("nan"))):
            soma.step(inputs=inputs, targets=targets)
        assert soma.homeostasis.loss_ema == initial_ema

    def test_too_many_consecutive_skips_raises(self) -> None:
        soma, inputs, targets = self._soma_with_target()
        with patch.object(SOMA, "_compute_loss",
                          return_value=(torch.tensor(float("inf")), float("inf"))):
            for _ in range(soma.config.max_consecutive_skipped_steps):
                soma.step(inputs=inputs, targets=targets)
            with pytest.raises(ValueError, match="consecutive non-finite"):
                soma.step(inputs=inputs, targets=targets)

    def test_skip_counter_resets_after_finite_step(self) -> None:
        soma, inputs, targets = self._soma_with_target()
        with patch.object(SOMA, "_compute_loss",
                          return_value=(torch.tensor(float("inf")), float("inf"))):
            for _ in range(soma.config.max_consecutive_skipped_steps):
                soma.step(inputs=inputs, targets=targets)
        # One finite step should reset the counter.
        soma.step(inputs=inputs, targets=targets)
        # Now M more inf steps should be tolerated again.
        with patch.object(SOMA, "_compute_loss",
                          return_value=(torch.tensor(float("inf")), float("inf"))):
            for _ in range(soma.config.max_consecutive_skipped_steps):
                soma.step(inputs=inputs, targets=targets)
            with pytest.raises(ValueError, match="consecutive non-finite"):
                soma.step(inputs=inputs, targets=targets)
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_system/test_soma.py::TestNonFiniteLossSkip -v`
Expected: FAIL — currently homeostasis.update raises (or, with Task 4 already in, returns None which SOMA.step doesn't handle yet).

**Step 3: Modify SOMA.step**

In `src/soma/system.py`:

(a) Add the counter to `__init__` (find the existing __init__ and append after `self.last_curiosity = 0.0` or similar):

```python
        self._consecutive_skipped_steps: int = 0
```

(b) Add the import at the top if not already present:

```python
import math
```

(c) Replace the `step` method body (around lines 180-234). Keep the existing structure; insert the skip check between `_compute_loss` and the learning block:

```python
    def step(
        self,
        inputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
        *,
        rng: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Run one interaction step; returns a dict with outputs/loss/etc."""
        outputs, activations = execute_graph(
            self.graph, inputs=inputs, current_step=self.global_step
        )

        context_tensor = next(iter(outputs.values())) if outputs else None
        self._update_working_memory(context_tensor)

        loss: torch.Tensor | None = None
        loss_value: float | None = None
        if targets is not None:
            loss, loss_value = self._compute_loss(outputs, targets)

        # If the loss is non-finite, skip the entire learning/growth/curiosity
        # pipeline for this step. Bumps a counter; raises if the counter
        # exceeds `max_consecutive_skipped_steps` so train_service's
        # CrashBackoff can detect a stuck training state.
        if loss_value is not None and not math.isfinite(loss_value):
            self._consecutive_skipped_steps += 1
            if self._consecutive_skipped_steps > self.config.max_consecutive_skipped_steps:
                raise ValueError(
                    f"{self._consecutive_skipped_steps} consecutive non-finite "
                    f"losses; training is stuck (last loss={loss_value!r})"
                )
            result = {
                "outputs": outputs,
                "loss": loss_value,
                "curiosity": self.last_curiosity,
                "lr_multiplier": 1.0,
                "global_step": self.global_step,
                "num_nodes": self.graph.num_nodes,
                "num_edges": self.graph.num_edges,
                "experience_idx": None,
                "skipped": True,
            }
            self.global_step += 1
            return result

        experience = self._encode_episodic(inputs, outputs, targets, loss_value)

        lr_multiplier = 1.0
        if loss is not None and loss_value is not None:
            mult = self.homeostasis.update(self.graph, current_loss=loss_value)
            if mult is None:
                # Defensive: shouldn't happen because we already checked finite
                # above, but if homeostasis adds future rejection criteria, fall
                # through to skip cleanly.
                self._consecutive_skipped_steps += 1
                self.global_step += 1
                return {
                    "outputs": outputs,
                    "loss": loss_value,
                    "curiosity": self.last_curiosity,
                    "lr_multiplier": 1.0,
                    "global_step": self.global_step - 1,
                    "num_nodes": self.graph.num_nodes,
                    "num_edges": self.graph.num_edges,
                    "experience_idx": experience,
                    "skipped": True,
                }
            lr_multiplier = mult
            update_step(
                self.graph,
                loss,
                activations,
                self.config,
                lr_multiplier=lr_multiplier,
            )

        # Successful learning step — reset the skip counter.
        self._consecutive_skipped_steps = 0

        self.last_curiosity = self._update_curiosity(inputs, loss_value)

        self._maybe_grow(activations, loss_value, rng=rng)
        self._maybe_consolidate(rng=rng)

        if loss_value is not None:
            self._recent_errors.append(loss_value)
            if len(self._recent_errors) > self._recent_errors_cap:
                self._recent_errors = self._recent_errors[-self._recent_errors_cap :]

        result = {
            "outputs": outputs,
            "loss": loss_value,
            "curiosity": self.last_curiosity,
            "lr_multiplier": lr_multiplier,
            "global_step": self.global_step,
            "num_nodes": self.graph.num_nodes,
            "num_edges": self.graph.num_edges,
            "experience_idx": experience,
        }
        self.global_step += 1
        return result
```

**Step 4: Verify tests pass**

Run: `pytest tests/test_system/test_soma.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/soma/system.py tests/test_system/test_soma.py
git commit -m "feat(system): SOMA.step skips on non-finite loss + escalates after N skips"
```

---

## Task 6: Full test suite + lint check

**Files:**
- None modified directly; this is a verification task.

**Step 1: Run the full test suite**

Run: `pytest tests/ -q --tb=short`
Expected: PASS for everything that was passing before. 

**Sanity check considerations:**
- Integration tests in `tests/test_integration/` that train SOMA for many steps may now have different observable behavior because edge_weight_decay changes equilibrium values. If any of them assert specific weight magnitudes, they may need adjustment.
- The existing `test_overactive_node_reduces_gain` style tests in `test_learning.py` should still pass because they check directional movement, not absolute values.

**Step 2: If any test fails:**
- Read the failure carefully.
- If it's an outdated assertion about weight magnitude (e.g., "weight should be exactly X after Y steps"), update the assertion to use a range that matches the new equilibrium.
- If it's a behavioral test that legitimately fails because skip-on-inf changes things — fix forward, don't suppress.

**Step 3: Run lint**

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: PASS

**Step 4: Run mypy**

Run: `mypy src/soma/`
Expected: PASS (homeostasis.update return type changed from `float` to `float | None`; SOMA.step's lr_multiplier handling needs to typecheck cleanly)

**Step 5: If any of steps 1-4 produce failures, commit the fix(es) separately.**

---

## Task 7: Re-run diagnostic on a fresh-init SOMA to confirm stability

**Files:**
- Use existing `scripts/diagnose_nan.py` and add a small extension to forward-pass-train it for N steps.

**Step 1: Add a fresh-train sanity to diagnose_nan.py**

Append at the bottom of `scripts/diagnose_nan.py` (above `if __name__`):

```python
def fresh_train_sanity(n_steps: int = 200, device: str = "cpu") -> None:
    """Train a fresh SOMA for N steps and confirm no clamp saturation."""
    from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
    print(f"\n{'='*70}\nFRESH TRAIN ({n_steps} steps, device={device})\n{'='*70}")
    cfg = SOMAConfig(seed=42)
    soma = SOMA(cfg, device=device)
    corpus = ["the quick brown fox", "jumps over the lazy dog"] * 50
    tokenizer = train_bpe_tokenizer(corpus, vocab_size=cfg.vocab_size)
    encoder = TextEncoder(tokenizer, embed_dim=cfg.text_embed_dim, device=device)
    sensor_id = next(n.id for n in soma.graph.all_nodes() if n.node_type.value == "SENSOR")
    output_id = next(n.id for n in soma.graph.all_nodes() if n.node_type.value == "OUTPUT")
    skipped = 0
    for step in range(n_steps):
        text = corpus[step % len(corpus)]
        embeds = encoder.encode(text)  # list[Tensor], one per token
        if len(embeds) < 2:
            continue
        result = soma.step(
            inputs={sensor_id: embeds[0]},
            targets={output_id: embeds[1]},
        )
        if result.get("skipped"):
            skipped += 1
    print(f"completed {n_steps} steps; skipped={skipped}")
    edge_weights = torch.stack([e.weight.detach().flatten() for e in soma.graph.all_edges()])
    n_at_clamp = int((edge_weights.abs() >= cfg.max_edge_weight - 0.01).sum().item())
    print(f"edges at weight-clamp boundary: {n_at_clamp}/{soma.graph.num_edges} "
          f"({100*n_at_clamp/max(soma.graph.num_edges,1):.1f}%)")
    activations = torch.tensor([float(n.activation_ema) for n in soma.graph.all_nodes()])
    print(f"node activation_ema: max={float(activations.abs().max()):.3e}  "
          f"non_finite={int((~torch.isfinite(activations)).sum())}")
    assert n_at_clamp / max(soma.graph.num_edges, 1) < 0.5, \
        "more than half of edges saturated within 200 steps — fixes failed"
    assert torch.isfinite(activations).all(), \
        "non-finite activations within 200 steps — fixes failed"
    print("✓ stability sanity passed")
```

Modify `main()` to call it:
```python
def main() -> None:
    ckpt_dir = Path("checkpoints")
    corrupt = ckpt_dir / "current.corrupt_1776082129.pt.bak"
    if corrupt.exists():
        inspect_state(corrupt, "CORRUPT (was step ~53740)")
    fresh_train_sanity(n_steps=200)
```

**Step 2: Run it**

Run: `python scripts/diagnose_nan.py`
Expected: corrupt-state snapshot AND `✓ stability sanity passed`

**Step 3: If sanity fails, iterate.**

- If activations still go non-finite within 200 steps: the fixes are insufficient. Likely needs tighter `grad_clip_max_norm` (try 0.5) or stronger `edge_weight_decay` (try 0.999). Tune via config, NOT by re-touching the carveout code.
- If skipped count is high (>20%): training is in a degenerate regime even with fixes. Worth a look but not a blocker — the goal is to stop crashing, not to converge.

**Step 4: Commit the diagnostic extension**

```bash
git add scripts/diagnose_nan.py
git commit -m "chore(diagnose): fresh-train sanity check for stability fixes"
```

---

## Task 8: Archive corrupt checkpoints, reset training state

**Files:**
- Move: all `checkpoints/step_*.pt`, `checkpoints/current.*` (except keep `current.corrupt_*.bak` as evidence)
- Reset: `.soma-loop/state/train_heartbeat.json`, `.soma-loop/state/train_permanent_failure.json`

**Step 1: Move existing checkpoints to an archive**

```bash
mkdir -p checkpoints/pre-fix-archive-2026-04-13
mv checkpoints/step_*.pt checkpoints/pre-fix-archive-2026-04-13/
mv checkpoints/current.pt checkpoints/current.txt checkpoints/current.encoder.pt \
   checkpoints/current.pre-fresh.pt checkpoints/current.pre-fresh.txt \
   checkpoints/pre-fix-archive-2026-04-13/ 2>/dev/null || true
ls checkpoints/
```

(Note: `current.corrupt_1776082129.pt.bak` is preserved at the top level as evidence for the diagnostic; leave it alone.)

**Step 2: Clear stale state files**

```bash
rm -f .soma-loop/state/train_permanent_failure.json
rm -f .soma-loop/state/train_crash.json
rm -f .soma-loop/state/train_heartbeat.json
ls .soma-loop/state/
```

(Keep `consecutive_failures.json`, `baseline.json`, `last_tick.json`, etc.)

**Step 3: This step has no test step — verification is via Task 9 startup.**

---

## Task 9: Clear STOP, restart train_service from fresh

**Files:**
- Remove: `.soma-loop/STOP`
- Spawn: `scripts/train_service.py` (will fresh-init since current.pt is gone)

**Step 1: Remove STOP**

```bash
rm .soma-loop/STOP
ls .soma-loop/STOP 2>&1
```

Expected: `No such file or directory`

**Step 2: Start train_service**

```bash
cd /e/Documents/Projects/SOMA
python -u scripts/train_service.py > .soma-loop/logs/train_service_post_fix_$(date +%s).log 2>&1 &
```

**Step 3: Verify it's running and advancing**

After a few seconds (use ScheduleWakeup if needed for longer):

```bash
cat .soma-loop/state/train_heartbeat.json
tail -10 .soma-loop/logs/train_service_post_fix_*.log
```

Expected: status=running, step advancing past 1, no `step crashed` lines.

**Step 4: Watch for the first checkpoint write (step 5000 default)**

Use ScheduleWakeup with delaySeconds=300-1200 (5-20 min) to come back and verify the service is still alive and has written a fresh checkpoint.

**Step 5: If the service crashes within 100 steps, ROLLBACK:**

```bash
git revert HEAD~4..HEAD --no-commit  # revert all 4 fix commits
git commit -m "revert: training stability fixes — broke fresh init"
touch .soma-loop/STOP
# Document the failure mode in reports/tick-summaries.md as a fresh BLOCKED entry.
```

**Step 6: If the service runs for ≥1000 steps without skips, mark the fix successful**

Document in `reports/tick-summaries.md` (append a new section):

```markdown
## 2026-04-13 [HH:MM] EDT — training stability fixes shipped

Three commits land:
- `<sha1>` feat(config): edge_weight_decay, grad_clip_max_norm, max_consecutive_skipped_steps
- `<sha2>` fix(core): edge weight decay
- `<sha3>` fix(core): gradient clipping
- `<sha4>` refactor(homeostasis): None instead of raise
- `<sha5>` feat(system): SOMA.step skip + escalation

Pre-fix corrupt checkpoint preserved at `checkpoints/current.corrupt_1776082129.pt.bak`
and rotated step files at `checkpoints/pre-fix-archive-2026-04-13/`.
Fresh init confirmed: 1000 steps clean, 0 skips, edge weights well below clamp.
STOP cleared. Loop tick scheduler will resume on next 30-min fire.
```

Commit:

```bash
git add reports/tick-summaries.md
git commit -m "docs(loop): training stability fixes shipped, loop resumed"
```

---

## Task 10: Push (best-effort) and resume monitoring

**Step 1: Try to push**

```bash
git push origin main 2>&1 | tail -10
```

If it fails with `fatal: could not read Username` (no tty), document with operator on wake. Commits are local but visible.

**Step 2: Schedule a wake-up to monitor**

Use ScheduleWakeup with delaySeconds=1800 (30 min) to come back, check heartbeat, check the next scheduled tick fired and ran cleanly. If it did, mark Task 26 complete.

---

## Risk register

| Risk | Mitigation |
|------|-----------|
| Edge decay too aggressive — useful weights also decay | Default 0.9999 chosen so equilibrium ≈ s*t for co-active edge; configurable. Task 7 fresh-train sanity catches this. |
| Gradient clipping too tight — training stagnates | Default `max_norm=1.0` is standard. Configurable. Sanity test catches non-progression. |
| Skip counter masks real issues | Counter raises after 50 skips → train_service crashes → CrashBackoff → permanent_failure → watchdog notices. Same escalation path as before, just with more grace. |
| Existing integration tests fail because of changed equilibrium | Task 6 explicitly handles. Integration tests check directional behavior, not absolute weight values. |
| Fresh init still has the underlying bug we didn't see | If Task 7 sanity fails, that's signal to dig deeper before shipping. Plan accommodates iteration. |
| Push failure leaves operator in the dark | Documented in reports/tick-summaries.md; visible on wake. |
