# Developmental SOMA PoC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a "virtual baby" that develops intelligence through text interaction — SOMA is the brain (learns, adapts, grows structurally), a small LLM is the mouth (translates SOMA's state to language).

**Architecture:** SOMA processes text input, computes prediction error (predicted vs actual node activations), uses error to drive Hebbian learning + neurogenesis + pruning. A `verbalize_state()` function serializes SOMA's internal state (top activations, working memory, curiosity, associations) into structured text. A weak LLM (1-3B params via Ollama) receives ONLY this state packet and generates responses. The LLM never sees raw input — it is dependent on SOMA's development.

**Tech Stack:** PyTorch (existing SOMA), Ollama (local LLM), sentence-transformers (embedding), Python 3.11+

---

### Task 1: Create developmental module scaffold + verbalize_state()

**Files:**
- Create: `src/soma/developmental/__init__.py`
- Create: `src/soma/developmental/verbalize.py`
- Test: `tests/test_developmental/test_verbalize.py`

**Step 1: Write the failing test**

```python
# tests/test_developmental/__init__.py
# (empty)

# tests/test_developmental/test_verbalize.py
"""Tests for SOMA state verbalization."""
from __future__ import annotations

import torch

from soma.core.config import SOMAConfig
from soma.developmental.verbalize import verbalize_state
from soma.system import SOMA


def _make_soma(device: str = "cpu") -> SOMA:
    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        seed=42,
    )
    return SOMA(config, device=torch.device(device))


class TestVerbalizeState:
    def test_returns_string(self) -> None:
        soma = _make_soma()
        result = verbalize_state(soma)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_developmental_stage(self) -> None:
        soma = _make_soma()
        result = verbalize_state(soma)
        assert "Developmental Stage" in result

    def test_contains_graph_stats(self) -> None:
        soma = _make_soma()
        result = verbalize_state(soma)
        assert "nodes" in result.lower()
        assert "edges" in result.lower()

    def test_contains_novelty(self) -> None:
        soma = _make_soma()
        result = verbalize_state(soma)
        assert "Novelty" in result or "novelty" in result

    def test_after_step_has_activations(self) -> None:
        soma = _make_soma()
        # Run a step to populate activations
        inputs = {"text": torch.randn(64)}
        soma.step(inputs, eval_mode=True)
        result = verbalize_state(soma)
        assert "Active Nodes" in result or "active" in result.lower()

    def test_working_memory_section(self) -> None:
        soma = _make_soma()
        inputs = {"text": torch.randn(64)}
        soma.step(inputs, eval_mode=True)
        result = verbalize_state(soma)
        assert "Working Memory" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_developmental/test_verbalize.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'soma.developmental'"

**Step 3: Write minimal implementation**

```python
# src/soma/developmental/__init__.py
"""Developmental SOMA — brain-inspired AI that grows through interaction."""

# src/soma/developmental/verbalize.py
"""Serialize SOMA's internal state to structured text for LLM consumption."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from soma.core.node import NodeType

if TYPE_CHECKING:
    from soma.system import SOMA


def _developmental_stage(step: int) -> str:
    """Map global step to human-readable developmental stage."""
    if step < 100:
        return "blank-slate"
    if step < 500:
        return "early-plasticity"
    if step < 2000:
        return "pattern-recognition"
    if step < 5000:
        return "association-formation"
    return "mature"


def _format_active_nodes(soma: SOMA, top_k: int = 5) -> str:
    """Format the most active nodes with their activation levels."""
    nodes = soma.graph.most_active_nodes(top_k)
    if not nodes:
        return "  (no activations yet)"
    lines = []
    for i, node in enumerate(nodes, 1):
        lines.append(
            f"  {i}. {node.node_type.value} node "
            f"(ema={node.activation_ema:.3f}, "
            f"maturity={node.maturity:.2f})"
        )
    return "\n".join(lines)


def _format_working_memory(soma: SOMA) -> str:
    """Format working memory slot usage."""
    wm = soma.working_memory
    used = int((wm.usage > 0.1).sum().item())
    total = wm.usage.shape[0]
    if used == 0:
        return f"  {used}/{total} slots occupied"
    # Show top slots by usage
    top_indices = torch.argsort(wm.usage, descending=True)[:min(3, used)]
    lines = [f"  {used}/{total} slots occupied"]
    for idx in top_indices:
        usage_val = wm.usage[idx].item()
        age_val = wm.age[idx].item()
        if usage_val > 0.1:
            lines.append(
                f"  - slot {idx.item()}: usage={usage_val:.2f}, "
                f"age={int(age_val)} steps"
            )
    return "\n".join(lines)


def _format_graph_stats(soma: SOMA) -> str:
    """Format graph structure summary."""
    graph = soma.graph
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    by_type = {}
    for nt in NodeType:
        count = len(graph.nodes_by_type(nt))
        if count > 0:
            by_type[nt.value] = count
    type_str = ", ".join(f"{v}={c}" for v, c in sorted(by_type.items()))
    return f"  {n_nodes} nodes ({type_str}), {n_edges} edges"


def _format_growth_log(soma: SOMA, recent: int = 5) -> str:
    """Format recent structural changes."""
    if not soma.growth_log:
        return "  (no structural changes yet)"
    entries = list(soma.growth_log)[-recent:]
    lines = []
    for entry in entries:
        event = entry.get("event", "unknown")
        step = entry.get("step", "?")
        lines.append(f"  - step {step}: {event}")
    return "\n".join(lines)


def verbalize_state(soma: SOMA) -> str:
    """Serialize SOMA's internal state to structured text.

    This output is designed to be consumed by an LLM as a context
    block, giving it a window into SOMA's developmental state.
    """
    step = soma.global_step
    stage = _developmental_stage(step)
    novelty = soma.last_curiosity

    # Homeostatic state
    lr_mult = soma.homeostasis.global_lr_multiplier
    loss_ema = soma.homeostasis.loss_ema

    sections = [
        "[SOMA Internal State]",
        f"Developmental Stage: {stage} (step {step})",
        f"Novelty Score: {novelty:.3f}",
        f"Stability: lr_mult={lr_mult:.3f}, loss_ema={loss_ema:.4f}",
        "",
        "Graph Structure:",
        _format_graph_stats(soma),
        "",
        "Active Nodes (top 5):",
        _format_active_nodes(soma),
        "",
        "Working Memory:",
        _format_working_memory(soma),
        "",
        "Recent Growth Events:",
        _format_growth_log(soma),
    ]

    # Episodic memory stats
    em = soma.episodic_memory
    n_valid = int(em.valid.sum().item())
    sections.extend([
        "",
        f"Episodic Memory: {n_valid}/{em.capacity} experiences stored",
    ])

    return "\n".join(sections)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_developmental/test_verbalize.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add src/soma/developmental/__init__.py src/soma/developmental/verbalize.py tests/test_developmental/__init__.py tests/test_developmental/test_verbalize.py
git commit -m "feat(developmental): add verbalize_state() for SOMA internal state serialization"
```

---

### Task 2: Prediction loop — predict next activation, compute error

**Files:**
- Create: `src/soma/developmental/prediction.py`
- Test: `tests/test_developmental/test_prediction.py`

**Step 1: Write the failing test**

```python
# tests/test_developmental/test_prediction.py
"""Tests for next-activation prediction loop."""
from __future__ import annotations

import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.system import SOMA


def _make_config() -> SOMAConfig:
    return SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        seed=42,
    )


class TestPredictiveSOMA:
    def test_init(self) -> None:
        config = _make_config()
        ps = PredictiveSOMA(config, device=torch.device("cpu"))
        assert ps.soma is not None
        assert ps.prediction_error == 0.0

    def test_process_returns_result(self) -> None:
        config = _make_config()
        ps = PredictiveSOMA(config, device=torch.device("cpu"))
        result = ps.process_input(torch.randn(64))
        assert "activations" in result
        assert "prediction_error" in result
        assert "novelty" in result
        assert isinstance(result["prediction_error"], float)

    def test_prediction_error_changes_after_inputs(self) -> None:
        config = _make_config()
        ps = PredictiveSOMA(config, device=torch.device("cpu"))
        # First input — no prior prediction, error should be 0
        r1 = ps.process_input(torch.randn(64))
        # Second input — now there IS a prior prediction
        r2 = ps.process_input(torch.randn(64))
        # After second input, prediction error should be non-zero
        # (predicted activation from step 1 vs actual from step 2)
        assert r2["prediction_error"] >= 0.0

    def test_cumulative_error_tracked(self) -> None:
        config = _make_config()
        ps = PredictiveSOMA(config, device=torch.device("cpu"))
        ps.process_input(torch.randn(64))
        ps.process_input(torch.randn(64))
        ps.process_input(torch.randn(64))
        assert len(ps.error_history) == 3

    def test_graph_grows_after_many_inputs(self) -> None:
        config = _make_config()
        # Lower thresholds for test
        config.synaptogenesis_interval = 5
        config.neurogenesis_interval = 10
        ps = PredictiveSOMA(config, device=torch.device("cpu"))
        initial_edges = len(ps.soma.graph.edges)
        for _ in range(20):
            ps.process_input(torch.randn(64))
        # Graph should have grown (synaptogenesis at least)
        final_edges = len(ps.soma.graph.edges)
        assert final_edges >= initial_edges
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_developmental/test_prediction.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'soma.developmental.prediction'"

**Step 3: Write minimal implementation**

```python
# src/soma/developmental/prediction.py
"""Predictive processing loop for developmental SOMA.

Core idea: SOMA predicts the next input's activation pattern.
Prediction error drives Hebbian learning, neurogenesis, pruning,
and curiosity — unifying all brain-inspired mechanisms under one
signal (Free Energy Principle).
"""
from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn as nn

from soma.core.config import SOMAConfig
from soma.core.execution import execute_graph
from soma.system import SOMA


class PredictiveSOMA(nn.Module):
    """Wraps SOMA with a next-activation prediction loop.

    Each input:
    1. Compare predicted activation (from last step) with actual
    2. Compute prediction error
    3. Update SOMA via step() (Hebbian learning, growth, etc.)
    4. Generate prediction for next input
    """

    def __init__(
        self,
        config: SOMAConfig,
        *,
        device: torch.device | None = None,
        error_history_size: int = 1000,
    ) -> None:
        super().__init__()
        self.soma = SOMA(config, device=device)
        self.config = config
        self.device = device or torch.device("cpu")

        # Prediction head: linear projection from activation space
        # to predict next activation pattern
        output_dim = config.sensor_output_dim
        self.prediction_head = nn.Linear(output_dim, output_dim).to(self.device)

        # State
        self._last_prediction: torch.Tensor | None = None
        self._last_activations: dict[str, torch.Tensor] = {}
        self.prediction_error: float = 0.0
        self.error_history: deque[float] = deque(maxlen=error_history_size)

    def _aggregate_activations(
        self, activations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reduce per-node activations to a single summary vector."""
        if not activations:
            return torch.zeros(
                self.config.sensor_output_dim, device=self.device,
            )
        # Mean of all node activations (resized to sensor_output_dim)
        vecs = []
        target_dim = self.config.sensor_output_dim
        for act in activations.values():
            if act.shape[-1] != target_dim:
                # Simple truncate or pad
                if act.shape[-1] > target_dim:
                    act = act[..., :target_dim]
                else:
                    pad = torch.zeros(
                        target_dim - act.shape[-1], device=act.device,
                    )
                    act = torch.cat([act, pad], dim=-1)
            vecs.append(act.detach())
        return torch.stack(vecs).mean(dim=0)

    def process_input(
        self,
        input_tensor: torch.Tensor,
        *,
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Process one input through the prediction loop.

        Parameters
        ----------
        input_tensor:
            Encoded input (shape: sensor_output_dim).
        targets:
            Optional targets for supervised learning signal.

        Returns
        -------
        dict with keys: activations, prediction_error, novelty,
            outputs, global_step
        """
        input_tensor = input_tensor.to(self.device)
        inputs = {self.config.input_modalities[0]: input_tensor}

        # Step SOMA (execute graph, learn, grow, consolidate)
        step_result = self.soma.step(inputs, targets=targets)

        # Get current activation summary
        # Use output activations as the summary
        outputs = step_result.get("outputs", {})
        if outputs:
            first_key = next(iter(outputs))
            current_summary = outputs[first_key].detach()
            # Resize to sensor_output_dim if needed
            target_dim = self.config.sensor_output_dim
            if current_summary.shape[-1] != target_dim:
                if current_summary.shape[-1] > target_dim:
                    current_summary = current_summary[..., :target_dim]
                else:
                    pad = torch.zeros(
                        target_dim - current_summary.shape[-1],
                        device=current_summary.device,
                    )
                    current_summary = torch.cat([current_summary, pad], dim=-1)
        else:
            current_summary = torch.zeros(
                self.config.sensor_output_dim, device=self.device,
            )

        # Compute prediction error
        if self._last_prediction is not None:
            error = torch.nn.functional.mse_loss(
                self._last_prediction, current_summary,
            )
            self.prediction_error = error.item()
        else:
            self.prediction_error = 0.0

        self.error_history.append(self.prediction_error)

        # Generate prediction for next input
        self._last_prediction = self.prediction_head(
            current_summary.detach(),
        )

        return {
            "activations": step_result.get("outputs", {}),
            "prediction_error": self.prediction_error,
            "novelty": step_result.get("curiosity", 0.0),
            "outputs": outputs,
            "global_step": step_result.get("global_step", 0),
            "num_nodes": step_result.get("num_nodes", 0),
            "num_edges": step_result.get("num_edges", 0),
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_developmental/test_prediction.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/soma/developmental/prediction.py tests/test_developmental/test_prediction.py
git commit -m "feat(developmental): add PredictiveSOMA with next-activation prediction loop"
```

---

### Task 3: DevelopmentTracker — log metrics over time

**Files:**
- Create: `src/soma/developmental/tracker.py`
- Test: `tests/test_developmental/test_tracker.py`

**Step 1: Write the failing test**

```python
# tests/test_developmental/test_tracker.py
"""Tests for development metric tracking."""
from __future__ import annotations

from soma.developmental.tracker import DevelopmentTracker


class TestDevelopmentTracker:
    def test_init(self) -> None:
        tracker = DevelopmentTracker()
        assert tracker.step_count == 0

    def test_record_step(self) -> None:
        tracker = DevelopmentTracker()
        tracker.record(
            step=0,
            prediction_error=0.5,
            novelty=0.8,
            num_nodes=20,
            num_edges=40,
            wm_occupancy=0,
            episodic_count=0,
        )
        assert tracker.step_count == 1

    def test_history_accessible(self) -> None:
        tracker = DevelopmentTracker()
        for i in range(10):
            tracker.record(
                step=i,
                prediction_error=0.5 - i * 0.04,
                novelty=0.8 - i * 0.05,
                num_nodes=20 + i,
                num_edges=40 + i * 2,
                wm_occupancy=i,
                episodic_count=i,
            )
        assert tracker.step_count == 10
        assert len(tracker.history["prediction_error"]) == 10
        # Error should decrease over time
        assert tracker.history["prediction_error"][-1] < tracker.history["prediction_error"][0]

    def test_summary(self) -> None:
        tracker = DevelopmentTracker()
        for i in range(5):
            tracker.record(
                step=i,
                prediction_error=0.5,
                novelty=0.3,
                num_nodes=20,
                num_edges=40,
                wm_occupancy=5,
                episodic_count=i,
            )
        summary = tracker.summary()
        assert "prediction_error" in summary
        assert "num_nodes" in summary
        assert isinstance(summary["prediction_error"]["mean"], float)

    def test_save_load_json(self, tmp_path) -> None:
        tracker = DevelopmentTracker()
        for i in range(3):
            tracker.record(
                step=i, prediction_error=0.1 * i,
                novelty=0.5, num_nodes=20, num_edges=40,
                wm_occupancy=0, episodic_count=0,
            )
        path = tmp_path / "tracker.json"
        tracker.save(path)
        loaded = DevelopmentTracker.load(path)
        assert loaded.step_count == 3
        assert loaded.history["prediction_error"] == tracker.history["prediction_error"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_developmental/test_tracker.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'soma.developmental.tracker'"

**Step 3: Write minimal implementation**

```python
# src/soma/developmental/tracker.py
"""Track developmental metrics over time."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class DevelopmentTracker:
    """Records SOMA development metrics at each step.

    Tracks prediction error, novelty, graph size, memory usage,
    and other indicators of developmental progress.
    """

    METRICS = [
        "prediction_error",
        "novelty",
        "num_nodes",
        "num_edges",
        "wm_occupancy",
        "episodic_count",
    ]

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = defaultdict(list)
        self._steps: list[int] = []

    @property
    def step_count(self) -> int:
        return len(self._steps)

    def record(
        self,
        step: int,
        prediction_error: float,
        novelty: float,
        num_nodes: int,
        num_edges: int,
        wm_occupancy: int,
        episodic_count: int,
        **extra: float,
    ) -> None:
        """Record metrics for one step."""
        self._steps.append(step)
        self.history["prediction_error"].append(prediction_error)
        self.history["novelty"].append(novelty)
        self.history["num_nodes"].append(float(num_nodes))
        self.history["num_edges"].append(float(num_edges))
        self.history["wm_occupancy"].append(float(wm_occupancy))
        self.history["episodic_count"].append(float(episodic_count))
        for key, val in extra.items():
            self.history[key].append(float(val))

    def summary(self, last_n: int | None = None) -> dict[str, dict[str, float]]:
        """Compute summary statistics for tracked metrics."""
        result: dict[str, dict[str, float]] = {}
        for key, values in self.history.items():
            subset = values[-last_n:] if last_n else values
            if not subset:
                continue
            result[key] = {
                "mean": sum(subset) / len(subset),
                "min": min(subset),
                "max": max(subset),
                "last": subset[-1],
                "count": len(subset),
            }
        return result

    def save(self, path: str | Path) -> None:
        """Save tracker state to JSON."""
        data = {
            "steps": self._steps,
            "history": dict(self.history),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DevelopmentTracker:
        """Load tracker state from JSON."""
        data = json.loads(Path(path).read_text())
        tracker = cls()
        tracker._steps = data["steps"]
        for key, values in data["history"].items():
            tracker.history[key] = values
        return tracker
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_developmental/test_tracker.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/soma/developmental/tracker.py tests/test_developmental/test_tracker.py
git commit -m "feat(developmental): add DevelopmentTracker for metric history"
```

---

### Task 4: InteractionLoop — human <-> SOMA <-> LLM

**Files:**
- Create: `src/soma/developmental/interaction.py`
- Test: `tests/test_developmental/test_interaction.py`

**Step 1: Write the failing test**

```python
# tests/test_developmental/test_interaction.py
"""Tests for the interaction loop."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop


def _make_config() -> SOMAConfig:
    return SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        seed=42,
    )


class TestInteractionLoop:
    def test_init(self) -> None:
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        assert loop.predictive_soma is not None
        assert loop.tracker is not None

    def test_encode_text(self) -> None:
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        # Train tokenizer with some text
        loop.train_tokenizer(["hello world", "test input"])
        vec = loop.encode_text("hello world")
        assert isinstance(vec, torch.Tensor)
        assert vec.shape[-1] == 64  # sensor_output_dim

    def test_process_without_llm(self) -> None:
        """Test SOMA processing without calling the LLM."""
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        loop.train_tokenizer(["hello world", "how are you"])
        result = loop.process_input("hello world", call_llm=False)
        assert "soma_state" in result
        assert "prediction_error" in result
        assert result["response"] is None  # No LLM call

    @patch("soma.developmental.interaction.requests.post")
    def test_process_with_llm(self, mock_post) -> None:
        """Test full loop with mocked LLM."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "I'm learning about this."},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        loop.train_tokenizer(["hello world", "test"])
        result = loop.process_input("hello world", call_llm=True)
        assert result["response"] == "I'm learning about this."
        assert mock_post.called

    def test_tracker_records_steps(self) -> None:
        loop = InteractionLoop(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        loop.train_tokenizer(["one", "two", "three"])
        loop.process_input("one", call_llm=False)
        loop.process_input("two", call_llm=False)
        assert loop.tracker.step_count == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_developmental/test_interaction.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'soma.developmental.interaction'"

**Step 3: Write minimal implementation**

```python
# src/soma/developmental/interaction.py
"""Main interaction loop: human <-> SOMA <-> LLM."""
from __future__ import annotations

from typing import Any

import requests
import torch

from soma.core.config import SOMAConfig
from soma.developmental.prediction import PredictiveSOMA
from soma.developmental.tracker import DevelopmentTracker
from soma.developmental.verbalize import verbalize_state
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer

_SYSTEM_PROMPT = """\
You are the voice of a developing mind. Your responses must reflect \
ONLY the internal state provided below. If the state shows high \
novelty, express curiosity and uncertainty. If associations are \
strong, make connections. If working memory is sparse, be brief. \
You are not a knowledgeable assistant — you are a developing \
intelligence expressing what it currently understands and feels.

Do NOT answer from general knowledge. Only reflect what the \
internal state tells you."""


class InteractionLoop:
    """Orchestrates the human <-> SOMA <-> LLM interaction.

    Text input → encode → SOMA processes (predict, learn, grow) →
    verbalize state → LLM generates response → text output.
    """

    def __init__(
        self,
        config: SOMAConfig,
        llm_model: str,
        llm_api_base: str = "http://localhost:11434",
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.device = device or torch.device("cpu")

        self.predictive_soma = PredictiveSOMA(
            config, device=self.device,
        )
        self.tracker = DevelopmentTracker()
        self._encoder: TextEncoder | None = None

    def train_tokenizer(
        self,
        corpus: list[str],
        vocab_size: int | None = None,
    ) -> None:
        """Train BPE tokenizer from corpus text."""
        vs = vocab_size or self.config.vocab_size
        tokenizer = train_bpe_tokenizer(corpus, vocab_size=vs)
        self._encoder = TextEncoder(
            tokenizer,
            embed_dim=self.config.text_embed_dim,
            device=self.device,
        )

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to tensor via BPE + embedding."""
        if self._encoder is None:
            raise RuntimeError(
                "Tokenizer not trained. Call train_tokenizer() first."
            )
        # encode_batch returns (T, embed_dim), mean-pool to single vector
        tokens = self._encoder.encode_batch(text)
        return tokens.mean(dim=0)

    def _call_llm(self, soma_state: str) -> str:
        """Call the LLM with SOMA's verbalized state."""
        payload: dict[str, Any] = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": soma_state},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": 256, "temperature": 0.7},
        }
        resp = requests.post(
            f"{self.llm_api_base}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def process_input(
        self,
        text: str,
        *,
        call_llm: bool = True,
    ) -> dict[str, Any]:
        """Process one text input through the full loop.

        Parameters
        ----------
        text:
            Human input text.
        call_llm:
            If False, skip LLM call (useful for testing/warmup).

        Returns
        -------
        dict with: response, soma_state, prediction_error, novelty,
            global_step
        """
        # Encode text
        input_vec = self.encode_text(text)

        # Process through predictive SOMA
        result = self.predictive_soma.process_input(input_vec)

        # Verbalize state
        soma_state = verbalize_state(self.predictive_soma.soma)

        # Track metrics
        soma = self.predictive_soma.soma
        wm_occ = int((soma.working_memory.usage > 0.1).sum().item())
        ep_count = int(soma.episodic_memory.valid.sum().item())

        self.tracker.record(
            step=result["global_step"],
            prediction_error=result["prediction_error"],
            novelty=result["novelty"],
            num_nodes=result["num_nodes"],
            num_edges=result["num_edges"],
            wm_occupancy=wm_occ,
            episodic_count=ep_count,
        )

        # Call LLM
        response = None
        if call_llm:
            response = self._call_llm(soma_state)

        return {
            "response": response,
            "soma_state": soma_state,
            "prediction_error": result["prediction_error"],
            "novelty": result["novelty"],
            "global_step": result["global_step"],
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_developmental/test_interaction.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/soma/developmental/interaction.py tests/test_developmental/test_interaction.py
git commit -m "feat(developmental): add InteractionLoop for human-SOMA-LLM interaction"
```

---

### Task 5: Ablation harness — measure SOMA's contribution

**Files:**
- Create: `src/soma/developmental/ablation.py`
- Test: `tests/test_developmental/test_ablation.py`

**Step 1: Write the failing test**

```python
# tests/test_developmental/test_ablation.py
"""Tests for the ablation harness."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from soma.core.config import SOMAConfig
from soma.developmental.ablation import AblationHarness


def _make_config() -> SOMAConfig:
    return SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        seed=42,
    )


class TestAblationHarness:
    def test_init(self) -> None:
        harness = AblationHarness(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        assert harness.loop is not None

    @patch("soma.developmental.interaction.requests.post")
    def test_run_ablation(self, mock_post) -> None:
        mock_resp = MagicMock()
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_r = MagicMock()
            if call_count % 2 == 1:
                mock_r.json.return_value = {
                    "message": {"content": "With SOMA: I remember cooking."},
                }
            else:
                mock_r.json.return_value = {
                    "message": {"content": "Without SOMA: I don't know."},
                }
            mock_r.raise_for_status = MagicMock()
            return mock_r
        mock_post.side_effect = side_effect

        harness = AblationHarness(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        harness.loop.train_tokenizer(["hello", "world"])

        results = harness.run(
            inputs=["hello", "world"],
            references=["greeting", "planet"],
        )
        assert "with_soma" in results
        assert "without_soma" in results
        assert "delta" in results
        assert len(results["with_soma"]) == 2
        assert len(results["without_soma"]) == 2

    def test_empty_state_text(self) -> None:
        harness = AblationHarness(
            config=_make_config(),
            llm_model="test-model",
            llm_api_base="http://localhost:11434",
        )
        empty = harness.empty_state_text()
        assert "blank" in empty.lower() or "no" in empty.lower()
        assert isinstance(empty, str)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_developmental/test_ablation.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'soma.developmental.ablation'"

**Step 3: Write minimal implementation**

```python
# src/soma/developmental/ablation.py
"""Ablation harness — measure SOMA's developmental contribution.

Runs every test input twice:
  1. With SOMA state (LLM sees full verbalized state)
  2. Without SOMA state (LLM sees empty/default state)

The delta between conditions is the measured contribution of
SOMA's development.
"""
from __future__ import annotations

from typing import Any

import requests

from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop, _SYSTEM_PROMPT


class AblationHarness:
    """Compare LLM responses with and without SOMA state."""

    def __init__(
        self,
        config: SOMAConfig,
        llm_model: str,
        llm_api_base: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        self.loop = InteractionLoop(
            config=config,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            **kwargs,
        )
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base

    def empty_state_text(self) -> str:
        """Return a blank/default SOMA state for the ablation baseline."""
        return (
            "[SOMA Internal State]\n"
            "Developmental Stage: blank-slate (step 0)\n"
            "Novelty Score: 0.000\n"
            "Stability: lr_mult=1.000, loss_ema=0.0000\n"
            "\n"
            "Graph Structure:\n"
            "  (no data)\n"
            "\n"
            "Active Nodes (top 5):\n"
            "  (no activations yet)\n"
            "\n"
            "Working Memory:\n"
            "  0/32 slots occupied\n"
            "\n"
            "Recent Growth Events:\n"
            "  (no structural changes yet)\n"
            "\n"
            "Episodic Memory: 0/10000 experiences stored"
        )

    def _call_llm(self, state_text: str) -> str:
        """Call LLM with given state text."""
        payload: dict[str, Any] = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": state_text},
            ],
            "stream": False,
            "think": False,
            "options": {"num_predict": 256, "temperature": 0.0},
        }
        resp = requests.post(
            f"{self.llm_api_base}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def run(
        self,
        inputs: list[str],
        references: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run ablation: with SOMA vs without SOMA.

        Parameters
        ----------
        inputs:
            List of text inputs to process.
        references:
            Optional reference answers for scoring.

        Returns
        -------
        dict with: with_soma (list of responses), without_soma
            (list of responses), delta (dict of metrics)
        """
        with_soma: list[str] = []
        without_soma: list[str] = []
        soma_states: list[str] = []

        empty_state = self.empty_state_text()

        for text in inputs:
            # Process through SOMA (learning happens)
            result = self.loop.process_input(text, call_llm=False)
            state = result["soma_state"]
            soma_states.append(state)

            # With SOMA: LLM sees real state
            resp_with = self._call_llm(state)
            with_soma.append(resp_with)

            # Without SOMA: LLM sees empty state
            resp_without = self._call_llm(empty_state)
            without_soma.append(resp_without)

        # Compute delta metrics
        delta: dict[str, float] = {
            "avg_len_with": sum(len(r) for r in with_soma) / max(len(with_soma), 1),
            "avg_len_without": sum(len(r) for r in without_soma) / max(len(without_soma), 1),
        }

        # If references provided, compute token F1
        if references:
            from benchmarks.industry.longmemeval.metrics import token_f1

            f1_with = [
                token_f1(hyp, ref)
                for hyp, ref in zip(with_soma, references)
            ]
            f1_without = [
                token_f1(hyp, ref)
                for hyp, ref in zip(without_soma, references)
            ]
            delta["f1_with_soma"] = sum(f1_with) / len(f1_with)
            delta["f1_without_soma"] = sum(f1_without) / len(f1_without)
            delta["f1_delta"] = delta["f1_with_soma"] - delta["f1_without_soma"]

        return {
            "with_soma": with_soma,
            "without_soma": without_soma,
            "soma_states": soma_states,
            "delta": delta,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_developmental/test_ablation.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/soma/developmental/ablation.py tests/test_developmental/test_ablation.py
git commit -m "feat(developmental): add AblationHarness for measuring SOMA's contribution"
```

---

### Task 6: CLI entry point — interactive "talk to SOMA"

**Files:**
- Create: `src/soma/developmental/cli.py`
- Modify: (none — standalone script)

**Step 1: Write the CLI**

```python
# src/soma/developmental/cli.py
"""Interactive CLI for talking to the developing SOMA.

Usage::

    python -m soma.developmental.cli
    python -m soma.developmental.cli --model qwen2:1.5b-instruct-q8_0
    python -m soma.developmental.cli --load state.pt
"""
from __future__ import annotations

import argparse
import sys

import torch

from soma.core.config import SOMAConfig
from soma.developmental.interaction import InteractionLoop
from soma.developmental.verbalize import verbalize_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Talk to the developing SOMA",
    )
    parser.add_argument(
        "--model",
        default="qwen2:1.5b-instruct-q8_0",
        help="Ollama model name",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:11434",
        help="Ollama API base URL",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: auto)",
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Print SOMA state before each response",
    )
    parser.add_argument(
        "--consolidate-every",
        type=int,
        default=50,
        help="Run consolidation every N interactions",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    config = SOMAConfig(
        vocab_size=256,
        text_embed_dim=64,
        sensor_output_dim=64,
        max_input_tokens=128,
        initial_integrator_count=4,
        initial_associator_count=8,
        consolidation_interval=args.consolidate_every,
        seed=42,
    )

    loop = InteractionLoop(
        config=config,
        llm_model=args.model,
        llm_api_base=args.api_base,
        device=device,
    )

    # Warm up tokenizer with seed corpus
    seed_corpus = [
        "Hello, how are you?",
        "I'm interested in learning about the world.",
        "Tell me about yourself.",
        "What do you think about that?",
        "That's interesting, tell me more.",
    ]
    loop.train_tokenizer(seed_corpus)

    print()
    print("=== SOMA Developmental Interface ===")
    print("Type messages to interact. SOMA will develop through conversation.")
    print("Commands: /state (show internal state), /stats (show metrics), /quit")
    print()

    interaction_count = 0
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input == "/quit":
                break

            if user_input == "/state":
                print()
                print(verbalize_state(loop.predictive_soma.soma))
                print()
                continue

            if user_input == "/stats":
                summary = loop.tracker.summary()
                print()
                for metric, stats in summary.items():
                    print(
                        f"  {metric}: mean={stats['mean']:.4f}, "
                        f"last={stats['last']:.4f}"
                    )
                print()
                continue

            interaction_count += 1

            # Process input
            result = loop.process_input(user_input, call_llm=True)

            if args.show_state:
                print()
                print(result["soma_state"])
                print()

            # Show response
            print(f"SOMA: {result['response']}")

            # Show development info
            pe = result["prediction_error"]
            nov = result["novelty"]
            step = result["global_step"]
            print(
                f"  [step {step} | "
                f"pred_error={pe:.4f} | "
                f"novelty={nov:.4f}]"
            )
            print()

    except KeyboardInterrupt:
        print("\n")

    # Save tracker
    if interaction_count > 0:
        tracker_path = "soma_development_log.json"
        loop.tracker.save(tracker_path)
        print(f"Development log saved to {tracker_path}")
        print(f"Total interactions: {interaction_count}")

        summary = loop.tracker.summary()
        if "prediction_error" in summary:
            print(
                f"Final prediction error: "
                f"{summary['prediction_error']['last']:.4f} "
                f"(mean: {summary['prediction_error']['mean']:.4f})"
            )


if __name__ == "__main__":
    main()
```

**Step 2: Verify it runs (smoke test)**

Run: `python -c "from soma.developmental.cli import main; print('CLI importable')"`
Expected: "CLI importable"

**Step 3: Commit**

```bash
git add src/soma/developmental/cli.py
git commit -m "feat(developmental): add interactive CLI for talking to SOMA"
```

---

### Task 7: Run all tests + lint

**Step 1: Run full test suite**

Run: `pytest tests/test_developmental/ -v`
Expected: All 19 tests PASS

**Step 2: Lint**

Run: `ruff check src/soma/developmental/ tests/test_developmental/`
Expected: No errors (fix any that appear)

**Step 3: Type check**

Run: `mypy src/soma/developmental/`
Expected: No errors (fix any that appear)

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore(developmental): fix lint and type errors"
```

---

## Summary

| Task | Component | Tests | New Files |
|------|-----------|-------|-----------|
| 1 | `verbalize_state()` | 6 | 2 src + 2 test |
| 2 | `PredictiveSOMA` | 5 | 1 src + 1 test |
| 3 | `DevelopmentTracker` | 5 | 1 src + 1 test |
| 4 | `InteractionLoop` | 5 | 1 src + 1 test |
| 5 | `AblationHarness` | 3 | 1 src + 1 test |
| 6 | CLI entry point | 1 (smoke) | 1 src |
| 7 | Lint + type check | - | - |

**Total: 25 tests across 5 modules, 7 new source files.**

After this plan completes, you can:
1. `python -m soma.developmental.cli` to talk to the developing SOMA
2. Run ablation tests to measure SOMA's contribution
3. Track development metrics over time
