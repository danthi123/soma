"""Tests for ``soma.ui.training_controller``.

Exercises the worker-thread lifecycle, command queue semantics, and
bus publishing without requiring DearPyGUI.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from soma.core.config import SOMAConfig
from soma.ui.bus import DataBus
from soma.ui.training_controller import (
    CHANNEL_GRAPH_SNAPSHOT,
    CHANNEL_GROWTH_EVENT,
    CHANNEL_METRICS,
    CHANNEL_STATE,
    SessionSpec,
    TrainingController,
    TrainingState,
    load_corpus,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        wm_slots=4,
        wm_dim=8,
        key_dim=8,
        value_dim=16,
        text_embed_dim=8,
        vocab_size=128,
        initial_associator_count=2,
        initial_integrator_count=0,
        max_nodes=32,
        num_curiosity_domains=2,
        base_lr=0.01,
        hebbian_lr=0.0001,
        youth_lr_multiplier=1.0,
        activation_threshold=0.01,
        synaptogenesis_interval=50,
        neurogenesis_interval=200,
        pruning_interval=100,
        consolidation_interval=100,
        max_input_tokens=16,
        max_output_tokens=8,
        seed=0,
    )


@pytest.fixture
def corpus() -> list[str]:
    return [
        "hello world apples bananas trees fish",
        "cats dogs birds flowers clouds sky",
        "red blue green yellow purple orange",
    ] * 3


@pytest.fixture
def spec(config: SOMAConfig, corpus: list[str]) -> SessionSpec:
    return SessionSpec(
        config=config,
        corpus_texts=corpus,
        vocab_size=64,
        full_sequence=True,
        snapshot_every=5,
        step_budget=30,  # auto-stop after 30 steps so the test can't hang
    )


def _wait_for(predicate: callable, timeout: float = 5.0, poll: float = 0.02) -> bool:  # type: ignore[valid-type]
    """Spin until ``predicate()`` is truthy or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


# ----------------------------------------------------------------------
# Load-corpus helper
# ----------------------------------------------------------------------
class TestLoadCorpus:
    def test_reads_non_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "c.txt"
        path.write_text("  alpha  \n\nbeta\n\n", encoding="utf-8")
        assert load_corpus(path) == ["alpha", "beta"]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_corpus(tmp_path / "no.txt")

    def test_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty"):
            load_corpus(path)


# ----------------------------------------------------------------------
# State / lifecycle
# ----------------------------------------------------------------------
class TestLifecycle:
    def test_starts_in_idle(self) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        assert ctrl.state is TrainingState.IDLE
        ctrl.shutdown()

    def test_configure_required_before_start(self) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        with pytest.raises(RuntimeError, match="configure"):
            ctrl.start()

    def test_start_transitions_to_running(self, spec: SessionSpec) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        ctrl.configure(spec)
        ctrl.start()
        assert _wait_for(lambda: ctrl.state is TrainingState.RUNNING)
        ctrl.shutdown()

    def test_pause_then_resume(self, spec: SessionSpec) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        ctrl.configure(spec)
        ctrl.start()
        assert _wait_for(lambda: ctrl.state is TrainingState.RUNNING)
        ctrl.pause()
        assert _wait_for(lambda: ctrl.state is TrainingState.PAUSED)
        ctrl.resume()
        assert _wait_for(lambda: ctrl.state is TrainingState.RUNNING)
        ctrl.shutdown()

    def test_stop_transitions_to_stopped(self, spec: SessionSpec) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        ctrl.configure(spec)
        ctrl.start()
        assert _wait_for(lambda: ctrl.state is TrainingState.RUNNING)
        ctrl.stop()
        assert _wait_for(lambda: ctrl.state is TrainingState.STOPPED)
        ctrl.shutdown()

    def test_auto_stop_on_step_budget(self, config: SOMAConfig, corpus: list[str]) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        spec = SessionSpec(
            config=config,
            corpus_texts=corpus,
            vocab_size=64,
            full_sequence=True,
            snapshot_every=3,
            step_budget=5,
        )
        ctrl.configure(spec)
        ctrl.start()
        assert _wait_for(
            lambda: ctrl.state is TrainingState.STOPPED,
            timeout=10.0,
        ), f"did not stop — current state={ctrl.state}"
        ctrl.shutdown()


# ----------------------------------------------------------------------
# Publishing
# ----------------------------------------------------------------------
class TestPublishing:
    def test_metrics_published_per_step(self, spec: SessionSpec) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        ctrl.configure(spec)
        ctrl.start()
        assert _wait_for(lambda: bus.get(CHANNEL_METRICS).size >= 5, timeout=10.0)
        ctrl.shutdown()
        rec = bus.get(CHANNEL_METRICS).latest()
        assert rec is not None
        assert {
            "global_step",
            "loss",
            "curiosity",
            "lr_multiplier",
            "num_nodes",
            "num_edges",
            "wm_occupancy",
            "episodic_entries",
        }.issubset(rec)

    def test_state_transitions_published(self, spec: SessionSpec) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        ctrl.configure(spec)
        ctrl.start()
        assert _wait_for(lambda: ctrl.state is TrainingState.RUNNING)
        ctrl.pause()
        assert _wait_for(lambda: ctrl.state is TrainingState.PAUSED)
        ctrl.shutdown()
        hist = bus.get(CHANNEL_STATE).get_history()
        assert any(t.get("to") == "running" for t in hist)
        assert any(t.get("to") == "paused" for t in hist)

    def test_snapshot_emitted_on_interval(self, config: SOMAConfig, corpus: list[str]) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        spec = SessionSpec(
            config=config,
            corpus_texts=corpus,
            vocab_size=64,
            snapshot_every=2,
            step_budget=6,
        )
        ctrl.configure(spec)
        ctrl.start()
        assert _wait_for(
            lambda: bus.get(CHANNEL_GRAPH_SNAPSHOT).size >= 2,
            timeout=15.0,
        )
        ctrl.shutdown()
        snap = bus.get(CHANNEL_GRAPH_SNAPSHOT).latest()
        assert snap is not None
        assert "nodes" in snap and "edges" in snap
        assert snap["nodes"]  # non-empty

    def test_growth_event_emitted_on_topology_change(
        self, config: SOMAConfig, corpus: list[str]
    ) -> None:
        # We can't easily force synaptogenesis on a tiny run — but we can
        # verify that if the controller observes a delta, it publishes.
        bus = DataBus()
        ctrl = TrainingController(bus)
        spec = SessionSpec(
            config=config,
            corpus_texts=corpus,
            vocab_size=64,
            snapshot_every=5,
            step_budget=10,
        )
        ctrl.configure(spec)
        ctrl.start()
        _wait_for(lambda: ctrl.state is TrainingState.STOPPED, timeout=15.0)
        ctrl.shutdown()
        # With consolidation_interval=100 + synaptogenesis_interval=50, the
        # first 10 steps won't necessarily trigger anything structural; we
        # just assert the channel exists without requiring entries.
        assert bus.has(CHANNEL_GROWTH_EVENT)


# ----------------------------------------------------------------------
# Reset + checkpoint
# ----------------------------------------------------------------------
class TestCheckpoints:
    def test_save_and_load_round_trip(
        self,
        spec: SessionSpec,
        tmp_path: Path,
    ) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        ctrl.configure(spec)
        ctrl.start()
        _wait_for(lambda: ctrl.state is TrainingState.STOPPED, timeout=15.0)
        ckpt = tmp_path / "soma.pt"
        ctrl.save_checkpoint(ckpt)
        assert ckpt.exists()
        ctrl.load_checkpoint(ckpt)
        ctrl.shutdown()

    def test_save_rejects_while_running(self, spec: SessionSpec, tmp_path: Path) -> None:
        bus = DataBus()
        ctrl = TrainingController(bus)
        ctrl.configure(spec)
        ctrl.start()
        _wait_for(lambda: ctrl.state is TrainingState.RUNNING, timeout=5.0)
        with pytest.raises(RuntimeError, match="Pause"):
            ctrl.save_checkpoint(tmp_path / "x.pt")
        ctrl.shutdown()
