"""Background training controller — runs SOMA.step on a worker thread.

The UI thread never calls ``SOMA.step`` directly (too slow to block the
DPG render loop). Instead:

- The main thread enqueues :class:`TrainingCommand` messages.
- A dedicated worker thread owns the :class:`SOMA` instance and consumes
  commands one at a time, publishing per-step metrics onto the
  :class:`DataBus` and a full graph snapshot every ``snapshot_every``
  steps (tunable by the UI).

State machine::

    IDLE --start--> RUNNING --pause--> PAUSED --resume--> RUNNING
      ^                 |                |
      |                 +---stop---------+
      |                                  |
      +------------------reset-----------+

``reset`` returns to IDLE with the same SOMA + feeder.  ``rebuild`` (a
harder reset) discards the SOMA and creates a fresh one from the current
config.

This module is import-safe without DearPyGUI installed.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from soma.core.config import SOMAConfig
from soma.io.dataset_feeders import Sample, TextDatasetFeeder
from soma.io.text_decoder import TextDecoder
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.system import SOMA
from soma.ui.bus import DataBus

LOGGER = logging.getLogger("soma.ui.controller")


# ----------------------------------------------------------------------
# State + commands
# ----------------------------------------------------------------------
class TrainingState(Enum):
    """Lifecycle state of the worker thread."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class TrainingCommand(Enum):
    """Messages the UI thread sends to the worker."""

    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    RESET = "reset"  # wipe SOMA + feeder, keep config
    STEP_ONCE = "step_once"  # run a single step while paused


# ----------------------------------------------------------------------
# Channel names (module-level so tests + panels share them)
# ----------------------------------------------------------------------
CHANNEL_METRICS = "metrics"
CHANNEL_GROWTH_EVENT = "growth_event"
CHANNEL_GRAPH_SNAPSHOT = "graph_snapshot"
CHANNEL_CHAT_RESPONSE = "chat_response"
CHANNEL_STATE = "state"
CHANNEL_LOG = "log"


@dataclass
class SessionSpec:
    """Everything needed to construct a training session from scratch."""

    config: SOMAConfig
    corpus_texts: list[str]
    vocab_size: int = 512
    full_sequence: bool = True
    reset_wm_between_samples: bool = True
    snapshot_every: int = 200
    # When > 0, the worker auto-stops after this many total steps. 0 = unlimited.
    step_budget: int = 0
    # Factory hook to support fake SOMAs / encoders in tests.
    soma_factory: Callable[[SOMAConfig], SOMA] | None = None
    device: str = "cpu"


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------
class TrainingController:
    """Owns the worker thread + SOMA instance + feeder.

    Lifecycle::

        controller = TrainingController(bus)
        controller.configure(SessionSpec(...))
        controller.start()      # -> RUNNING
        controller.pause()      # -> PAUSED
        controller.resume()     # -> RUNNING
        controller.stop()       # -> STOPPED
        controller.shutdown()   # join worker thread
    """

    def __init__(self, bus: DataBus) -> None:
        self.bus = bus
        self._state = TrainingState.IDLE
        self._state_lock = threading.RLock()
        self._cmd_queue: queue.Queue[TrainingCommand] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._shutdown = threading.Event()
        self._paused_event = threading.Event()
        self._paused_event.set()  # unset means "pause"; set means "run freely"

        # Worker-owned state (touched only from inside the worker thread).
        self._spec: SessionSpec | None = None
        self._soma: SOMA | None = None
        self._encoder: TextEncoder | None = None
        self._decoder: TextDecoder | None = None
        self._feeder: TextDatasetFeeder | None = None
        self._sample_iter: Iterator[Sample] | None = None
        self._total_steps: int = 0

        # Ensure every channel we'll publish on exists before panels subscribe.
        for name in (
            CHANNEL_METRICS,
            CHANNEL_GROWTH_EVENT,
            CHANNEL_GRAPH_SNAPSHOT,
            CHANNEL_CHAT_RESPONSE,
            CHANNEL_STATE,
            CHANNEL_LOG,
        ):
            bus.create_channel(name)

    # ------------------------------------------------------------------
    # Public API (main thread)
    # ------------------------------------------------------------------
    @property
    def state(self) -> TrainingState:
        with self._state_lock:
            return self._state

    @property
    def soma(self) -> SOMA | None:
        """Worker-owned SOMA. Safe to read from the main thread for quick
        inspections (num_nodes, last_curiosity) because those fields
        don't race badly — but never call .step() from outside the worker."""
        return self._soma

    @property
    def encoder(self) -> TextEncoder | None:
        return self._encoder

    @property
    def decoder(self) -> TextDecoder | None:
        return self._decoder

    def configure(self, spec: SessionSpec) -> None:
        """Install a session spec. Must be called before :meth:`start`."""
        if self.state is TrainingState.RUNNING:
            raise RuntimeError("Cannot reconfigure while RUNNING; stop first")
        self._spec = spec

    def start(self) -> None:
        """Launch the worker thread if not already running."""
        if self._spec is None:
            raise RuntimeError("Call configure() before start()")
        if self._worker is not None and self._worker.is_alive():
            self._cmd_queue.put(TrainingCommand.START)
            return
        self._shutdown.clear()
        self._paused_event.set()
        self._worker = threading.Thread(
            target=self._run_worker,
            name="soma-training-worker",
            daemon=True,
        )
        self._worker.start()
        self._cmd_queue.put(TrainingCommand.START)

    def pause(self) -> None:
        self._cmd_queue.put(TrainingCommand.PAUSE)

    def resume(self) -> None:
        self._cmd_queue.put(TrainingCommand.RESUME)

    def stop(self) -> None:
        self._cmd_queue.put(TrainingCommand.STOP)

    def reset(self) -> None:
        self._cmd_queue.put(TrainingCommand.RESET)

    def step_once(self) -> None:
        self._cmd_queue.put(TrainingCommand.STEP_ONCE)

    def set_snapshot_every(self, n: int) -> None:
        """Live-update how often the worker publishes graph snapshots.

        Safe to call from the UI thread: ``SessionSpec.snapshot_every`` is a
        plain int, the worker re-reads it on every step, and assignment is
        atomic under the GIL. A no-op if no session has been configured yet.
        """
        if self._spec is None:
            return
        self._spec.snapshot_every = max(1, int(n))

    def shutdown(self, join_timeout: float = 2.0) -> None:
        """Signal the worker to exit and wait for it."""
        self._shutdown.set()
        self._paused_event.set()
        self._cmd_queue.put(TrainingCommand.STOP)
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=join_timeout)

    # ------------------------------------------------------------------
    # Interactive / checkpoint helpers (main thread reads, worker writes)
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> None:
        """Save the current SOMA state (safe to call while paused or idle)."""
        if self._soma is None:
            raise RuntimeError("No SOMA to save - configure and start first")
        if self.state is TrainingState.RUNNING:
            raise RuntimeError("Pause before saving a checkpoint")
        self._soma.save_state(path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load SOMA state. Must be stopped or paused first."""
        if self._soma is None:
            raise RuntimeError("No SOMA to load into - configure first")
        if self.state is TrainingState.RUNNING:
            raise RuntimeError("Pause before loading a checkpoint")
        self._soma.load_state(path)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------
    def _run_worker(self) -> None:
        """Main worker loop. Runs on the worker thread."""
        try:
            self._ensure_soma()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Worker failed to build SOMA: %s", exc)
            self._publish_log(f"ERROR building SOMA: {exc}")
            self._set_state(TrainingState.STOPPED)
            return

        while not self._shutdown.is_set():
            # Drain any pending command messages (non-blocking if IDLE/PAUSED).
            try:
                cmd = self._cmd_queue.get(timeout=0.05)
            except queue.Empty:
                cmd = None

            if cmd is TrainingCommand.START:
                self._set_state(TrainingState.RUNNING)
                self._paused_event.set()
            elif cmd is TrainingCommand.PAUSE:
                self._paused_event.clear()
                self._set_state(TrainingState.PAUSED)
            elif cmd is TrainingCommand.RESUME:
                self._paused_event.set()
                self._set_state(TrainingState.RUNNING)
            elif cmd is TrainingCommand.STOP:
                self._paused_event.set()
                self._set_state(TrainingState.STOPPED)
                # Stay in the loop waiting for new commands (START / RESET).
            elif cmd is TrainingCommand.RESET:
                self._reset_soma()
                self._set_state(TrainingState.IDLE)
            elif cmd is TrainingCommand.STEP_ONCE:
                self._one_step()

            # When RUNNING (and not paused mid-loop), take one SOMA step.
            # Paused events let future coordination points break out, but
            # for now a PAUSE arriving between steps is enough granularity.
            if self.state is TrainingState.RUNNING and self._paused_event.is_set():
                self._one_step()

    # ------------------------------------------------------------------
    # Worker-side helpers
    # ------------------------------------------------------------------
    def _ensure_soma(self) -> None:
        assert self._spec is not None
        if self._soma is None:
            spec = self._spec
            if spec.soma_factory is not None:
                self._soma = spec.soma_factory(spec.config)
            else:
                self._soma = SOMA(spec.config, device=torch.device(spec.device))
            # Encoder + decoder MUST live on the same device as SOMA — the
            # feeder produces tensors off the encoder's embedding weight,
            # which go straight into SOMA.step. A CPU/CUDA mismatch here
            # blows up on the first edge transmit.
            device = torch.device(spec.device)
            tokenizer = train_bpe_tokenizer(spec.corpus_texts, vocab_size=spec.vocab_size)
            self._encoder = TextEncoder(
                tokenizer,
                embed_dim=spec.config.text_embed_dim,
                max_seq_len=spec.config.max_input_tokens,
                device=device,
            )
            self._decoder = TextDecoder(
                tokenizer, embed_dim=spec.config.text_embed_dim, device=device
            )
            # Tie so generated tokens map back through the learned embedding.
            self._decoder.tie_weights(self._encoder)
            self._feeder = TextDatasetFeeder(
                self._encoder,
                spec.corpus_texts,
                chunk_size=max(4, min(16, spec.config.max_input_tokens // 2)),
            )
            self._sample_iter = iter(self._feeder)
            self._total_steps = 0
            self._publish_log("worker ready")

    def _reset_soma(self) -> None:
        """Drop SOMA + feeder and rebuild on next START."""
        self._soma = None
        self._encoder = None
        self._decoder = None
        self._feeder = None
        self._sample_iter = None
        self._total_steps = 0
        self._publish_log("reset complete - ready for start")

    def _one_step(self) -> None:
        """Run a single SOMA.step and publish metrics / growth events."""
        assert self._spec is not None
        if self._soma is None:
            self._ensure_soma()
        assert self._soma is not None
        assert self._sample_iter is not None

        spec = self._spec
        first_out = spec.config.output_modalities[0]

        # Lazily pull one token pair from the current sample; refresh sample
        # when we exhaust its tokens.
        inputs, targets = self._next_token_pair(first_out, spec.reset_wm_between_samples)
        if inputs is None or targets is None:
            return
        try:
            result = self._soma.step(inputs=inputs, targets=targets)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("SOMA.step failed: %s", exc)
            self._publish_log(f"ERROR in step: {exc}")
            self._set_state(TrainingState.STOPPED)
            return

        self._total_steps += 1
        self._publish_metrics(result)
        self._maybe_publish_growth(result)
        if self._total_steps % spec.snapshot_every == 0:
            self._publish_graph_snapshot()
        if spec.step_budget > 0 and self._total_steps >= spec.step_budget:
            self._publish_log(f"step budget reached ({spec.step_budget}); stopping")
            self._set_state(TrainingState.STOPPED)
            self._paused_event.clear()

    _current_sample: Sample | None = None
    _current_sample_idx: int = 0

    def _next_token_pair(
        self, output_modality: str, reset_wm: bool
    ) -> tuple[dict[str, torch.Tensor] | None, dict[str, torch.Tensor] | None]:
        """Yield one (inputs, targets) pair, walking the current sample token-by-token."""
        assert self._sample_iter is not None
        assert self._soma is not None
        assert self._spec is not None

        if (
            self._current_sample is None
            or self._current_sample_idx >= self._current_sample.target.shape[0]
        ):
            # Pull next sample.
            try:
                self._current_sample = next(self._sample_iter)
            except StopIteration:
                assert self._feeder is not None
                self._sample_iter = iter(self._feeder)
                self._current_sample = next(self._sample_iter)
            self._current_sample_idx = 0
            if reset_wm and self._spec.full_sequence:
                self._soma.working_memory.clear()

        sample = self._current_sample
        if sample.target.numel() == 0:
            self._current_sample = None
            return None, None

        t = self._current_sample_idx if self._spec.full_sequence else 0
        step_inputs: dict[str, torch.Tensor] = {}
        for modality, seq in sample.inputs.items():
            if seq.numel() == 0:
                continue
            idx = min(t, seq.shape[0] - 1)
            step_inputs[modality] = seq[idx].detach()
        step_targets = {output_modality: sample.target[t].detach()}
        self._current_sample_idx += 1
        if not self._spec.full_sequence:
            # Force next iteration to pull a fresh sample.
            self._current_sample = None
            self._current_sample_idx = 0
        if not step_inputs:
            return None, None
        return step_inputs, step_targets

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------
    def _publish_metrics(self, result: dict[str, Any]) -> None:
        assert self._soma is not None
        record = {
            "global_step": self._soma.global_step,
            "loss": float(result["loss"]) if result["loss"] is not None else None,
            "curiosity": float(result["curiosity"]),
            "lr_multiplier": float(result["lr_multiplier"]),
            "num_nodes": int(result["num_nodes"]),
            "num_edges": int(result["num_edges"]),
            "wm_occupancy": float(self._soma.working_memory.occupancy()),
            "episodic_entries": int(self._soma.episodic_memory.num_valid),
        }
        self.bus.publish(CHANNEL_METRICS, record)

    _last_node_count: int = 0
    _last_edge_count: int = 0

    def _maybe_publish_growth(self, result: dict[str, Any]) -> None:
        """Emit a growth event if the node or edge count changed."""
        nodes = int(result["num_nodes"])
        edges = int(result["num_edges"])
        if nodes != self._last_node_count or edges != self._last_edge_count:
            delta_nodes = nodes - self._last_node_count
            delta_edges = edges - self._last_edge_count
            event_type = self._classify_growth(delta_nodes, delta_edges)
            self.bus.publish(
                CHANNEL_GROWTH_EVENT,
                {
                    "global_step": int(result["global_step"]),
                    "type": event_type,
                    "delta_nodes": delta_nodes,
                    "delta_edges": delta_edges,
                    "num_nodes": nodes,
                    "num_edges": edges,
                },
            )
            self._last_node_count = nodes
            self._last_edge_count = edges

    @staticmethod
    def _classify_growth(delta_nodes: int, delta_edges: int) -> str:
        if delta_nodes > 0:
            return "neurogenesis"
        if delta_nodes < 0 or delta_edges < 0:
            return "pruning"
        if delta_edges > 0:
            return "synaptogenesis"
        return "none"

    def _publish_graph_snapshot(self) -> None:
        """Send a full topology snapshot for the graph viewer to pick up."""
        assert self._soma is not None
        graph = self._soma.graph
        nodes = [
            {
                "id": n.id,
                "type": n.node_type.value,
                "activation_ema": float(n.activation_ema),
                "maturity": float(n.maturity),
            }
            for n in graph.all_nodes()
        ]
        edges = [
            {
                "source": e.source_id,
                "target": e.target_id,
                "weight": float(e.weight.item()),
            }
            for e in graph.all_edges()
        ]
        self.bus.publish(
            CHANNEL_GRAPH_SNAPSHOT,
            {
                "global_step": self._soma.global_step,
                "nodes": nodes,
                "edges": edges,
                "sensor_nodes": {m: n.id for m, n in graph.sensor_nodes.items()},
                "output_nodes": {m: n.id for m, n in graph.output_nodes.items()},
            },
        )

    def _publish_log(self, message: str) -> None:
        self.bus.publish(CHANNEL_LOG, message)

    def _set_state(self, state: TrainingState) -> None:
        with self._state_lock:
            old = self._state
            self._state = state
        if old is not state:
            self.bus.publish(CHANNEL_STATE, {"from": old.value, "to": state.value})


# ----------------------------------------------------------------------
# Convenience: load a corpus into a list of strings
# ----------------------------------------------------------------------
def load_corpus(path: str | Path) -> list[str]:
    """Read a text file into non-empty lines (same format as scripts/train.py)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus {p} not found")
    lines = [raw.strip() for raw in p.read_text(encoding="utf-8").splitlines() if raw.strip()]
    if not lines:
        raise ValueError(f"Corpus {p} has no non-empty lines")
    return lines


def iter_samples(feeder: TextDatasetFeeder) -> Iterable[Sample]:
    """Yield samples from a feeder — a thin wrapper for clarity."""
    yield from feeder
