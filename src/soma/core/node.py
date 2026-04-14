"""Node: the dynamic graph's processing unit.

Each Node is a small MLP (input_dim -> hidden_dim -> output_dim) with GELU,
an optional residual connection when ``input_dim == output_dim``, and a
homeostatic gain factor. Nodes have four types (SENSOR, ASSOCIATOR,
INTEGRATOR, OUTPUT) with different roles and default dimensions.

See whitepaper Section 3.1 for the full spec.

Key design choices:
- ``Node`` is an ``nn.Module`` because it owns learnable parameters
  (``linear1``/``linear2``) — using ``nn.Linear`` under the hood gives us
  clean param registration and batched input for free. The whitepaper's
  W1/b1/W2/b2 map to ``linear1.weight``/``linear1.bias`` etc.
- ``gain``, ``maturity``, and activation statistics are in-place updated
  by homeostasis / learning code; they are stored as Python floats so the
  update code stays readable. Serialization is explicit via ``to_dict``
  / ``from_dict``.
- SENSOR nodes are special: their ``forward()`` ignores incoming edges
  and returns whatever was set via ``set_input()``.
- ``current_step`` is always passed as a parameter (never a global).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from soma.core.config import SOMAConfig
from soma.core.ring_buffer import RingBuffer
from soma.core.utils import generate_uuid


class NodeType(Enum):
    """Functional role of a node in the graph."""

    SENSOR = "sensor"
    ASSOCIATOR = "associator"
    INTEGRATOR = "integrator"
    OUTPUT = "output"

    @property
    def is_boundary(self) -> bool:
        """True for I/O boundary nodes (SENSOR, OUTPUT) that cannot be pruned."""
        return self in (NodeType.SENSOR, NodeType.OUTPUT)


# Default per-type dimensions, per whitepaper Section 3.1. These apply when
# caller passes no explicit dims. A modality-specific encoder/decoder may
# override sensor/output dims.
_DEFAULT_DIMS: dict[NodeType, tuple[str, str, str]] = {
    NodeType.SENSOR: ("sensor_output_dim", "associator_hidden_dim", "sensor_output_dim"),
    NodeType.ASSOCIATOR: (
        "associator_input_dim",
        "associator_hidden_dim",
        "associator_output_dim",
    ),
    NodeType.INTEGRATOR: (
        "integrator_input_dim",
        "integrator_hidden_dim",
        "integrator_output_dim",
    ),
    NodeType.OUTPUT: ("associator_output_dim", "associator_hidden_dim", "text_embed_dim"),
}


class Node(nn.Module):
    """A single processing unit in the SOMA graph.

    Parameters
    ----------
    node_id:
        Unique identifier. If ``None``, a UUID is generated.
    node_type:
        Functional role (SENSOR/ASSOCIATOR/INTEGRATOR/OUTPUT).
    input_dim, hidden_dim, output_dim:
        MLP dimensions. Must be positive.
    creation_step:
        Global step at which this node was created. Drives maturity and
        pruning grace periods.
    config:
        System config; supplies ``activation_history_size``,
        ``default_target_activation``, ``gain_min``/``gain_max``.
    position:
        Optional soft position vector for locality-biased wiring. If
        ``None``, a small-random position of dim ``config.position_dim``
        is drawn.
    device:
        Optional device for all owned tensors.
    """

    def __init__(
        self,
        node_type: NodeType,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        creation_step: int,
        config: SOMAConfig,
        *,
        node_id: str | None = None,
        position: torch.Tensor | None = None,
        maturity: float = 0.0,
        gain: float = 1.0,
        target_activation: float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError(
                "Node requires positive dims: "
                f"input={input_dim}, hidden={hidden_dim}, output={output_dim}"
            )
        if creation_step < 0:
            raise ValueError(f"creation_step must be non-negative, got {creation_step}")
        if not 0.0 <= maturity <= 1.0:
            raise ValueError(f"maturity must be in [0, 1], got {maturity}")
        if gain < config.gain_min or gain > config.gain_max:
            raise ValueError(f"gain {gain} out of bounds [{config.gain_min}, {config.gain_max}]")

        self.id: str = node_id if node_id is not None else generate_uuid()
        self.node_type: NodeType = node_type
        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim

        # Learnable MLP: x -> GELU(linear1(x)) -> linear2(...).
        # ``linear1.weight`` has shape (hidden_dim, input_dim), matching W1
        # in the whitepaper pseudocode (``h = W1 @ x + b1``).
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        # Position buffer: non-learnable, participates in state_dict.
        if position is None:
            position = torch.randn(config.position_dim) * 0.1
        else:
            position = position.detach().clone().float()
        self.register_buffer("position", position)

        if device is not None:
            self.to(device)

        # Activation tracking (host-side floats — fast to read/modify).
        self.activation_history: RingBuffer = RingBuffer(config.activation_history_size)
        self.activation_ema: float = 0.0
        self.creation_step: int = creation_step
        self.last_active_step: int = creation_step
        self.maturity: float = maturity

        # Most recent output tensor produced by this node (or ``None`` before
        # the first forward pass). Read by downstream consumers that need the
        # actual activation vector, not just its magnitude — e.g.,
        # ``SOMA.chat`` aggregating OUTPUT-node state into a soft prompt.
        # Kept as a plain attribute (not a buffer) because it is not part of
        # the node's persistent state — checkpoints restore activations by
        # replaying, not by storing per-step tensors. Per-node cost is a
        # single detached tensor; at scale only OUTPUT nodes (a handful) are
        # actually read, so total footprint is trivial.
        self.last_activation: torch.Tensor | None = None

        # Homeostatic parameters.
        self.target_activation: float = (
            target_activation if target_activation is not None else config.default_target_activation
        )
        self.gain: float = gain
        self._gain_min: float = config.gain_min
        self._gain_max: float = config.gain_max
        self._activation_threshold: float = config.activation_threshold

        # SENSOR-only state: most-recent externally-injected input.
        self._sensor_input: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        current_step: int,
    ) -> torch.Tensor:
        """Compute one forward pass.

        Parameters
        ----------
        inputs:
            Mapping from source node id to an already-weighted activation
            vector. The weighting (edge weight + projection) is done by
            ``Edge.transmit`` before this call. For SENSOR nodes, ``inputs``
            is ignored and the injected sensor input is used.
        current_step:
            Global step, recorded as ``last_active_step`` when this node's
            output magnitude exceeds the activation threshold.

        Returns
        -------
        Activation vector of shape ``(output_dim,)``.
        """
        if self.node_type is NodeType.SENSOR:
            return self._sensor_forward(current_step)

        # Aggregate incoming signals (sum — whitepaper 3.1).
        x = self._aggregate_inputs(inputs)

        # Small MLP with residual + gain (whitepaper 3.1).
        h: torch.Tensor = F.gelu(self.linear1(x))
        h = self.linear2(h)
        h = h * self.gain
        if self.input_dim == self.output_dim:
            # Residual connection — x already has output_dim == input_dim.
            h = h + x

        self._record_activation(h, current_step)
        return h

    # ------------------------------------------------------------------
    # Sensor-specific helpers
    # ------------------------------------------------------------------
    def set_input(self, data: torch.Tensor) -> None:
        """Store external input for a SENSOR node (consumed on next forward).

        We deliberately do NOT detach here: callers that pass the output of
        an upstream ``nn.Module`` (e.g., ``TextEncoder``) need gradients to
        flow back through the sensor into the encoder. Callers that want a
        detached snapshot should detach before calling.
        """
        if self.node_type is not NodeType.SENSOR:
            raise RuntimeError(
                f"set_input only valid on SENSOR nodes; this node is {self.node_type.value}"
            )
        if data.shape[-1] != self.output_dim:
            raise ValueError(
                f"SENSOR expects last-dim={self.output_dim}, got shape {tuple(data.shape)}"
            )
        self._sensor_input = data

    def get_input_activation(self) -> torch.Tensor:
        """Return the pending sensor input, or zeros if nothing is set."""
        if self.node_type is not NodeType.SENSOR:
            raise RuntimeError(
                "get_input_activation only valid on SENSOR nodes; "
                f"this node is {self.node_type.value}"
            )
        if self._sensor_input is None:
            # Return zeros on the same device as the node's params.
            ref = self.linear1.weight
            return torch.zeros(self.output_dim, device=ref.device, dtype=ref.dtype)
        return self._sensor_input

    def clear_input(self) -> None:
        """Discard any pending sensor input."""
        self._sensor_input = None

    # ------------------------------------------------------------------
    # Homeostasis helpers
    # ------------------------------------------------------------------
    def clamp_gain(self) -> None:
        """Clamp ``self.gain`` back into the configured range."""
        if self.gain < self._gain_min:
            self.gain = self._gain_min
        elif self.gain > self._gain_max:
            self.gain = self._gain_max

    def advance_maturity(self, increment: float) -> None:
        """Move maturity toward 1.0 by ``increment``, clipping at 1.0."""
        if increment < 0.0:
            raise ValueError(f"maturity increment must be non-negative, got {increment}")
        self.maturity = min(1.0, self.maturity + increment)

    # ------------------------------------------------------------------
    # Serialization (explicit — ``state_dict`` only covers nn.Parameter/buffers)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize non-learnable scalar state (IDs, stats, maturity)."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "creation_step": self.creation_step,
            "last_active_step": self.last_active_step,
            "maturity": self.maturity,
            "target_activation": self.target_activation,
            "gain": self.gain,
            "activation_ema": self.activation_ema,
            "activation_history": self.activation_history.to_list(),
            "activation_history_capacity": self.activation_history.capacity,
            "gain_min": self._gain_min,
            "gain_max": self._gain_max,
            "activation_threshold": self._activation_threshold,
        }

    def load_scalar_state(self, state: dict[str, Any]) -> None:
        """Restore scalar state from ``to_dict`` output. Learnable params and
        buffers are restored via the module's own ``load_state_dict``.
        """
        self.id = state["id"]
        self.creation_step = int(state["creation_step"])
        self.last_active_step = int(state["last_active_step"])
        self.maturity = float(state["maturity"])
        self.target_activation = float(state["target_activation"])
        self.gain = float(state["gain"])
        self.activation_ema = float(state["activation_ema"])
        cap = int(state["activation_history_capacity"])
        history = [float(v) for v in state["activation_history"]]
        self.activation_history = (
            RingBuffer.from_list(history, capacity=cap) if history else RingBuffer(cap)
        )
        self._gain_min = float(state.get("gain_min", self._gain_min))
        self._gain_max = float(state.get("gain_max", self._gain_max))
        self._activation_threshold = float(
            state.get("activation_threshold", self._activation_threshold)
        )

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def make(
        cls,
        node_type: NodeType,
        config: SOMAConfig,
        creation_step: int,
        *,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        node_id: str | None = None,
        position: torch.Tensor | None = None,
        maturity: float = 0.0,
        device: torch.device | str | None = None,
    ) -> Node:
        """Construct a Node with whitepaper default dimensions for its type."""
        in_key, hid_key, out_key = _DEFAULT_DIMS[node_type]
        resolved_in = input_dim if input_dim is not None else int(getattr(config, in_key))
        resolved_hid = hidden_dim if hidden_dim is not None else int(getattr(config, hid_key))
        resolved_out = output_dim if output_dim is not None else int(getattr(config, out_key))
        return cls(
            node_type=node_type,
            input_dim=resolved_in,
            hidden_dim=resolved_hid,
            output_dim=resolved_out,
            creation_step=creation_step,
            config=config,
            node_id=node_id,
            position=position,
            maturity=maturity,
            device=device,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"id={self.id[:8]}..., type={self.node_type.value}, "
            f"dims=({self.input_dim},{self.hidden_dim},{self.output_dim})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _aggregate_inputs(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum the incoming activations, yielding a vector of shape (input_dim,).

        All inputs must match ``self.input_dim`` on the last axis. An empty
        ``inputs`` dict produces a zero vector (the caller typically skips
        empty-input nodes, but we keep the contract well-defined).
        """
        ref = self.linear1.weight  # for device/dtype
        if not inputs:
            return torch.zeros(self.input_dim, device=ref.device, dtype=ref.dtype)
        acc: torch.Tensor | None = None
        for source_id, signal in inputs.items():
            if signal.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Node {self.id[:8]} got input from {source_id[:8]} with "
                    f"last-dim={signal.shape[-1]}, expected {self.input_dim}"
                )
            acc = signal if acc is None else acc + signal
        assert acc is not None  # non-empty inputs dict ensures acc is set
        return acc

    def _sensor_forward(self, current_step: int) -> torch.Tensor:
        signal = self.get_input_activation()
        self._record_activation(signal, current_step)
        return signal

    def _record_activation(self, activation: torch.Tensor, current_step: int) -> None:
        magnitude = float(activation.detach().norm().item())
        self.activation_history.append(magnitude)
        # EMA — matches whitepaper (0.99 decay).
        self.activation_ema = 0.99 * self.activation_ema + 0.01 * magnitude
        if magnitude > self._activation_threshold:
            self.last_active_step = current_step
        # Detach the tensor we stash so downstream readers can't
        # accidentally route gradients through stale state; the live
        # forward-pass tensor is still returned from ``forward`` for the
        # executor to route into subsequent waves.
        self.last_activation = activation.detach()
