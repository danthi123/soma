"""SOMA — the integrated system main class.

Whitepaper Section 9. Wires together:
- Processing graph (core.Graph + execution)
- Memory systems (WorkingMemory + EpisodicMemory)
- Meta-cognitive modules (Curiosity + Homeostasis + DevelopmentSchedule)
- Growth engine (synaptogenesis + neurogenesis)
- Consolidation cycle (artificial sleep)
- Text I/O encoders and decoders

The class exposes a ``step(inputs, targets=None)`` method that matches
the whitepaper's loop semantics and a ``save_state`` / ``load_state``
pair for checkpointing.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F  # noqa: N812

from soma.consolidation.cycle import ExperienceUnpacker, consolidation_cycle
from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.execution import execute_graph, execute_graph_batched
from soma.core.graph import Graph
from soma.core.learning import update_step
from soma.core.node import Node, NodeType
from soma.growth.neurogenesis import neurogenesis
from soma.growth.pruning import pruning
from soma.growth.synaptogenesis import synaptogenesis
from soma.memory.episodic_memory import EpisodicMemory
from soma.memory.working_memory import WorkingMemory
from soma.metacognition.curiosity import CuriosityModule
from soma.metacognition.development import DevelopmentSchedule
from soma.metacognition.homeostasis import HomeostaticRegulator


def _current_soma_version() -> str:
    """Best-effort SOMA source version string (git describe) for the bundle manifest.

    Returns ``"unknown"`` if the repo isn't a git checkout or git is missing —
    never raises. Called once per ``save_state`` so a failed subprocess can't
    block a checkpoint write.
    """
    try:
        import subprocess

        return (
            subprocess.check_output(
                ["git", "describe", "--always", "--dirty"],
                cwd=str(Path(__file__).resolve().parents[2]),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 — any failure degrades gracefully
        return "unknown"


def _snapshot_edges(graph: Graph, *, step: int) -> dict[str, dict[str, Any]]:
    """Capture per-edge context for attribution on prune events.

    Called BEFORE pruning runs since the edge objects are gone from the
    graph by the time we'd want to read their ``source_id`` / ``strength``
    / ``creation_step``. Returns ``{edge_id: {source, target, age, final_strength}}``.
    """
    return {
        e.id: {
            "source": e.source_id,
            "target": e.target_id,
            "age": int(step - e.creation_step),
            "final_strength": float(e.strength),
        }
        for e in graph.all_edges()
    }


def _snapshot_nodes(graph: Graph) -> dict[str, dict[str, Any]]:
    """Capture per-node context for attribution on prune events.

    Returns ``{node_id: {node_type}}`` — small today, extensible without
    breaking downstream event consumers.
    """
    return {n.id: {"node_type": n.node_type.name} for n in graph.all_nodes()}


def _neurogenesis_trigger_ratio(recent_errors: Sequence[float]) -> float:
    """Recompute the recent/baseline error ratio that triggered a neurogenesis.

    Mirrors the computation inside ``neurogenesis()`` using the default
    windows (100 / 1000) so the ratio stamped on the event matches the
    trigger condition. Returns 0.0 if the baseline is empty or non-positive
    (shouldn't happen at emission time, but defensive).
    """
    recent = list(recent_errors)[-100:]
    baseline = list(recent_errors)[-1000:]
    if not recent or not baseline:
        return 0.0
    recent_mean = sum(recent) / len(recent)
    baseline_mean = sum(baseline) / len(baseline)
    if baseline_mean <= 0.0:
        return 0.0
    return recent_mean / baseline_mean


class SOMA:
    """Integrated SOMA system — main class.

    The constructor wires an initial graph (one SENSOR + OUTPUT per
    modality, plus ``initial_associator_count`` associators connected to
    every sensor and output) and instantiates all sub-modules. Most of
    the training-loop logic lives in ``step``.
    """

    def __init__(
        self,
        config: SOMAConfig,
        *,
        experience_unpacker: ExperienceUnpacker | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.config = config
        self.device = device
        self.global_step: int = 0
        self._recent_errors: list[float] = []
        self._recent_errors_cap: int = 1000

        self.graph = Graph()
        self._initialize_seed_graph(config)
        # Move the entire graph (nodes + edges, including projections) onto
        # the requested device. Subsequent synaptogenesis / neurogenesis
        # calls need to respect the same device — see _try_add_edge and
        # the growth functions (they now call .to(device) on new modules).
        if device is not None:
            self.graph.to(device)

        self.working_memory = WorkingMemory.from_config(config, device=device)
        self.episodic_memory = EpisodicMemory.from_config(config, device=device)
        self.curiosity = CuriosityModule(
            input_dim=config.sensor_output_dim,
            num_domains=config.num_curiosity_domains,
            device=device,
        )
        self.homeostasis = HomeostaticRegulator.from_config(config)
        self.development = DevelopmentSchedule()

        # Default unpacker: first half of episodic value is sensor input
        # for the first declared modality; second half is target output.
        self.experience_unpacker: ExperienceUnpacker = (
            experience_unpacker
            if experience_unpacker is not None
            else self._default_experience_unpacker
        )

        # Curiosity scores emitted by ``step`` — handy for logging.
        self.last_curiosity: float = 0.0

        # Counts how many consecutive non-finite-loss steps have been
        # skipped. Reset to 0 on any successful step. Raised once it
        # exceeds config.max_consecutive_skipped_steps so train_service's
        # CrashBackoff can detect a stuck training state.
        self._consecutive_skipped_steps: int = 0

        # Append-only structural-event journal. Every synaptogenesis /
        # neurogenesis / pruning event gets a dict appended here by
        # ``record_growth_event``; the bounded deque keeps memory flat
        # over long runs (oldest events age out once we hit the cap).
        # Persisted via ``save_state`` / ``load_state``.
        self.growth_log: deque[dict[str, Any]] = deque(maxlen=10_000)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _initialize_seed_graph(self, config: SOMAConfig) -> None:
        """Seed the graph with one sensor + one output per modality, plus associators."""
        sensor_dim = config.sensor_output_dim
        assoc_in = config.associator_input_dim
        assoc_hid = config.associator_hidden_dim
        assoc_out = config.associator_output_dim
        # Output nodes share the sensor dim at init for simplicity; dedicated
        # modalities can widen them later via add_node.
        output_dim = sensor_dim

        # One SENSOR per input modality.
        sensors: list[Node] = []
        for modality in config.input_modalities:
            node = Node(
                node_type=NodeType.SENSOR,
                input_dim=sensor_dim,
                hidden_dim=sensor_dim * 2,
                output_dim=sensor_dim,
                creation_step=0,
                config=config,
                device=self.device,
            )
            self.graph.add_node(node, modality=modality)
            sensors.append(node)

        # One OUTPUT per output modality.
        outputs: list[Node] = []
        for modality in config.output_modalities:
            node = Node(
                node_type=NodeType.OUTPUT,
                input_dim=assoc_out,
                hidden_dim=assoc_hid,
                output_dim=output_dim,
                creation_step=0,
                config=config,
                device=self.device,
            )
            self.graph.add_node(node, modality=modality)
            outputs.append(node)

        # Associator layer wired sensor -> assoc -> output.
        for _ in range(max(1, config.initial_associator_count)):
            node = Node(
                node_type=NodeType.ASSOCIATOR,
                input_dim=assoc_in,
                hidden_dim=assoc_hid,
                output_dim=assoc_out,
                creation_step=0,
                config=config,
                device=self.device,
            )
            self.graph.add_node(node)
            for sensor in sensors:
                self._try_add_edge(sensor, node)
            for out_node in outputs:
                self._try_add_edge(node, out_node)

    def _try_add_edge(
        self,
        source: Node,
        target: Node,
        *,
        initial_weight: float = 0.1,
    ) -> None:
        if self.graph.has_edge(source.id, target.id):
            return
        self.graph.add_edge(
            Edge(
                source_id=source.id,
                target_id=target.id,
                source_output_dim=source.output_dim,
                target_input_dim=target.input_dim,
                creation_step=self.global_step,
                initial_weight=initial_weight,
            )
        )

    # ------------------------------------------------------------------
    # Step (main training/interaction loop)
    # ------------------------------------------------------------------
    def step(
        self,
        inputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
        *,
        rng: torch.Generator | None = None,
        eval_mode: bool = False,
    ) -> dict[str, Any]:
        """Run one interaction step; returns a dict with outputs/loss/etc.

        When ``eval_mode=True``, skip weight updates, growth, and
        consolidation. Used by evaluation harnesses that need loss /
        output metrics without mutating the graph. Working-memory and
        curiosity are still updated (cheap, read-only for graph
        structure). ``global_step`` still advances so downstream
        metrics stay monotonic.
        """
        exec_fn = execute_graph_batched if self.config.use_batched_executor else execute_graph
        outputs, activations = exec_fn(self.graph, inputs=inputs, current_step=self.global_step)

        context_tensor = next(iter(outputs.values())) if outputs else None
        self._update_working_memory(context_tensor)

        loss: torch.Tensor | None = None
        loss_value: float | None = None
        if targets is not None:
            loss, loss_value = self._compute_loss(outputs, targets)

        # Skip the entire learning/growth/curiosity pipeline if loss is
        # non-finite. Bumps the skip counter; raises after
        # config.max_consecutive_skipped_steps so train_service's
        # CrashBackoff can flag a stuck training state. In eval_mode
        # the counter is neither incremented nor inspected — eval
        # callers expect a tolerant read-only pass.
        if loss_value is not None and not math.isfinite(loss_value):
            if not eval_mode:
                self._consecutive_skipped_steps += 1
                if self._consecutive_skipped_steps > self.config.max_consecutive_skipped_steps:
                    raise ValueError(
                        f"{self._consecutive_skipped_steps} consecutive non-finite "
                        f"losses; training is stuck (last loss={loss_value!r})"
                    )
            result_skipped: dict[str, Any] = {
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
            return result_skipped

        experience = self._encode_episodic(inputs, outputs, targets, loss_value)

        lr_multiplier = 1.0
        if loss is not None and loss_value is not None:
            mult = self.homeostasis.update(self.graph, current_loss=loss_value)
            # Defensive: if homeostasis ever rejects a value we already
            # passed the finite-check on (e.g., future expansion), treat
            # like a skip so the service keeps breathing.
            if mult is None:
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
            if not eval_mode:
                update_step(
                    self.graph,
                    loss,
                    activations,
                    self.config,
                    lr_multiplier=lr_multiplier,
                )

        # Successful learning step (or no-target inference): reset counter.
        # Never modify the counter in eval_mode — eval is read-only for
        # training-mode state.
        if not eval_mode:
            self._consecutive_skipped_steps = 0

        self.last_curiosity = self._update_curiosity(inputs, loss_value)

        if not eval_mode:
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

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------
    def _update_working_memory(self, context: torch.Tensor | None) -> None:
        if context is None:
            return
        if context.shape[-1] != self.working_memory.wm_dim:
            # Map sensor-sized outputs into WM dim with a simple pad/slice.
            resized = self._resize_for_wm(context)
        else:
            resized = context
        wm_read = self.working_memory.read(resized)
        self.working_memory.write(resized.detach(), wm_read.detach())
        self.working_memory.step()

    def _resize_for_wm(self, vec: torch.Tensor) -> torch.Tensor:
        target = self.working_memory.wm_dim
        flat = vec.detach().flatten()
        if flat.numel() >= target:
            return flat[:target]
        padded = torch.zeros(target, device=flat.device, dtype=flat.dtype)
        padded[: flat.numel()] = flat
        return padded

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, float]:
        components: list[torch.Tensor] = []
        for modality, target in targets.items():
            out = outputs.get(modality)
            if out is None:
                continue
            components.append(F.mse_loss(out, target))
        if not components:
            zero = torch.zeros((), requires_grad=True)
            return zero, 0.0
        loss = torch.stack(components).mean()
        return loss, float(loss.item())

    def _encode_episodic(
        self,
        inputs: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None,
        loss_value: float | None,
    ) -> int | None:
        if not inputs:
            return None
        # Build a value vector = concat(first_input, first_target_or_output).
        first_input = next(iter(inputs.values())).detach().flatten()
        if targets is not None and targets:
            complement = next(iter(targets.values())).detach().flatten()
        elif outputs:
            complement = next(iter(outputs.values())).detach().flatten()
        else:
            return None

        vec = torch.cat([first_input, complement])
        if vec.numel() < self.episodic_memory.value_dim:
            pad = torch.zeros(
                self.episodic_memory.value_dim - vec.numel(),
                device=vec.device,
                dtype=vec.dtype,
            )
            vec = torch.cat([vec, pad])
        vec = vec[: self.episodic_memory.value_dim]

        surprise = loss_value if loss_value is not None else 0.0
        return self.episodic_memory.encode(
            experience=vec,
            prediction_error=float(surprise),
            current_step=self.global_step,
        )

    def _update_curiosity(
        self,
        inputs: dict[str, torch.Tensor],
        loss_value: float | None,
    ) -> float:
        if not inputs:
            return 0.0
        first = next(iter(inputs.values())).detach()
        if first.ndim > 1:
            first = first.mean(dim=0)
        if first.shape[-1] != self.curiosity.input_dim:
            first = self._resize_to(first, self.curiosity.input_dim)
        return self.curiosity.compute_curiosity(
            first,
            prediction_error=float(loss_value) if loss_value is not None else 0.0,
        )

    def _resize_to(self, vec: torch.Tensor, target_dim: int) -> torch.Tensor:
        flat = vec.detach().flatten()
        if flat.numel() >= target_dim:
            return flat[:target_dim]
        padded = torch.zeros(target_dim, device=flat.device, dtype=flat.dtype)
        padded[: flat.numel()] = flat
        return padded

    def record_growth_event(self, event_type: str, **fields: Any) -> None:
        """Append a structural event to ``growth_log``.

        ``step`` is stamped automatically from ``global_step``; extra
        ``fields`` are stored verbatim (kept JSON-friendly by convention
        so the journal round-trips through ``torch.save``).
        """
        self.growth_log.append(
            {
                "step": self.global_step,
                "event_type": event_type,
                **fields,
            }
        )

    def _maybe_grow(
        self,
        activations: dict[str, torch.Tensor],
        loss_value: float | None,
        *,
        rng: torch.Generator | None,
    ) -> None:
        config = self.config
        if self.global_step > 0:
            if (
                self.global_step % config.synaptogenesis_interval == 0
                and self.homeostasis.allow_synaptogenesis
            ):
                new_edges = synaptogenesis(
                    self.graph,
                    activations,
                    step=self.global_step,
                    config=config,
                    rng=rng,
                )
                for edge in new_edges:
                    self.record_growth_event(
                        "synaptogenesis",
                        edge_id=edge.id,
                        source=edge.source_id,
                        target=edge.target_id,
                    )
            if (
                self.global_step % config.neurogenesis_interval == 0
                and self.homeostasis.allow_neurogenesis
            ):
                new_node = neurogenesis(
                    self.graph,
                    self._recent_errors,
                    step=self.global_step,
                    config=config,
                    rng=rng,
                )
                if new_node is not None:
                    ratio = _neurogenesis_trigger_ratio(self._recent_errors)
                    self.record_growth_event(
                        "neurogenesis",
                        node_id=new_node.id,
                        node_type=new_node.node_type.name,
                        trigger_ratio=ratio,
                    )
            if self.global_step % config.pruning_interval == 0:
                # Snapshot edge/node metadata BEFORE pruning so we can attach
                # source/target/age/final_strength to prune events — these
                # fields are irretrievable once the edge is gone from the graph.
                edge_ctx = _snapshot_edges(self.graph, step=self.global_step)
                node_ctx = _snapshot_nodes(self.graph)
                result = pruning(self.graph, step=self.global_step, config=config)
                for edge_id in result.removed_edge_ids:
                    self.record_growth_event(
                        "prune_edge", edge_id=edge_id, **edge_ctx.get(edge_id, {})
                    )
                for node_id in result.removed_node_ids:
                    self.record_growth_event(
                        "prune_node", node_id=node_id, **node_ctx.get(node_id, {})
                    )

    def _maybe_consolidate(self, *, rng: torch.Generator | None) -> None:
        if (
            self.global_step > 0
            and self.global_step % self.config.consolidation_interval == 0
            and self.episodic_memory.num_valid > 0
        ):
            # Snapshot edge/node context before consolidation so prune events
            # during the cycle carry source/target/age the same way online
            # _maybe_grow events do.
            edge_ctx = _snapshot_edges(self.graph, step=self.global_step)
            node_ctx = _snapshot_nodes(self.graph)
            result = consolidation_cycle(
                self.graph,
                self.episodic_memory,
                current_step=self.global_step,
                config=self.config,
                experience_unpacker=self.experience_unpacker,
                rng=rng,
            )
            # Drain structural events from the cycle into the journal so
            # pruning / myelination / (opt-in) neurogenesis that happens
            # during artificial sleep shows up in the archaeology trace
            # the same way online _maybe_grow events do.
            for edge_id in result.removed_edge_ids:
                self.record_growth_event("prune_edge", edge_id=edge_id, **edge_ctx.get(edge_id, {}))
            for node_id in result.removed_node_ids:
                self.record_growth_event("prune_node", node_id=node_id, **node_ctx.get(node_id, {}))
            for new_node_id, chain_ids in zip(
                result.myelination_new_node_ids,
                result.myelination_chain_ids,
                strict=False,
            ):
                self.record_growth_event(
                    "myelinate",
                    new_node_id=new_node_id,
                    chain_node_ids=list(chain_ids),
                )
            if result.neurogenesis_node_id is not None:
                self.record_growth_event(
                    "neurogenesis",
                    node_id=result.neurogenesis_node_id,
                )

    # ------------------------------------------------------------------
    # Interactive helpers
    # ------------------------------------------------------------------
    def interactive_session(
        self,
        text_input: str,
        text_encoder: Callable[[str], torch.Tensor],
        text_decoder: Callable[[torch.Tensor], str],
        *,
        max_output_tokens: int | None = None,
    ) -> str:
        """Run a text-in / text-out interaction.

        Parameters
        ----------
        text_input:
            UTF-8 string.
        text_encoder:
            Callable returning a ``(T, embed_dim)`` tensor.
        text_decoder:
            Callable taking an ``(embed_dim,)`` tensor and returning a string.
        max_output_tokens:
            Optional cap; defaults to ``config.max_output_tokens``.
        """
        max_tokens = (
            max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        )
        encoded = text_encoder(text_input)
        if encoded.ndim != 2:
            raise ValueError(f"text_encoder must return (T, embed_dim), got {tuple(encoded.shape)}")

        pieces: list[str] = []
        last_output: torch.Tensor | None = None
        sensor_dim = self.graph.get_sensor("text").output_dim
        for step in range(max_tokens):
            # Feed one token at a time; if input is exhausted, loop the
            # last produced embedding back in (autoregressive).
            if step < encoded.shape[0]:
                vec = encoded[step]
            elif last_output is not None:
                vec = last_output
            else:
                break
            if vec.shape[-1] != sensor_dim:
                vec = self._resize_to(vec, sensor_dim)
            result = self.step(inputs={"text": vec})
            outputs = result["outputs"]
            if "text" not in outputs:
                continue
            last_output = outputs["text"]
            piece = text_decoder(last_output)
            if piece == "<EOS>":
                break
            pieces.append(piece)
        return "".join(pieces)

    # ------------------------------------------------------------------
    # Defaults / serialization
    # ------------------------------------------------------------------
    def _default_experience_unpacker(
        self,
        experience: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        half = experience.shape[-1] // 2
        first_in = self.config.input_modalities[0]
        first_out = self.config.output_modalities[0]
        sensor = self.graph.get_sensor(first_in)
        out = self.graph.get_output(first_out)
        input_part = self._resize_to(experience[:half], sensor.output_dim)
        target_part = self._resize_to(experience[half:], out.output_dim)
        return {first_in: input_part}, {first_out: target_part}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_state(self, path: str | Path) -> None:
        """Write a full checkpoint to ``path`` (``torch.save`` format).

        The file is wrapped in the versioned brain-bundle envelope
        (``format="soma-brain"``, ``schema_version``, SOMA/torch/tokenizers
        versions, git sha, timestamp) and CPU-normalized so the same file
        loads identically on cpu/cuda without a device prefix mismatch.
        """
        # Local import avoids any future circular-import issue if
        # brain_bundle ever needs to reach back into soma.system.
        from soma.core.brain_bundle import to_cpu_state, wrap_payload

        payload = {
            "global_step": self.global_step,
            "recent_errors": list(self._recent_errors),
            "graph": self.graph.serialize(),
            "working_memory": self.working_memory.state_dict(),
            "episodic_memory": self.episodic_memory.state_dict(),
            "curiosity_state_dict": self.curiosity.state_dict(),
            "curiosity_histories": self.curiosity.to_dict(),
            "homeostasis": self.homeostasis.state_dict(),
            "development": self.development.to_dict(),
            "config": self.config.to_dict(),
            "last_curiosity": self.last_curiosity,
            # Serialize as a plain list — deque reconstituted on load.
            "growth_log": list(self.growth_log),
        }
        payload = to_cpu_state(payload)
        wrapped = wrap_payload(payload, soma_version=_current_soma_version())
        torch.save(wrapped, str(path))

    def load_state(self, path: str | Path) -> None:
        """Load a checkpoint from ``path``.

        Accepts both the v1 envelope (``format="soma-brain"``) and the
        pre-envelope legacy format (loose dict, schema 0). In the legacy
        case the payload is upgraded through the migration dispatch to the
        current schema version.
        """
        from soma.core.brain_bundle import migrate_payload, unwrap_payload

        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and raw.get("format") == "soma-brain":
            state, meta = unwrap_payload(raw)
            from_schema = int(meta["schema_version"])
        else:
            # Legacy single-file format — treat the whole blob as payload, schema=0.
            state = dict(raw)
            from_schema = 0
        state, _ = migrate_payload(state, from_schema=from_schema)

        self.config = SOMAConfig.from_dict(state["config"])
        self.global_step = int(state["global_step"])
        self._recent_errors = [float(v) for v in state["recent_errors"]]
        self.graph = Graph.deserialize(state["graph"], config=self.config)
        # Graph.deserialize rebuilds Nodes/Edges on the default device (CPU);
        # move the freshly-built graph back onto SOMA's device so subsequent
        # steps don't cross cuda/cpu. The other modules below preserve their
        # own device via load_state_dict, but this one gets fully replaced.
        self.graph.to(self.device)
        self.working_memory.load_state_dict(state["working_memory"])
        self.episodic_memory.load_state_dict(state["episodic_memory"])
        self.curiosity.load_state_dict(state["curiosity_state_dict"])
        self.curiosity.load_histories(state["curiosity_histories"])
        self.homeostasis.load_state_dict(state["homeostasis"])
        # ``development`` was added in brain-bundle Task 7. Older checkpoints
        # omit it — keep the constructor's default schedule in that case so
        # the migrator doesn't have to invent a payload.
        if "development" in state:
            self.development.from_dict(state["development"])
        self.last_curiosity = float(state["last_curiosity"])
        # ``growth_log`` was added in brain-bundle Task 8. Older
        # checkpoints omit it — default to an empty journal so pre-Task-8
        # bundles load without error.
        self.growth_log = deque(state.get("growth_log", []), maxlen=10_000)

    # ------------------------------------------------------------------
    # Directory-shaped brain bundle (brain.pt + tokenizer + encoder + manifest)
    # ------------------------------------------------------------------
    def save_bundle(
        self,
        dir_path: str | Path,
        *,
        tokenizer: Any = None,
        encoder: Any = None,
        decoder: Any = None,
        verbalizer: Any = None,
        llm_identity: str | None = None,
    ) -> None:
        """Write a directory-shaped brain bundle to ``dir_path``.

        The bundle is a folder containing at minimum ``brain.pt`` (the
        usual versioned envelope from :meth:`save_state`) and
        ``manifest.json`` (schema + versions + vocab size). When a
        ``tokenizer`` is supplied it's saved to ``tokenizer.json`` and
        its ``vocab_size`` is stamped into the manifest. When an
        ``encoder`` / ``decoder`` is supplied, each is saved alongside
        as ``encoder.pt`` / ``decoder.pt`` with just enough metadata to
        reconstruct on load. When a ``verbalizer`` is supplied it is
        written to the ``verbalizer/`` subdir (spec.json + weights.pt)
        and its ``spec.llm_name`` is stamped into ``manifest.llm_identity``
        for swap-detection — an explicit ``llm_identity`` argument still
        wins if both are provided. The encoder / decoder / verbalizer
        arguments are all optional so callers that only want to persist
        the graph + tokenizer don't have to fabricate modules.
        """
        from soma.core.brain_bundle import write_manifest

        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)
        self.save_state(str(out / "brain.pt"))

        if tokenizer is not None:
            tokenizer.save(str(out / "tokenizer.json"))

        if encoder is not None:
            torch.save(
                {
                    "state_dict": encoder.state_dict(),
                    "embed_dim": encoder.embed_dim,
                    "max_seq_len": encoder.max_seq_len,
                },
                str(out / "encoder.pt"),
            )

        if decoder is not None:
            torch.save({"state_dict": decoder.state_dict()}, str(out / "decoder.pt"))

        if verbalizer is not None:
            verbalizer.save(out / "verbalizer")
            # Stamp LLM identity from the verbalizer spec unless the caller
            # passed an explicit llm_identity override.
            if llm_identity is None:
                llm_identity = verbalizer.spec.llm_name

        vocab_size = tokenizer.get_vocab_size() if tokenizer is not None else self.config.vocab_size
        interface_spec = {
            "sensor_by_modality": dict(self.graph._sensor_by_modality),
            "output_by_modality": dict(self.graph._output_by_modality),
            "sensor_output_dim": self.config.sensor_output_dim,
            "output_dim": self.config.integrator_output_dim,
            "text_embed_dim": self.config.text_embed_dim,
        }
        write_manifest(
            out,
            soma_version=_current_soma_version(),
            vocab_size=vocab_size,
            llm_identity=llm_identity,
            interface_spec=interface_spec,
        )

    def load_bundle(self, dir_path: str | Path) -> tuple[Any, Any, Any]:
        """Load a directory-shaped brain bundle from ``dir_path``.

        Returns ``(tokenizer, encoder, verbalizer)``; any of the three
        may be ``None`` if the corresponding sidecar isn't present.
        Decoder reconstruction is deferred — callers that need a
        decoder can load ``decoder.pt`` themselves since its shape is
        specific to the downstream head. The manifest's ``vocab_size``
        is cross-checked against the tokenizer to fail loud on mismatch.

        Note: The return type changed from 2-tuple to 3-tuple in Phase 2
        Task 8 to surface an optional verbalizer. Callers that previously
        did ``tok, enc = soma.load_bundle(...)`` must update to
        ``tok, enc, verb = soma.load_bundle(...)``.
        """
        from tokenizers import Tokenizer

        from soma.core.brain_bundle import read_manifest
        from soma.io.text_encoder import TextEncoder
        from soma.io.verbalizer import SomaVerbalizer

        src = Path(dir_path)
        manifest = read_manifest(src)
        self.load_state(str(src / "brain.pt"))

        tok: Any = None
        enc: Any = None
        verb: Any = None

        tokenizer_path = src / "tokenizer.json"
        if tokenizer_path.exists():
            tok = Tokenizer.from_file(str(tokenizer_path))
            if tok.get_vocab_size() != int(manifest["vocab_size"]):
                raise ValueError(
                    f"Tokenizer vocab {tok.get_vocab_size()} doesn't match manifest "
                    f"{manifest['vocab_size']}"
                )

        encoder_path = src / "encoder.pt"
        if encoder_path.exists() and tok is not None:
            blob = torch.load(str(encoder_path), map_location="cpu", weights_only=False)
            enc = TextEncoder(
                tok,
                embed_dim=int(blob["embed_dim"]),
                max_seq_len=int(blob["max_seq_len"]),
                device=self.device,
            )
            enc.load_state_dict(blob["state_dict"])

        verbalizer_dir = src / "verbalizer"
        if (verbalizer_dir / "spec.json").exists():
            verb = SomaVerbalizer.load(verbalizer_dir)

        return tok, enc, verb
