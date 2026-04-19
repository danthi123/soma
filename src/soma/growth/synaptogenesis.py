"""Synaptogenesis: create new edges between co-activated nodes.

Whitepaper Section 5.1. Called periodically (every
``config.synaptogenesis_interval`` steps). For each pair of nodes that
were both active this step, we compute a connection probability based on:

- Coactivation strength (product of activation magnitudes).
- Locality bonus (``exp(-distance / locality_scale)``) — nearby nodes
  in position space are more likely to wire together.
- Global ``synaptogenesis_rate`` scaling factor.

Edges are always directed; we consider both ``a -> b`` and ``b -> a`` as
independent candidates. An edge is only proposed if one doesn't already
exist in that direction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch

from soma.core.config import SOMAConfig
from soma.core.edge import Edge
from soma.core.graph import Graph
from soma.core.utils import generate_uuid


def synaptogenesis(
    graph: Graph,
    activations: Mapping[str, torch.Tensor],
    step: int,
    config: SOMAConfig,
    *,
    rng: torch.Generator | None = None,
    pe_ema: Mapping[tuple[str, str], float] | None = None,
    pe_counts: Mapping[tuple[str, str], int] | None = None,
    rate_scale: float = 1.0,
) -> list[Edge]:
    """Propose and add new edges between co-active nodes.

    Returns the list of newly-created ``Edge`` instances (may be empty).
    Mutation order within the call is deterministic given ``rng``.

    When ``config.synaptogenesis_supervision == "pe_conditional"``, a
    pair is only admitted if the running EMA of (loss[t] - loss[t-1])
    observed when the pair was co-active satisfies both:

    - ``pe_counts[key] >= config.synaptogenesis_pe_min_observations``
      (enough evidence to act on), AND
    - ``pe_ema[key] < config.synaptogenesis_pe_threshold`` (the pair's
      co-activation has historically preceded PE reduction).

    ``key`` is the sorted tuple ``(min(a, b), max(a, b))`` — co-activation
    is symmetric, so both directional candidates share one evidence slot.
    Missing/None EMA or counts are treated as universal cold start and
    block all admissions under the gate. When supervision is ``"none"``
    the kwargs are ignored even if supplied, matching legacy behavior.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")

    threshold = config.activation_threshold
    rate = config.synaptogenesis_rate * max(0.0, float(rate_scale))
    locality = config.locality_scale
    supervision_on = config.synaptogenesis_supervision == "pe_conditional"
    if supervision_on:
        ema_map = pe_ema if pe_ema is not None else {}
        count_map = pe_counts if pe_counts is not None else {}
        min_obs = config.synaptogenesis_pe_min_observations
        pe_threshold = config.synaptogenesis_pe_threshold
        new_node_grace = config.synaptogenesis_supervision_new_node_grace
    else:
        ema_map = {}
        count_map = {}
        min_obs = 0
        pe_threshold = 0.0
        new_node_grace = 0

    active_ids: list[str] = []
    magnitudes: dict[str, float] = {}
    for nid, act in activations.items():
        # RMS magnitude (per-channel scale) rather than raw L2 norm, so
        # coactivation doesn't grow linearly with dim. A 64-dim tensor
        # at unit per-channel magnitude has norm=8 which would make
        # coact=64 and drive synaptogenesis prob to 1 for nearly every
        # pair regardless of rate. RMS keeps coact ~1 for unit signals.
        tensor = act.detach()
        numel = tensor.numel()
        mag = float(tensor.norm().item()) / math.sqrt(numel) if numel > 0 else 0.0
        if mag > threshold and nid in graph.nodes:
            active_ids.append(nid)
            magnitudes[nid] = mag

    if len(active_ids) < 2:
        return []

    # Pre-compute positions as CPU float32 tensors for distance math.
    positions: dict[str, torch.Tensor] = {
        nid: _to_cpu_float(cast(torch.Tensor, graph.nodes[nid].position)) for nid in active_ids
    }

    new_edges: list[Edge] = []
    # Iterate both directional pairs (a -> b and b -> a are sampled
    # independently since edges are directed).
    for source_id in active_ids:
        source_node = graph.nodes[source_id]
        for target_id in active_ids:
            if source_id == target_id:
                continue
            if graph.has_edge(source_id, target_id):
                continue
            target_node = graph.nodes[target_id]

            # PE-supervised gate (Direction 1): filter out pairs whose
            # co-activation has no history of reducing prediction error.
            # The gate runs BEFORE the coact / locality / probability
            # computation so filtered pairs never consume rng draws —
            # keeping determinism of the legacy path intact on
            # supervision='none' while pruning wasted work under
            # 'pe_conditional'.
            if supervision_on:
                pair_key = (source_id, target_id) if source_id <= target_id else (
                    target_id,
                    source_id,
                )
                # New-node waiver: if either endpoint was created within
                # the last ``new_node_grace`` steps, the pair is in its
                # "honeymoon" window. We waive the cold-start
                # (min_observations) check for such pairs so fresh
                # neurogenesis nodes can wire into the graph without
                # waiting to accumulate their own EMA evidence. Fixes
                # the full_pe regression from Phase 1 where
                # neurogenesis-added nodes couldn't wire via synap for
                # 5 observations each.
                #
                # Important: if a fresh pair somehow HAS accumulated
                # enough observations (count >= min_obs), the EMA check
                # still applies — the waiver is specifically about the
                # cold-start phase, not about overriding evidence once
                # we have it. In practice fresh-node pairs will almost
                # always be cold-start, so this branch does what you'd
                # expect.
                # Only treat nodes as "fresh" if they were created by
                # neurogenesis (creation_step > 0). Initial seed nodes
                # have creation_step=0 by convention; applying the
                # waiver to them would disable supervision for every
                # pair during the first ``new_node_grace`` steps — a
                # critical warm-up window where the EMA is trying to
                # accumulate its first observations. Empirically
                # confirmed in the waiver rerun: applying the waiver
                # to initial nodes regressed synap_only_pe on all 8
                # v0.5 regimes.
                src_fresh = (
                    new_node_grace > 0
                    and source_node.creation_step > 0
                    and (step - source_node.creation_step) < new_node_grace
                )
                tgt_fresh = (
                    new_node_grace > 0
                    and target_node.creation_step > 0
                    and (step - target_node.creation_step) < new_node_grace
                )
                is_fresh = src_fresh or tgt_fresh
                pair_count = int(count_map.get(pair_key, 0))
                is_cold_start = pair_count < min_obs
                if is_cold_start:
                    if not is_fresh:
                        continue  # normal cold-start reject
                    # Fresh cold-start pair: waive both checks, fall
                    # through to the rng draw. EMA is zero by default
                    # (unreliable anyway on 0 observations) so the
                    # threshold check would wrongly reject on any
                    # threshold >= 0.
                else:
                    pair_ema = float(ema_map.get(pair_key, 0.0))
                    if pair_ema >= pe_threshold:
                        continue

            source_mag = magnitudes[source_id]
            target_mag = magnitudes[target_id]
            coact = source_mag * target_mag

            dist = float((positions[source_id] - positions[target_id]).norm().item())
            # Hard positional-locality filter (2026-04-19): when enabled,
            # reject pairs whose distance exceeds the cutoff. Tests the
            # hypothesis (derived from why_neuro_only_works.md) that
            # neurogenesis's positional-neighbor wiring is the key to
            # its reliable multi-seed benefit on v0.5; applying the
            # same discipline to synap should make synap's admissions
            # useful instead of arbitrary.
            if config.synaptogenesis_max_distance > 0.0 and (
                dist > config.synaptogenesis_max_distance
            ):
                continue
            locality_bonus = float(torch.exp(torch.tensor(-dist / locality)).item())

            # Clamp probability to [0, 1]. Without this, when activations
            # are large (magnitude >> 1 during instability or spikes),
            # coact * rate easily exceeds 1 and every random draw
            # succeeds — creating edges between every co-active pair and
            # driving a positive-feedback loop: more edges -> bigger
            # activations -> more edges. Seen during Apr 13 run where
            # a single synaptogenesis event added 200+ edges mid-spike.
            prob = min(1.0, coact * locality_bonus * rate)
            if prob <= 0.0:
                continue

            draw = float(torch.rand(1, generator=rng).item())
            if draw >= prob:
                continue

            # Build and insert the new edge.
            init_weight = 0.01 * float(torch.randn(1, generator=rng).item())
            edge = Edge(
                source_id=source_id,
                target_id=target_id,
                source_output_dim=source_node.output_dim,
                target_input_dim=target_node.input_dim,
                creation_step=step,
                initial_weight=init_weight,
                edge_id=generate_uuid(),
            )
            graph.add_edge(edge)
            edge.last_active_step = step
            new_edges.append(edge)

    return new_edges


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    """Detach -> CPU -> float32 -> flatten. Stable for distance math."""
    detached = tensor.detach()
    cpu = detached.cpu()
    flat = cpu.reshape(-1)
    return flat.to(torch.float32)
