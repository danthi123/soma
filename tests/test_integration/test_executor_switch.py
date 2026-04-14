"""Integration parity test: SOMA.step with sequential vs batched executor.

Spins up two SOMA instances from the same small config + seed, syncs
their graphs bit-for-bit via ``copy.deepcopy``, runs one ``step`` through
each with identical inputs/targets, and asserts that the reported loss
agrees to ``atol=1e-5``. The two systems diverge only in the executor
they call during the forward pass; the backward + optimizer update that
follow operate on identical gradients produced by identical math.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.system import SOMA


def _tiny_config() -> SOMAConfig:
    return SOMAConfig(
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


def test_soma_step_with_batched_flag_matches_sequential() -> None:
    base_config = _tiny_config()
    soma_seq = SOMA(base_config, device=torch.device("cpu"))

    # Build the batched system from a deepcopy of the sequential one: the
    # graph (with all its UUIDs + weights + buffers) is byte-identical
    # before we run step. Simply toggling the config flag on the copy is
    # what the batched executor keys off of.
    soma_bat = copy.deepcopy(soma_seq)
    soma_bat.config = replace(base_config, use_batched_executor=True)

    inputs = {"text": torch.randn(base_config.sensor_output_dim)}
    targets = {"text": torch.zeros(base_config.sensor_output_dim)}
    r_seq = soma_seq.step(inputs=inputs, targets=targets)
    r_bat = soma_bat.step(inputs=inputs, targets=targets)

    # Losses match to float tolerance (the batched executor affects the
    # forward pass only; update_step runs identical math on identical
    # gradients).
    assert r_seq["loss"] == pytest.approx(r_bat["loss"], rel=1e-5, abs=1e-7)
