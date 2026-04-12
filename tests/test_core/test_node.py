"""Tests for ``soma.core.node``."""

from __future__ import annotations

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.core.node import Node, NodeType


@pytest.fixture
def config() -> SOMAConfig:
    return SOMAConfig()


class TestNodeType:
    def test_is_boundary(self) -> None:
        assert NodeType.SENSOR.is_boundary
        assert NodeType.OUTPUT.is_boundary
        assert not NodeType.ASSOCIATOR.is_boundary
        assert not NodeType.INTEGRATOR.is_boundary


class TestConstruction:
    def test_defaults_from_make(self, config: SOMAConfig) -> None:
        node = Node.make(NodeType.ASSOCIATOR, config, creation_step=0)
        assert node.node_type is NodeType.ASSOCIATOR
        assert node.input_dim == config.associator_input_dim
        assert node.hidden_dim == config.associator_hidden_dim
        assert node.output_dim == config.associator_output_dim
        assert node.maturity == pytest.approx(0.0)
        assert node.gain == pytest.approx(1.0)
        assert node.target_activation == pytest.approx(config.default_target_activation)
        assert node.creation_step == 0
        assert node.last_active_step == 0
        assert node.activation_ema == pytest.approx(0.0)
        assert len(node.id) == 32
        assert node.position.shape == (config.position_dim,)

    def test_explicit_dims(self, config: SOMAConfig) -> None:
        node = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=10,
            hidden_dim=20,
            output_dim=10,
            creation_step=5,
            config=config,
        )
        assert node.input_dim == 10
        assert node.hidden_dim == 20
        assert node.output_dim == 10
        assert node.creation_step == 5

    def test_rejects_bad_dims(self, config: SOMAConfig) -> None:
        with pytest.raises(ValueError, match="positive dims"):
            Node(
                node_type=NodeType.ASSOCIATOR,
                input_dim=0,
                hidden_dim=8,
                output_dim=4,
                creation_step=0,
                config=config,
            )

    def test_rejects_negative_creation_step(self, config: SOMAConfig) -> None:
        with pytest.raises(ValueError, match="creation_step"):
            Node(
                node_type=NodeType.ASSOCIATOR,
                input_dim=4,
                hidden_dim=8,
                output_dim=4,
                creation_step=-1,
                config=config,
            )

    def test_rejects_bad_maturity(self, config: SOMAConfig) -> None:
        with pytest.raises(ValueError, match="maturity"):
            Node(
                node_type=NodeType.ASSOCIATOR,
                input_dim=4,
                hidden_dim=8,
                output_dim=4,
                creation_step=0,
                config=config,
                maturity=1.5,
            )

    def test_rejects_gain_out_of_range(self, config: SOMAConfig) -> None:
        with pytest.raises(ValueError, match="gain"):
            Node(
                node_type=NodeType.ASSOCIATOR,
                input_dim=4,
                hidden_dim=8,
                output_dim=4,
                creation_step=0,
                config=config,
                gain=100.0,  # > gain_max
            )


class TestForward:
    def test_forward_with_matching_dims(self, config: SOMAConfig) -> None:
        node = Node.make(NodeType.ASSOCIATOR, config, creation_step=0)
        x = torch.randn(node.input_dim)
        y = node.forward({"src-1": x}, current_step=1)
        assert y.shape == (node.output_dim,)
        assert torch.isfinite(y).all()

    def test_forward_aggregates_multiple_inputs(self, config: SOMAConfig) -> None:
        node = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=4,
            hidden_dim=8,
            output_dim=4,
            creation_step=0,
            config=config,
        )
        x1 = torch.ones(4)
        x2 = torch.ones(4)
        y_both = node.forward({"a": x1, "b": x2}, current_step=1)
        y_single = node.forward({"a": x1 + x2}, current_step=2)
        # Summed inputs should yield the same output as a single combined input
        # (up to residual / EMA updates, which don't affect the deterministic MLP).
        assert torch.allclose(y_both, y_single, atol=1e-6)

    def test_forward_empty_inputs_returns_zero_vector(self, config: SOMAConfig) -> None:
        node = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=4,
            hidden_dim=8,
            output_dim=4,
            creation_step=0,
            config=config,
        )
        # Zero input -> linear1(0) = b1, then linear2 then residual add zero.
        y = node.forward({}, current_step=1)
        assert y.shape == (4,)
        # Nothing should blow up on empty inputs.
        assert torch.isfinite(y).all()

    def test_forward_updates_activation_state(self, config: SOMAConfig) -> None:
        node = Node.make(NodeType.ASSOCIATOR, config, creation_step=0)
        x = torch.randn(node.input_dim) * 5.0  # big signal
        y = node.forward({"src": x}, current_step=10)
        assert len(node.activation_history) == 1
        assert node.activation_ema > 0.0
        expected_mag = float(y.norm().item())
        # EMA after one step: 0.99 * 0 + 0.01 * mag.
        assert node.activation_ema == pytest.approx(0.01 * expected_mag, rel=1e-4)
        if expected_mag > config.activation_threshold:
            assert node.last_active_step == 10
        else:
            assert node.last_active_step == 0  # unchanged

    def test_forward_rejects_mismatched_input_dim(self, config: SOMAConfig) -> None:
        node = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=4,
            hidden_dim=8,
            output_dim=4,
            creation_step=0,
            config=config,
        )
        with pytest.raises(ValueError, match="last-dim"):
            node.forward({"src": torch.randn(7)}, current_step=0)

    def test_residual_applied_only_when_dims_match(self, config: SOMAConfig) -> None:
        # Same-dim path: residual adds x.
        same_dim = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=4,
            hidden_dim=8,
            output_dim=4,
            creation_step=0,
            config=config,
        )
        # Zero out linear layers to isolate residual.
        with torch.no_grad():
            same_dim.linear1.weight.zero_()
            same_dim.linear1.bias.zero_()
            same_dim.linear2.weight.zero_()
            same_dim.linear2.bias.zero_()
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = same_dim.forward({"src": x}, current_step=0)
        # linear output is zero, gain*0 = 0, residual adds x -> y == x.
        assert torch.allclose(y, x)

        # Different-dim node: no residual.
        diff_dim = Node(
            node_type=NodeType.INTEGRATOR,
            input_dim=4,
            hidden_dim=8,
            output_dim=6,
            creation_step=0,
            config=config,
        )
        with torch.no_grad():
            diff_dim.linear1.weight.zero_()
            diff_dim.linear1.bias.zero_()
            diff_dim.linear2.weight.zero_()
            diff_dim.linear2.bias.zero_()
        y = diff_dim.forward({"src": torch.tensor([1.0, 2.0, 3.0, 4.0])}, current_step=0)
        assert torch.allclose(y, torch.zeros(6))


class TestSensor:
    def test_set_and_get_input(self, config: SOMAConfig) -> None:
        sensor = Node.make(NodeType.SENSOR, config, creation_step=0)
        data = torch.randn(config.sensor_output_dim)
        sensor.set_input(data)
        retrieved = sensor.get_input_activation()
        assert torch.equal(retrieved, data)

    def test_forward_uses_sensor_input(self, config: SOMAConfig) -> None:
        sensor = Node.make(NodeType.SENSOR, config, creation_step=0)
        data = torch.randn(config.sensor_output_dim)
        sensor.set_input(data)
        y = sensor.forward({"ignored": torch.zeros(10)}, current_step=3)
        assert torch.equal(y, data)
        assert len(sensor.activation_history) == 1

    def test_sensor_without_input_returns_zero(self, config: SOMAConfig) -> None:
        sensor = Node.make(NodeType.SENSOR, config, creation_step=0)
        y = sensor.forward({}, current_step=1)
        assert torch.equal(y, torch.zeros(config.sensor_output_dim))

    def test_set_input_rejected_on_non_sensor(self, config: SOMAConfig) -> None:
        node = Node.make(NodeType.ASSOCIATOR, config, creation_step=0)
        with pytest.raises(RuntimeError, match="SENSOR"):
            node.set_input(torch.zeros(4))

    def test_set_input_checks_shape(self, config: SOMAConfig) -> None:
        sensor = Node.make(NodeType.SENSOR, config, creation_step=0)
        with pytest.raises(ValueError, match="last-dim"):
            sensor.set_input(torch.zeros(config.sensor_output_dim + 1))

    def test_clear_input(self, config: SOMAConfig) -> None:
        sensor = Node.make(NodeType.SENSOR, config, creation_step=0)
        sensor.set_input(torch.randn(config.sensor_output_dim))
        sensor.clear_input()
        y = sensor.forward({}, current_step=1)
        assert torch.equal(y, torch.zeros(config.sensor_output_dim))


class TestHomeostasis:
    def test_clamp_gain(self, config: SOMAConfig) -> None:
        node = Node.make(NodeType.ASSOCIATOR, config, creation_step=0)
        node.gain = 0.05  # below gain_min=0.1
        node.clamp_gain()
        assert node.gain == pytest.approx(config.gain_min)
        node.gain = 50.0  # above gain_max=10.0
        node.clamp_gain()
        assert node.gain == pytest.approx(config.gain_max)

    def test_advance_maturity(self, config: SOMAConfig) -> None:
        node = Node.make(NodeType.ASSOCIATOR, config, creation_step=0)
        node.advance_maturity(0.25)
        assert node.maturity == pytest.approx(0.25)
        node.advance_maturity(0.9)
        assert node.maturity == pytest.approx(1.0)  # clamped
        with pytest.raises(ValueError, match="non-negative"):
            node.advance_maturity(-0.1)


class TestSerialization:
    def test_to_dict_and_reload(self, config: SOMAConfig) -> None:
        node = Node.make(NodeType.ASSOCIATOR, config, creation_step=42)
        # Mutate some state so we're testing non-default values.
        node.maturity = 0.7
        node.gain = 2.0
        node.activation_ema = 0.35
        node.last_active_step = 100
        node.activation_history.extend([0.1, 0.2, 0.3])
        state = node.to_dict()

        rebuilt = Node(
            node_type=NodeType.ASSOCIATOR,
            input_dim=node.input_dim,
            hidden_dim=node.hidden_dim,
            output_dim=node.output_dim,
            creation_step=0,  # will be overwritten by load
            config=config,
        )
        rebuilt.load_scalar_state(state)
        rebuilt.load_state_dict(node.state_dict())

        assert rebuilt.id == node.id
        assert rebuilt.maturity == pytest.approx(0.7)
        assert rebuilt.gain == pytest.approx(2.0)
        assert rebuilt.activation_ema == pytest.approx(0.35)
        assert rebuilt.last_active_step == 100
        assert rebuilt.activation_history.get_all() == [0.1, 0.2, 0.3]
        # Learnable params match.
        assert torch.equal(rebuilt.linear1.weight, node.linear1.weight)
        assert torch.equal(rebuilt.linear2.bias, node.linear2.bias)
        assert torch.equal(rebuilt.position, node.position)

    def test_to_dict_contains_node_type(self, config: SOMAConfig) -> None:
        node = Node.make(NodeType.OUTPUT, config, creation_step=0)
        state = node.to_dict()
        assert state["node_type"] == "output"
