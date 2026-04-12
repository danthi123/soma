"""Tests for ``soma.core.edge.Edge``."""

from __future__ import annotations

import pytest
import torch

from soma.core.edge import Edge


class TestConstruction:
    def test_defaults(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
        )
        assert edge.source_id == "src"
        assert edge.target_id == "tgt"
        assert edge.source_output_dim == 4
        assert edge.target_input_dim == 4
        assert edge.creation_step == 0
        assert edge.last_active_step == 0
        assert edge.coactivation_count == 0
        assert edge.strength == pytest.approx(0.01)
        assert edge.projection is None  # same dims -> no projection
        assert float(edge.weight.item()) == pytest.approx(0.01)
        assert edge.weight.requires_grad

    def test_projection_created_when_dims_differ(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=8,
            creation_step=0,
        )
        assert edge.projection is not None
        assert edge.projection.in_features == 4
        assert edge.projection.out_features == 8

    def test_rejects_self_loop(self) -> None:
        with pytest.raises(ValueError, match="self-loop"):
            Edge(
                source_id="same",
                target_id="same",
                source_output_dim=4,
                target_input_dim=4,
                creation_step=0,
            )

    def test_rejects_empty_ids(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Edge(
                source_id="",
                target_id="tgt",
                source_output_dim=4,
                target_input_dim=4,
                creation_step=0,
            )

    def test_rejects_negative_creation_step(self) -> None:
        with pytest.raises(ValueError, match="creation_step"):
            Edge(
                source_id="src",
                target_id="tgt",
                source_output_dim=4,
                target_input_dim=4,
                creation_step=-1,
            )

    def test_id_generated_when_missing(self) -> None:
        e1 = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
        )
        e2 = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
        )
        assert e1.id != e2.id


class TestTransmit:
    def test_same_dim_passthrough(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
            initial_weight=2.0,
        )
        x = torch.tensor([1.0, -1.0, 0.5, 0.0])
        y = edge.transmit(x)
        assert y.shape == (4,)
        assert torch.allclose(y, x * 2.0)

    def test_projection_applied(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=6,
            creation_step=0,
            initial_weight=1.0,
        )
        assert edge.projection is not None
        x = torch.randn(4)
        y = edge.transmit(x)
        assert y.shape == (6,)
        # Should equal projection(x) * weight == projection(x) * 1.0
        assert torch.allclose(y, edge.projection(x))

    def test_forward_is_transmit(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=3,
            target_input_dim=3,
            creation_step=0,
        )
        x = torch.randn(3)
        assert torch.allclose(edge(x), edge.transmit(x))

    def test_rejects_mismatched_source_dim(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
        )
        with pytest.raises(ValueError, match="source last-dim"):
            edge.transmit(torch.randn(5))

    def test_transmit_flows_gradients(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
            initial_weight=0.5,
        )
        x = torch.randn(4, requires_grad=False)
        y = edge.transmit(x)
        loss = y.sum()
        loss.backward()
        assert edge.weight.grad is not None
        # d/dw sum(w * x) = sum(x)
        assert float(edge.weight.grad.item()) == pytest.approx(float(x.sum().item()), abs=1e-5)


class TestStatUpdates:
    def test_mark_active(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
        )
        edge.mark_active(42)
        assert edge.last_active_step == 42

    def test_increment_coactivation(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
        )
        edge.increment_coactivation()
        edge.increment_coactivation()
        assert edge.coactivation_count == 2

    def test_update_strength_ema(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
            initial_weight=2.0,
        )
        # strength seeded at 0.01 (== initial_weight=0.01 default), but here
        # we set initial_weight=2.0 so strength should seed at 2.0.
        assert edge.strength == pytest.approx(2.0)
        edge.update_strength(signal_magnitude=3.0, decay=0.5)
        # 0.5 * 2.0 + 0.5 * |2.0 * 3.0| = 1.0 + 3.0 = 4.0
        assert edge.strength == pytest.approx(4.0)

    def test_update_strength_rejects_bad_decay(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
        )
        with pytest.raises(ValueError, match="decay"):
            edge.update_strength(1.0, decay=0.0)
        with pytest.raises(ValueError, match="decay"):
            edge.update_strength(1.0, decay=1.5)

    def test_clamp_weight(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
            initial_weight=7.5,
        )
        edge.clamp_weight(max_abs=5.0)
        assert float(edge.weight.item()) == pytest.approx(5.0)
        with torch.no_grad():
            edge.weight.fill_(-9.0)
        edge.clamp_weight(max_abs=5.0)
        assert float(edge.weight.item()) == pytest.approx(-5.0)

    def test_clamp_weight_rejects_non_positive(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=4,
            creation_step=0,
        )
        with pytest.raises(ValueError, match="positive"):
            edge.clamp_weight(0.0)


class TestSerialization:
    def test_round_trip(self) -> None:
        edge = Edge(
            source_id="src",
            target_id="tgt",
            source_output_dim=4,
            target_input_dim=6,
            creation_step=3,
            initial_weight=0.25,
        )
        edge.mark_active(10)
        edge.increment_coactivation()
        edge.update_strength(1.0, decay=0.9)
        snapshot = edge.to_dict()

        rebuilt = Edge(
            source_id="placeholder_src",
            target_id="placeholder_tgt",
            source_output_dim=4,
            target_input_dim=6,
            creation_step=0,
            initial_weight=0.0,
        )
        rebuilt.load_scalar_state(snapshot)
        rebuilt.load_state_dict(edge.state_dict())

        assert rebuilt.id == edge.id
        assert rebuilt.source_id == edge.source_id
        assert rebuilt.target_id == edge.target_id
        assert rebuilt.last_active_step == edge.last_active_step
        assert rebuilt.coactivation_count == edge.coactivation_count
        assert rebuilt.strength == pytest.approx(edge.strength)
        assert float(rebuilt.weight.item()) == pytest.approx(float(edge.weight.item()))
        assert rebuilt.projection is not None
        assert torch.equal(rebuilt.projection.weight, edge.projection.weight)
