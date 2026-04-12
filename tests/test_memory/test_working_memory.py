"""Tests for ``soma.memory.working_memory.WorkingMemory``."""

from __future__ import annotations

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.memory.working_memory import WorkingMemory


class TestConstruction:
    def test_defaults(self) -> None:
        wm = WorkingMemory()
        assert wm.num_slots == 32
        assert wm.wm_dim == 128
        assert wm.decay_rate == pytest.approx(0.95)
        assert wm._slots().shape == (32, 128)
        assert wm._usage().shape == (32,)
        assert wm._age().shape == (32,)

    def test_from_config(self) -> None:
        config = SOMAConfig(wm_slots=16, wm_dim=64, wm_decay_rate=0.9)
        wm = WorkingMemory.from_config(config)
        assert wm.num_slots == 16
        assert wm.wm_dim == 64
        assert wm.decay_rate == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "num_slots, wm_dim, decay_rate, match",
        [
            (0, 128, 0.95, "num_slots"),
            (32, 0, 0.95, "wm_dim"),
            (32, 128, 0.0, "decay_rate"),
            (32, 128, 1.5, "decay_rate"),
        ],
    )
    def test_validation(self, num_slots: int, wm_dim: int, decay_rate: float, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            WorkingMemory(num_slots=num_slots, wm_dim=wm_dim, decay_rate=decay_rate)

    def test_learnable_params(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        params = dict(wm.named_parameters())
        # query_proj, write_gate, erase_gate each have weight + bias.
        assert "query_proj.weight" in params
        assert "query_proj.bias" in params
        assert "write_gate.weight" in params
        assert "erase_gate.weight" in params


class TestWrite:
    def test_write_when_gate_open(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        # Force the write gate to always output a high value.
        with torch.no_grad():
            wm.write_gate.weight.zero_()
            wm.write_gate.bias.fill_(5.0)  # sigmoid(5) ~= 0.993

        content = torch.randn(8)
        context = torch.randn(8)
        wrote = wm.write(content, context)
        assert wrote is True
        # Some slot should now contain content.
        matches = [torch.allclose(wm._slots()[i], content) for i in range(wm.num_slots)]
        assert any(matches)

    def test_no_write_when_gate_closed(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        with torch.no_grad():
            wm.write_gate.weight.zero_()
            wm.write_gate.bias.fill_(-5.0)  # sigmoid(-5) ~= 0.007
        wrote = wm.write(torch.randn(8), torch.randn(8))
        assert wrote is False
        # Slots still empty.
        assert torch.allclose(wm._slots(), torch.zeros_like(wm._slots()))

    def test_write_picks_least_used_slot(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        with torch.no_grad():
            wm.write_gate.weight.zero_()
            wm.write_gate.bias.fill_(5.0)
            # Pre-fill slot 0,1,2 with high usage; slot 3 is least-used.
            wm._usage()[0] = 0.9
            wm._usage()[1] = 0.8
            wm._usage()[2] = 0.7
            wm._usage()[3] = 0.0
        content = torch.arange(8, dtype=torch.float32)
        wm.write(content, torch.zeros(8))
        assert torch.allclose(wm._slots()[3], content)
        assert wm._usage()[3] == pytest.approx(1.0)
        assert wm._age()[3].item() == 0

    def test_write_shape_mismatch_raises(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        with pytest.raises(ValueError, match="shape"):
            wm.write(torch.randn(7), torch.randn(8))


class TestRead:
    def test_read_shape(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        out = wm.read(torch.randn(8))
        assert out.shape == (8,)

    def test_read_is_linear_combination_of_slots(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        # Zero-init query_proj so query maps to zero -> softmax is uniform
        # over slots -> read is the mean of slot contents.
        with torch.no_grad():
            wm.query_proj.weight.zero_()
            wm.query_proj.bias.zero_()
            for i in range(4):
                wm._slots()[i] = torch.ones(8) * float(i)
        out = wm.read(torch.randn(8))
        expected_mean = (0 + 1 + 2 + 3) / 4.0
        assert torch.allclose(out, torch.full((8,), expected_mean), atol=1e-5)


class TestStep:
    def test_usage_decays(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8, decay_rate=0.5)
        with torch.no_grad():
            wm._usage().fill_(1.0)
        wm.step()
        assert torch.allclose(wm._usage(), torch.ones(4) * 0.5)

    def test_age_increments(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        wm.step()
        wm.step()
        assert torch.equal(wm._age(), torch.full((4,), 2, dtype=torch.long))

    def test_fade_on_low_usage(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8, fade_threshold=0.2, fade_factor=0.5)
        with torch.no_grad():
            wm._slots().fill_(1.0)
            wm._usage().fill_(0.0)  # well below threshold -> fade kicks in
        wm.step()
        assert torch.allclose(wm._slots(), torch.full((4, 8), 0.5))

    def test_no_fade_when_above_threshold(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8, fade_threshold=0.2)
        with torch.no_grad():
            wm._slots().fill_(1.0)
            wm._usage().fill_(0.5)  # above threshold -> slot stays intact
        wm.step()
        assert torch.allclose(wm._slots(), torch.ones(4, 8))


class TestIntrospection:
    def test_clear(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8)
        with torch.no_grad():
            wm._slots().fill_(7.0)
            wm._usage().fill_(1.0)
            wm._age().fill_(3)
        wm.clear()
        assert torch.allclose(wm._slots(), torch.zeros(4, 8))
        assert torch.allclose(wm._usage(), torch.zeros(4))
        assert torch.equal(wm._age(), torch.zeros(4, dtype=torch.long))

    def test_occupancy(self) -> None:
        wm = WorkingMemory(num_slots=4, wm_dim=8, fade_threshold=0.2)
        with torch.no_grad():
            wm._usage()[0] = 0.9
            wm._usage()[1] = 0.0
            wm._usage()[2] = 0.5
            wm._usage()[3] = 0.1
        assert wm.occupancy() == pytest.approx(0.5)  # 2 of 4 slots occupied


class TestSerialization:
    def test_state_dict_round_trip(self) -> None:
        wm1 = WorkingMemory(num_slots=4, wm_dim=8)
        with torch.no_grad():
            wm1._slots().fill_(2.0)
            wm1._usage().fill_(0.5)
            wm1._age().fill_(7)
            wm1.write_gate.bias.fill_(1.5)
        state = wm1.state_dict()

        wm2 = WorkingMemory(num_slots=4, wm_dim=8)
        wm2.load_state_dict(state)
        assert torch.allclose(wm2._slots(), wm1._slots())
        assert torch.allclose(wm2._usage(), wm1._usage())
        assert torch.equal(wm2._age(), wm1._age())
        assert torch.allclose(wm2.write_gate.bias, wm1.write_gate.bias)
