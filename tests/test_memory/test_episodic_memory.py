"""Tests for ``soma.memory.episodic_memory.EpisodicMemory``."""

from __future__ import annotations

import pytest
import torch

from soma.core.config import SOMAConfig
from soma.memory.episodic_memory import EpisodicMemory, Retrieval


class TestConstruction:
    def test_defaults(self) -> None:
        em = EpisodicMemory()
        assert em.capacity == 10_000
        assert em.key_dim == 128
        assert em.value_dim == 256
        assert em.num_valid == 0

    def test_from_config(self) -> None:
        config = SOMAConfig(episodic_capacity=64, key_dim=16, value_dim=32)
        em = EpisodicMemory.from_config(config)
        assert em.capacity == 64
        assert em.key_dim == 16
        assert em.value_dim == 32

    @pytest.mark.parametrize(
        "capacity, key_dim, value_dim, age_scale, match",
        [
            (0, 8, 16, 2000.0, "capacity"),
            (4, 0, 16, 2000.0, "key_dim"),
            (4, 8, -1, 2000.0, "key_dim"),
            (4, 8, 16, 0.0, "age_scale"),
        ],
    )
    def test_validation(
        self,
        capacity: int,
        key_dim: int,
        value_dim: int,
        age_scale: float,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            EpisodicMemory(
                capacity=capacity,
                key_dim=key_dim,
                value_dim=value_dim,
                age_scale=age_scale,
            )


class TestEncode:
    def test_basic_encode(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=8, value_dim=16)
        experience = torch.randn(16)
        idx = em.encode(experience, prediction_error=0.5, current_step=10)
        assert idx == 0
        assert em.num_valid == 1
        assert torch.allclose(em._values()[0], experience)
        assert em._surprise()[0].item() == pytest.approx(0.5)
        assert em._timestamps()[0].item() == 10

    def test_wraps_around_write_head(self) -> None:
        em = EpisodicMemory(capacity=2, key_dim=8, value_dim=16)
        em.encode(torch.randn(16), 0.1, 1)
        em.encode(torch.randn(16), 0.2, 2)
        # Third encode wraps to index 0.
        replacement = torch.ones(16) * 3.0
        idx = em.encode(replacement, 0.3, 3)
        assert idx == 0
        assert torch.allclose(em._values()[0], replacement)
        assert em.num_valid == 2

    def test_shape_check(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=8, value_dim=16)
        with pytest.raises(ValueError, match="value shape"):
            em.encode(torch.randn(8), 0.0, 1)

    def test_rejects_negative_step(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=8, value_dim=16)
        with pytest.raises(ValueError, match="current_step"):
            em.encode(torch.randn(16), 0.0, -1)


class TestRetrieve:
    def test_empty_returns_empty(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=8, value_dim=16)
        results = em.retrieve(torch.randn(8))
        assert results == []

    def test_returns_most_similar(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        # Manually populate with known keys so similarity is deterministic.
        with torch.no_grad():
            em._keys()[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            em._keys()[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
            em._keys()[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
            em._values()[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
            em._values()[1] = torch.tensor([0.0, 20.0, 0.0, 0.0])
            em._values()[2] = torch.tensor([0.0, 0.0, 30.0, 0.0])
            em._valid()[0:3] = True

        query = torch.tensor([0.1, 0.1, 1.0, 0.1])  # closest to index 2
        results = em.retrieve(query, top_k=2)
        assert len(results) == 2
        # Top result should be index 2's value.
        assert torch.allclose(results[0].value, torch.tensor([0.0, 0.0, 30.0, 0.0]))
        assert results[0].similarity > results[1].similarity

    def test_top_k_bounded_by_valid_count(self) -> None:
        em = EpisodicMemory(capacity=8, key_dim=4, value_dim=4)
        em.encode(torch.randn(4), 0.1, 1)
        results = em.retrieve(torch.randn(4), top_k=5)
        assert len(results) == 1  # only one valid entry

    def test_query_shape_check(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=8, value_dim=16)
        with pytest.raises(ValueError, match="key shape"):
            em.retrieve(torch.randn(4))

    def test_retrieval_namedtuple_exposes_fields(self) -> None:
        em = EpisodicMemory(capacity=2, key_dim=4, value_dim=4)
        em.encode(torch.ones(4), 1.0, 1)
        results = em.retrieve(torch.ones(4))
        assert isinstance(results[0], Retrieval)
        assert hasattr(results[0], "value")
        assert hasattr(results[0], "similarity")


class TestSampleForReplay:
    def test_empty_returns_empty(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=8, value_dim=16)
        assert em.sample_for_replay(current_step=100) == []

    def test_all_zero_surprise_returns_empty(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=8, value_dim=16)
        for _ in range(4):
            em.encode(torch.randn(16), prediction_error=0.0, current_step=100)
        assert em.sample_for_replay(current_step=500) == []

    def test_prefers_high_surprise(self) -> None:
        torch.manual_seed(0)
        em = EpisodicMemory(capacity=8, key_dim=4, value_dim=4)
        # Surprise pattern: mostly low, one very high.
        highlight = torch.full((4,), 99.0)
        em.encode(torch.zeros(4), 0.01, 0)
        em.encode(torch.zeros(4), 0.01, 100)
        em.encode(highlight, 10.0, 1000)
        em.encode(torch.zeros(4), 0.01, 1500)
        # Sample many times; the high-surprise entry should dominate.
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for seed in range(50):
            rng = torch.Generator().manual_seed(seed)
            samples = em.sample_for_replay(current_step=2000, batch_size=1, rng=rng)
            if samples:
                # Find which slot's value we got back.
                for i in range(4):
                    if torch.allclose(samples[0], em._values()[i]):
                        counts[i] += 1
                        break
        assert counts[2] > counts[0]
        assert counts[2] > counts[3]

    def test_increments_replay_count(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        em.encode(torch.ones(4), 5.0, 100)
        before = int(em._replay_count()[0].item())
        em.sample_for_replay(current_step=1000, batch_size=1)
        after = int(em._replay_count()[0].item())
        assert after == before + 1

    def test_batch_size_caps_at_num_valid(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        em.encode(torch.ones(4), 5.0, 100)
        em.encode(torch.ones(4) * 2, 5.0, 200)
        samples = em.sample_for_replay(current_step=1000, batch_size=10)
        assert len(samples) == 2


class TestDecay:
    def test_invalidates_old_replayed(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        em.encode(torch.ones(4), 1.0, 0)
        with torch.no_grad():
            em._replay_count()[0] = 10
        dropped = em.decay_old_entries(current_step=1_000_000, max_age=100, min_replays=5)
        assert dropped == 1
        assert em.num_valid == 0

    def test_preserves_old_unreplayed(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        em.encode(torch.ones(4), 1.0, 0)
        # Not replayed -> must not drop.
        dropped = em.decay_old_entries(current_step=1_000_000, max_age=100, min_replays=5)
        assert dropped == 0
        assert em.num_valid == 1

    def test_preserves_recent_replayed(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        em.encode(torch.ones(4), 1.0, 999)
        with torch.no_grad():
            em._replay_count()[0] = 10
        # Recent -> must not drop even if replayed a lot.
        dropped = em.decay_old_entries(current_step=1000, max_age=100, min_replays=5)
        assert dropped == 0


class TestClearAndSerialization:
    def test_clear(self) -> None:
        em = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        em.encode(torch.ones(4), 1.0, 100)
        em.clear()
        assert em.num_valid == 0
        assert int(em.write_head.item()) == 0

    def test_state_dict_round_trip(self) -> None:
        em1 = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        em1.encode(torch.randn(4), 1.5, 100)
        em1.encode(torch.randn(4), 2.5, 200)
        state = em1.state_dict()

        em2 = EpisodicMemory(capacity=4, key_dim=4, value_dim=4)
        em2.load_state_dict(state)
        assert em2.num_valid == em1.num_valid
        assert torch.equal(em2._keys(), em1._keys())
        assert torch.equal(em2._values(), em1._values())
        assert torch.equal(em2._surprise(), em1._surprise())
        assert torch.equal(em2._timestamps(), em1._timestamps())
        assert int(em2.write_head.item()) == int(em1.write_head.item())
