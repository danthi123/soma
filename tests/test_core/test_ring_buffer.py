"""Tests for ``soma.core.ring_buffer.RingBuffer``."""

from __future__ import annotations

import pytest

from soma.core.ring_buffer import RingBuffer


class TestConstruction:
    def test_starts_empty(self) -> None:
        rb = RingBuffer(5)
        assert rb.capacity == 5
        assert len(rb) == 0
        assert rb.is_empty
        assert not rb.is_full
        assert rb.get_all() == []

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "10"])
    def test_rejects_bad_capacity(self, bad: object) -> None:
        with pytest.raises(ValueError, match="positive int"):
            RingBuffer(bad)  # type: ignore[arg-type]


class TestAppend:
    def test_below_capacity(self) -> None:
        rb = RingBuffer(5)
        rb.append(1.0)
        rb.append(2.0)
        rb.append(3.0)
        assert len(rb) == 3
        assert rb.get_all() == [1.0, 2.0, 3.0]
        assert rb.last() == pytest.approx(3.0)

    def test_exact_capacity(self) -> None:
        rb = RingBuffer(3)
        rb.extend([1.0, 2.0, 3.0])
        assert rb.is_full
        assert rb.get_all() == [1.0, 2.0, 3.0]
        assert rb.last() == pytest.approx(3.0)

    def test_wrap_around_order(self) -> None:
        rb = RingBuffer(3)
        rb.extend([1.0, 2.0, 3.0, 4.0, 5.0])
        # Oldest two were evicted.
        assert rb.is_full
        assert rb.get_all() == [3.0, 4.0, 5.0]
        assert rb.last() == pytest.approx(5.0)

    def test_coerces_ints_to_float(self) -> None:
        rb = RingBuffer(3)
        rb.append(7)  # type: ignore[arg-type]
        assert rb.get_all() == [7.0]
        assert isinstance(rb.get_all()[0], float)


class TestStatistics:
    def test_mean(self) -> None:
        rb = RingBuffer(5)
        assert rb.mean() == pytest.approx(0.0)  # empty buffer -> 0
        rb.extend([2.0, 4.0, 6.0])
        assert rb.mean() == pytest.approx(4.0)

    def test_mean_after_wrap(self) -> None:
        rb = RingBuffer(3)
        rb.extend([10.0, 20.0, 30.0, 40.0])  # [20, 30, 40]
        assert rb.mean() == pytest.approx(30.0)

    def test_variance(self) -> None:
        rb = RingBuffer(5)
        assert rb.variance() == pytest.approx(0.0)  # empty
        rb.append(1.0)
        assert rb.variance() == pytest.approx(0.0)  # one element
        rb.extend([2.0, 3.0, 4.0, 5.0])
        # mean = 3.0 -> sq diffs [4, 1, 0, 1, 4], /5 = 2.0
        assert rb.variance() == pytest.approx(2.0)

    def test_min_max(self) -> None:
        rb = RingBuffer(4)
        rb.extend([3.0, -1.0, 7.0, 2.0])
        assert rb.min() == pytest.approx(-1.0)
        assert rb.max() == pytest.approx(7.0)

    def test_min_max_empty_raises(self) -> None:
        rb = RingBuffer(3)
        with pytest.raises(ValueError, match="empty"):
            rb.min()
        with pytest.raises(ValueError, match="empty"):
            rb.max()


class TestDundersAndClear:
    def test_last_empty_raises(self) -> None:
        rb = RingBuffer(3)
        with pytest.raises(IndexError, match="empty"):
            rb.last()

    def test_iter(self) -> None:
        rb = RingBuffer(4)
        rb.extend([1.0, 2.0, 3.0])
        assert list(iter(rb)) == [1.0, 2.0, 3.0]

    def test_contains(self) -> None:
        rb = RingBuffer(3)
        rb.extend([1.0, 2.0, 3.0])
        assert 2.0 in rb
        assert 4.0 not in rb
        assert "hi" not in rb  # type: ignore[operator]

    def test_clear(self) -> None:
        rb = RingBuffer(3)
        rb.extend([1.0, 2.0, 3.0, 4.0])
        rb.clear()
        assert rb.is_empty
        assert rb.get_all() == []
        # Reusable after clearing.
        rb.append(5.0)
        assert rb.get_all() == [5.0]

    def test_repr(self) -> None:
        rb = RingBuffer(3)
        rb.extend([1.0, 2.0])
        assert "RingBuffer" in repr(rb)
        assert "capacity=3" in repr(rb)


class TestSerialization:
    def test_round_trip_below_capacity(self) -> None:
        rb = RingBuffer(5)
        rb.extend([1.0, 2.0, 3.0])
        rebuilt = RingBuffer.from_list(rb.to_list(), capacity=5)
        assert rebuilt.get_all() == rb.get_all()
        assert rebuilt.capacity == 5

    def test_round_trip_after_wrap(self) -> None:
        rb = RingBuffer(3)
        rb.extend([1.0, 2.0, 3.0, 4.0, 5.0])  # [3, 4, 5]
        rebuilt = RingBuffer.from_list(rb.to_list(), capacity=3)
        assert rebuilt.get_all() == [3.0, 4.0, 5.0]
        # Next append should evict the oldest surviving entry.
        rebuilt.append(6.0)
        assert rebuilt.get_all() == [4.0, 5.0, 6.0]

    def test_from_list_infers_capacity(self) -> None:
        rb = RingBuffer.from_list([1.0, 2.0, 3.0])
        assert rb.capacity == 3
        assert rb.is_full

    def test_from_list_empty_requires_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            RingBuffer.from_list([])

    def test_from_list_truncates_if_over_capacity(self) -> None:
        rb = RingBuffer.from_list([1.0, 2.0, 3.0, 4.0, 5.0], capacity=3)
        assert rb.get_all() == [3.0, 4.0, 5.0]
