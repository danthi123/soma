"""Tests for ``soma.ui.bus`` — pub/sub ring-buffer data bus."""

from __future__ import annotations

import threading

import pytest

from soma.ui.bus import DataBus, DataChannel


class TestDataChannel:
    def test_publish_and_history(self) -> None:
        ch = DataChannel("x", max_history=5)
        for i in range(3):
            ch.publish(i)
        assert ch.get_history() == [0, 1, 2]
        assert ch.latest() == 2
        assert ch.size == 3

    def test_ring_buffer_drops_oldest(self) -> None:
        ch = DataChannel("x", max_history=3)
        for i in range(10):
            ch.publish(i)
        assert ch.get_history() == [7, 8, 9]

    def test_throttle(self) -> None:
        ch = DataChannel("x", max_history=10, throttle_steps=3)
        for i in range(10):
            ch.publish(i)
        # Throttle 3 -> accepts on step counts 3, 6, 9 (indices 2, 5, 8).
        assert ch.get_history() == [2, 5, 8]

    def test_subscribe_and_unsubscribe(self) -> None:
        ch = DataChannel("x")
        seen: list[int] = []
        cb = seen.append
        ch.subscribe(cb)
        ch.publish(1)
        ch.publish(2)
        ch.unsubscribe(cb)
        ch.publish(3)
        assert seen == [1, 2]

    def test_subscriber_exceptions_are_swallowed(self) -> None:
        ch = DataChannel("x")

        def bad(_value: int) -> None:
            raise RuntimeError("nope")

        good_log: list[int] = []
        ch.subscribe(bad)
        ch.subscribe(good_log.append)
        ch.publish(42)
        assert good_log == [42]  # bad subscriber didn't block the good one

    def test_clear(self) -> None:
        ch = DataChannel("x")
        for i in range(5):
            ch.publish(i)
        ch.clear()
        assert ch.size == 0
        assert ch.latest() is None

    def test_thread_safe_concurrent_publish(self) -> None:
        ch = DataChannel("x", max_history=10_000)

        def produce() -> None:
            for i in range(500):
                ch.publish(i)

        threads = [threading.Thread(target=produce) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 4 threads * 500 items = 2000 total (all fit in max_history=10000).
        assert ch.size == 2000

    def test_invalid_config(self) -> None:
        with pytest.raises(ValueError):
            DataChannel("x", max_history=0)
        with pytest.raises(ValueError):
            DataChannel("x", throttle_steps=0)


class TestDataBus:
    def test_create_and_get(self) -> None:
        bus = DataBus()
        ch = bus.create_channel("a")
        assert bus.get("a") is ch
        assert bus.has("a")
        assert bus.channel_names == ["a"]

    def test_create_is_idempotent_on_name(self) -> None:
        bus = DataBus()
        ch1 = bus.create_channel("x")
        ch2 = bus.create_channel("x")
        assert ch1 is ch2

    def test_get_missing_raises(self) -> None:
        bus = DataBus()
        with pytest.raises(KeyError):
            bus.get("nope")

    def test_publish_auto_creates_channel(self) -> None:
        bus = DataBus()
        bus.publish("new", 1)
        assert bus.has("new")
        assert bus.get("new").latest() == 1

    def test_subscribe_auto_creates_channel(self) -> None:
        bus = DataBus()
        seen: list[int] = []
        bus.subscribe("new", seen.append)
        bus.publish("new", 42)
        assert seen == [42]
