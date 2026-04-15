from datetime import datetime

import pytest

from soma.training.online_verbalizer import ChatExchange, ReplayBuffer


def test_chat_exchange_dataclass_fields():
    ex = ChatExchange(user_text="hi", response="hello")
    assert ex.user_text == "hi"
    assert ex.response == "hello"
    assert isinstance(ex.ts, datetime)


def test_chat_exchange_is_frozen():
    ex = ChatExchange(user_text="hi", response="hello")
    with pytest.raises((AttributeError, TypeError)):
        ex.user_text = "changed"  # type: ignore[misc]


def test_replay_buffer_default_empty():
    buf = ReplayBuffer(capacity=4)
    assert len(buf) == 0


def test_replay_buffer_add_grows_length():
    buf = ReplayBuffer(capacity=4)
    buf.add(ChatExchange(user_text="a", response="b"))
    assert len(buf) == 1


def test_replay_buffer_evicts_oldest_at_capacity():
    buf = ReplayBuffer(capacity=2)
    buf.add(ChatExchange(user_text="a", response="x"))
    buf.add(ChatExchange(user_text="b", response="y"))
    buf.add(ChatExchange(user_text="c", response="z"))
    assert len(buf) == 2
    user_texts = [ex.user_text for ex in buf.entries]
    assert "a" not in user_texts  # oldest evicted
    assert user_texts == ["b", "c"]


def test_replay_buffer_sample_returns_at_most_n():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.add(ChatExchange(user_text=f"u{i}", response=f"r{i}"))
    sampled = buf.sample(n=3)
    assert len(sampled) == 3
    assert all(isinstance(s, ChatExchange) for s in sampled)
    # Must be drawn from the buffer's entries (no fabricated elements).
    user_texts = {ex.user_text for ex in buf.entries}
    for s in sampled:
        assert s.user_text in user_texts


def test_replay_buffer_sample_n_larger_than_buffer_returns_all():
    buf = ReplayBuffer(capacity=10)
    buf.add(ChatExchange(user_text="a", response="x"))
    buf.add(ChatExchange(user_text="b", response="y"))
    sampled = buf.sample(n=10)
    assert len(sampled) == 2


def test_replay_buffer_sample_from_empty_returns_empty():
    buf = ReplayBuffer(capacity=4)
    sampled = buf.sample(n=3)
    assert sampled == []


def test_replay_buffer_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=0)
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=-5)
