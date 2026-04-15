from dataclasses import replace
from datetime import datetime

import pytest
import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.system import SOMA
from soma.training.online_verbalizer import (
    ChatExchange,
    OnlineVerbalizerTrainer,
    ReplayBuffer,
)
from soma.training.verbalizer_bootstrap import VerbalizerTrainer


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


# --- Shared fixtures for OnlineVerbalizerTrainer tests (T3/T4) -------------


class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, labels=None, **_):
        # Simple causal-mean-pool + lm_head (so prefix influences every position).
        B, T, D = inputs_embeds.shape
        # Causal cumulative mean:
        cumsum = inputs_embeds.cumsum(dim=1)
        counts = torch.arange(
            1, T + 1, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        ).view(1, T, 1)
        pooled = cumsum / counts
        logits = self.lm_head(pooled)
        if labels is None:
            return type("Out", (), {"logits": logits, "loss": None})()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return type("Out", (), {"logits": logits, "loss": loss})()


class _TinyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = [min(ord(c) % 32, 31) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _soma_cfg() -> SOMAConfig:
    return SOMAConfig(
        sensor_output_dim=8,
        associator_input_dim=8,
        associator_hidden_dim=16,
        associator_output_dim=8,
        integrator_input_dim=16,
        integrator_hidden_dim=16,
        integrator_output_dim=16,
        position_dim=4,
        wm_slots=2,
        wm_dim=8,
        episodic_capacity=4,
        key_dim=8,
        value_dim=8,
        vocab_size=128,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )


_shared_tokenizer = train_bpe_tokenizer(
    iter(["hello world", "the quick brown fox", "lorem ipsum"]),
    vocab_size=128,
)


def _fresh_inner_trainer() -> VerbalizerTrainer:
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    encoder = TextEncoder(
        _shared_tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name="mock",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    chat_head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    return VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=_shared_tokenizer,
        encoder=encoder,
    )


# --- T3 tests: OnlineVerbalizerTrainer __init__ + record -------------------


def test_online_trainer_stores_inner_and_config():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    assert online.inner is inner
    assert online.config is inner.config


def test_online_trainer_builds_replay_buffer_from_config():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    assert isinstance(online.replay_buffer, ReplayBuffer)
    assert online.replay_buffer.capacity == inner.config.replay_buffer_capacity


def test_online_trainer_initial_is_not_diverged():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    assert online.is_diverged is False


def test_online_trainer_overrides_inner_lr():
    """Constructor should swap the inner optimizer's LR to the online rate
    (preserving Adam moments — same param_group, just LR edit)."""
    inner = _fresh_inner_trainer()
    # Sanity: before online trainer, inner uses bootstrap LR.
    assert inner.optim.param_groups[0]["lr"] == inner.config.verbalizer_lr
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    # After construction, LR is online_verbalizer_lr.
    assert inner.optim.param_groups[0]["lr"] == inner.config.online_verbalizer_lr
    assert online.inner.optim.param_groups[0]["lr"] == inner.config.online_verbalizer_lr


def test_online_trainer_record_adds_to_buffer():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.record(user_text="hi", response="hello")
    assert len(online.replay_buffer) == 1
    assert online.replay_buffer.entries[0].user_text == "hi"
    assert online.replay_buffer.entries[0].response == "hello"


# --- T4 tests: sample_batch ------------------------------------------------


def test_sample_batch_returns_user_texts():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.record(user_text="u0", response="r0")
    online.record(user_text="u1", response="r1")
    online.record(user_text="u2", response="r2")

    batch = online.sample_batch(batch_size=2)
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert all(isinstance(t, str) for t in batch)
    # Most-recent exchange must be first (contract for training).
    assert batch[0] == "u2"
    # Second slot must come from prior turns.
    assert batch[1] in {"u0", "u1"}


def test_sample_batch_with_empty_buffer_returns_empty():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    batch = online.sample_batch(batch_size=4)
    assert batch == []


def test_sample_batch_caps_at_buffer_size():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.record(user_text="u0", response="r0")
    batch = online.sample_batch(batch_size=4)
    assert len(batch) == 1
    assert batch[0] == "u0"


def test_sample_batch_size_one_returns_just_latest():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    for i in range(5):
        online.record(user_text=f"u{i}", response=f"r{i}")
    batch = online.sample_batch(batch_size=1)
    assert batch == ["u4"]


# --- T5 tests: step() ----------------------------------------------------


def test_step_records_and_trains():
    import torch

    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)

    before = [p.detach().clone() for p in inner.verbalizer.parameters()]
    online.step(user_text="hello there", response="hi!")
    after = [p.detach().clone() for p in inner.verbalizer.parameters()]

    assert len(online.replay_buffer) == 1
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))
    assert len(online._loss_history) >= 1


def test_step_skips_training_when_diverged():
    """When is_diverged is set, step records but does NOT train."""
    import torch

    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.is_diverged = True

    before = [p.detach().clone() for p in inner.verbalizer.parameters()]
    online.step(user_text="hi", response="yo")
    after = [p.detach().clone() for p in inner.verbalizer.parameters()]

    assert len(online.replay_buffer) == 1  # recorded
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a), "trained despite is_diverged=True"
    # And no entry in loss history either.
    assert len(online._loss_history) == 0


def test_step_excludes_non_finite_losses_from_history():
    """If inner.train_step returns NaN (e.g. guarded by its own NaN check),
    the mean should be computed over FINITE losses only. If ALL losses are
    NaN, nothing appends to history.
    """
    from unittest.mock import patch

    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)

    # First record so there's something to sample.
    online.record(user_text="u0", response="r0")

    # Monkeypatch inner.train_step to always return NaN.
    with patch.object(inner, "train_step", return_value=float("nan")):
        online.step(user_text="hi", response="yo")

    # Buffer grew (record happened) but no entry in loss history
    # because all samples' losses were filtered out.
    assert len(online.replay_buffer) == 2
    assert len(online._loss_history) == 0


# --- T6 tests: _check_divergence + reset_divergence_guard ----------------


def test_divergence_detects_rising_loss():
    inner = _fresh_inner_trainer()
    cfg = replace(inner.config, divergence_window=4, divergence_threshold=0.5)
    online = OnlineVerbalizerTrainer(inner=inner, config=cfg)

    # Inject losses: first half mean 0.5, second half mean 2.0 → rise 1.5
    online._loss_history.extend([0.4, 0.6, 1.8, 2.2])
    online._check_divergence()
    assert online.is_diverged


def test_divergence_stable_loss_is_not_flagged():
    inner = _fresh_inner_trainer()
    cfg = replace(inner.config, divergence_window=4, divergence_threshold=0.5)
    online = OnlineVerbalizerTrainer(inner=inner, config=cfg)

    online._loss_history.extend([1.0, 1.05, 0.95, 1.02])
    online._check_divergence()
    assert not online.is_diverged


def test_divergence_requires_full_window():
    """Partial history (< window) should not trip the monitor even if
    the rise is large."""
    inner = _fresh_inner_trainer()
    cfg = replace(inner.config, divergence_window=4, divergence_threshold=0.5)
    online = OnlineVerbalizerTrainer(inner=inner, config=cfg)

    online._loss_history.extend([0.1, 5.0, 10.0])  # only 3 entries
    online._check_divergence()
    assert not online.is_diverged


def test_reset_divergence_guard_clears_flag_and_history():
    inner = _fresh_inner_trainer()
    online = OnlineVerbalizerTrainer(inner=inner, config=inner.config)
    online.is_diverged = True
    online._loss_history.extend([10.0] * online.config.divergence_window)

    online.reset_divergence_guard()
    assert not online.is_diverged
    assert len(online._loss_history) == 0
