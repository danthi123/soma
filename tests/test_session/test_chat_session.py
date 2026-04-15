from datetime import datetime
from pathlib import Path

import pytest
import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.session.chat_session import ChatSession, ChatTurn
from soma.system import SOMA


# Reuse Phase 4's mock pattern. The forward() path mirrors the richer
# _TinyCausalLM used in test_online_verbalizer.py so the T7 tests that
# route through VerbalizerTrainer.train_step (→ compute_lm_loss, which
# needs out.loss) actually exercise training without crashing.
class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, labels=None, **_):
        # Causal cumulative-mean pool so the prefix influences every position
        # (gives the verbalizer a non-degenerate gradient signal under
        # teacher-forcing).
        B, T, D = inputs_embeds.shape
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

    @torch.no_grad()
    def generate(self, *, inputs_embeds, attention_mask=None, max_new_tokens=4, **_):
        embeds = inputs_embeds
        out = []
        for _ in range(max_new_tokens):
            res = self.forward(inputs_embeds=embeds)
            logits = res.logits
            nid = logits[:, -1, :].argmax(dim=-1)
            out.append(nid)
            embeds = torch.cat([embeds, self.embed(nid).unsqueeze(1)], dim=1)
        return torch.stack(out, dim=1)


class _TinyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str = "pt",
        padding: bool = False,
    ) -> dict:
        del return_tensors
        if isinstance(text, str):
            ids = [min(ord(c) % 32, 31) for c in text]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        encoded = [[min(ord(c) % 32, 31) for c in t] for t in text]
        if padding:
            max_len = max((len(row) for row in encoded), default=0)
            padded = [row + [self.pad_token_id] * (max_len - len(row)) for row in encoded]
            attn = [[1] * len(row) + [0] * (max_len - len(row)) for row in encoded]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        return {"input_ids": torch.tensor(encoded, dtype=torch.long)}

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids.flatten().tolist())


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


# Module-scoped tokenizer to avoid expensive re-train per test (see Phase 4 T4 pattern).
_shared_tokenizer = train_bpe_tokenizer(
    iter(["hello world", "the quick brown fox", "lorem ipsum dolor sit amet"]),
    vocab_size=128,
)


def _build_session(system_prompt: str | None = None) -> ChatSession:
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
    return ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=_shared_tokenizer,
        encoder=encoder,
        system_prompt=system_prompt,
    )


def test_chat_turn_dataclass_fields():
    t = ChatTurn(role="user", text="hi")
    assert t.role == "user"
    assert t.text == "hi"
    assert isinstance(t.ts, datetime)


def test_chat_turn_role_validated():
    with pytest.raises(ValueError, match="role"):
        ChatTurn(role="system", text="hi")  # type: ignore[arg-type]


def test_session_stores_components():
    s = _build_session()
    assert s.soma is not None
    assert s.verbalizer is not None
    assert s.chat_head is not None
    assert s.tokenizer is not None
    assert s.encoder is not None
    assert s.history == []


def test_session_with_system_prompt_pre_warms_soma():
    """A session with a system_prompt should run the prompt through SOMA at
    init, so the very first respond() call sees a state already tinted by
    the system context. Verify by comparing OUTPUT activations with vs.
    without a system_prompt."""
    s_with = _build_session(system_prompt="you are a helpful assistant")
    s_without = _build_session(system_prompt=None)

    acts_with = s_with.soma._current_output_activations()
    acts_without = s_without.soma._current_output_activations()

    # Guard: if both empty, the test is vacuous — log and warn.
    assert len(acts_with) > 0 or len(acts_without) > 0, (
        "no OUTPUT activations on either; system_prompt warm has nothing to verify"
    )

    # When non-empty, verify the warmed and unwarmed states differ.
    if acts_with and acts_without:
        any_diff = any(
            not torch.allclose(acts_with[k], acts_without[k])
            for k in acts_with
            if k in acts_without
        )
        assert any_diff, "system_prompt warm did not change OUTPUT state"


# ---------------------------------------------------------------------------
# T2: _feed_text_through_soma — explicit side-effect tests
# ---------------------------------------------------------------------------


def test_feed_text_advances_global_step():
    s = _build_session()
    before = s.soma.global_step
    s._feed_text_through_soma("hello world this is a test")
    after = s.soma.global_step
    assert after > before, f"soma.global_step did not advance ({before} → {after})"


def test_feed_text_does_not_train_soma():
    s = _build_session()
    node = next(iter(s.soma.graph.nodes.values()))
    before = next(node.parameters()).detach().clone()
    s._feed_text_through_soma("some words to push through")
    after = next(node.parameters()).detach().clone()
    assert torch.allclose(before, after), (
        "SOMA params drifted during _feed_text_through_soma — no_grad/eval_mode contract broken"
    )


def test_feed_text_evolves_output_activations():
    """Two distinct inputs through SOMA should produce different OUTPUT
    activations afterward — proves WM/last_activation actually advance."""
    s = _build_session()
    s._feed_text_through_soma("warm up text alpha")
    acts_first = {k: v.clone() for k, v in s.soma._current_output_activations().items()}
    s._feed_text_through_soma("very different second text beta gamma")
    acts_second = s.soma._current_output_activations()
    if acts_first and acts_second:
        any_changed = any(
            not torch.allclose(acts_first[k], acts_second[k])
            for k in acts_first
            if k in acts_second
        )
        assert any_changed, (
            "OUTPUT activations identical after two distinct feeds — "
            "Node.last_activation isn't being updated by soma.step"
        )


# ---------------------------------------------------------------------------
# T3: respond — single-turn flow
# ---------------------------------------------------------------------------


def test_respond_returns_string():
    s = _build_session()
    out = s.respond(user_text="hello")
    assert isinstance(out, str)


def test_respond_appends_user_and_assistant_turns():
    s = _build_session()
    assert len(s.history) == 0
    s.respond(user_text="hi")
    assert len(s.history) == 2
    assert s.history[0].role == "user"
    assert s.history[0].text == "hi"
    assert s.history[1].role == "assistant"
    assert isinstance(s.history[1].text, str)


def test_respond_max_new_tokens_passes_through():
    s = _build_session()
    out = s.respond(user_text="hi", max_new_tokens=2)
    # Mock decoder: each token id → 1 char. Response length == max_new_tokens.
    assert len(out) == 2


# ---------------------------------------------------------------------------
# T4: response back-loop
# ---------------------------------------------------------------------------


def test_respond_feeds_response_back_into_soma():
    """After respond(), SOMA's global_step should reflect TWO text-pushes:
    one for the user input, one for the assistant response. Compare to a
    parallel session where only the user input was fed.
    """
    # Session A: full respond (user push + LLM gen + response push)
    s_a = _build_session()
    before_a = s_a.soma.global_step
    _ = s_a.respond(user_text="hi", max_new_tokens=3)
    delta_a = s_a.soma.global_step - before_a

    # Session B: user push only, no respond
    s_b = _build_session()
    before_b = s_b.soma.global_step
    s_b._feed_text_through_soma("hi")
    delta_b = s_b.soma.global_step - before_b

    assert delta_a > delta_b, (
        f"SOMA didn't see the assistant response — delta_a={delta_a} vs "
        f"user-only delta_b={delta_b}. Back-loop missing or no-op."
    )


# ---------------------------------------------------------------------------
# T5: system_prompt pre-warm contract hardening
# ---------------------------------------------------------------------------


def test_session_without_system_prompt_does_not_warm_soma():
    s = _build_session(system_prompt=None)
    assert s.soma.global_step == 0, (
        f"SOMA pre-warmed without a system_prompt — global_step={s.soma.global_step}"
    )


def test_session_with_system_prompt_advances_global_step():
    s = _build_session(system_prompt="be helpful and concise")
    assert s.soma.global_step > 0, "system_prompt did not warm SOMA — global_step still zero"


def test_session_with_empty_string_system_prompt_does_not_warm():
    """Empty string is falsy → treated as no system prompt."""
    s = _build_session(system_prompt="")
    assert s.soma.global_step == 0


# ---------------------------------------------------------------------------
# T6: multi-turn coherence — state evolves across turns
# ---------------------------------------------------------------------------


def test_multi_turn_global_step_accumulates():
    """After 3 distinct turns, SOMA's global_step should have advanced
    by (user-feed + LLM-response-feed) × 3 turns worth of per-token steps.

    Note on tested signals: ``Node.last_activation`` is overwritten every
    soma.step to the latest input's output (point-in-time snapshot, not
    accumulator) and WM slots with this tiny-graph test config stay near
    zero because the write gate doesn't fire on micro-magnitude inputs.
    The genuinely monotonic signal at unit-test scale is ``global_step``.
    Rich multi-turn coherence on the activation side is validated by the
    T9 SmolLM2 smoke where the real LLM produces meaningfully varied
    responses that WM can actually register.
    """
    s = _build_session()
    assert s.soma.global_step == 0

    s.respond(user_text="apples are red")
    step_after_t1 = s.soma.global_step
    assert step_after_t1 > 0

    s.respond(user_text="bananas are yellow")
    step_after_t2 = s.soma.global_step
    assert step_after_t2 > step_after_t1

    s.respond(user_text="grapes are purple")
    step_after_t3 = s.soma.global_step
    assert step_after_t3 > step_after_t2


def test_history_grows_two_per_respond_call():
    """History interleaves user/assistant correctly across multiple turns."""
    s = _build_session()
    assert len(s.history) == 0
    s.respond(user_text="a")
    assert len(s.history) == 2
    s.respond(user_text="b")
    assert len(s.history) == 4
    s.respond(user_text="c")
    assert len(s.history) == 6
    assert [t.role for t in s.history] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [t.text for t in s.history if t.role == "user"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# T7: save (SOMA bundle + chat_history sidecar) + load_history
# ---------------------------------------------------------------------------


def test_save_writes_chat_history_json(tmp_path: Path):
    import json

    s = _build_session()
    s.respond(user_text="hello")
    s.respond(user_text="world")
    s.save(out_dir=tmp_path)
    history_path = tmp_path / "chat_history.json"
    assert history_path.exists()
    data = json.loads(history_path.read_text())
    assert isinstance(data, list)
    assert len(data) == 4  # 2 turns × (user + assistant)
    assert all(item["role"] in ("user", "assistant") for item in data)
    assert all("text" in item for item in data)
    assert all("ts" in item for item in data)


def test_save_writes_soma_bundle_core_files(tmp_path: Path):
    """Verify save() delegates to SOMA.save_bundle — brain.pt + manifest."""
    s = _build_session()
    s.save(out_dir=tmp_path)
    assert (tmp_path / "brain.pt").exists()
    assert (tmp_path / "manifest.json").exists()


def test_load_history_round_trips(tmp_path: Path):
    """Save from session A, load into a fresh session B, history matches."""
    s_a = _build_session()
    s_a.respond(user_text="hi")
    s_a.respond(user_text="how are you")
    s_a.save(out_dir=tmp_path)

    s_b = _build_session()
    s_b.load_history(out_dir=tmp_path)
    assert len(s_b.history) == 4
    assert s_b.history[0].text == s_a.history[0].text
    assert s_b.history[0].role == s_a.history[0].role
    assert s_b.history[3].role == "assistant"


def test_load_history_rejects_missing_sidecar(tmp_path: Path):
    """If chat_history.json doesn't exist, load_history raises."""
    s = _build_session()
    with pytest.raises(FileNotFoundError, match="chat_history"):
        s.load_history(out_dir=tmp_path)


# ---------------------------------------------------------------------------
# T7 (Phase 6): optional online_trainer integration
# ---------------------------------------------------------------------------


def _build_session_with_online_trainer():
    """Build a session with an OnlineVerbalizerTrainer attached.

    Reuses _build_session() and wires up an inner VerbalizerTrainer +
    OnlineVerbalizerTrainer on top of the same components.
    """
    from soma.training.online_verbalizer import OnlineVerbalizerTrainer
    from soma.training.verbalizer_bootstrap import VerbalizerTrainer

    s = _build_session()
    inner = VerbalizerTrainer(
        soma=s.soma,
        verbalizer=s.verbalizer,
        chat_head=s.chat_head,
        config=s.soma.config,
        tokenizer=s.tokenizer,
        encoder=s.encoder,
    )
    online = OnlineVerbalizerTrainer(inner=inner, config=s.soma.config)
    s.online_trainer = online
    return s, online


def test_session_without_online_trainer_does_not_train_verbalizer():
    """Pure Phase 5 behavior: verbalizer params are frozen during chat."""
    s = _build_session()
    # Explicitly None to be sure.
    s.online_trainer = None
    before = [p.detach().clone() for p in s.verbalizer.parameters()]
    s.respond(user_text="hi")
    after = [p.detach().clone() for p in s.verbalizer.parameters()]
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a), "verbalizer drifted without online trainer"


def test_session_with_online_trainer_updates_verbalizer_after_respond():
    s, online = _build_session_with_online_trainer()
    before = [p.detach().clone() for p in s.verbalizer.parameters()]
    s.respond(user_text="hi there how are you")
    after = [p.detach().clone() for p in s.verbalizer.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True)), (
        "verbalizer did not move after respond with online trainer"
    )


def test_session_with_online_trainer_fills_replay_buffer():
    s, online = _build_session_with_online_trainer()
    assert len(online.replay_buffer) == 0
    s.respond(user_text="turn one")
    assert len(online.replay_buffer) == 1
    s.respond(user_text="turn two")
    assert len(online.replay_buffer) == 2
    # Buffer entry records both user_text and the response.
    assert online.replay_buffer.entries[0].user_text == "turn one"
    assert isinstance(online.replay_buffer.entries[0].response, str)


def test_session_init_accepts_online_trainer_kwarg():
    """The __init__ signature should accept online_trainer directly, not
    just via attribute assignment after construction."""
    from soma.io.text_encoder import TextEncoder
    from soma.training.online_verbalizer import OnlineVerbalizerTrainer
    from soma.training.verbalizer_bootstrap import VerbalizerTrainer

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
    inner = VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
        tokenizer=_shared_tokenizer,
        encoder=encoder,
    )
    online = OnlineVerbalizerTrainer(inner=inner, config=cfg)

    # Pass online_trainer via __init__, not via attribute assignment.
    session = ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=_shared_tokenizer,
        encoder=encoder,
        online_trainer=online,
    )
    assert session.online_trainer is online


# ---------------------------------------------------------------------------
# T9 (Phase 6): save/load online_state.json sidecar
# ---------------------------------------------------------------------------


def test_save_writes_online_state_when_trainer_attached(tmp_path: Path):
    import json

    s, online = _build_session_with_online_trainer()
    s.respond(user_text="hi")
    s.respond(user_text="how are you")
    s.save(out_dir=tmp_path)
    state_path = tmp_path / "online_state.json"
    assert state_path.exists(), "online_state.json missing after save"

    data = json.loads(state_path.read_text())
    assert "replay_buffer" in data
    assert "loss_history" in data
    assert "is_diverged" in data
    assert len(data["replay_buffer"]) == 2
    assert data["replay_buffer"][0]["user_text"] == "hi"
    assert isinstance(data["loss_history"], list)
    assert len(data["loss_history"]) >= 1
    assert data["is_diverged"] is False


def test_save_skips_online_state_when_trainer_is_none(tmp_path: Path):
    """If the session has no online_trainer, no online_state.json is written."""
    s = _build_session()
    s.respond(user_text="hi")
    s.save(out_dir=tmp_path)
    assert not (tmp_path / "online_state.json").exists()


def test_load_online_state_restores_buffer_and_history(tmp_path: Path):
    from soma.training.online_verbalizer import OnlineVerbalizerTrainer
    from soma.training.verbalizer_bootstrap import VerbalizerTrainer

    # Session A: train + save.
    s_a, online_a = _build_session_with_online_trainer()
    s_a.respond(user_text="alpha")
    s_a.respond(user_text="beta")
    s_a.respond(user_text="gamma")
    s_a.save(out_dir=tmp_path)

    assert len(online_a.replay_buffer) == 3
    assert len(online_a._loss_history) >= 1

    # Session B: fresh online trainer; load the state.
    s_b = _build_session()
    inner_b = VerbalizerTrainer(
        soma=s_b.soma,
        verbalizer=s_b.verbalizer,
        chat_head=s_b.chat_head,
        config=s_b.soma.config,
        tokenizer=s_b.tokenizer,
        encoder=s_b.encoder,
    )
    online_b = OnlineVerbalizerTrainer(inner=inner_b, config=s_b.soma.config)
    s_b.online_trainer = online_b

    s_b.load_online_state(out_dir=tmp_path, online_trainer=online_b)
    assert len(online_b.replay_buffer) == 3
    user_texts = [ex.user_text for ex in online_b.replay_buffer.entries]
    assert user_texts == ["alpha", "beta", "gamma"]
    # Loss history roundtrip.
    assert list(online_b._loss_history) == list(online_a._loss_history)
    # is_diverged roundtrip.
    assert online_b.is_diverged == online_a.is_diverged


def test_load_online_state_rejects_missing_sidecar(tmp_path: Path):
    from soma.training.online_verbalizer import OnlineVerbalizerTrainer
    from soma.training.verbalizer_bootstrap import VerbalizerTrainer

    s = _build_session()
    inner = VerbalizerTrainer(
        soma=s.soma,
        verbalizer=s.verbalizer,
        chat_head=s.chat_head,
        config=s.soma.config,
        tokenizer=s.tokenizer,
        encoder=s.encoder,
    )
    online = OnlineVerbalizerTrainer(inner=inner, config=s.soma.config)
    with pytest.raises(FileNotFoundError, match="online_state"):
        s.load_online_state(out_dir=tmp_path, online_trainer=online)


def test_load_online_state_preserves_is_diverged_flag(tmp_path: Path):
    """Save a session whose trainer is diverged; load into a fresh
    trainer and verify the flag round-trips."""
    s_a, online_a = _build_session_with_online_trainer()
    online_a.is_diverged = True
    s_a.save(out_dir=tmp_path)

    from soma.training.online_verbalizer import OnlineVerbalizerTrainer
    from soma.training.verbalizer_bootstrap import VerbalizerTrainer

    s_b = _build_session()
    inner_b = VerbalizerTrainer(
        soma=s_b.soma,
        verbalizer=s_b.verbalizer,
        chat_head=s_b.chat_head,
        config=s_b.soma.config,
        tokenizer=s_b.tokenizer,
        encoder=s_b.encoder,
    )
    online_b = OnlineVerbalizerTrainer(inner=inner_b, config=s_b.soma.config)
    s_b.load_online_state(out_dir=tmp_path, online_trainer=online_b)
    assert online_b.is_diverged is True
