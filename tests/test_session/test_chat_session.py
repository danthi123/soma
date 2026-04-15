from datetime import datetime

import pytest
import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.session.chat_session import ChatSession, ChatTurn
from soma.system import SOMA


# Reuse Phase 4's mock pattern.
class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, **_):
        return self.lm_head(inputs_embeds)

    @torch.no_grad()
    def generate(self, *, inputs_embeds, attention_mask=None, max_new_tokens=4, **_):
        embeds = inputs_embeds
        out = []
        for _ in range(max_new_tokens):
            logits = self.forward(inputs_embeds=embeds)
            nid = logits[:, -1, :].argmax(dim=-1)
            out.append(nid)
            embeds = torch.cat([embeds, self.embed(nid).unsqueeze(1)], dim=1)
        return torch.stack(out, dim=1)


class _TinyTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = [min(ord(c) % 32, 31) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

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
    assert after > before, (
        f"soma.global_step did not advance ({before} → {after})"
    )


def test_feed_text_does_not_train_soma():
    s = _build_session()
    node = next(iter(s.soma.graph.nodes.values()))
    before = next(node.parameters()).detach().clone()
    s._feed_text_through_soma("some words to push through")
    after = next(node.parameters()).detach().clone()
    assert torch.allclose(before, after), (
        "SOMA params drifted during _feed_text_through_soma — "
        "no_grad/eval_mode contract broken"
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
