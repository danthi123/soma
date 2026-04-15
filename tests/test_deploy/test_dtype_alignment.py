"""Phase 7 T3: fp16 LLM + fp32 verbalizer prefix dtype alignment.

SomaVerbalizer trains in fp32 for numerical stability, but on CUDA the
deploy layer loads HF causal LMs in fp16 to fit consumer-GPU VRAM. When
the session concats ``[prefix_fp32, token_embeds_fp16]`` without first
aligning dtypes, HF/PyTorch silently promote the full sequence to fp32
inside matmul kernels — doubling VRAM and defeating the whole point of
the fp16 load.

These tests exercise the exact code paths (SOMA.chat + ChatSession.respond)
with a fp16 mock LLM + fp32 verbalizer. Pre-fix, the concat path in both
sites hits a RuntimeError on the silent-upcast kernels (or, failing that,
produces an inputs_embeds tensor that is mixed-dtype and promoted). Post-
fix, the prefix is cast to match token_embeds.dtype and the generation
path succeeds cleanly.

We also include a gradient-flow sanity check — ``.to(dtype=...)`` is
autograd-aware, so verbalizer training through the cast should still
propagate gradients into the projector weights.
"""

from __future__ import annotations

import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.text_encoder import TextEncoder, train_bpe_tokenizer
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.session.chat_session import ChatSession
from soma.system import SOMA

# ---------------------------------------------------------------------------
# Minimal fp16 mocks.
#
# Intentionally duplicated (YAGNI) rather than refactoring the fp32 helpers
# in tests/test_session/test_chat_session.py — the only shape these tests
# need is "everything is fp16 end-to-end from embed() through generate()".
# The fp32 mocks cast internally in ways that would mask the bug we're
# guarding against; keeping a purpose-built fp16 harness isolates the
# regression surface.
# ---------------------------------------------------------------------------


class _TinyFP16CausalLM(nn.Module):
    """Mock HF causal LM whose weights are fp16 end-to-end.

    Calling ``get_input_embeddings()(input_ids)`` returns an fp16 tensor,
    matching what ``AutoModelForCausalLM.from_pretrained(torch_dtype=fp16)``
    produces on CUDA. ``forward`` / ``generate`` do NOT re-cast their
    inputs, so a fp32 prefix concatenated with fp16 token_embeds would
    either RuntimeError (dtype-mismatched matmul) or silently promote —
    either way the test fails without the fix.
    """

    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model).to(torch.float16)
        self.lm_head = nn.Linear(d_model, vocab).to(torch.float16)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, position_ids=None, **_):
        # Strict dtype check: the real fp16 HF kernels will RuntimeError on
        # mixed-dtype inputs; we mirror that here so the pre-fix test fails
        # deterministically instead of masking the bug via silent upcast.
        if inputs_embeds.dtype != torch.float16:
            raise RuntimeError(
                f"expected fp16 inputs_embeds, got {inputs_embeds.dtype}; "
                "prefix-dtype alignment fix is missing"
            )
        return self.lm_head(inputs_embeds)

    @torch.no_grad()
    def generate(self, *, inputs_embeds, attention_mask=None, max_new_tokens=4, **_):
        if inputs_embeds.dtype != torch.float16:
            raise RuntimeError(
                f"expected fp16 inputs_embeds, got {inputs_embeds.dtype}; "
                "prefix-dtype alignment fix is missing"
            )
        embeds = inputs_embeds
        gens = []
        for _ in range(max_new_tokens):
            logits = self.lm_head(embeds)
            next_id = logits[:, -1, :].argmax(dim=-1)
            gens.append(next_id)
            next_embed = self.embed(next_id).unsqueeze(1)
            embeds = torch.cat([embeds, next_embed], dim=1)
        return torch.stack(gens, dim=1)


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


# Shared tokenizer so each test doesn't re-train BPE.
_shared_tokenizer = train_bpe_tokenizer(
    iter(["hello world", "the quick brown fox", "lorem ipsum dolor sit amet"]),
    vocab_size=128,
)


def _build_fp16_session() -> ChatSession:
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    encoder = TextEncoder(
        _shared_tokenizer,
        embed_dim=cfg.text_embed_dim,
        max_seq_len=cfg.max_input_tokens,
    )
    spec = VerbalizerSpec(
        soma_output_dim=cfg.sensor_output_dim,
        llm_name="mock-fp16",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    # Verbalizer stays in fp32 — this is the whole point of the test.
    verbalizer = SomaVerbalizer(spec)
    assert next(verbalizer.parameters()).dtype == torch.float32
    chat_head = ChatHead(
        model=_TinyFP16CausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    return ChatSession(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        tokenizer=_shared_tokenizer,
        encoder=encoder,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chat_session_respond_with_fp16_llm_and_fp32_verbalizer():
    """ChatSession.respond must align dtypes before concat.

    Pre-fix: _TinyFP16CausalLM.generate raises RuntimeError because
    inputs_embeds ends up fp32 (prefix is fp32, concat promotes). Post-fix:
    prefix is cast to fp16 before concat, the generate path runs cleanly,
    and we get back a non-empty response string.
    """
    session = _build_fp16_session()
    response = session.respond(
        user_text="hi",
        max_new_tokens=2,
        min_new_tokens=1,
        do_sample=False,
    )
    assert isinstance(response, str)
    assert len(response) > 0


def test_soma_chat_with_fp16_llm_and_fp32_verbalizer():
    """SOMA.chat (the public single-turn method) must also align dtypes.

    Mirrors the session-level test but exercises SOMA.chat directly, which
    has its own copy of the concat logic. Both code sites need the fix.
    """
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name="mock-fp16",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    assert next(verbalizer.parameters()).dtype == torch.float32
    chat_head = ChatHead(
        model=_TinyFP16CausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    response = soma.chat(
        user_text="hello",
        verbalizer=verbalizer,
        chat_head=chat_head,
        max_new_tokens=2,
        min_new_tokens=1,
        do_sample=False,
    )
    assert isinstance(response, str)
    assert len(response) > 0


def test_prefix_cast_preserves_verbalizer_gradient_flow():
    """`.to(dtype=...)` is autograd-aware — a backward pass through the
    cast should still populate verbalizer parameter grads.

    Mimics the concat+LM-head path used during verbalizer training with
    a fp16 LLM: project fp32 verbalizer -> cast to fp16 -> concat -> fp16
    linear -> scalar loss -> backward. Verbalizer params must have non-
    None, non-zero grads, otherwise the fix would silently break T7
    online training on fp16 deployments.
    """
    cfg = _soma_cfg()
    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name="mock-fp16",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    model = _TinyFP16CausalLM(vocab=32, d_model=16)
    pooled = torch.randn(1, cfg.integrator_output_dim)
    prefix = verbalizer(pooled)
    assert prefix.dtype == torch.float32

    token_embeds = model.get_input_embeddings()(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert token_embeds.dtype == torch.float16

    # Apply the same cast the fix applies. Gradient must flow through .to().
    prefix_cast = prefix.to(dtype=token_embeds.dtype)
    inputs_embeds = torch.cat([prefix_cast, token_embeds], dim=1)
    out = model.forward(inputs_embeds=inputs_embeds)
    loss = out.float().sum()
    loss.backward()

    any_nonzero = False
    for p in verbalizer.parameters():
        assert p.grad is not None, "verbalizer param received no grad through .to(dtype=)"
        if p.grad.abs().sum().item() > 0:
            any_nonzero = True
    assert any_nonzero, "all verbalizer grads were zero after backward through fp16 cast"
