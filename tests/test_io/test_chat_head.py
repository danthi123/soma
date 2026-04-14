import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.io.chat_head import (
    ChatHead,
    build_attention_mask,
    build_attention_mask_from_pad,
    build_position_ids,
)
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.system import SOMA


class _TinyCausalLM(nn.Module):
    """Mock HF causal LM: embedding + single Linear lm_head."""

    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, position_ids=None):
        return self.lm_head(inputs_embeds)

    @torch.no_grad()
    def generate(self, *, inputs_embeds, attention_mask=None, max_new_tokens=4, **kw):
        embeds = inputs_embeds
        generated = []
        for _ in range(max_new_tokens):
            logits = self.forward(inputs_embeds=embeds)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated.append(next_id)
            next_embed = self.embed(next_id).unsqueeze(1)
            embeds = torch.cat([embeds, next_embed], dim=1)
        return torch.stack(generated, dim=1)


class _TinyTokenizer:
    """Mock HF tokenizer: char-level, vocab=32."""

    def __init__(self) -> None:
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt") -> dict:
        ids = [min(ord(c) % 32, 31) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids.flatten().tolist())


def test_chat_head_freezes_base_model():
    model = _TinyCausalLM()
    assert all(p.requires_grad for p in model.parameters())
    head = ChatHead(model=model, tokenizer=_TinyTokenizer())
    assert not any(p.requires_grad for p in head.model.parameters())


def test_chat_head_stores_model_and_tokenizer():
    model = _TinyCausalLM()
    tok = _TinyTokenizer()
    head = ChatHead(model=model, tokenizer=tok)
    assert head.model is model
    assert head.tokenizer is tok


def test_chat_head_exposes_hidden_size():
    head = ChatHead(model=_TinyCausalLM(d_model=16), tokenizer=_TinyTokenizer())
    assert head.hidden_size == 16


def test_chat_head_generate_takes_inputs_embeds():
    head = ChatHead(model=_TinyCausalLM(), tokenizer=_TinyTokenizer())
    inputs_embeds = torch.randn(1, 5, 16)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    out = head.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=3,
    )
    assert out.shape == (1, 3)


def test_chat_head_generate_decodes_to_text():
    head = ChatHead(model=_TinyCausalLM(), tokenizer=_TinyTokenizer())
    text = head.generate_text(
        inputs_embeds=torch.randn(1, 4, 16),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        max_new_tokens=5,
    )
    assert isinstance(text, str)
    # Mock tokenizer decodes one char per token id. 5 tokens → 5 chars.
    # Proves the decode path actually ran rather than returning "".
    assert len(text) == 5


def test_position_ids_cover_prefix_plus_tokens():
    ids = build_position_ids(num_prefix=4, num_tokens=7, batch_size=2)
    assert ids.shape == (2, 11)
    assert torch.all(ids[0] == torch.arange(11))
    assert torch.all(ids[1] == torch.arange(11))


def test_attention_mask_ones_for_prefix_and_tokens():
    mask = build_attention_mask(num_prefix=4, num_tokens=7, batch_size=1)
    assert mask.shape == (1, 11)
    assert torch.all(mask == 1)


def test_attention_mask_zero_for_padded_tokens():
    pad_mask = torch.tensor([[1, 1, 1, 0, 0]])  # 3 real + 2 pad
    mask = build_attention_mask_from_pad(num_prefix=4, pad_mask=pad_mask)
    assert mask.shape == (1, 9)
    assert torch.all(mask[0, :4] == 1)
    assert torch.all(mask[0, 4:] == pad_mask[0])


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
        vocab_size=16,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )


def test_soma_chat_end_to_end_with_mock_llm():
    cfg = _soma_cfg()
    soma = SOMA(cfg, device=torch.device("cpu"))
    spec = VerbalizerSpec(
        soma_output_dim=cfg.integrator_output_dim,
        llm_name="mock",
        llm_hidden_dim=16,
        num_prefix_tokens=4,
        proj_hidden_dim=16,
    )
    verbalizer = SomaVerbalizer(spec)
    chat_head = ChatHead(model=_TinyCausalLM(vocab=32, d_model=16), tokenizer=_TinyTokenizer())
    response = soma.chat(
        user_text="hello",
        verbalizer=verbalizer,
        chat_head=chat_head,
        max_new_tokens=3,
    )
    assert isinstance(response, str)
