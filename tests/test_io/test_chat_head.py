import torch
from torch import nn

from soma.io.chat_head import ChatHead


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
