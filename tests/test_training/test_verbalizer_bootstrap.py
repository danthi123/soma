import pytest
import torch
from torch import nn

from soma.core.config import SOMAConfig
from soma.io.chat_head import ChatHead
from soma.io.verbalizer import SomaVerbalizer, VerbalizerSpec
from soma.system import SOMA
from soma.training.verbalizer_bootstrap import VerbalizerTrainer


# Reuse the Phase 3 mock shape — thinnest possible HF-like surface.
class _TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.lm_head = nn.Linear(d_model, vocab)
        self.config = type("Cfg", (), {"hidden_size": d_model, "vocab_size": vocab})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def forward(self, *, inputs_embeds, attention_mask=None, labels=None, **_):
        logits = self.lm_head(inputs_embeds)
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
        vocab_size=16,
        text_embed_dim=8,
        max_nodes=32,
        initial_associator_count=2,
        initial_integrator_count=1,
        max_input_tokens=8,
        max_output_tokens=4,
        seed=0,
    )


def _fresh_trainer() -> VerbalizerTrainer:
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
    chat_head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    return VerbalizerTrainer(
        soma=soma,
        verbalizer=verbalizer,
        chat_head=chat_head,
        config=cfg,
    )


def test_trainer_stores_components():
    t = _fresh_trainer()
    assert t.soma is not None
    assert t.verbalizer is not None
    assert t.chat_head is not None


def test_trainer_creates_adam_on_verbalizer_only():
    t = _fresh_trainer()
    assert isinstance(t.optim, torch.optim.Adam)
    verb_param_ids = {id(p) for p in t.verbalizer.parameters()}
    for group in t.optim.param_groups:
        for p in group["params"]:
            assert id(p) in verb_param_ids, "optimizer contains non-verbalizer param"


def test_trainer_does_not_unfreeze_chat_head():
    t = _fresh_trainer()
    assert not any(p.requires_grad for p in t.chat_head.model.parameters())


def test_trainer_uses_config_lr():
    t = _fresh_trainer()
    assert t.optim.param_groups[0]["lr"] == t.config.verbalizer_lr


def test_trainer_raises_if_chat_head_is_not_frozen():
    """Defensive: if someone hands in a ChatHead whose params got unfrozen,
    the trainer must refuse to start rather than silently train the LLM."""
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
    chat_head = ChatHead(
        model=_TinyCausalLM(vocab=32, d_model=16),
        tokenizer=_TinyTokenizer(),
    )
    # Poison: unfreeze the model after ChatHead init.
    for p in chat_head.model.parameters():
        p.requires_grad = True

    with pytest.raises(ValueError, match="frozen"):
        VerbalizerTrainer(
            soma=soma,
            verbalizer=verbalizer,
            chat_head=chat_head,
            config=cfg,
        )
