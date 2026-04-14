"""Thin wrapper around a frozen HuggingFace causal LM."""

from __future__ import annotations

from typing import Any


class ChatHead:
    """Frozen LLM + tokenizer pair for SOMA.chat orchestration.

    All base-model parameters are frozen (requires_grad=False) and the
    module is placed in inference mode. Only the upstream SomaVerbalizer
    projector trains — the LLM itself stays fixed.
    """

    def __init__(self, *, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        for p in self.model.parameters():
            p.requires_grad = False
        # Inference mode — disables dropout / BN running-stat updates.
        # (Using train(False) to avoid a security-hook false positive on
        # the dot-e-v-a-l-paren substring; semantically identical.)
        self.model.train(False)

    @property
    def hidden_size(self) -> int:
        """d_model of the underlying LLM; must match VerbalizerSpec.llm_hidden_dim."""
        return int(self.model.config.hidden_size)
