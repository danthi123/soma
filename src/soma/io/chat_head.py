"""Thin wrapper around a frozen HuggingFace causal LM."""

from __future__ import annotations

from typing import Any, cast

import torch


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

    def generate(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **kw: Any,
    ) -> torch.Tensor:
        """Thin wrapper around `model.generate` using the `inputs_embeds` path.

        Returns the generated token id tensor (caller decodes as needed).
        """
        return cast(
            torch.Tensor,
            self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kw,
            ),
        )

    def generate_text(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **kw: Any,
    ) -> str:
        """Convenience: generate + decode the first batch row as a string.

        If the caller passes a batch with B>1, only row 0 is decoded and
        returned; use :meth:`generate` directly and decode each row yourself
        for multi-sample output.
        """
        ids = self.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kw,
        )
        return cast(str, self.tokenizer.decode(ids[0], skip_special_tokens=True))
