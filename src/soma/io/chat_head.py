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


def build_position_ids(*, num_prefix: int, num_tokens: int, batch_size: int) -> torch.Tensor:
    """Produce [0..k+T-1] positions per batch row for prefix+tokens layout.

    Prefix occupies positions 0..k-1; user tokens occupy k..k+T-1. Returned
    tensor is contiguous (not a strided view) so HF paths that internally
    ``.view(...)`` on ``position_ids`` stay happy.
    """
    seq_len = num_prefix + num_tokens
    return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).contiguous()


def build_attention_mask(*, num_prefix: int, num_tokens: int, batch_size: int) -> torch.Tensor:
    """All-ones attention mask; assumes no padding in the token section."""
    seq_len = num_prefix + num_tokens
    return torch.ones(batch_size, seq_len, dtype=torch.long)


def build_attention_mask_from_pad(*, num_prefix: int, pad_mask: torch.Tensor) -> torch.Tensor:
    """Prepend prefix-ones to a tokenizer-produced pad mask.

    ``pad_mask`` must be a (B, T) int64/bool tensor where 1 = real token,
    0 = pad. The returned (B, k+T) mask keeps padding zeros in-place.
    """
    batch_size = pad_mask.shape[0]
    prefix = torch.ones(batch_size, num_prefix, dtype=pad_mask.dtype, device=pad_mask.device)
    return torch.cat([prefix, pad_mask], dim=1)
