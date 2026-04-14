"""Bootstrap training loop for the SomaVerbalizer.

Only the verbalizer's projector trains. SOMA stays frozen (via
``torch.no_grad`` during state production — enforced in later tasks),
and the ChatHead's LLM stays frozen (via Phase 3's ``requires_grad=False``
contract — enforced at trainer init).
"""

from __future__ import annotations

from typing import Any, cast

import torch

from soma.core.config import SOMAConfig


class VerbalizerTrainer:
    """Bootstrap-trains a SomaVerbalizer against a frozen LLM via LM loss.

    The only trainable parameters are the verbalizer's; SOMA's Hebbian
    graph and the ChatHead's LLM both stay fixed by construction.
    """

    def __init__(
        self,
        *,
        soma: Any,
        verbalizer: Any,
        chat_head: Any,
        config: SOMAConfig,
    ) -> None:
        self.soma = soma
        self.verbalizer = verbalizer
        self.chat_head = chat_head
        self.config = config

        # Sanity: ChatHead must already be frozen (Phase 3 invariant).
        if any(p.requires_grad for p in self.chat_head.model.parameters()):
            raise ValueError(
                "ChatHead model parameters must be frozen before bootstrap "
                "training. Construct ChatHead via its __init__ (which sets "
                "requires_grad=False on every param) and do not unfreeze."
            )

        # Adam on verbalizer only — SOMA has its own Hebbian learning path
        # that runs inside soma.step(); that path must not be reached during
        # bootstrap (text_to_state in T4 wraps soma.step in torch.no_grad).
        self.optim = torch.optim.Adam(
            self.verbalizer.parameters(),
            lr=config.verbalizer_lr,
        )


def compute_lm_loss(
    *,
    chat_head: Any,
    prefix: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Causal-LM cross-entropy conditioned on a continuous prefix.

    ``prefix`` is the (B, k, d_model) soft-prompt from the verbalizer;
    ``token_ids`` is the (B, T) target text. Returns a scalar loss.

    Prefix positions are masked via ``-100`` labels so only token-position
    predictions contribute to the loss. Gradient flows through ``prefix``
    back into the verbalizer; the LLM stays frozen (no grads recorded).
    """
    batch_size, num_prefix, _ = prefix.shape
    _, num_tokens = token_ids.shape

    token_embeds = chat_head.model.get_input_embeddings()(token_ids)
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

    prefix_labels = torch.full(
        (batch_size, num_prefix),
        -100,
        dtype=torch.long,
        device=token_ids.device,
    )
    labels = torch.cat([prefix_labels, token_ids], dim=1)

    attn_mask = torch.ones(
        batch_size,
        num_prefix + num_tokens,
        dtype=torch.long,
        device=token_ids.device,
    )

    out = chat_head.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        labels=labels,
    )
    return cast(torch.Tensor, out.loss)
